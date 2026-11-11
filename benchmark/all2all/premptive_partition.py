"""Benchmark a co-activation-aware expert placement against contiguous placement.

Bisects the experts so that few tokens need experts from both halves, then times
the plain alltoall and X-MoE's Redundancy-Bypassing Dispatch (RBD) under each
placement.

Usage: premptive_partition.py RUN_DIR LAYER (one process per GPU under SLURM).

srun --ntasks-per-node=4 \
  python -u benchmark/all2all/premptive_partition.py logs/2026-10-16_14-07-50 0
"""
import os
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import numpy as np
import networkx as nx
from collections import Counter
from itertools import combinations
from benchmark.all2all.even_vs_uneven import (
    benchmark, get_send_recv_splits, init_dist, H, WARMUP, ITERATIONS)
import pulp


# Step 1: Count how often each group of experts appears together
def count_groups(routing):
    """Count tokens per distinct expert set.

    Args:
        routing: [num_tokens, k] tensor of expert ids.

    Returns:
        Counter mapping each sorted tuple of unique expert ids to the number of
        tokens routed to exactly that set.
    """
    groups = Counter()
    for row in routing.tolist():  # tolist(): 0-d tensors hash by id, never dedupe
        group = tuple(sorted(set(row)))
        groups[group] += 1
    return groups          # {(3, 17, 42, ...): count, ...}

# Step 2: Turn groups into a weighted graph (pairs that appear together)
def build_graph(groups, n_experts=256):
    """Build the expert co-activation graph.

    Args:
        groups: Counter from count_groups().
        n_experts: Number of experts; nodes are 0..n_experts-1.

    Returns:
        nx.Graph where edge (a, b) is weighted by the number of tokens routed
        to both a and b.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_experts))
    for group, count in groups.items():
        for a, b in combinations(group, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += count
            else:
                G.add_edge(a, b, weight=count)
    return G

def exact_split(groups, n_experts=256, init=None):
    """Bisect the experts with an ILP that minimizes split tokens.

    Unlike split_in_half(), each split token counts once instead of once per cut
    edge. CBC stops after 300 s, so the result may be feasible but not optimal.

    Args:
        groups: Counter from count_groups().
        n_experts: Number of experts.
        init: Set of experts in half A to warm-start from. Optional, but without
            it CBC usually finds no feasible point in time.

    Returns:
        Set of experts in half A.

    Raises:
        RuntimeError: If CBC finds no solution.
    """
    prob = pulp.LpProblem("bisection", pulp.LpMinimize)

    # x[e] = 1 if expert e is in half A
    x = [pulp.LpVariable(f"x{e}", cat="Binary") for e in range(n_experts)]
    # y[g] = 1 if group g is split across halves. Continuous is still exact (x is
    # binary and y is minimized) and leaves CBC 256 binaries instead of ~2.8k.
    y = {g: pulp.LpVariable(f"y{i}", lowBound=0, upBound=1) for i, g in enumerate(groups)}

    # Goal: minimize the number of split tokens
    prob += pulp.lpSum(count * y[g] for g, count in groups.items())

    # Exactly half the experts in A
    prob += pulp.lpSum(x) == n_experts // 2

    # A group is split if any member differs from its first member
    for g in groups:
        first = g[0]
        for e in g[1:]:
            prob += y[g] >= x[e] - x[first]
            prob += y[g] >= x[first] - x[e]

    # Without a warm start CBC finds no feasible point within the time limit and
    # x.value() is garbage (e.g. a 254/2 "split").
    if init is not None:
        for e in range(n_experts):
            x[e].setInitialValue(int(e in init))
        for g in groups:
            y[g].setInitialValue(int(len({e in init for e in g}) > 1))

    prob.solve(pulp.PULP_CBC_CMD(timeLimit=300, msg=False, warmStart=init is not None))
    if prob.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        raise RuntimeError(f"CBC found no solution: {pulp.LpSolution[prob.sol_status]}")
    return {e for e in range(n_experts) if x[e].value() > 0.5}

# Step 3: Split into two halves of 128, keeping heavy pairs together
def split_in_half(G, tries=20):
    """Bisect G with Kernighan-Lin, keeping the best of several seeds.

    Minimizes the weighted edge cut, a proxy for split tokens.

    Args:
        G: Graph from build_graph().
        tries: Number of seeds to try.

    Returns:
        (A, B) sets of experts.
    """
    best_halves, best_cut = None, float("inf")
    for seed in range(tries):
        A, B = nx.algorithms.community.kernighan_lin_bisection(
            G, weight="weight", seed=seed)
        cut = nx.cut_size(G, A, B, weight="weight")
        if cut < best_cut:
            best_halves, best_cut = (A, B), cut
    return best_halves

# Step 4: How many tokens need experts from both halves?
def count_split_tokens(groups, A):
    """Return how many tokens in groups use experts both inside and outside A."""
    split = 0
    for group, count in groups.items():
        in_A = [e in A for e in group]
        if any(in_A) and not all(in_A):
            split += count
    return split

# Step 5: X-MoE Redundancy-Bypassing Dispatch (RBD), arXiv 2508.13337.
# A token routed to several experts on the same node crosses the network once: a
# randomly chosen "pilot" copy goes to one of those experts' GPUs (stage 1, the only
# alltoall that touches inter-node links), then the node-local "replicas" are copied
# from the pilot and sent to the remaining experts' GPUs (stage 2, intra-node alltoall).
def rbd_counts(ids, E, world, gpus_per_node, gen):
    """Count the rows this rank's tokens add to each RBD stage.

    Args:
        ids: [num_tokens, k] tensor of expert ids. Experts are placed
            contiguously, E // world per rank.
        E: Number of experts.
        world: Number of ranks.
        gpus_per_node: Ranks per node.
        gen: torch.Generator for the random pilot choice.

    Returns:
        stage1: [world] pilots this rank sends, indexed by destination rank.
        stage2: [world, world] replicas that this rank's pilots induce, indexed
            [pilot rank, replica rank]. Sum across ranks for the global matrix.
    """
    ids = ids.long()
    dst_rank = ids // (E // world)
    dst_node = dst_rank // gpus_per_node
    # Random pilot per (token, node); X-MoE notes always taking the lowest expert id
    # skews stage-1 load.
    keys = torch.rand(ids.shape, generator=gen)

    stage1 = torch.zeros(world, dtype=torch.long)
    stage2 = torch.zeros(world * world, dtype=torch.long)
    for node in dst_node.unique().tolist():
        on_node = dst_node == node
        has = on_node.any(1)
        pilot_slot = torch.where(on_node, keys, 2.0).argmin(1, keepdim=True)
        pilot_rank = dst_rank.gather(1, pilot_slot)
        stage1 += torch.bincount(pilot_rank[has, 0], minlength=world)

        replica = on_node.scatter(1, pilot_slot, False)
        src = pilot_rank.expand_as(dst_rank)[replica]
        stage2 += torch.bincount(src * world + dst_rank[replica], minlength=world * world)
    return stage1, stage2.view(world, world)

def gather_counts(send, world):
    """All-gather each rank's send counts into one matrix.

    Args:
        send: [world] tensor of rows this rank sends to each rank.
        world: Number of ranks.

    Returns:
        [src, dst] CPU tensor.
    """
    counts = [torch.empty_like(send) for _ in range(world)]
    dist.all_gather(counts, send)
    return torch.stack(counts).cpu()

def get_rbd_splits(ids, E, rank, world, gpus_per_node, device, gen, scale=1):
    """Compute global RBD row counts and this rank's alltoall splits.

    Collective: every rank must call it.

    Args:
        ids, E, world, gpus_per_node, gen: As in rbd_counts().
        rank: This rank.
        device: Device for the collectives.
        scale: Factor applied to every row count.

    Returns:
        C1: [src, dst] row counts for stage 1.
        S2: [src, dst] row counts for stage 2.
        splits: (send1, recv1, send2, recv2) for benchmark_rbd(). The stage-2
            splits are indexed by rank within the node.
    """
    stage1, stage2 = rbd_counts(ids, E, world, gpus_per_node, gen)
    C1 = gather_counts(stage1.to(device), world) * scale
    S2 = stage2.to(device)
    dist.all_reduce(S2)
    S2 = S2.cpu() * scale

    lo = rank // gpus_per_node * gpus_per_node
    local = slice(lo, lo + gpus_per_node)
    assert S2[rank].sum() == S2[rank, local].sum(), "replicas must stay on-node"
    splits = (C1[rank].tolist(), C1[:, rank].tolist(),
              S2[rank, local].tolist(), S2[local, rank].tolist())
    return C1, S2, splits

def report_traffic(name, C, gpus_per_node):
    """Print inter- and intra-node traffic in MiB.

    Assumes bf16 rows of width H and excludes self-sends. Unlike benchmark(),
    uses each rank's actual counts instead of assuming they all match rank 0.

    Args:
        name: Label for the printed line.
        C: [src, dst] row-count matrix.
        gpus_per_node: Ranks per node.
    """
    node = torch.arange(C.shape[0]) // gpus_per_node
    inter = C[node[:, None] != node[None, :]].sum().item()
    intra = C.sum().item() - inter - C.diag().sum().item()
    mib = H * 2 / 2**20  # bf16 rows
    print(f"  {name}: {inter * mib:.1f} MiB inter-node, {intra * mib:.1f} MiB intra-node (off-rank)")

def benchmark_rbd(name, send1, recv1, send2, recv2, local_group):
    """Time the two-stage RBD dispatch with random bf16 payloads.

    Rank 0 prints the mean time per stage over ITERATIONS runs (after WARMUP),
    taking the max across ranks.

    Args:
        name: Label for the printed line.
        send1, recv1: Stage-1 splits over the global group.
        send2, recv2: Stage-2 splits over local_group.
        local_group: Process group of this rank's node.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    x1 = torch.randn(sum(send1), H, device=device, dtype=torch.bfloat16)
    y1 = torch.empty(sum(recv1), H, device=device, dtype=x1.dtype)
    # Replicas are rebuilt from pilots this rank received; which pilot doesn't matter for timing.
    idx = torch.randint(max(sum(recv1), 1), (sum(send2),), device=device)
    x2 = torch.empty(sum(send2), H, device=device, dtype=x1.dtype)
    y2 = torch.empty(sum(recv2), H, device=device, dtype=x1.dtype)

    def dispatch(ev=None):
        """Run one RBD dispatch.

        Args:
            ev: Optional CUDA events [start, after stage 1, after stage 2] to record.
        """
        if ev: ev[0].record()
        dist.all_to_all_single(y1, x1, recv1, send1)                 # stage 1: pilots
        if ev: ev[1].record()
        torch.index_select(y1, 0, idx, out=x2)                       # rebuild replicas
        dist.all_to_all_single(y2, x2, recv2, send2, group=local_group)  # stage 2
        if ev: ev[2].record()

    for _ in range(WARMUP):
        dispatch()

    events = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(ITERATIONS)]
    dist.barrier()
    torch.cuda.synchronize()
    for ev in events:
        dispatch(ev)
    torch.cuda.synchronize()

    s1 = sum(ev[0].elapsed_time(ev[1]) for ev in events) / ITERATIONS
    s2 = sum(ev[1].elapsed_time(ev[2]) for ev in events) / ITERATIONS
    ms = torch.tensor([s1, s2, s1 + s2], device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)

    if rank == 0:
        s1, s2, tot = ms.tolist()
        print(f"{name}: {tot:.3f} ms | stage1 (pilots) {s1:.3f} ms | stage2 (replicas) {s2:.3f} ms")

def run_placement(name, ids, rank , local_group, scale):
    """Benchmark one expert placement with the plain alltoall and with RBD.

    Rank 0 also prints the traffic of each. Reads E, world, device and
    gpus_per_node from module globals.

    Args:
        name: Label for the printed lines.
        ids: [num_tokens, k] tensor of expert ids under this placement.
        rank: This rank.
        local_group: Process group of this rank's node.
        scale: Factor applied to every row count.
    """
    gen = torch.Generator().manual_seed(rank)  # RBD pilot choice
    
    send_splits, recv_splits = get_send_recv_splits(ids, E, rank, world, device, scale)
    benchmark(name, send_splits, recv_splits)

    C1, S2, rbd_splits = get_rbd_splits(ids, E, rank, world, gpus_per_node, device, gen, scale)
    benchmark_rbd(f"{name} + RBD", *rbd_splits, local_group)

    C = gather_counts(torch.bincount(ids.long().flatten().to(device) // (E // world),
                                     minlength=world), world) * scale
    if rank == 0:
        report_traffic("plain alltoall", C, gpus_per_node)
        report_traffic("RBD stage 1   ", C1, gpus_per_node)
        report_traffic("RBD stage 2   ", S2, gpus_per_node)

def solve_optimal_split(groups, E, total):
    """Bisect the experts on rank 0 and broadcast the halves to every rank.

    Uses split_in_half(). Solving once keeps the placement identical on every
    rank. Rank 0 prints the solve time and the split-token count. Reads rank
    from module globals.

    Args:
        groups: Counter from count_groups(), over all ranks.
        E: Number of experts.
        total: Total number of tokens, for the printed percentage.

    Returns:
        (A, B) sets of experts.
    """
    halves = [None, None]

    if rank == 0:
        strt = time.time()
        G = build_graph(groups, E)
        halves[0], halves[1] = split_in_half(G)
        dur = time.time() - strt
        print(f"Time to find optimal version: {dur}", flush=True)

    dist.broadcast_object_list(halves, src=0)
    A, B = halves

    if rank == 0:
        split = count_split_tokens(groups, A)
        print(f"Half A: {len(A)} experts, Half B: {len(B)} experts")
        print(f"Tokens split across halves: {split} / {total} ({split / total:.1%})")
    
    return A, B

def mock_prempetive_placement(routing, A, B):
    """Renumber expert ids so half A comes first, then half B.

    Expert A_sorted[i] becomes i and B_sorted[j] becomes len(A) + j, so with
    contiguous placement half A lands on the first half of the ranks. Reads
    A_sorted and B_sorted from module globals.

    Args:
        routing: [num_tokens, k] tensor of expert ids.
        A, B: Sets of experts in each half.

    Returns:
        [num_tokens, k] tensor of renumbered expert ids.

    Raises:
        IndexError: If an expert is in neither half.
    """
    routing_new = []
    for route in routing:
        new_experts = []

        for expert in route:
            expert_id = expert.item()
            if expert_id in A:
                new_expert = A_sorted.index(expert_id)
            elif expert_id in B:
                new_expert = len(A_sorted) + B_sorted.index(expert_id)
            else:
                raise IndexError()
            
            new_experts.append(new_expert)
        
        routing_new.append(new_experts)
        
    routing_new = torch.tensor(routing_new)
    return routing_new

# Run it
if __name__ == "__main__":
    run_dir = sys.argv[1]
    layer = int(sys.argv[2])

    E, H = 256, 2048
    SCALE = 1

    rank, world, device = init_dist()
    gpus_per_node = world // int(os.environ["SLURM_NNODES"])
    local_group, _ = dist.new_subgroups(group_size=gpus_per_node)  # RBD stage 2

    # Routing: array of shape [num_tokens, k], each row = experts one token used
    # With more ranks than recorded, shard each recording's tokens across
    # world // recorded ranks so the global workload stays the same.
    recorded = sum(f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir))
    assert world % recorded == 0, f"world {world} not a multiple of {recorded} recorded ranks"
    shards = world // recorded
    routing = torch.load(f"{run_dir}/experts_rank{rank // shards}.pt", map_location="cpu")[layer]
    routing = routing.tensor_split(shards)[rank % shards]

    # Convert routing into groups
    # Gather every rank's groups so the split is computed on the global routing
    all_groups = [None] * world
    dist.all_gather_object(all_groups, count_groups(routing))
    groups = sum(all_groups, Counter())
    total = sum(groups.values())
    
    # 1. run the regular version
    A_contiguous = set(range(E // 2)) # Contiguous split in half
    if rank == 0:
        split = count_split_tokens(groups, A_contiguous)
        print(f"Contiguous: tokens split across halves: {split} / {total} ({split / total:.1%})")
    run_placement("regular", routing, rank, local_group, SCALE)

    # 2. run the optimal version (non-exact)
    A, B = solve_optimal_split(groups, E, total)
    # sort each half split
    A_sorted, B_sorted = sorted(list(A)), sorted(list(B))
    routing_new = mock_prempetive_placement(routing, A, B)
    # map orderd number to the split
    run_placement("optimized-split (non-exact)", routing_new, rank, local_group, SCALE)

    # # Rank 0 hosts the TCPStore; don't let it exit while others are still solving.
    dist.barrier()
    dist.destroy_process_group()