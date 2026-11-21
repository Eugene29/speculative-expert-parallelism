"""Shared helpers for the benchmarks: launch, recorded routing, inputs and timing.

Usage:
    from benchmark.utils import init_dist, check_node_layout, load_routing, make_inputs, time_ms
    from benchmark.utils import expert_counts, busiest_remote, replicate_hot
"""
import os
import socket
from collections.abc import Callable

import torch
import torch.distributed as dist


def init_dist() -> tuple[int, int, torch.device]:
    """Init NCCL from SLURM env.

    Returns:
        rank: this rank.
        world: number of ranks.
        device: this rank's GPU.
    """
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    return rank, world, device


def check_node_layout(gpus_per_node: int, min_nodes: int = 2) -> list[str]:
    """Check ranks fill at least min_nodes nodes in blocks of gpus_per_node.

    Block order (ranks 0-3 on node 0, 4-7 on node 1, ...) is what makes
    rank + gpus_per_node the same local rank on a different node.

    Returns:
        hosts: hostname of every rank.
    """
    world = dist.get_world_size()
    if world % gpus_per_node != 0:
        raise RuntimeError(f"{world} ranks is not a multiple of {gpus_per_node} per node.")

    num_nodes = world // gpus_per_node
    if num_nodes < min_nodes:
        raise RuntimeError(f"Expected at least {min_nodes} nodes, got {num_nodes}. "
                           f"Use srun -N {min_nodes} or more.")

    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())

    node_hosts = hosts[::gpus_per_node]
    if len(set(node_hosts)) != num_nodes:
        raise RuntimeError(f"Some nodes share a host: {node_hosts}.")
    return hosts


def load_routing(
    run_dir: str,
    layer: int,
    rank: int,
    world: int,
    num_experts: int,
) -> torch.Tensor:
    """Load this rank's share of one layer's recorded routing.

    run_dir/experts_rank{r}.pt holds a [num_layers, num_tokens, k] tensor of expert
    ids for recorded rank r, written by scripts/ds_moe.py --record. With more ranks than
    recordings, each recording's tokens are split across world // recordings ranks, in
    rank order. With fewer, each rank replays recordings // world recordings back to
    back. Either way the global workload stays the same.

    Returns:
        route: [num_tokens, k] expert ids of this rank's tokens, as int64.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    if recorded == 0 or (world % recorded != 0 and recorded % world != 0):
        raise RuntimeError(f"{world} ranks and {recorded} recordings in {run_dir} "
                           f"don't divide each other.")

    shards, merged = max(world // recorded, 1), max(recorded // world, 1)
    first = rank // shards * merged
    route = torch.cat([
        torch.load(f"{run_dir}/experts_rank{r}.pt", map_location="cpu")[layer]
        for r in range(first, first + merged)
    ])
    route = route.tensor_split(shards)[rank % shards].long()

    if route.numel() > 0 and (route.min() < 0 or route.max() >= num_experts):
        raise RuntimeError(f"Recorded expert ids span {route.min()}..{route.max()}, "
                           f"outside 0..{num_experts - 1}.")
    return route


def expert_counts(run_dir: str, num_experts: int) -> torch.Tensor:
    """Count the copies every expert gets in each recording and layer.

    A recording is one batch of tokens: run_dir/experts_rank{r}.pt, as load_routing
    reads it.

    Returns:
        counts: [num_recordings, num_layers, num_experts] copies routed to each
            expert, as int64.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    counts = []
    for r in range(recorded):
        route = torch.load(f"{run_dir}/experts_rank{r}.pt", map_location="cpu").long()
        layer = torch.arange(route.shape[0]).view(-1, 1, 1)
        flat = (layer * num_experts + route).flatten()
        counts.append(flat.bincount(minlength=route.shape[0] * num_experts).view(-1, num_experts))
    return torch.stack(counts)


def busiest_remote(
    counts: torch.Tensor,
    num_hot: int,
    rank: int,
    world: int,
    gpus_per_node: int,
) -> torch.Tensor:
    """Pick the busiest experts that this rank's node does not hold.

    A replica of an expert the node already holds would move no copy, so those
    experts are never picked.

    Args:
        counts: [num_experts] copies routed to each expert, the same on every rank of
            a node.

    Returns:
        hot: [num_hot] expert ids on other nodes, busiest first.
    """
    per_node = counts.numel() // world * gpus_per_node
    node = rank // gpus_per_node
    counts = counts.clone()
    counts[node * per_node : (node + 1) * per_node] = -1
    return counts.topk(num_hot).indices


def replicate_hot(
    route: torch.Tensor,
    hot: torch.Tensor,
    rank: int,
    world: int,
    num_experts: int,
    gpus_per_node: int,
) -> tuple[torch.Tensor, int]:
    """Reroute this rank's copies of hot experts to replicas on its own node.

    Every node gets one replica of each hot expert, dealt over its ranks in turn. A
    replica is a new expert id after its rank's own experts, so every rank still holds
    one contiguous block of the same size and the dispatch functions run unchanged. A
    copy whose expert already lives on the token's node keeps its route.

    Args:
        route: [num_tokens, k] experts each token on this rank is routed to.
        hot: [num_hot] expert ids to replicate, the same on every rank of a node, as
            busiest_remote() picks them.

    Returns:
        route: [num_tokens, k] routes in the grown numbering.
        num_experts: Experts plus replicas across all ranks.
    """
    per_rank = num_experts // world
    grown = per_rank + -(-hot.numel() // gpus_per_node)
    node_first = rank // gpus_per_node * gpus_per_node

    index = torch.full((num_experts,), -1, device=route.device)
    index[hot] = torch.arange(hot.numel(), device=route.device)
    index = index[route]
    replica = (node_first + index % gpus_per_node) * grown + per_rank + index // gpus_per_node
    home = route // per_rank * grown + route % per_rank
    remote = route // per_rank // gpus_per_node != rank // gpus_per_node
    return torch.where((index >= 0) & remote, replica, home), world * grown


def make_inputs(
    route: torch.Tensor,
    H: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make random tokens and weights for the given routing.

    The recording holds only expert ids, so the token values are made up; they don't
    affect communication cost.

    Returns:
        tokens: [num_tokens, H] bf16 tokens.
        weights: [num_tokens, k] weights, each row summing to 1.
    """
    num_tokens, k = route.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.randn(num_tokens, H, generator=gen, device=device, dtype=torch.bfloat16)
    logits = torch.randn(num_tokens, k, generator=gen, device=device)
    weights = logits.softmax(dim=-1)
    return tokens, weights


def time_ms(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
) -> list[float]:
    """Time fn on every rank and report each call's slowest rank. Collective.

    Runs warmup untimed calls, then iterations timed ones. All ranks line up at a
    barrier before each timed call.

    Returns:
        times: Milliseconds of each timed call on the rank that took longest for it.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        dist.barrier()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(starts, ends)],
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    return times.tolist()
