"""Choose an expert placement that makes RBD (or plain all2all) cheaper, given routing.

Assumes the routing of a step is known before it runs and moving experts is free,
which gives an upper bound on what placement can save. A placement maps each
expert id to a slot; slot p lives on rank p // experts_per_rank, as spec_ep.rbd
expects. Running a layer under a placement only means relabeling its routing:
route -> slots[route].

The cost model estimates a whole forward pass (plan, dispatch, identity experts and
combine) from per-rank row counts:
    ms = fixed + NET_ROW_MS_PER_NODE * nodes * busiest rank's network rows
               + NVLINK_ROW_MS * busiest rank's NVLink rows
where each count is the larger of rows sent and received, the NVLink rows are those
in the all2all that also carries the network rows (RBD stage 1, or plain's only
all2all), and fixed differs for RBD and plain. Copies a rank sends to itself are
free. The constants are a non-negative least-squares fit to 132 measured
end-to-end times on Perlmutter (36 on 2 nodes, 96 on 4 nodes, 4 A100 per node,
H=2048, bf16, recorded layers 0, 20 and 39, plain and RBD with every pilot rule,
with and without placement), with leave-one-out error 0.53 ms on average (about 7%)
and 2.8 ms at most. The network cost per row doubled from 2 to 4 nodes; the model
assumes it keeps growing linearly with the node count, which is untested beyond 4.
Stage 2 rows got zero weight in the fit, so they are counted but not charged.
The "even" pilot rule's additional count all-gather is not included in this fit. The
model overstated placement's measured gain about 2x on 4 nodes.

even_placement balances routed expert-input rows across all ranks, keeping exactly
E / world experts on each rank. It ignores network locality and pilot policy.

Five searches build on pilot_swap_delta, an exact all-pairs formula for the change
in pilots from swapping two experts:
    colocate: makes the single best pilot-cutting swap until none helps, ignoring
        balance.
    partition_placement: Kernighan-Lin partitioning of the token hypergraph, with
        each token's source node as a fixed vertex, so its connectivity objective
        is the pilot count. Node loads stay within a cap, then each node's experts
        are balanced over its ranks.
    kahypar_placement: the same hypergraph partitioned by KaHyPar, then greedy
        swaps (swap_refine), with or without a node load cap.
    hypergraph_placement: the same hypergraph partitioned by Mt-KaHyPar, then
        repaired to E / nodes experts and the load cap per node, optionally
        refined by Kernighan-Lin passes.
    greedy_placement: swaps pairs of experts while the cost model goes down. Each
        round it scores a few candidate swaps exactly and keeps the best: the swaps
        that remove the most pilots, swaps that move an expert off the bottleneck
        rank, and a few random swaps. It can start from the other two.

Usage:
    from spec_ep.placement import greedy_placement, partition_placement
    start = partition_placement(route, source_rank, E, world, gpus_per_node)
    slots, _ = greedy_placement(route, source_rank, E, world, gpus_per_node,
                                start=start)
    route = slots[route]
"""
from typing import NamedTuple

import torch

# Fitted end-to-end cost model, in milliseconds, for H=2048 bf16 rows. See above.
NET_ROW_MS_PER_NODE = 9.92e-5
NVLINK_ROW_MS = 2.73e-4
RBD_FIXED_MS = 2.97
PLAIN_FIXED_MS = 2.36


class RankLoads(NamedTuple):
    """Rows each rank moves in one direction, for a batch of placements.

    Every field is [num_placements, world].

    Attributes:
        net_out: Rows sent to other nodes.
        net_in: Rows received from other nodes.
        nvl_out: Rows sent to other ranks on this node, in the all2all that also
            carries the network rows (RBD stage 1, or plain's only all2all).
        nvl_in: Rows received from other ranks on this node, in that all2all.
        stage2_out: RBD replica rows sent to other ranks on this node. 0 for plain.
        stage2_in: RBD replica rows received from other ranks on this node.
    """

    net_out: torch.Tensor
    net_in: torch.Tensor
    nvl_out: torch.Tensor
    nvl_in: torch.Tensor
    stage2_out: torch.Tensor
    stage2_in: torch.Tensor


def _count(rank: torch.Tensor, mask: torch.Tensor, world: int) -> torch.Tensor:
    """Count masked entries per (placement, rank).

    Args:
        rank: [M, ...] rank of each entry.
        mask: [M, ...] entries to count.
        world: Number of ranks.

    Returns:
        counts: [M, world] entries per rank.
    """
    M = rank.shape[0]
    flat = (rank + world * torch.arange(M, device=rank.device).view(-1, *[1] * (rank.dim() - 1)))
    return torch.bincount(flat[mask], minlength=M * world).view(M, world)


def rank_loads(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    rbd: bool = True,
    pilot: str = "rail",
) -> RankLoads:
    """Count the rows each rank sends and receives, for each placement.

    Args:
        route: [num_tokens, k] experts of every token in the EP group.
        source_rank: [num_tokens] rank each token starts on.
        slots: [M, E] slot of each expert, for M placements.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        rbd: Count RBD's traffic if True, plain all2all's if False.
        pilot: RBD's pilot rule, as in spec_ep.rbd.RBD_combine_dispatch.

    Returns:
        loads: Rows per rank for each placement.
    """
    M, E = slots.shape
    world = E // experts_per_rank
    G = gpus_per_node
    k = route.shape[1]
    expert_rank = slots[:, route] // experts_per_rank  # [M, T, k]
    node = expert_rank // G
    source = source_rank.view(1, -1, 1).expand_as(expert_rank)
    direct = node == source // G
    nvl = direct & (expert_rank != source)
    zeros = torch.zeros(M, world, dtype=torch.long, device=route.device)

    if not rbd:
        remote = ~direct
        return RankLoads(
            _count(source, remote, world),
            _count(expert_rank, remote, world),
            _count(source, nvl, world),
            _count(expert_rank, nvl, world),
            zeros,
            zeros,
        )

    # The pilot of a token's copies on a node is its first copy there, in route order.
    slot_index = torch.arange(k, device=route.device)
    pilot_slot = torch.zeros_like(expert_rank)
    for n in range(world // G):
        on_n = node == n
        first = torch.where(on_n, slot_index, k).amin(dim=-1, keepdim=True)
        pilot_slot = torch.where(on_n, first.expand_as(pilot_slot), pilot_slot)
    is_pilot = (pilot_slot == slot_index) & ~direct
    if pilot == "even":
        source_order = source_rank.argsort(stable=True)
        pilot_rank = node * G
        for n in range(world // G):
            on_node = node == n
            has_pilot = (on_node & is_pilot).any(dim=-1)
            ordinal = has_pilot[:, source_order].long().cumsum(dim=-1) - 1
            position = torch.empty_like(ordinal)
            position[:, source_order] = ordinal
            pilot_rank = torch.where(on_node, n * G + position.unsqueeze(-1) % G, pilot_rank)
    elif pilot == "rail":
        pilot_rank = node * G + source % G
    elif pilot == "most":
        held = (expert_rank.unsqueeze(-1) == expert_rank.unsqueeze(-2)).sum(dim=-1)
        score = held * G + (G - 1 - (expert_rank - source) % G)
        same_node = node.unsqueeze(-1) == node.unsqueeze(-2)
        best = torch.where(same_node, score.unsqueeze(-2), -1).argmax(dim=-1)
        pilot_rank = expert_rank.gather(-1, best)
    elif pilot == "first":
        pilot_rank = expert_rank.gather(-1, pilot_slot)
    else:
        raise ValueError(f'Unknown pilot rule: {pilot!r}.')
    replica = ~direct & (expert_rank != pilot_rank)
    return RankLoads(
        _count(source, is_pilot, world),
        _count(pilot_rank, is_pilot, world),
        _count(source, nvl, world),
        _count(expert_rank, nvl, world),
        _count(pilot_rank, replica, world),
        _count(expert_rank, replica, world),
    )


def _row_costs(num_nodes: int) -> tuple[float, float]:
    """Per-row network and NVLink costs of the fitted model.

    Args:
        num_nodes: Nodes in the EP group.

    Returns:
        net_ms: Cost of one network row.
        nvlink_ms: Cost of one NVLink row.
    """
    return NET_ROW_MS_PER_NODE * num_nodes, NVLINK_ROW_MS


def estimate_ms(
    loads: RankLoads,
    num_nodes: int,
    rbd: bool = True,
    mean_weight: float = 0.0,
) -> torch.Tensor:
    """Estimate a forward pass's time from per-rank row counts, with the fitted model.

    Args:
        loads: From rank_loads().
        num_nodes: Nodes in the EP group.
        rbd: Use RBD's fixed cost if True, plain all2all's if False.
        mean_weight: Adds this times each term at the average rank. 0 gives the
            fitted model. The search uses a small value so that swaps cutting total
            traffic still count while the busiest rank stays the same.

    Returns:
        ms: [M] estimated milliseconds per placement.
    """
    net_ms, nvlink_ms = _row_costs(num_nodes)
    net = torch.maximum(loads.net_out, loads.net_in).float() * net_ms
    nvl = torch.maximum(loads.nvl_out, loads.nvl_in).float() * nvlink_ms
    fixed = RBD_FIXED_MS if rbd else PLAIN_FIXED_MS
    return (fixed + net.amax(dim=1) + nvl.amax(dim=1)
            + mean_weight * (net.mean(dim=1) + nvl.mean(dim=1)))


def pilot_swap_delta(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
) -> torch.Tensor:
    """Exact change in the total pilot count from swapping each pair of experts.

    A token has a pilot on each other node that holds at least one of its experts.
    Swapping a and b on different nodes only matters for tokens with one of them:
    a token with a but not b gains a pilot if it had nothing on b's node, and loses
    one if a was its only expert on a's node. Summed over tokens, both terms are
    matrix products, so every pair is scored at once.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [E] current slot of each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.

    Returns:
        delta: [E, E] pilots added (negative: removed) by swapping experts a and b.
    """
    E = slots.numel()
    num_nodes = E // experts_per_rank // gpus_per_node
    expert_node = slots // experts_per_rank // gpus_per_node
    has = torch.zeros(route.shape[0], E, device=route.device)
    has.scatter_(1, route, 1.0)
    on_node = torch.nn.functional.one_hot(expert_node, num_nodes).float()  # [E, N]
    per_node = has @ on_node  # [T, N] experts of each token on each node
    remote = torch.arange(num_nodes, device=route.device) != (source_rank // gpus_per_node)[:, None]

    gain = ((per_node == 0) & remote).float()  # [T, N] adding an expert here adds a pilot
    only = (per_node == 1) & remote
    loss = has * only.float()[:, expert_node]  # [T, E] removing e removes a pilot

    gain_to = (has.T @ gain)[:, expert_node]  # [a, b]: tokens of a that gain on b's node
    loss_both = loss.T @ has  # [a, b]: tokens losing a's pilot that also have b
    one_side = gain_to - loss.sum(0)[:, None] + loss_both
    delta = one_side + one_side.T
    return torch.where(expert_node[:, None] != expert_node[None, :], delta, 0.0)


def colocate(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    max_swaps: int = 5000,
) -> torch.Tensor:
    """Swap experts between nodes to remove pilots, ignoring load balance.

    Each step makes the swap that removes the most pilots, from pilot_swap_delta,
    until no swap removes any.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [E] starting slot of each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        max_swaps: Most swaps to make.

    Returns:
        slots: [E] slot of each expert.
    """
    E = slots.numel()
    for _ in range(max_swaps):
        delta = pilot_swap_delta(route, source_rank, slots, experts_per_rank, gpus_per_node)
        value, index = delta.flatten().min(dim=0)
        if value >= 0:
            break
        a, b = divmod(index.item(), E)
        slots = slots.clone()
        slots[a], slots[b] = slots[b].clone(), slots[a].clone()
    return slots


def _kl_pass(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    load: torch.Tensor,
    cap: float,
) -> tuple[torch.Tensor, int]:
    """One Kernighan-Lin pass over the token hypergraph's pilot count.

    Makes the best swap between unlocked experts on different nodes, even if it
    adds pilots, then locks both, until every expert is locked. Keeps the prefix of
    swaps with the largest total gain, so a pass can cross a hill that single best
    swaps cannot, such as moving a whole group of co-selected experts.

    A swap is allowed only if both nodes' loads end at or below cap, or at least no
    higher than the larger of the two was, so an overloaded start can only improve.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [E] starting slot of each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        load: [E] float copies routed to each expert, its vertex weight.
        cap: Most load a node may hold.

    Returns:
        slots: [E] slot of each expert after the best prefix.
        gain: Pilots removed, 0 if no prefix helps.
    """
    E = slots.numel()
    num_nodes = E // experts_per_rank // gpus_per_node
    locked = torch.zeros(E, dtype=torch.bool, device=slots.device)
    current, best, best_total, total = slots, slots, 0, 0
    for _ in range(E // 2):
        delta = pilot_swap_delta(route, source_rank, current, experts_per_rank, gpus_per_node)
        node = current // experts_per_rank // gpus_per_node
        node_load = torch.zeros(num_nodes, device=load.device).index_add_(0, node, load)
        here = node_load[node]  # [E] load of each expert's node
        moved = load[None, :] - load[:, None]  # [a, b] load a's node gains from swapping a, b
        after_a, after_b = here[:, None] + moved, here[None, :] - moved
        worst_after = torch.maximum(after_a, after_b)
        worst_before = torch.maximum(here[:, None], here[None, :])
        balanced = worst_after <= torch.clamp(worst_before, min=cap)
        allowed = ((~locked[:, None]) & (~locked[None, :]) & (node[:, None] != node[None, :])
                   & balanced)
        if not allowed.any():
            break
        value, index = torch.where(allowed, delta, float("inf")).flatten().min(dim=0)
        a, b = divmod(index.item(), E)
        current = current.clone()
        current[a], current[b] = current[b].clone(), current[a].clone()
        locked[a] = locked[b] = True
        total += int(value.item())
        if total < best_total:
            best, best_total = current, total
    return best, -best_total


def _balance_within_nodes(
    route: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
) -> torch.Tensor:
    """Reassign each node's experts to its ranks so their routed copies even out.

    Keeps every expert on its node, so pilots don't change. Experts go in order of
    load, each to the least loaded rank of its node that still has a free slot.

    Args:
        route: [num_tokens, k] experts of every token.
        slots: [E] slot of each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.

    Returns:
        slots: [E] slot of each expert.
    """
    E = slots.numel()
    L, G = experts_per_rank, gpus_per_node
    load = torch.bincount(route.flatten(), minlength=E).tolist()
    node = (slots // L // G).tolist()
    out = slots.clone()
    for n in range(E // L // G):
        rank_load, free = [0] * G, [L] * G
        for e in sorted((e for e in range(E) if node[e] == n), key=lambda e: -load[e]):
            g = min((g for g in range(G) if free[g]), key=lambda g: rank_load[g])
            out[e] = (n * G + g) * L + (L - free[g])
            rank_load[g] += load[e]
            free[g] -= 1
    return out


def even_placement(route: torch.Tensor, E: int, world: int) -> torch.Tensor:
    """Place experts to balance routed copies while keeping equal expert counts.

    Each expert keeps all its routed tokens; only its owning rank changes. Experts
    are assigned in descending token-count order to the least loaded rank that
    still has a free slot. This is a greedy baseline: indivisible expert loads and
    the fixed number of slots per rank can prevent perfectly equal token counts,
    and the result is not guaranteed to be the global optimum.

    All ranks compete for experts, regardless of node. The placement can be used
    with plain all2all or any RBD pilot rule; it does not choose pilot ranks.

    Args:
        route: [num_tokens, k] experts of every token in the EP group.
        E: Total number of experts. Must be positive and divisible by world.
        world: Number of ranks in the EP group. Must be positive.

    Returns:
        slots: [E] slot of each expert, a permutation of 0 through E - 1 on
            route's device. Each rank owns exactly E / world slots. Apply it with
            slots[route], keeping the tokens and their routing weights unchanged.
    """
    if world <= 0 or E <= 0 or E % world:
        raise ValueError("E and world must be positive, with E divisible by world.")
    contiguous = torch.arange(E, device=route.device)
    # Treat the whole EP group as one balancing domain, allowing cross-node moves.
    return _balance_within_nodes(route, contiguous, E // world, world)


def partition_placement(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    E: int,
    world: int,
    gpus_per_node: int,
    imbalance: float = 0.05,
    max_passes: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """Partition experts across nodes to minimize pilots, then balance ranks.

    The token hypergraph has a vertex per expert and a fixed vertex per node, which
    stands for the tokens that start there and is pinned to that node. Each token is
    a hyperedge over its experts and its source node's vertex. A hyperedge touching
    lambda nodes costs lambda - 1, the connectivity objective of hypergraph
    partitioners, which here is exactly the token's pilot count. pilot_swap_delta
    computes this objective's change for every swap, and Kernighan-Lin passes
    (_kl_pass) minimize it with exactly E / nodes experts per node. As in
    hypergraph partitioners, each expert's vertex weight is its load (copies routed
    to it), and no node may exceed (1 + imbalance) times the mean load.

    Three starts are tried: contiguous, colocate()'s placement, and a random one.
    The result with the fewest pilots among those within the load cap wins (or the
    least overloaded, if none is), and _balance_within_nodes then places each
    node's experts on its ranks.

    Args:
        route: [num_tokens, k] experts of every token in the EP group.
        source_rank: [num_tokens] rank each token starts on.
        E: Number of experts.
        world: Ranks in the EP group.
        gpus_per_node: Ranks per node.
        imbalance: Allowed node load above the mean, as a fraction. Use
            float("inf") to ignore balance and minimize pilots only.
        max_passes: Most Kernighan-Lin passes per start.
        seed: Seed for the random start.

    Returns:
        slots: [E] slot of each expert.
    """
    L, G = E // world, gpus_per_node
    contiguous = torch.arange(E, device=route.device)
    if world == G:
        return _balance_within_nodes(route, contiguous, L, G)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    starts = [contiguous, colocate(route, source_rank, contiguous, L, G),
              torch.randperm(E, generator=gen).to(route.device)]
    pilots = lambda s: rank_loads(route, source_rank, s[None], L, G).net_out.sum().item()
    load = torch.bincount(route.flatten(), minlength=E).float()
    cap = (1 + imbalance) * load.sum().item() / (E // L // G)
    num_nodes = E // L // G
    best, best_key = None, None
    for slots in starts:
        for _ in range(max_passes):
            slots, gain = _kl_pass(route, source_rank, slots, L, G, load, cap)
            if gain <= 0:
                break
        node_load = torch.zeros(num_nodes, device=load.device).index_add_(0, slots // L // G, load)
        key = (max(node_load.max().item() - cap, 0.0), pilots(slots))
        if best is None or key < best_key:
            best, best_key = slots, key
    return _balance_within_nodes(route, best, L, G)


_MTKAHYPAR = None  # mtkahypar.Initializer; Mt-KaHyPar may be initialized once per process.


def hypergraph_placement(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    E: int,
    world: int,
    gpus_per_node: int,
    weight: str = "load",
    imbalance: float = 0.05,
    preset: str = "QUALITY",
    threads: int = 16,
    seed: int = 0,
    refine_passes: int = 0,
) -> tuple[torch.Tensor, int, int]:
    """Partition experts across nodes with Mt-KaHyPar to minimize pilots, then balance ranks.

    The same token hypergraph as partition_placement: a vertex per expert, a vertex
    per node fixed to that node, and a hyperedge per token over its experts and its
    source node's vertex, so the connectivity (km1) objective is the pilot count.
    Identical tokens are merged into one hyperedge weighted by their count.

    Node balance follows weight:
        "count": every vertex weighs 1 and no imbalance is allowed, so each node
            gets exactly E / nodes experts, whatever their loads.
        "load": each expert weighs its load (copies routed to it) and no node may
            exceed (1 + imbalance) times the mean load. Mt-KaHyPar does not fix the
            number of experts per node, so moves are then made, each the one adding
            the fewest pilots while keeping the load cap where possible, until every
            node holds E / nodes.
    Either way, swaps then bring any node over (1 + imbalance) times the mean load
    back under it, each the one adding the fewest pilots (pilot_swap_delta).
    Optional Kernighan-Lin passes (_kl_pass, as in partition_placement) then refine
    the result within the cap, and _balance_within_nodes places each node's experts
    on its ranks.

    Needs the mtkahypar module (pip install mtkahypar) on the path.

    Args:
        route: [num_tokens, k] experts of every token in the EP group.
        source_rank: [num_tokens] rank each token starts on.
        E: Number of experts.
        world: Ranks in the EP group.
        gpus_per_node: Ranks per node.
        weight: "count" or "load", as above.
        imbalance: Allowed node load above the mean, as a fraction. Use
            float("inf") to skip the load repair.
        preset: Name of a mtkahypar.PresetType, such as "DEFAULT", "QUALITY" or
            "HIGHEST_QUALITY".
        threads: Mt-KaHyPar threads. Used only by the first call in a process.
        seed: Mt-KaHyPar's random seed. Results also vary run to run with threads.
        refine_passes: Most Kernighan-Lin passes after the repair. 0 skips them.

    Returns:
        slots: [E] slot of each expert.
        moves: Experts moved after Mt-KaHyPar to fix the number per node.
        swaps: Expert swaps made after the moves to bring node loads back under the cap.
    """
    global _MTKAHYPAR
    import mtkahypar

    L, G = E // world, gpus_per_node
    num_nodes, per_node = world // G, E // (world // G)
    device = route.device
    contiguous = torch.arange(E, device=device)
    if num_nodes == 1:
        return _balance_within_nodes(route, contiguous, L, G), 0, 0
    if weight not in ("count", "load"):
        raise ValueError(f'weight must be "count" or "load", got {weight!r}.')
    if _MTKAHYPAR is None:
        _MTKAHYPAR = mtkahypar.initialize(threads, False)
    mtkahypar.set_seed(seed)

    pins = torch.cat([route.sort(dim=1).values, E + (source_rank // G)[:, None]], dim=1)
    edges, counts = pins.unique(dim=0, return_counts=True)
    load = torch.bincount(route.flatten(), minlength=E)
    if weight == "count":
        node_weights = [1] * (E + num_nodes)
        epsilon = 0.0
    else:
        # Fixed vertices weigh 1, not 0: zero-weight fixed vertices crash Mt-KaHyPar 1.7.
        node_weights = load.tolist() + [1] * num_nodes
        epsilon = min(imbalance, 1.0)

    context = _MTKAHYPAR.context_from_preset(getattr(mtkahypar.PresetType, preset))
    context.set_partitioning_parameters(num_nodes, epsilon, mtkahypar.Objective.KM1)
    context.logging = False
    hypergraph = _MTKAHYPAR.create_hypergraph(context, E + num_nodes, edges.shape[0],
                                              edges.tolist(), node_weights, counts.tolist())
    hypergraph.add_fixed_vertices([-1] * E + list(range(num_nodes)), num_nodes)
    partitioned = hypergraph.partition(context)
    node = torch.tensor([partitioned.block_id(v) for v in range(E)], device=device)

    slots, moves, swaps = _repair_nodes(route, source_rank, node, load, L, G, imbalance)
    if refine_passes:
        return kl_refine(route, source_rank, slots, L, G, imbalance, refine_passes), moves, swaps
    return _balance_within_nodes(route, slots, L, G), moves, swaps


def _repair_nodes(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    node: torch.Tensor,
    load: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    imbalance: float,
) -> tuple[torch.Tensor, int, int]:
    """Turn a node assignment into a placement with E / nodes experts and capped load per node.

    Moves experts off nodes with too many, each the move adding the fewest pilots
    while keeping the load cap where possible. Then swaps experts, keeping counts,
    until every node is within (1 + imbalance) times the mean load: the swap adding
    the fewest pilots that leaves both nodes within it, else the one leaving the
    busiest node lightest.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        node: [E] node of each expert, any number per node.
        load: [E] copies routed to each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        imbalance: Allowed node load above the mean, as a fraction. float("inf")
            skips the swaps.

    Returns:
        slots: [E] slot of each expert, before balancing ranks within nodes.
        moves: Experts moved to fix the number per node.
        swaps: Swaps made to bring node loads under the cap.
    """
    L, G = experts_per_rank, gpus_per_node
    E, device = node.numel(), node.device
    num_nodes = E // L // G
    per_node = E // num_nodes
    node = node.clone()
    # Move experts off nodes with too many, each the move adding the fewest pilots.
    has = torch.zeros(route.shape[0], E, device=device)
    has.scatter_(1, route, 1.0)
    remote = (torch.arange(num_nodes, device=device)
              != (source_rank // G)[:, None])  # [T, N]
    cap_load = (1 + imbalance) * load.sum().item() / num_nodes
    moves = 0
    while True:
        size = torch.bincount(node, minlength=num_nodes)
        if (size <= per_node).all():
            break
        on_node = torch.nn.functional.one_hot(node, num_nodes).float()
        count = has @ on_node  # [T, N] experts of each token on each node
        gain = ((count == 0) & remote).float()  # adding an expert here adds a pilot
        only = ((count == 1) & remote).float()  # removing its expert removes a pilot
        delta = has.T @ gain - (has * only[:, node]).sum(0)[:, None]  # [e, target node]
        node_load = torch.zeros(num_nodes, device=device).index_add_(0, node, load.float())
        allowed = (size[node] > per_node)[:, None] & (size < per_node)[None, :]
        within = allowed & (node_load[None, :] + load[:, None] <= cap_load)
        pick = within if within.any() else allowed
        e, target = divmod(torch.where(pick, delta, float("inf")).flatten().argmin().item(),
                           num_nodes)
        node[e] = target
        moves += 1

    # Any slot order within a node; _balance_within_nodes picks the ranks.
    contiguous = torch.arange(E, device=device)
    slots = torch.empty_like(contiguous)
    slots[node.argsort(stable=True)] = contiguous

    # The moves can overload nodes. Swap experts, keeping counts, until every node is
    # within the cap: the swap adding the fewest pilots that leaves both nodes within
    # it, else the one leaving the busiest node lightest.
    swaps = 0
    load = load.float()
    for _ in range(E):
        node = slots // L // G
        node_load = torch.zeros(num_nodes, device=device).index_add_(0, node, load)
        if node_load.max() <= cap_load:
            break
        moved = load[None, :] - load[:, None]  # [a, b] load a's node gains from swapping
        after = torch.maximum(node_load[node][:, None] + moved,
                              node_load[node][None, :] - moved)
        busiest = node == node_load.argmax()
        pairs = (busiest[:, None] ^ busiest[None, :]) & (node[:, None] != node[None, :])
        within = pairs & (after <= cap_load)
        if within.any():
            delta = pilot_swap_delta(route, source_rank, slots, L, G)
            a, b = divmod(torch.where(within, delta, float("inf")).argmin().item(), E)
        else:
            a, b = divmod(torch.where(pairs, after, float("inf")).argmin().item(), E)
            if after[a, b] >= node_load.max():
                break
        slots = slots.clone()
        slots[a], slots[b] = slots[b].clone(), slots[a].clone()
        swaps += 1

    return slots, moves, swaps


def kl_refine(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    imbalance: float = 0.05,
    max_passes: int = 20,
) -> torch.Tensor:
    """Cut pilots of a placement with Kernighan-Lin passes, then balance ranks.

    The passes of partition_placement, from a given start: swaps keep E / nodes
    experts per node and node loads within (1 + imbalance) times the mean, or at
    least no worse than before.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [E] starting slot of each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        imbalance: Allowed node load above the mean, as a fraction.
        max_passes: Most passes.

    Returns:
        slots: [E] slot of each expert.
    """
    L, G = experts_per_rank, gpus_per_node
    load = torch.bincount(route.flatten(), minlength=slots.numel()).float()
    cap = (1 + imbalance) * load.sum().item() / (slots.numel() // L // G)
    for _ in range(max_passes):
        slots, gain = _kl_pass(route, source_rank, slots, L, G, load, cap)
        if gain <= 0:
            break
    return _balance_within_nodes(route, slots, L, G)


def placement_cost(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    rbd: bool = True,
    pilot: str = "rail",
    mean_weight: float = 0.0,
    chunk: int = 32,
) -> torch.Tensor:
    """Estimated time of each placement, in chunks to bound memory.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [M, E] placements to score.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        rbd: Score RBD if True, plain all2all if False.
        pilot: RBD's pilot rule, as in spec_ep.rbd.RBD_combine_dispatch.
        mean_weight: Passed to estimate_ms().
        chunk: Placements scored at once.

    Returns:
        ms: [M] estimated milliseconds per placement.
    """
    num_nodes = slots.shape[1] // experts_per_rank // gpus_per_node
    return torch.cat([
        estimate_ms(rank_loads(route, source_rank, part, experts_per_rank, gpus_per_node, rbd,
                               pilot), num_nodes, rbd, mean_weight)
        for part in slots.split(chunk)
    ])


def _search(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    L: int,
    gpus_per_node: int,
    rbd: bool,
    pilot: str,
    max_swaps: int,
    candidates: int,
    mean_weight: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, list[float]]:
    """Swap experts from a starting placement while the search objective drops.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [E] starting slot of each expert.
        L: Slots per rank.
        gpus_per_node: Ranks per node.
        rbd: Optimize for RBD if True, plain all2all if False.
        pilot: RBD's pilot rule.
        max_swaps: Most swaps to make.
        candidates: Swaps of each kind scored per round.
        mean_weight: Weight of the average rank in the search objective.
        gen: Generator for the random candidates.

    Returns:
        slots: [E] placement with the lowest fitted-model estimate seen.
        history: Fitted-model estimate of the start and after each accepted swap.
    """
    E, device = slots.numel(), slots.device
    score = lambda s: placement_cost(route, source_rank, s, L, gpus_per_node, rbd, pilot,
                                     mean_weight)
    model = lambda s: placement_cost(route, source_rank, s[None], L, gpus_per_node, rbd,
                                     pilot).item()
    cost = score(slots[None]).item()
    history = [model(slots)]
    best_slots = slots

    for _ in range(max_swaps if E // L > 1 else 0):
        pairs = [torch.randint(E, (candidates, 2), generator=gen).to(device)]

        if rbd:
            delta = pilot_swap_delta(route, source_rank, slots, L, gpus_per_node)
            best = delta.flatten().topk(candidates, largest=False).indices
            pairs.append(torch.stack([best // E, best % E], dim=1))

        # Move experts off the busiest rank of the costlier term.
        loads = rank_loads(route, source_rank, slots[None], L, gpus_per_node, rbd, pilot)
        net_ms, nvlink_ms = _row_costs(E // L // gpus_per_node)
        net = torch.maximum(loads.net_out, loads.net_in)[0] * net_ms
        nvl = torch.maximum(loads.nvl_out, loads.nvl_in)[0] * nvlink_ms
        hot = (net if net.max() >= nvl.max() else nvl).argmax()
        on_hot = (slots // L == hot).nonzero().flatten()
        off_hot = (slots // L != hot).nonzero().flatten()
        pick = torch.randint(on_hot.numel(), (candidates,), generator=gen).to(device)
        other = torch.randint(off_hot.numel(), (candidates,), generator=gen).to(device)
        pairs.append(torch.stack([on_hot[pick], off_hot[other]], dim=1))

        pairs = torch.cat(pairs)
        pair_rank = slots[pairs] // L
        pairs = pairs[pair_rank[:, 0] != pair_rank[:, 1]]  # same-rank swaps do nothing
        trials = slots.repeat(pairs.shape[0], 1)
        rows = torch.arange(pairs.shape[0], device=device)
        trials[rows, pairs[:, 0]] = slots[pairs[:, 1]]
        trials[rows, pairs[:, 1]] = slots[pairs[:, 0]]
        costs = score(trials)

        i = costs.argmin()
        if costs[i] >= cost * (1 - 1e-4):
            break
        slots, cost = trials[i], costs[i].item()
        history.append(model(slots))
        if history[-1] < min(history[:-1]):
            best_slots = slots
    return best_slots, history


def greedy_placement(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    E: int,
    world: int,
    gpus_per_node: int,
    rbd: bool = True,
    pilot: str = "rail",
    max_swaps: int = 2000,
    candidates: int = 32,
    mean_weight: float = 1.0,
    warm_start: bool = True,
    start: torch.Tensor | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, list[float]]:
    """Swap experts greedily while the estimated communication time drops.

    Searches from the contiguous placement, slot e for expert e, and for RBD also
    from colocate()'s placement, and returns the placement with the lowest
    fitted-model estimate seen.

    Args:
        route: [num_tokens, k] experts of every token in the EP group.
        source_rank: [num_tokens] rank each token starts on.
        E: Number of experts.
        world: Ranks in the EP group.
        gpus_per_node: Ranks per node.
        rbd: Optimize for RBD if True, for plain all2all if False. Plain gets no
            pilot-based candidates or warm start, since it has no pilots.
        pilot: RBD's pilot rule, as in spec_ep.rbd.RBD_combine_dispatch.
        max_swaps: Most swaps per search.
        candidates: Swaps of each kind scored per round.
        mean_weight: Weight of the average rank in the search objective, passed to
            estimate_ms(). history reports the fitted model, without it.
        warm_start: For RBD, also search from colocate()'s placement.
        start: [E] another placement to search from, such as
            partition_placement()'s. None searches only the others.
        seed: Seed for the random candidates.

    Returns:
        slots: [E] slot of each expert, the best one seen.
        history: Fitted-model estimate of the contiguous placement, then after each
            accepted swap of the search that found slots.
    """
    L = E // world
    gen = torch.Generator(device="cpu").manual_seed(seed)
    contiguous = torch.arange(E, device=route.device)
    args = (route, source_rank)
    rest = (L, gpus_per_node, rbd, pilot, max_swaps, candidates, mean_weight, gen)
    slots, history = _search(*args, contiguous, *rest)
    starts = [] if start is None else [start]
    if rbd and warm_start and world > gpus_per_node:
        starts.append(colocate(route, source_rank, contiguous, L, gpus_per_node))
    for begin in starts:
        warm_slots, warm_history = _search(*args, begin, *rest)
        if min(warm_history) < min(history):
            slots, history = warm_slots, history[:1] + warm_history
    return slots, history


def swap_refine(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    slots: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
    imbalance: float | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, int]:
    """Greedy pilot-cutting swaps, as in the fixed-vertex experiment's refine().

    Each pass visits every expert in random order and makes its best swap with an
    expert on another node if that removes pilots (pilot_swap_delta). Passes repeat
    until one makes no swap. With imbalance, a swap is allowed only if both nodes'
    loads end at or below (1 + imbalance) times the mean, or at least no higher than
    the larger of the two was.

    Args:
        route: [num_tokens, k] experts of every token.
        source_rank: [num_tokens] rank each token starts on.
        slots: [E] starting slot of each expert.
        experts_per_rank: Slots per rank.
        gpus_per_node: Ranks per node.
        imbalance: Allowed node load above the mean, as a fraction. None ignores load.
        seed: Seed for the visiting order.

    Returns:
        slots: [E] slot of each expert, ranks not rebalanced.
        swaps: Swaps made.
    """
    L, G = experts_per_rank, gpus_per_node
    E = slots.numel()
    num_nodes = E // L // G
    load = torch.bincount(route.flatten(), minlength=E).float()
    cap = None if imbalance is None else (1 + imbalance) * load.sum().item() / num_nodes
    gen = torch.Generator(device="cpu").manual_seed(seed)
    swaps, improved = 0, True
    while improved:
        improved = False
        for e in torch.randperm(E, generator=gen).tolist():
            node = slots // L // G
            delta = pilot_swap_delta(route, source_rank, slots, L, G)[e]
            allowed = node != node[e]
            if cap is not None:
                node_load = torch.zeros(num_nodes, device=load.device).index_add_(0, node, load)
                moved = load - load[e]  # load e's node gains from swapping e with each f
                after = torch.maximum(node_load[node[e]] + moved, node_load[node] - moved)
                before = torch.maximum(node_load[node[e]], node_load[node])
                allowed &= after <= torch.clamp(before, min=cap)
            value, f = torch.where(allowed, delta, float("inf")).min(dim=0)
            if value >= 0:
                continue
            slots = slots.clone()
            slots[e], slots[f] = slots[f].clone(), slots[e].clone()
            swaps += 1
            improved = True
    return slots, swaps


def kahypar_placement(
    route: torch.Tensor,
    source_rank: torch.Tensor,
    E: int,
    world: int,
    gpus_per_node: int,
    balance: bool = True,
    imbalance: float = 0.05,
    config: str | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, int, int, int]:
    """Fixed-vertex hypergraph partition with KaHyPar, then greedy swaps.

    The approach of the exp/fixed-vertex-partition branch (fixed_vertex.hypergraph):
    the same token hypergraph as hypergraph_placement, partitioned by KaHyPar (not
    Mt-KaHyPar) with the connectivity objective, then swap_refine().
        balance=False: as on the branch. Every vertex weighs 1 and no imbalance is
            allowed, so each node gets exactly E / nodes experts whatever their
            loads, and swap_refine ignores load.
        balance=True: each expert weighs its load and no node may exceed
            (1 + imbalance) times the mean load. The result is repaired to E / nodes
            experts and the load cap per node as in hypergraph_placement, and
            swap_refine keeps the cap.
    _balance_within_nodes then places each node's experts on its ranks.

    Needs the kahypar module on the path and KaHyPar's km1_kKaHyPar_sea20.ini.

    Args:
        route: [num_tokens, k] experts of every token in the EP group.
        source_rank: [num_tokens] rank each token starts on.
        E: Number of experts.
        world: Ranks in the EP group.
        gpus_per_node: Ranks per node.
        balance: Balance node loads, as above.
        imbalance: Allowed node load above the mean, as a fraction, with balance.
        config: Path of the KaHyPar .ini. None reads the KAHYPAR_INI variable.
        seed: KaHyPar's and swap_refine's seed.

    Returns:
        slots: [E] slot of each expert.
        moves: Experts moved to fix the number per node (0 without balance).
        swaps: Swaps made to bring node loads under the cap (0 without balance).
        refine_swaps: Swaps swap_refine made.
    """
    import os

    import kahypar

    L, G = E // world, gpus_per_node
    num_nodes, per_node = world // G, E // (world // G)
    device = route.device
    load = torch.bincount(route.flatten(), minlength=E)
    pins = torch.cat([route.sort(dim=1).values, E + (source_rank // G)[:, None]], dim=1)
    edges, counts = pins.unique(dim=0, return_counts=True)
    k = edges.shape[1]
    if balance:
        # KaHyPar rejects weight 0, which unrouted experts would have.
        node_weights = load.clamp(min=1).tolist() + [1] * num_nodes
        block = int((1 + imbalance) * load.sum().item() / num_nodes) + 1
    else:
        node_weights = [1] * (E + num_nodes)
        block = per_node + 1
    hypergraph = kahypar.Hypergraph(E + num_nodes, edges.shape[0],
                                    list(range(0, edges.shape[0] * k + 1, k)),
                                    edges.flatten().tolist(), num_nodes, counts.tolist(),
                                    node_weights)
    for n in range(num_nodes):
        hypergraph.fixNodeToBlock(E + n, n)
    context = kahypar.Context()
    context.loadINIconfiguration(config or os.environ["KAHYPAR_INI"])
    context.setK(num_nodes)
    context.setCustomTargetBlockWeights([block] * num_nodes)
    context.setEpsilon(0.0)
    context.setSeed(seed)
    context.suppressOutput(True)
    kahypar.partition(hypergraph, context)
    node = torch.tensor([hypergraph.blockID(v) for v in range(E)], device=device)

    if balance:
        slots, moves, swaps = _repair_nodes(route, source_rank, node, load, L, G, imbalance)
    else:
        if (torch.bincount(node, minlength=num_nodes) != per_node).any():
            raise RuntimeError("KaHyPar did not give every node E / nodes experts.")
        slots = torch.empty_like(node)
        slots[node.argsort(stable=True)] = torch.arange(E, device=device)
        moves = swaps = 0
    slots, refine_swaps = swap_refine(route, source_rank, slots, L, G,
                                      imbalance if balance else None, seed)
    return _balance_within_nodes(route, slots, L, G), moves, swaps, refine_swaps
