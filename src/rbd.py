"""Redundancy-Bypassing Dispatch (RBD) for expert-parallel MoE communication.

Based on X-MoE (arXiv 2508.13337). In a plain all2all, a token routed to several
experts on one node crosses the network once per expert, and so does each output on
the way back. RBD sends one row per (token, other node) between nodes: the pilot.
The pilot's rank then copies it to the token's experts on that node, the replicas.
Tokens routed to experts on their own node skip the pilot step. Which rank on a node
receives the pilot is the pilot rule; see rbd_combine_dispatch().

rbd_combine_dispatch() is make_plan(), then five stages around the experts:
    1. stage1_dispatch, over ep_group: send each token straight to its experts on
       this node, and one pilot per other node it needs. One all2all carries both,
       so the NVLink and network traffic run together.
    2. stage2_dispatch, over local_group: each rank copies the pilots it received
       to the experts they need on its node.
    3. stage2_combine: replica outputs return to their pilot's rank, which scales
       and sums them into one row per pilot.
    4. stage1_combine: pilot sums and same-node outputs return to the source rank.
    5. final_merge: the source rank scales the same-node outputs and sums each
       token's rows.

make_plan() costs one all2all of fixed-size counts, a single host sync to read every
split size, and one small all2all of pilot metadata. The "even" pilot rule also
all-gathers per-node pilot counts before planning. Expert ids are never sent: rows
are sorted by expert inside each block, so per-expert counts give the ids.

Letters used throughout: E experts in total, L experts per rank, G ranks per node,
H token width, k experts per token.

TODO: Support backprop. The all2alls don't carry gradients, so tokens and expert
parameters get no gradient through rbd_combine_dispatch.

Usage:
    from spec_ep.rbd import rbd_combine_dispatch
    python -m pytest tests/test_rbd.py -v
"""
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

from spec_ep.ops import all2all, gather_sum

__all__ = [
    "PILOT_RULES",
    "Plan",
    "token_slices",
    "PlanStart",
    "start_plans",
    "finish_plans",
    "make_plans",
    "make_plan",
    "stage1_dispatch",
    "stage2_dispatch",
    "stage2_combine",
    "stage1_combine",
    "final_merge",
    "rbd_combine_dispatch",
]

PILOT_RULES = ("rail", "first", "most", "even")


class Plan(NamedTuple):
    """Where every row goes in both stages. Stage 1 is over ep_group, stage 2 over local_group."""

    send1_token: torch.Tensor  # [sum(send1)] token each stage 1 row carries
    send1: list[int]  # stage 1 rows sent to each rank of ep_group
    recv1: list[int]  # stage 1 rows received from each rank of ep_group
    direct: tuple[int, int]  # stage 1 buffer rows from this node; the rest are pilots
    send2_row: torch.Tensor  # [sum(send2)] stage 1 row each stage 2 row copies
    send2: list[int]  # stage 2 rows sent to each rank of local_group
    recv2: list[int]  # stage 2 rows received from each rank of local_group
    expert_ids: torch.Tensor  # [num_inputs] expert of each direct, then stage 2, row
    merge_index: torch.Tensor  # [num_pilots, k] returned stage 2 row per copy, or -1
    merge_weight: torch.Tensor  # [num_pilots, k] combine weight of each of those copies
    pilot_row: torch.Tensor  # [num_pilots] stage 1 row of each received pilot
    final_index: torch.Tensor  # [num_tokens, k] returned stage 1 row per copy, or -1
    final_weight: torch.Tensor  # [num_tokens, k] combine weight; 1 for pre-scaled pilots


def _expert_ids(counts: torch.Tensor, first_expert: int, total: int) -> torch.Tensor:
    """Expand [num_blocks, experts_per_rank] row counts into one expert id per row.

    total is counts.sum(), passed in so no host sync is needed.
    """
    num_blocks, experts_per_rank = counts.shape
    ids = torch.arange(first_expert, first_expert + experts_per_rank, device=counts.device)
    return ids.repeat(num_blocks).repeat_interleave(counts.flatten(), output_size=total)


def _select_pilot_rank(
    expert_rank: torch.Tensor,
    expert_node: torch.Tensor,
    same_node: torch.Tensor,
    first_slot: torch.Tensor,
    rank: int,
    gpus_per_node: int,
    pilot_rule: str,
    pilot_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """[num_tokens, k] pilot rank for each copy's node, by the pilot rule.

    same_node[t, j, i] says copies j and i of token t share a node. pilot_offset,
    needed for "even", counts the pilots lower source ranks send to each node.
    Entries for same-node copies are computed too, but the caller ignores them.
    """
    G = gpus_per_node
    if pilot_rule == "even":
        first = first_slot == torch.arange(expert_rank.shape[1], device=expert_rank.device)
        pilot_rank = expert_node * G
        for n in range(pilot_offset.numel()):
            on_node = expert_node == n
            ordinal = (first & on_node).sum(dim=1).cumsum(dim=0) - 1
            receiver = n * G + (pilot_offset[n] + ordinal) % G
            pilot_rank = torch.where(on_node, receiver.unsqueeze(1), pilot_rank)
    elif pilot_rule == "rail":
        pilot_rank = expert_node * G + rank % G
    elif pilot_rule == "first":
        pilot_rank = expert_rank.gather(1, first_slot)
    elif pilot_rule == "most":
        held = (expert_rank.unsqueeze(2) == expert_rank.unsqueeze(1)).sum(dim=2)
        after_rail = (expert_rank - rank) % G
        score = held * G + (G - 1 - after_rail)
        best = torch.where(same_node, score.unsqueeze(1), -1).argmax(dim=2)
        pilot_rank = expert_rank.gather(1, best)
    else:
        raise ValueError(f"pilot_rule must be one of {PILOT_RULES}, got {pilot_rule!r}.")

    return pilot_rank


def _even_pilot_offsets(
    expert_node: torch.Tensor,
    is_pilot: torch.Tensor,
    num_nodes: int,
    ep_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """[num_nodes] pilots that lower source ranks send to each node. Collective."""
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    node_counts = torch.zeros(num_nodes, dtype=torch.long, device=expert_node.device)
    node_counts.scatter_add_(0, expert_node.flatten(), is_pilot.long().flatten())
    all_counts = node_counts.new_empty(world * num_nodes)
    with record_function("comm/both"):
        dist.all_gather_into_tensor(all_counts, node_counts, group=ep_group)
    return all_counts.view(world, num_nodes)[:rank].sum(dim=0)


def _exchange_counts(
    slots: torch.Tensor,
    world: int,
    C: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count rows per (rank, column) and all2all the [world, C] counts. Collective.

    slots holds rank * C + column per row; world * C marks a row not counted.

    Returns:
        counts: [world, C] this rank's counts for each destination.
        recv_counts: [world, C] each source's counts for this rank.
    """
    unused = world * C
    # scatter_add_ rather than bincount, which syncs to find its output size.
    counts = torch.zeros(unused + 1, dtype=torch.long, device=slots.device)
    counts.scatter_add_(0, slots, torch.ones_like(slots))
    counts = counts[:unused].view(world, C)
    recv_counts = torch.empty_like(counts)
    with record_function("comm/both"):
        dist.all_to_all_single(recv_counts, counts, group=ep_group)
    return counts, recv_counts


def _send_pilot_meta(
    metas: list[torch.Tensor],
    pilot_send: list[list[int]],
    pilot_recv: list[list[int]],
    ep_group: dist.ProcessGroup | None,
) -> list[torch.Tensor]:
    """Send every slice's pilot metadata to the pilots' ranks in one all2all. Collective.

    Args:
        metas: Per slice, the [num_pilots_sent, 2k] metadata of its pilots, by rank.
        pilot_send: Per slice, the pilots sent to each rank of ep_group.
        pilot_recv: Per slice, the pilots received from each rank of ep_group.

    Returns:
        received: Per slice, the [num_pilots, 2k] metadata of the pilots it receives,
            by source rank.
    """
    if len(metas) == 1:
        return [all2all(metas[0], pilot_send[0], pilot_recv[0], ep_group, link="inter")]
    # One buffer by rank, and by slice within a rank, on both sides.
    world, degree = len(pilot_send[0]), len(metas)
    by_slice = [meta.split(sizes) for meta, sizes in zip(metas, pilot_send)]
    buffer = torch.cat([by_slice[c][r] for r in range(world) for c in range(degree)])
    received = all2all(
        buffer,
        [sum(sizes[r] for sizes in pilot_send) for r in range(world)],
        [sum(sizes[r] for sizes in pilot_recv) for r in range(world)],
        ep_group,
        link="inter",
    )
    pieces = received.split([pilot_recv[c][r] for r in range(world) for c in range(degree)])
    return [torch.cat(pieces[c::degree]) for c in range(degree)]


def token_slices(num_tokens: int, degree: int) -> list[slice]:
    """Cut num_tokens tokens into degree contiguous slices of near-equal size.

    Returns:
        slices: Slice c is tokens num_tokens * c // degree up to
            num_tokens * (c + 1) // degree.
    """
    bounds = [num_tokens * c // degree for c in range(degree + 1)]
    return [slice(a, b) for a, b in zip(bounds, bounds[1:])]


class PlanStart(NamedTuple):
    """Every slice's plan up to what its stage 1 dispatch needs; finish_plans() does the rest."""

    send1_token: list[torch.Tensor]  # per slice: [sum(send1)] token each stage 1 row carries
    sizes: list[list[int]]  # per slice: send1, recv1, pilot_send, pilot_recv, send2, recv2
    order: list[torch.Tensor]  # per slice: its copies sorted by stage 1 row
    meta: list[torch.Tensor]  # per slice: [num_pilots, 2k] metadata of its received pilots
    weights: torch.Tensor  # [num_tokens, k] combine weights
    direct: torch.Tensor  # [num_tokens, k] whether a copy's expert is on this node
    is_pilot: torch.Tensor  # [num_tokens, k] whether a copy is its node's pilot
    direct_in: torch.Tensor  # [world, degree, L] direct rows per source rank and expert
    stage2_in: torch.Tensor  # [degree, G, L] stage 2 rows per source rank and expert
    rank: int  # this rank in ep_group
    gpus_per_node: int  # ranks per node

    @property
    def send1(self) -> list[list[int]]:
        """Per slice: stage 1 rows sent to each rank of ep_group."""
        return [size[: self.direct_in.shape[0]] for size in self.sizes]

    @property
    def recv1(self) -> list[list[int]]:
        """Per slice: stage 1 rows received from each rank of ep_group."""
        world = self.direct_in.shape[0]
        return [size[world : 2 * world] for size in self.sizes]


def start_plans(
    route: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    degree: int = 1,
) -> PlanStart:
    """Plan every slice of token_slices(num_tokens, degree) as far as its stage 1 dispatch.

    Collective over ep_group; sends no tokens. The slices share one exchange of counts
    and one host sync, which reads the split sizes of all of them: the counts get one
    block of columns per slice. One small all2all then carries the pilot metadata of
    every slice. What is left, finish_plans(), is host-bound and needs no rank but
    this one, so a caller can start the stage 1 dispatches first and finish the plans
    while they are on the network. The arguments are those of rbd_combine_dispatch().

    Returns:
        start: The stage 1 rows and split sizes of every slice, and what
            finish_plans() needs.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    G, L = gpus_per_node, num_experts // world
    node = rank // G
    num_tokens, k = route.shape
    device = route.device
    route = route.long()

    expert_rank = route // L
    expert_node = expert_rank // G
    local_expert = route % L
    direct = expert_node == node

    same = expert_node.unsqueeze(2) == expert_node.unsqueeze(1)
    pilot_slot = same.to(torch.uint8).argmax(dim=2)
    is_pilot = (pilot_slot == torch.arange(k, device=device)) & ~direct
    pilot_offset = None
    if pilot_rule == "even":
        pilot_offset = _even_pilot_offsets(expert_node, is_pilot, world // G, ep_group)
    pilot_rank = _select_pilot_rank(
        expert_rank, expert_node, same, pilot_slot, rank, G, pilot_rule, pilot_offset
    )

    # Columns of one slice's counts: a row per expert, the pilots, then stage 2.
    pilot_column = L
    stage2_send_start = pilot_column + 1
    stage2_recv_start = stage2_send_start + G
    C = stage2_recv_start + G * L
    slices = token_slices(num_tokens, degree)
    lengths = torch.tensor([s.stop - s.start for s in slices], device=device)
    token_slice = torch.arange(degree, device=device).repeat_interleave(lengths, output_size=num_tokens)
    first_column = (token_slice * C).unsqueeze(1)
    unused = world * degree * C
    stage1 = torch.where(
        direct,
        expert_rank * degree * C + first_column + local_expert,
        torch.where(is_pilot, pilot_rank * degree * C + first_column + pilot_column, unused),
    )
    to_pilot = torch.where(
        ~direct,
        pilot_rank * degree * C + first_column + stage2_send_start + expert_rank % G,
        unused,
    )
    to_expert = torch.where(
        ~direct,
        expert_rank * degree * C + first_column + stage2_recv_start
        + (pilot_rank % G) * L + local_expert,
        unused,
    )
    slots = torch.cat([stage1.flatten(), to_pilot.flatten(), to_expert.flatten()])
    counts, recv_counts = _exchange_counts(slots, world, degree * C, ep_group)
    counts, recv_counts = counts.view(world, degree, C), recv_counts.view(world, degree, C)

    stage2_in = recv_counts[:, :, stage2_recv_start:].view(world, degree, G, L).sum(dim=0)
    sizes = torch.cat(
        [
            counts[:, :, :stage2_send_start].sum(dim=2).T,
            recv_counts[:, :, :stage2_send_start].sum(dim=2).T,
            counts[:, :, pilot_column].T,
            recv_counts[:, :, pilot_column].T,
            recv_counts[:, :, stage2_send_start:stage2_recv_start].sum(dim=0),
            stage2_in.sum(dim=2),
        ],
        dim=1,
    ).tolist()

    stage1_key = torch.where(
        direct,
        expert_rank * (L + 1) + local_expert,
        torch.where(is_pilot, pilot_rank * (L + 1) + pilot_column, world * (L + 1)),
    )
    node_expert = (expert_rank % G) * L + local_expert
    lo, hi = node * G, (node + 1) * G

    # Per slice: sort its stage 1 rows, and gather what its pilots' ranks must know.
    orders, metas, tokens = [], [], []
    for size, s in zip(sizes, slices):
        send1 = size[:world]
        order = stage1_key[s].flatten().sort(stable=True).indices
        send_lo = sum(send1[:lo])
        send_hi = send_lo + sum(send1[lo:hi])
        pilot_copy = torch.cat([order[:send_lo], order[send_hi : sum(send1)]])
        tokens.append(order[: sum(send1)] // k)
        pilot_token = pilot_copy // k
        covered = same[s].view(-1, k)[pilot_copy]
        # Node-local experts and weights of each pilot; weights ride as int32 bits.
        metas.append(
            torch.cat(
                [
                    torch.where(covered, node_expert[s][pilot_token], -1).int(),
                    torch.where(covered, weights[s][pilot_token].float(), 0.0).view(torch.int32),
                ],
                dim=1,
            )
        )
        orders.append(order)
    metas = _send_pilot_meta(
        metas,
        [size[2 * world : 3 * world] for size in sizes],
        [size[3 * world : 4 * world] for size in sizes],
        ep_group,
    )
    return PlanStart(
        tokens, sizes, orders, metas, weights, direct, is_pilot, recv_counts[:, :, :L],
        stage2_in, rank, G,
    )


def finish_plans(start: PlanStart) -> list[Plan]:
    """Build every slice's Plan from start_plans(). No communication and no host sync.

    Returns:
        plans: One Plan per slice, each as if its tokens were a layer of their own.
    """
    slices = token_slices(start.weights.shape[0], len(start.sizes))
    return [
        _finish_plan(
            start.sizes[c], start.order[c], start.meta[c], start.weights[s], start.direct[s],
            start.is_pilot[s], start.direct_in[:, c], start.stage2_in[c], start.rank,
            start.gpus_per_node,
        )
        for c, s in enumerate(slices)
    ]


def make_plans(
    route: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    degree: int = 1,
) -> list[Plan]:
    """Plan every slice of token_slices(num_tokens, degree) at once. Sends no tokens.

    Collective over ep_group: start_plans(), then finish_plans(). The arguments are
    those of rbd_combine_dispatch().

    Returns:
        plans: One Plan per slice, each as if its tokens were a layer of their own.
    """
    return finish_plans(
        start_plans(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule, degree)
    )


def _finish_plan(
    sizes: list[int],
    order: torch.Tensor,
    meta: torch.Tensor,
    weights: torch.Tensor,
    direct: torch.Tensor,
    is_pilot: torch.Tensor,
    direct_in: torch.Tensor,
    stage2_in: torch.Tensor,
    rank: int,
    gpus_per_node: int,
) -> Plan:
    """Build one slice's Plan from its split sizes and the metadata of its pilots.

    Args:
        sizes: The slice's send1, recv1, pilot_send and pilot_recv per rank of
            ep_group, then send2 and recv2 per rank of the node.
        order: [num_tokens * k] the slice's copies sorted by stage 1 row: by
            destination rank, then expert, with pilots after a rank's experts, and
            the copies that send no row last.
        meta: [num_pilots, 2k] for each received pilot, the node-local expert of
            each of its copies, or -1 for a copy on another node, then the combine
            weights as int32 bits.
        direct: [num_tokens, k] whether the copy's expert is on this node.
        is_pilot: [num_tokens, k] whether the copy is its node's pilot.
        direct_in: [world, L] direct rows each rank sends to each expert here.
        stage2_in: [G, L] stage 2 rows each rank of the node sends to each expert here.

    Returns:
        plan: The slice's indices, split sizes and weights.
    """
    world, G = direct_in.shape[0], gpus_per_node
    L = direct_in.shape[1]
    node = rank // G
    num_tokens, k = direct.shape
    device = order.device
    send1, recv1 = sizes[:world], sizes[world : 2 * world]
    pilot_recv = sizes[3 * world : 4 * world]
    send2, recv2 = sizes[4 * world : 4 * world + G], sizes[4 * world + G :]

    num_send1 = sum(send1)
    lo, hi = node * G, (node + 1) * G
    recv_lo = sum(recv1[:lo])
    num_direct = sum(recv1[lo:hi])
    copy_expert, merge_weight = meta[:, :k], meta[:, k:].view(torch.float32)

    num_pilots = sum(pilot_recv)
    pilot_row = torch.arange(num_pilots, device=device)
    pilot_row = pilot_row + num_direct * (pilot_row >= recv_lo)

    num_send2 = sum(send2)
    stage2_key = torch.where(copy_expert >= 0, copy_expert, G * L).flatten()
    order2 = stage2_key.sort(stable=True).indices[:num_send2]
    send2_row = pilot_row[order2 // k]
    merge_index = torch.full((num_pilots * k,), -1, dtype=torch.long, device=device)
    merge_index[order2] = torch.arange(num_send2, device=device)

    expert_ids = torch.cat(
        [
            _expert_ids(direct_in[lo:hi], rank * L, num_direct),
            _expert_ids(stage2_in, rank * L, sum(recv2)),
        ]
    )

    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=device)
    position = position.view(num_tokens, k)
    final_index = torch.where(position < num_send1, position, -1)
    final_weight = torch.where(direct, weights.float(), is_pilot.float())

    return Plan(
        send1_token=order[:num_send1] // k,
        send1=send1,
        recv1=recv1,
        direct=(recv_lo, recv_lo + num_direct),
        send2_row=send2_row,
        send2=send2,
        recv2=recv2,
        expert_ids=expert_ids,
        merge_index=merge_index.view(num_pilots, k),
        merge_weight=merge_weight,
        pilot_row=pilot_row,
        final_index=final_index,
        final_weight=final_weight,
    )


def make_plan(
    route: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
) -> Plan:
    """Decide where every row goes and tell the receiving ranks. Sends no tokens.

    Collective over ep_group, with a single host sync to read every split size.
    The arguments are those of rbd_combine_dispatch().

    Returns:
        plan: The indices, split sizes and weights the five stages need.
    """
    return make_plans(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule)[0]


def stage1_dispatch(
    tokens: torch.Tensor,
    plan: Plan,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Send direct rows and pilots to their ranks. Collective over ep_group.

    Returns:
        received: [sum(recv1), H] rows by source rank. This node's direct rows are
            the block plan.direct; the rest are pilots.
    """
    with record_function("autoep/rbd_stage1_dispatch"):
        return all2all(tokens[plan.send1_token], plan.send1, plan.recv1, ep_group)


def stage2_dispatch(
    received: torch.Tensor,
    plan: Plan,
    local_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Copy received pilots to their experts on this node. Collective over local_group.

    Returns:
        expert_inputs: [num_inputs, H] direct rows, then stage 2 rows, in the order of
            plan.expert_ids.
    """
    with record_function("autoep/rbd_stage2_dispatch"):
        start, end = plan.direct
        num_direct = end - start
        expert_inputs = received.new_empty((num_direct + sum(plan.recv2), received.shape[1]))
        all2all(
            received[plan.send2_row],
            plan.send2,
            plan.recv2,
            local_group,
            out=expert_inputs[num_direct:],
            link="intra",
        )
        expert_inputs[:num_direct].copy_(received[start:end])
    return expert_inputs


def stage2_combine(
    outputs: torch.Tensor,
    plan: Plan,
    local_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Return replica outputs to their pilot's rank and sum them. Collective over local_group.

    Returns:
        back: [sum(recv1), H] the stage 1 buffer rebuilt: direct outputs in their
            block, and each pilot's weighted sum in its row.
    """
    with record_function("autoep/rbd_stage2_combine"):
        start, end = plan.direct
        num_direct = end - start
        outputs = outputs.contiguous()
        returned = all2all(
            outputs[num_direct:], plan.recv2, plan.send2, local_group, link="intra"
        )
        back = outputs.new_empty((sum(plan.recv1), outputs.shape[1]))
        back[start:end].copy_(outputs[:num_direct])
        gather_sum(
            returned, plan.merge_index, plan.merge_weight, out=back, out_rows=plan.pilot_row
        )
    return back


def stage1_combine(
    back: torch.Tensor,
    plan: Plan,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return direct outputs and pilot sums to their source rank. Collective over ep_group.

    Returns:
        returned: [sum(send1), H] rows in the stage 1 send order.
    """
    with record_function("autoep/rbd_stage1_combine"):
        return all2all(back, plan.recv1, plan.send1, ep_group)


def final_merge(returned: torch.Tensor, plan: Plan) -> torch.Tensor:
    """Scale the direct outputs and sum each token's rows. Pilot sums get weight 1.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs.
    """
    with record_function("autoep/rbd_final_merge"):
        return gather_sum(returned, plan.final_index, plan.final_weight)


def rbd_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs.

    Collective over ep_group and local_group. Every rank must call it, including
    ranks with no tokens.

    Each token reaches its experts by one of two paths, and returns the same way:
        Same node: source rank -> expert rank, in stage 1.
        Other node: source rank -> pilot rank in stage 1, then pilot rank -> expert
            rank in stage 2. One row per (token, node) crosses between nodes.

    Weights are applied on the rank that gathers a token's copies: the pilot rank for
    other-node copies and the source rank for same-node copies. Pilot sums therefore
    cross between nodes already scaled.

    The pilot rule picks which rank on each other node receives the pilot:
        "rail" (default): the rank with this rank's local index. Every rank
            receives pilots only from its peers at the same local index, which
            spreads them evenly, at the cost of one more NVLink copy when that rank
            holds none of the token's experts.
        "even": balance pilot arrivals across each node's ranks, with counts
            differing by at most one. Costs one extra all-gather of per-node pilot
            counts, and may add stage 2 traffic compared with "most".
        "first": the rank of the token's first copy on that node, in route order.
            Ranks holding popular top-1 experts receive the most pilots.
        "most": the rank holding the most of the token's copies on that node, so
            the fewest replicas cross NVLink in stage 2. It cuts stage 2 rows but
            ranks of popular experts receive the most pilots and cause imbalance.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights, where weights[t, j] scales the
            output of expert route[t, j] for token t.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to. Experts must be
            placed contiguously, num_experts // world per rank of ep_group, so
            expert e lives on rank e // (num_experts // world).
        num_experts: Total number of experts across all ranks.
        gpus_per_node: Ranks per node. Ranks must be numbered contiguously by node.
        local_group: Process group of this rank's node.
        ep_group: Process group holding the experts. None means the global group.
        pilot_rule: "rail", "even", "first" or "most", as described above.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    with record_function("autoep/rbd_plan"):
        plan = make_plan(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule)
    received = stage1_dispatch(tokens, plan, ep_group)
    expert_inputs = stage2_dispatch(received, plan, local_group)
    with record_function("autoep/rbd_experts"):
        outputs = expert(expert_inputs, plan.expert_ids)
    back = stage2_combine(outputs, plan, local_group)
    returned = stage1_combine(back, plan, ep_group)
    return final_merge(returned, plan)
