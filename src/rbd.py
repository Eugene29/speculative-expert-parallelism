"""Redundancy-Bypassing Dispatch (RBD) for expert-parallel MoE communication.

Based on X-MoE (arXiv 2508.13337). In a plain all2all, a token routed to several
experts on one node crosses the network once per expert, and so does each output on
the way back. RBD sends one row per (token, other node) between nodes: the pilot.
The pilot's rank then copies it to the token's experts on that node, the replicas.
Tokens routed to experts on their own node skip the pilot step. Which rank on a node
receives the pilot is the pilot rule; see rbd_combine_dispatch().

rbd_combine_dispatch() is a plan in two halves around the first of five stages:
    start_plans: the source rank's half. Where its rows go, and every split size.
    1. stage1_dispatch, over ep_group: send each token straight to its experts on
       this node, and one pilot per other node it needs. One all2all carries both,
       so the NVLink and network traffic run together.
    finish_plan: the receiving rank's half, read from the pilots that arrived.
    2. stage2_dispatch, over local_group: each rank copies the pilots it received
       to the experts they need on its node.
    3. stage2_combine: replica outputs return to their pilot's rank, which scales
       and sums them into one row per pilot.
    4. stage1_combine: pilot sums and same-node outputs return to the source rank.
    5. final_merge: the source rank scales the same-node outputs and sums each
       token's rows.

The plan follows DeepEP's: a layout computed where the tokens are, and rows that
carry what their receiver must know. It costs one all2all of fixed-size counts and a
single host sync to read every split size, which NCCL needs before it can receive.
Nothing else is exchanged. A stage 1 row is its token followed by the token's routing
and weights, the metadata columns, so a pilot's rank reads from the row itself which
experts to copy it to. Whatever depends only on an expert's id comes from a table
built once per layout (_layout). The "even" pilot rule also all-gathers per-node
pilot counts. Expert ids are never sent: rows are sorted by expert inside each block,
so per-expert counts give the ids.

Letters used throughout: E experts in total, L experts per rank, G ranks per node,
H token width, k experts per token, M metadata columns.

Forward only. The all2alls carry no gradients, so tokens and expert parameters get
none through rbd_combine_dispatch().

Usage:
    from spec_ep.rbd import rbd_combine_dispatch
    python -m pytest tests/test_rbd.py -v
"""
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

from spec_ep.ops import all2all, gather_sum

__all__ = [
    "PILOT_RULES",
    "Plan",
    "PlanStart",
    "token_slices",
    "start_plans",
    "with_meta",
    "finish_plan",
    "stage1_dispatch",
    "stage2_dispatch",
    "stage2_combine",
    "stage1_combine",
    "final_merge",
    "rbd_combine_dispatch",
]

PILOT_RULES = ("rail", "first", "most", "even")


class _Layout(NamedTuple):
    """One rank's place among the experts, and what follows from an expert's id alone."""

    world: int  # ranks of ep_group
    rank: int  # this rank in ep_group
    G: int  # ranks per node
    L: int  # experts per rank
    lanes: int  # 1: stage 1 rows by rank. 2: pilots, then this node's direct rows
    degree: int  # slices; the counts have one block of columns per slice
    by_expert: torch.Tensor  # [6, E] the rows that _layout() lists, per expert
    ids: torch.Tensor  # [2 * G * L] this rank's expert ids, G times for each stage

    @property
    def columns(self) -> int:
        """Count columns per (rank, slice): a rank's experts, its pilots, then stage 2."""
        return self.L + 1 + self.G + self.G * self.L


@cache
def _layout(
    world: int,
    rank: int,
    gpus_per_node: int,
    experts_per_rank: int,
    lanes: int,
    degree: int,
    device: torch.device,
) -> _Layout:
    """Build the tables of one rank's layout. Cached, so a layer builds its layout once.

    by_expert holds, for a copy routed to expert e:
        0: whether e is on this node, which makes the copy a direct row.
        1: the rank its pilot goes to under the "rail" rule; 0 for a direct row.
        2: the count slot of a direct row to e.
        3: its stage 2 send column at the pilot's rank; the unused slot if direct.
        4: its stage 2 receive slot at e's rank, for a pilot at the node's first
           rank; the unused slot if direct.
        5: the sort key of a direct row to e.

    Returns:
        layout: The tables and the numbers they were built from.
    """
    G, L = gpus_per_node, experts_per_rank
    layout = _Layout(world, rank, G, L, lanes, degree, None, None)
    per_rank = degree * layout.columns
    unused = world * per_rank
    expert = torch.arange(world * L, device=device)
    expert_rank, local_expert = expert // L, expert % L
    direct = expert_rank // G == rank // G
    lane = world if lanes == 2 else 0
    by_expert = torch.stack(
        [
            direct.long(),
            torch.where(direct, 0, expert_rank // G * G + rank % G),
            expert_rank * per_rank + local_expert,
            torch.where(direct, unused, L + 1 + expert_rank % G),
            torch.where(direct, unused, expert_rank * per_rank + L + 1 + G + local_expert),
            (expert_rank + lane) * L + local_expert,
        ]
    )
    ids = torch.arange(rank * L, (rank + 1) * L, device=device).repeat(2 * G)
    return layout._replace(by_expert=by_expert, ids=ids)


class PlanStart(NamedTuple):
    """One slice's plan as far as its source rank knows it; finish_plan() adds the rest."""

    send1_token: torch.Tensor  # [sum(send1)] token of each stage 1 row, among all the rank's tokens
    send1: list[int]  # stage 1 rows sent to each rank of ep_group
    recv1: list[int]  # stage 1 rows received from each rank of ep_group
    send2: list[int]  # stage 2 rows sent to each rank of local_group
    recv2: list[int]  # stage 2 rows received from each rank of local_group
    final_index: torch.Tensor  # [slice tokens, k] returned stage 1 row per copy, or -1
    final_weight: torch.Tensor  # [slice tokens, k] combine weight; 1 for pre-scaled pilots
    meta: torch.Tensor  # [num_tokens, 2k] int32 every token's routing, then its weights' bits
    direct_in: torch.Tensor  # [G, L] direct rows from each rank of the node, per expert here
    stage2_in: torch.Tensor  # [G, L] stage 2 rows from each rank of the node, per expert here
    layout: _Layout  # this rank's layout


class Plan(NamedTuple):
    """Where every row goes in both stages. Stage 1 is over ep_group, stage 2 over local_group."""

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
    """[num_tokens, k] pilot rank for each copy's node, by a pilot rule other than "rail".

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


def token_slices(num_tokens: int, degree: int) -> list[slice]:
    """Cut num_tokens tokens into degree contiguous slices of near-equal size.

    Returns:
        slices: Slice c is tokens num_tokens * c // degree up to
            num_tokens * (c + 1) // degree.
    """
    bounds = [num_tokens * c // degree for c in range(degree + 1)]
    return [slice(a, b) for a, b in zip(bounds, bounds[1:])]


def start_plans(
    route: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    degree: int = 1,
    lanes: int = 1,
) -> list[PlanStart]:
    """Plan the source rank's half of every slice of token_slices(num_tokens, degree).

    Collective over ep_group; sends no tokens. The slices are planned as one batch:
    one exchange of counts, with a block of columns per slice, one host sync that
    reads the split sizes of all of them, and one sort that puts every slice's stage 1
    rows in sending order. The arguments are those of rbd_combine_dispatch().

    Args:
        degree: Slices to plan.
        lanes: 1 orders a slice's stage 1 rows by destination rank, for one all2all
            over ep_group. 2 puts the pilots first and this node's direct rows after
            them, each by rank, for the two all2alls of spec_ep.two_lane.

    Returns:
        starts: One PlanStart per slice, each as if its tokens were a layer of their own.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    G, L = gpus_per_node, num_experts // world
    layout = _layout(world, rank, G, L, lanes, degree, route.device)
    num_tokens, k = route.shape
    route = route.long()

    # Classify every copy: a direct row, its node's pilot, or a copy the pilot covers.
    direct, pilot_rank, direct_slot, pilot_column, expert_slot, key = layout.by_expert[:, route]
    direct = direct.bool()
    expert_node = route // (G * L)
    same = expert_node.unsqueeze(2) == expert_node.unsqueeze(1)
    pilot_slot = same.to(torch.uint8).argmax(dim=2)
    is_pilot = (pilot_slot == torch.arange(k, device=route.device)) & ~direct
    if pilot_rule != "rail":
        offset = None
        if pilot_rule == "even":
            offset = _even_pilot_offsets(expert_node, is_pilot, world // G, ep_group)
        chosen = _select_pilot_rank(
            route // L, expert_node, same, pilot_slot, rank, G, pilot_rule, offset
        )
        pilot_rank = torch.where(direct, 0, chosen)

    # Count the rows of both stages per destination and exchange the counts.
    C = layout.columns
    per_rank = degree * C
    unused = world * per_rank
    sends = direct | is_pilot
    pilot_block = pilot_rank * per_rank
    slots = torch.stack(
        [
            torch.where(sends, torch.where(direct, direct_slot, pilot_block + L), unused),
            pilot_block + pilot_column,
            expert_slot + pilot_rank % G * L,
        ]
    )
    big = 2 * world * L
    key = torch.where(sends, torch.where(direct, key, pilot_rank * L), big)
    if degree > 1:
        token = torch.arange(num_tokens, device=route.device)
        token_slice = (((token + 1) * degree - 1) // max(num_tokens, 1)).unsqueeze(1)
        slots = slots + token_slice * C
        key = key + token_slice * (big + 1)
    counts = torch.zeros(unused + per_rank, dtype=torch.long, device=route.device)
    counts.scatter_add_(0, slots.flatten(), torch.ones_like(slots).flatten())
    counts = counts[:unused].view(world, per_rank)
    recv_counts = torch.empty_like(counts)
    with record_function("comm/both"):
        dist.all_to_all_single(recv_counts, counts, group=ep_group)
    counts, recv_counts = counts.view(world, degree, C), recv_counts.view(world, degree, C)

    # One sort gives every slice's stage 1 rows; a copy that sends none sorts last.
    # None of this needs the exchange, so the host does it while the counts travel.
    order = key.flatten().sort(stable=True).indices
    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=route.device)
    position = position.view(num_tokens, k)
    final_weight = torch.where(direct, weights.float(), 1.0)
    meta = torch.cat([route.int(), weights.float().contiguous().view(torch.int32)], dim=1)

    # The one host sync: every slice's send1, recv1, send2 and recv2.
    stage2_in = recv_counts[:, :, L + 1 + G :].view(world, degree, G, L).sum(dim=0)
    sizes = torch.cat(
        [
            counts[:, :, : L + 1].sum(dim=2).T,
            recv_counts[:, :, : L + 1].sum(dim=2).T,
            recv_counts[:, :, L + 1 : L + 1 + G].sum(dim=0),
            stage2_in.sum(dim=2),
        ],
        dim=1,
    ).tolist()
    lo = rank // G * G
    starts = []
    for c, (size, s) in enumerate(zip(sizes, token_slices(num_tokens, degree))):
        first, num_send1 = s.start * k, sum(size[:world])
        starts.append(
            PlanStart(
                send1_token=order[first : first + num_send1] // k,
                send1=size[:world],
                recv1=size[world : 2 * world],
                send2=size[2 * world : 2 * world + G],
                recv2=size[2 * world + G :],
                final_index=torch.where(sends[s], position[s] - first, -1),
                final_weight=final_weight[s],
                meta=meta,
                direct_in=recv_counts[lo : lo + G, c, :L],
                stage2_in=stage2_in[c],
                layout=layout,
            )
        )
    return starts


def with_meta(tokens: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Append every token's metadata to it, as M more columns of the tokens' dtype.

    Returns:
        rows: [num_tokens, H + M] the rows stage 1 gathers its send buffer from.
    """
    return torch.cat([tokens, meta.view(tokens.dtype)], dim=1)


def finish_plan(start: PlanStart, received: torch.Tensor) -> tuple[torch.Tensor, Plan]:
    """Plan the receiving rank's half of a slice from the stage 1 rows that arrived.

    No communication and no host sync. A pilot's metadata names all of its token's
    experts; the ones on this node are the copies this rank makes in stage 2.

    Args:
        start: The slice's plan from start_plans().
        received: [sum(recv1), H + M] stage 1 rows with their metadata columns.

    Returns:
        rows: [sum(recv1), H] the received rows without their metadata, a view.
        plan: The slice's indices, split sizes and weights.
    """
    layout = start.layout
    G, L, device = layout.G, layout.L, received.device
    lo = layout.rank // G * G
    k = start.final_index.shape[1]
    H = received.shape[1] - start.meta.shape[1] * 4 // received.element_size()
    num_direct = sum(start.recv1[lo : lo + G])
    num_pilots = sum(start.recv1) - num_direct
    first_direct = num_pilots if layout.lanes == 2 else sum(start.recv1[:lo])
    pilot_row = torch.arange(num_pilots, device=device)
    if first_direct < num_pilots:
        pilot_row = pilot_row + num_direct * (pilot_row >= first_direct)

    # Stage 2 rows: a pilot's copies on this node, sorted by rank and then expert.
    meta = received[pilot_row, H:].view(torch.int32)
    node_expert = meta[:, :k] - lo * L
    here = (node_expert >= 0) & (node_expert < G * L)
    num_send2 = sum(start.send2)
    order = torch.where(here, node_expert, G * L).flatten().sort(stable=True).indices[:num_send2]
    merge_index = torch.full((num_pilots * k,), -1, dtype=torch.long, device=device)
    merge_index[order] = torch.arange(num_send2, device=device)

    rows_in = torch.cat([start.direct_in, start.stage2_in]).flatten()
    plan = Plan(
        send1=start.send1,
        recv1=start.recv1,
        direct=(first_direct, first_direct + num_direct),
        send2_row=pilot_row[order // k],
        send2=start.send2,
        recv2=start.recv2,
        expert_ids=layout.ids.repeat_interleave(rows_in, output_size=num_direct + sum(start.recv2)),
        merge_index=merge_index.view(num_pilots, k),
        merge_weight=meta[:, k:].view(torch.float32),
        pilot_row=pilot_row,
        final_index=start.final_index,
        final_weight=start.final_weight,
    )
    return received[:, :H], plan


def stage1_dispatch(
    tokens: torch.Tensor,
    start: PlanStart,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Send direct rows and pilots to their ranks. Collective over ep_group.

    Returns:
        received: [sum(recv1), H + M] rows by source rank, with their metadata
            columns. finish_plan() says which rows are this node's direct rows.
    """
    with record_function("autoep/rbd_stage1_dispatch"):
        rows = with_meta(tokens, start.meta)[start.send1_token]
        return all2all(rows, start.send1, start.recv1, ep_group)


def stage2_dispatch(
    received: torch.Tensor,
    plan: Plan,
    local_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Copy received pilots to their experts on this node. Collective over local_group.

    Args:
        received: [sum(recv1), H] the rows finish_plan() returns.

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
        (start,) = start_plans(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule)
    received = stage1_dispatch(tokens, start, ep_group)
    with record_function("autoep/rbd_plan"):
        received, plan = finish_plan(start, received)
    expert_inputs = stage2_dispatch(received, plan, local_group)
    with record_function("autoep/rbd_experts"):
        outputs = expert(expert_inputs, plan.expert_ids)
    back = stage2_combine(outputs, plan, local_group)
    returned = stage1_combine(back, plan, ep_group)
    return final_merge(returned, plan)
