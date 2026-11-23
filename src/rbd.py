"""Redundancy-Bypassing Dispatch (RBD) for expert-parallel MoE communication.

Based on X-MoE (arXiv 2508.13337). In a plain all2all, a token routed to several
experts on one node crosses the network once per expert, and so does each output on
the way back. RBD sends one row per (token, other node) between nodes: the pilot.
The pilot's rank then copies it to the token's experts on that node, the replicas.
Tokens routed to experts on their own node skip the pilot step and go as direct rows.

rbd_combine_dispatch() is a plan and five stages:
    make_plan: where every row goes, and every split size.
    1. stage1_dispatch, over ep_group: send each token straight to its experts on
       this node, and one pilot per other node it needs. The experts run on the
       direct rows.
    2. stage2_dispatch, over local_group: each rank copies the pilots it received
       to the experts they need on its node. The experts run on those.
    3. stage2_combine: replica outputs return to their pilot's rank, which scales
       and sums them into one row per pilot.
    4. stage1_combine: pilot sums and same-node outputs return to the source rank.
    5. final_merge: the source rank scales the same-node outputs and sums each
       token's rows.

The plan is three Triton kernels. make_plan() runs _plan_kernel over this rank's
tokens: it classifies every copy, counts the rows of both stages per destination, and
gives every stage 1 row its place among the rows of its bucket, all with atomic adds.
The ranks exchange the counts in one small all2all, and one host sync reads every
split size, which NCCL needs before it can receive. _group_kernel then orders the rows
this rank's experts will get by expert, so the experts need no sort. A stage 1 row is
its token followed by the token's routing and weights, the metadata, so a pilot's rank
reads from the row itself which experts to copy it to: stage2_dispatch() runs
_fanout_kernel over the pilots that arrived. Expert ids are never sent. Rows of one
bucket share an expert, so the counts give the ids.

The expert function is called on the direct rows and on the stage 2 rows separately,
each time with rows grouped by expert, ids ascending.

Letters used throughout: E experts in total, L experts per rank, G ranks per node,
W ranks, H token width, k experts per token, M metadata columns, C count columns.

GPU only: the plan is Triton. Forward only: the all2alls carry no gradients.

Usage:
    from spec_ep.rbd import rbd_combine_dispatch
    python -m pytest tests/test_rbd.py -v
"""
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.profiler import record_function

from spec_ep.ops import all2all, gather_sum

__all__ = [
    "PILOT_RULES",
    "Plan",
    "Merge",
    "Expert",
    "make_plan",
    "with_meta",
    "stage1_dispatch",
    "run_direct",
    "stage2_dispatch",
    "run_experts",
    "stage2_combine",
    "stage1_combine",
    "final_merge",
    "rbd_combine_dispatch",
]

PILOT_RULES = ("rail", "first")
_BLOCK = 128  # tokens, pilots or rows per kernel program
Expert = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Plan(NamedTuple):
    """Where every row goes in both stages. Stage 1 is over ep_group, stage 2 over local_group."""

    send1_token: torch.Tensor  # [sum(send1)] token of each stage 1 row
    send1: list[int]  # stage 1 rows sent to each rank of ep_group
    recv1: list[int]  # stage 1 rows received from each rank of ep_group
    direct: tuple[int, int]  # received stage 1 rows from this node; the rest are pilots
    pilot_row: torch.Tensor  # [num_pilots] received stage 1 row of each pilot
    send2: list[int]  # stage 2 rows sent to each rank of local_group
    recv2: list[int]  # stage 2 rows received from each rank of local_group
    send2_start: torch.Tensor  # [G * L] first stage 2 row sent for each expert of this node
    first_expert: int  # first expert of this node
    direct_order: torch.Tensor  # [num_direct] received stage 1 row of each direct row, by expert
    direct_place: torch.Tensor  # [num_direct] where each direct row is in that order
    stage2_order: torch.Tensor  # [sum(recv2)] received stage 2 row of each stage 2 row, by expert
    stage2_place: torch.Tensor  # [sum(recv2)] where each stage 2 row is in that order
    expert_ids: torch.Tensor  # [num_direct + sum(recv2)] experts of the direct, then the stage 2, rows, each ascending
    final_index: torch.Tensor  # [num_tokens, k] returned stage 1 row per copy, or -1
    final_weight: torch.Tensor  # [num_tokens, k] combine weight; 1 for pre-scaled pilots
    meta: torch.Tensor  # [num_tokens, 2k] int32 every token's routing, then its weights' bits


class Merge(NamedTuple):
    """How a pilot's rank sums the replicas of each pilot it received."""

    index: torch.Tensor  # [num_pilots, k] returned stage 2 row per copy, or -1
    weight: torch.Tensor  # [num_pilots, k] combine weight of each of those copies


@triton.jit
def _plan_kernel(
    route,  # [T, K] int64
    weights,  # [T, K]
    counts,  # [W * C + 1] int64, zeroed. Rows per (destination rank, column), then a spare
    bucket,  # [T, K] int64 out: counts entry of the copy's stage 1 row; the spare if it sends none
    slot,  # [T, K] int64 out: the row's place among the rows of its bucket
    final_weight,  # [T, K] float32 out
    meta,  # [T, 2K] int32 out
    T,
    rank,
    G,
    L,
    C,
    spare,
    K: tl.constexpr,
    RAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Classify every copy of BLOCK tokens and count the rows it causes in both stages.

    A destination rank's C columns: direct rows per expert [L], its pilots [1], the
    stage 2 rows it forwards per (rank of its node, expert) [G * L], and the stage 2
    rows it receives per (pilot rank of its node, expert) [G * L].
    """
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = t < T
    one = tl.full([BLOCK], 1, tl.int64)
    for j in tl.static_range(K):
        e = tl.load(route + t * K + j, mask=live, other=0).to(tl.int32)
        w = tl.load(weights + t * K + j, mask=live, other=0.0).to(tl.float32)
        r = e // L
        x = e % L
        direct = r // G == rank // G
        # A node's pilot is the token's first copy there. It goes to the rank with this
        # rank's place on that node (rail), or to that copy's own rank (first).
        first = live
        pilot = r // G * G + rank % G if RAIL else r
        for i in tl.static_range(j):
            ri = tl.load(route + t * K + i, mask=live, other=0).to(tl.int32) // L
            same = ri // G == r // G
            if not RAIL:
                pilot = tl.where(same & first, ri, pilot)
            first = first & ~same

        sends = live & (direct | first)
        b = tl.where(sends, tl.where(direct, r * C + x, pilot * C + L), spare)
        s = tl.atomic_add(counts + b, one, mask=sends)
        tl.store(bucket + t * K + j, b.to(tl.int64), mask=live)
        tl.store(slot + t * K + j, tl.where(sends, s, 0), mask=live)
        remote = live & ~direct
        tl.atomic_add(counts + pilot * C + L + 1 + (r % G) * L + x, one, mask=remote)
        tl.atomic_add(counts + r * C + L + 1 + (G + pilot % G) * L + x, one, mask=remote)
        tl.store(final_weight + t * K + j, tl.where(direct, w, 1.0), mask=live)
        tl.store(meta + t * 2 * K + j, e, mask=live)
        tl.store(meta + t * 2 * K + K + j, w.to(tl.int32, bitcast=True), mask=live)


@triton.jit
def _group_kernel(
    runs,  # [2 * G * L] int64 rows per (part, rank of the node, expert); the parts are direct and stage 2
    order,  # [num_inputs] int64 out: received row of each row, by part and then expert
    place,  # [num_inputs] int64 out: where each row, as received, is in that order
    expert_ids,  # [num_inputs] int64 out: expert of each row in that order
    first_direct,
    num_direct,
    first_expert,
    G,
    L,
    RUNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Move one run of rows to its expert's group. One program per run.

    A part's rows arrive by rank and then expert, so the rows of one (rank, expert)
    are a run. In the order the experts take, a part's rows are by expert and then
    rank. Rows and places count from the start of their part, except that a direct
    row's received row is its row in the stage 1 buffer.
    """
    run = tl.program_id(0)
    i = tl.arange(0, RUNS)
    rows = tl.load(runs + i, mask=i < 2 * G * L, other=0)
    part = run // (G * L)
    x = run % L
    same_part = i // (G * L) == part
    received = tl.sum(tl.where(same_part & (i < run), rows, 0))
    grouped = tl.sum(tl.where(same_part & ((i % L < x) | ((i % L == x) & (i < run))), rows, 0))
    at = part * num_direct  # where the part starts in order, place and expert_ids
    count = tl.load(runs + run).to(tl.int32)
    for j0 in range(0, count, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        in_run = j < count
        tl.store(order + at + grouped + j, received + j + (1 - part) * first_direct, mask=in_run)
        tl.store(place + at + received + j, grouped + j, mask=in_run)
        tl.store(expert_ids + at + grouped + j, j * 0 + first_expert + x, mask=in_run)


@triton.jit
def _fanout_kernel(
    rows,  # [num_rows, row_width] int32 view of the received stage 1 rows
    pilot_row,  # [num_pilots] int64
    cursor,  # [G * L] int64 next stage 2 row to send for each expert of this node
    send2_row,  # [sum(send2)] int64 out: received row each stage 2 row copies
    merge_index,  # [num_pilots, K] int64 out
    merge_weight,  # [num_pilots, K] float32 out
    num_pilots,
    row_width,
    meta_at,
    first_expert,
    node_experts,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Give every copy that BLOCK pilots make on this node its stage 2 row."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = p < num_pilots
    row = tl.load(pilot_row + p, mask=live, other=0)
    one = tl.full([BLOCK], 1, tl.int64)
    for j in tl.static_range(K):
        x = tl.load(rows + row * row_width + meta_at + j, mask=live, other=0) - first_expert
        here = live & (x >= 0) & (x < node_experts)
        at = tl.atomic_add(cursor + x, one, mask=here)
        tl.store(send2_row + at, row, mask=here)
        tl.store(merge_index + p * K + j, tl.where(here, at, -1), mask=live)
        bits = tl.load(rows + row * row_width + meta_at + K + j, mask=live, other=0)
        tl.store(merge_weight + p * K + j, bits.to(tl.float32, bitcast=True), mask=live)


@cache
def _bucket_order(
    world: int,
    experts_per_rank: int,
    count_columns: int,
    lanes: int,
    device: torch.device,
) -> torch.Tensor:
    """List the stage 1 buckets, as entries of the counts, in sending order. Cached.

    Returns:
        order: [W * (L + 1)] One lane: by rank, a rank's experts and then its pilots.
            Two lanes: every rank's pilots, then every rank's experts.
    """
    first = torch.arange(world, device=device).unsqueeze(1) * count_columns
    experts = first + torch.arange(experts_per_rank, device=device)
    pilots = first + experts_per_rank
    if lanes == 2:
        return torch.cat([pilots.flatten(), experts.flatten()])
    return torch.cat([experts, pilots], dim=1).flatten()


def make_plan(
    route: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    lanes: int = 1,
) -> Plan:
    """Plan a layer's round trip. Collective over ep_group; sends no tokens.

    One kernel over the tokens, one exchange of counts, one host sync, and one kernel
    over the runs of rows this rank's experts get. The arguments are those of
    rbd_combine_dispatch().

    Args:
        lanes: 1 orders the stage 1 rows by destination rank, for one all2all over
            ep_group. 2 puts the pilots first and this node's direct rows after them,
            each by rank, for the two all2alls of spec_ep.two_lane.

    Returns:
        plan: The indices, split sizes and weights of both stages.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    G, L = gpus_per_node, num_experts // world
    C = L + 1 + 2 * G * L
    T, k = route.shape
    device = route.device

    counts = torch.zeros(world * C + 1, dtype=torch.long, device=device)
    bucket, slot = torch.empty((2, T, k), dtype=torch.long, device=device)
    final_weight = torch.empty((T, k), device=device)
    meta = torch.empty((T, 2 * k), dtype=torch.int32, device=device)
    if T:
        _plan_kernel[(triton.cdiv(T, _BLOCK),)](
            route.long().contiguous(), weights.contiguous(), counts, bucket, slot, final_weight,
            meta, T, rank, G, L, C, world * C, K=k, RAIL=pilot_rule == "rail", BLOCK=_BLOCK,
        )
    counts = counts[:-1].view(world, C)
    arrived = torch.empty_like(counts)
    with record_function("comm/both"):
        dist.all_to_all_single(arrived, counts, group=ep_group)

    # A stage 1 row's place: where its bucket starts in the sending order, plus its
    # slot. A copy that sends no row gets -1, which also parks its token in the spare
    # last entry of send1_token.
    order = _bucket_order(world, L, C, lanes, device)
    rows = counts.flatten()[order]
    start = bucket.new_full((world * C + 1,), -1)
    start[order] = rows.cumsum(0) - rows
    final_index = start[bucket] + slot
    send1_token = bucket.new_empty(T * k + 1)
    send1_token[final_index.flatten()] = torch.arange(T * k, device=device) // k

    # The one host sync: every split size. stage2 is [2G, L]: the rows this rank
    # forwards to each rank of its node, then the rows it gets from each, per expert.
    lo = rank // G * G
    stage2 = arrived[:, L + 1 :].view(world, 2 * G, L).sum(dim=0)
    sizes = torch.cat(
        [counts[:, : L + 1].sum(dim=1), arrived[:, : L + 1].sum(dim=1), stage2.sum(dim=1)]
    ).tolist()
    send1, recv1 = sizes[:world], sizes[world : 2 * world]
    send2, recv2 = sizes[2 * world : 2 * world + G], sizes[2 * world + G :]
    num_direct = sum(recv1[lo : lo + G])
    num_pilots = sum(recv1) - num_direct
    first_direct = num_pilots if lanes == 2 else sum(recv1[:lo])
    pilot_row = torch.arange(num_pilots, device=device)
    forward = stage2[:G].flatten()

    # Order the rows the experts get by expert: the direct rows, then the stage 2 rows.
    runs = torch.cat([arrived[lo : lo + G, :L], stage2[G:]]).flatten()
    grouped = bucket.new_empty((3, num_direct + sum(recv2)))
    _group_kernel[(2 * G * L,)](
        runs, grouped[0], grouped[1], grouped[2], first_direct, num_direct, rank * L, G, L,
        RUNS=triton.next_power_of_2(2 * G * L), BLOCK=_BLOCK,
    )
    return Plan(
        send1_token=send1_token[: sum(send1)],
        send1=send1,
        recv1=recv1,
        direct=(first_direct, first_direct + num_direct),
        pilot_row=pilot_row + num_direct * (pilot_row >= first_direct),
        send2=send2,
        recv2=recv2,
        send2_start=forward.cumsum(0) - forward,
        first_expert=lo * L,
        direct_order=grouped[0, :num_direct],
        direct_place=grouped[1, :num_direct],
        stage2_order=grouped[0, num_direct:],
        stage2_place=grouped[1, num_direct:],
        expert_ids=grouped[2],
        final_index=final_index,
        final_weight=final_weight,
        meta=meta,
    )


def with_meta(tokens: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Append every token's metadata to it, as M more columns of the tokens' dtype.

    Returns:
        rows: [num_tokens, H + M] the rows stage 1 gathers its send buffer from.
    """
    return torch.cat([tokens, meta.view(tokens.dtype)], dim=1)


def _width(received: torch.Tensor, plan: Plan) -> int:
    """H, the token width of received stage 1 rows, which also carry M metadata columns."""
    return received.shape[1] - plan.meta.shape[1] * 4 // received.element_size()


def stage1_dispatch(
    tokens: torch.Tensor,
    plan: Plan,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Send direct rows and pilots to their ranks. Collective over ep_group.

    Returns:
        received: [sum(recv1), H + M] rows by source rank, with their metadata
            columns. plan.direct says which are this node's direct rows.
    """
    with record_function("autoep/rbd_stage1_dispatch"):
        rows = with_meta(tokens, plan.meta)[plan.send1_token]
        return all2all(rows, plan.send1, plan.recv1, ep_group)


def run_experts(expert: Expert, rows: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Call the expert function, unless there are no rows.

    Returns:
        outputs: [num_rows, H] expert outputs, in the order of rows.
    """
    if not rows.shape[0]:
        return rows
    with record_function("autoep/rbd_experts"):
        return expert(rows, expert_ids)


def run_direct(
    expert: Expert,
    received: torch.Tensor,
    plan: Plan,
    back: torch.Tensor,
) -> None:
    """Run the experts on the direct rows and put the outputs in their rows of back.

    Args:
        received: [sum(recv1), H + M] the rows stage 1 received.
        back: [sum(recv1), H] the stage 1 buffer being rebuilt.
    """
    start, end = plan.direct
    rows = received[plan.direct_order, : _width(received, plan)]
    outputs = run_experts(expert, rows, plan.expert_ids[: end - start])
    torch.index_select(outputs, 0, plan.direct_place, out=back[start:end])


def _fan_out(received: torch.Tensor, plan: Plan) -> tuple[torch.Tensor, Merge]:
    """Plan the pilot rank's half from the pilots that arrived: one kernel, no host sync.

    A pilot's metadata names all of its token's experts. The ones on this node are the
    copies this rank makes in stage 2, sent by rank and then expert.

    Returns:
        send2_row: [sum(send2)] received row that each stage 2 row copies.
        merge: What stage2_combine() sums each pilot's replicas with.
    """
    num_pilots, k = plan.pilot_row.numel(), plan.final_index.shape[1]
    send2_row = plan.pilot_row.new_empty(sum(plan.send2))
    merge = Merge(
        plan.pilot_row.new_empty((num_pilots, k)), plan.final_weight.new_empty((num_pilots, k))
    )
    if num_pilots:
        rows = received.view(torch.int32)
        _fanout_kernel[(triton.cdiv(num_pilots, _BLOCK),)](
            rows, plan.pilot_row, plan.send2_start.clone(), send2_row, merge.index, merge.weight,
            num_pilots, rows.shape[1], rows.shape[1] - 2 * k, plan.first_expert,
            plan.send2_start.numel(), K=k, BLOCK=_BLOCK,
        )
    return send2_row, merge


def stage2_dispatch(
    received: torch.Tensor,
    plan: Plan,
    local_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, Merge]:
    """Copy received pilots to their experts on this node. Collective over local_group.

    Args:
        received: [sum(recv1), H + M] the rows stage 1 received.

    Returns:
        rows: [sum(recv2), H] the stage 2 rows, grouped by expert, in the order of
            plan.expert_ids[num_direct:].
        merge: What stage2_combine() sums each pilot's replicas with.
    """
    with record_function("autoep/rbd_plan"):
        send2_row, merge = _fan_out(received, plan)
    with record_function("autoep/rbd_stage2_dispatch"):
        rows = all2all(
            received[send2_row, : _width(received, plan)], plan.send2, plan.recv2, local_group,
            link="intra",
        )
        return rows[plan.stage2_order], merge


def stage2_combine(
    outputs: torch.Tensor,
    plan: Plan,
    merge: Merge,
    local_group: dist.ProcessGroup,
    back: torch.Tensor,
) -> None:
    """Return replica outputs to their pilot's rank and sum them. Collective over local_group.

    Args:
        outputs: [sum(recv2), H] expert outputs of the stage 2 rows, grouped by expert.
        back: [sum(recv1), H] the stage 1 buffer being rebuilt. Each pilot's weighted
            sum is written to its row.
    """
    with record_function("autoep/rbd_stage2_combine"):
        returned = all2all(
            outputs[plan.stage2_place], plan.recv2, plan.send2, local_group, link="intra"
        )
        gather_sum(returned, merge.index, merge.weight, out=back, out_rows=plan.pilot_row)


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
    expert: Expert,
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
        "first": the rank of the token's first copy on that node, in route order.
            Ranks holding popular top-1 experts receive the most pilots.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights, where weights[t, j] scales the
            output of expert route[t, j] for token t.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order. Called twice, on the direct
            rows and on the stage 2 rows, each time with the rows grouped by expert
            and the ids ascending, so it needs no sort.
        route: [num_tokens, k] experts each token is routed to. Experts must be
            placed contiguously, num_experts // world per rank of ep_group, so
            expert e lives on rank e // (num_experts // world).
        num_experts: Total number of experts across all ranks.
        gpus_per_node: Ranks per node. Ranks must be numbered contiguously by node.
        local_group: Process group of this rank's node.
        ep_group: Process group holding the experts. None means the global group.
        pilot_rule: "rail" or "first", as described above.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    with record_function("autoep/rbd_plan"):
        plan = make_plan(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule)
    received = stage1_dispatch(tokens, plan, ep_group)
    back = tokens.new_empty((sum(plan.recv1), tokens.shape[1]))
    run_direct(expert, received, plan, back)
    rows, merge = stage2_dispatch(received, plan, local_group)
    outputs = run_experts(expert, rows, plan.expert_ids[plan.direct_order.numel() :])
    stage2_combine(outputs, plan, merge, local_group, back)
    returned = stage1_combine(back, plan, ep_group)
    return final_merge(returned, plan)
