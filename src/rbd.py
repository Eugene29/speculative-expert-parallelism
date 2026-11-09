"""Redundancy-Bypassing Dispatch (RBD) for expert-parallel MoE communication.

Based on X-MoE (arXiv 2508.13337). In a plain all2all, a token routed to several
experts on one node crosses the network once per expert, and so does each output on
the way back. RBD sends one row per (token, other node) between nodes: the pilot.
The pilot's rank then copies it to the token's experts on that node, the replicas.
Tokens routed to experts on their own node skip the pilot step. Which rank on a node
receives the pilot is the pilot rule; see RBD_combine_dispatch().

Dispatch:
    1. Stage 1, over ep_group: send each token straight to its experts on this
       node, and one pilot per other node it needs. One all2all carries both, so
       the NVLink and network traffic run together.
    2. Stage 2, over local_group: each rank copies the pilots it received to the
       experts they need on its node.

Combine reverses it:
    3. Stage 2 back: replica outputs return to their pilot's rank, which scales and
       sums them into one row per pilot.
    4. Stage 1 back: pilot sums and same-node outputs return to the source rank,
       which scales the same-node outputs and sums each token's rows.

Planning costs one all2all of fixed-size counts, a single host sync to read every
split size, and one small all2all of pilot metadata. The "even" pilot rule also
all-gathers per-node pilot counts before planning. Expert ids are never sent: rows
are sorted by expert inside each block, so per-expert counts give the ids.

TODO: Support backprop. The all2alls don't carry gradients, so tokens and expert
parameters get no gradient through RBD_combine_dispatch.

Usage:
    python -m pytest tests/test_rbd.py -v
"""
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

__all__ = ["RBD_combine_dispatch"]


if triton is not None:

    @triton.jit
    def _gather_sum_kernel(
        src,
        index,
        weight,
        out,
        out_rows,
        H,
        src_stride,
        out_stride,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_OUT_ROWS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        in_row = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(K):
            i = tl.load(index + row * K + j)
            w = tl.load(weight + row * K + j, mask=i >= 0, other=0.0)
            # Padding slots (index -1) load nothing, so they never touch memory.
            x = tl.load(src + i * src_stride + cols, mask=in_row & (i >= 0), other=0.0)
            acc += w * x.to(tl.float32)
        dst = row
        if HAS_OUT_ROWS:
            dst = tl.load(out_rows + row)
        tl.store(out + dst * out_stride + cols, acc.to(out.dtype.element_ty), mask=in_row)


def _gather_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    out_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[out_rows[r]] = sum_j weight[r, j] * src[index[r, j]], skipping index -1.

    out_rows defaults to r and must be unique; other rows of out are untouched.
    Sums in float32 (float64 for float64 src). Triton on CUDA, else PyTorch ops;
    float64 always takes the PyTorch path.
    """
    num_out, K = index.shape
    H = src.shape[1]
    if out is None:
        out = src.new_empty((num_out, H))

    if triton is None or not src.is_cuda or src.dtype == torch.float64:
        # Point empty slots at an appended zero row so every slot can be gathered.
        padded = torch.cat([src, src.new_zeros((1, H))])
        safe = torch.where(index >= 0, index, src.shape[0])
        sum_dtype = torch.promote_types(src.dtype, torch.float32)
        valid_weight = torch.where(index >= 0, weight, 0).to(sum_dtype)
        merged = (padded[safe].to(sum_dtype) * valid_weight.unsqueeze(-1)).sum(dim=1)
        if out_rows is None:
            out[:num_out] = merged.to(out.dtype)
        else:
            out[out_rows] = merged.to(out.dtype)
        return out

    if num_out == 0 or H == 0:
        return out
    if src.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("gather_sum needs rows that are contiguous along H.")
    BLOCK = min(triton.next_power_of_2(H), 1024)
    grid = (num_out, triton.cdiv(H, BLOCK))
    _gather_sum_kernel[grid](
        src,
        index.contiguous(),
        weight.float().contiguous(),
        out,
        out_rows.contiguous() if out_rows is not None else index,
        H,
        src.stride(0),
        out.stride(0),
        K=K,
        BLOCK=BLOCK,
        HAS_OUT_ROWS=out_rows is not None,
        num_warps=4,
    )
    return out


class _Plan(NamedTuple):
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


def _all2all(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-size all2all of rows grouped by destination; returns rows by source."""
    if out is None:
        out = rows.new_empty((sum(recv), *rows.shape[1:]))
    dist.all_to_all_single(out, rows, recv, send, group=group)
    return out


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
    pilot: str,
    pilot_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """[num_tokens, k] pilot rank for each copy's node, by the pilot rule.

    same_node[t, j, i] says copies j and i of token t share a node. pilot_offset,
    needed for "even", counts the pilots lower source ranks send to each node.
    Entries for same-node copies are computed too, but the caller ignores them.
    """
    G = gpus_per_node
    if pilot == "even":
        if pilot_offset is None:
            raise ValueError('pilot="even" requires per-node pilot offsets.')
        # Round-robin over the node's ranks in (source rank, token) order, starting
        # where the lower source ranks left off.
        first = first_slot == torch.arange(expert_rank.shape[1], device=expert_rank.device)
        pilot_rank = expert_node * G
        for n in range(pilot_offset.numel()):
            on_node = expert_node == n
            ordinal = (first & on_node).sum(dim=1).cumsum(dim=0) - 1
            receiver = n * G + (pilot_offset[n] + ordinal) % G
            pilot_rank = torch.where(on_node, receiver.unsqueeze(1), pilot_rank)
    elif pilot == "rail":
        pilot_rank = expert_node * G + rank % G
    elif pilot == "first":
        pilot_rank = expert_rank.gather(1, first_slot)
    elif pilot == "most":
        # Score = copies held * G + a tie bonus below G that falls with cyclic
        # distance after the rail rank, so the bonus never outweighs an extra copy.
        held =(expert_rank.unsqueeze(2) == expert_rank.unsqueeze(1)).sum(dim=2)
        after_rail = (expert_rank - rank) % G
        score = held * G + (G - 1 - after_rail)
        best = torch.where(same_node, score.unsqueeze(1), -1).argmax(dim=2)
        pilot_rank = expert_rank.gather(1, best)
    else:
        raise ValueError(f'pilot must be "first", "rail", "most" or "even", got {pilot!r}.')

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
    Returns this rank's counts per destination and each source's counts for it.
    """
    unused = world * C
    # scatter_add_ rather than bincount, which syncs to find its output size.
    counts = torch.zeros(unused + 1, dtype=torch.long, device=slots.device)
    counts.scatter_add_(0, slots, torch.ones_like(slots))
    counts = counts[:unused].view(world, C)
    recv_counts = torch.empty_like(counts)
    dist.all_to_all_single(recv_counts, counts, group=ep_group)
    return counts, recv_counts


def _send_pilot_meta(
    pilot_copy: torch.Tensor,
    covered: torch.Tensor,
    node_expert: torch.Tensor,
    weights: torch.Tensor,
    pilot_send: list[int],
    pilot_recv: list[int],
    ep_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Send each pilot's node-local experts and weights to its rank. Collective.

    Weights ride as int32 bits so one all2all carries both. Returns, per received
    pilot and copy, the node-local expert (-1 if on another node) and its weight.
    """
    k = covered.shape[1]
    pilot_token = pilot_copy // k
    meta = torch.cat(
        [
            torch.where(covered, node_expert[pilot_token], -1).int(),
            torch.where(covered, weights[pilot_token].float(), 0.0).view(torch.int32),
        ],
        dim=1,
    )
    meta = _all2all(meta, pilot_send, pilot_recv, ep_group)
    return meta[:, :k], meta[:, k:].view(torch.float32)


def _make_plan(
    route: torch.Tensor,
    weights: torch.Tensor,
    E: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot: str = "rail",
) -> _Plan:
    """Decide where every row goes and tell the receiving ranks. Sends no tokens.

    Collective over ep_group, with a single host sync to read every split size.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    G, L = gpus_per_node, E // world
    node = rank // G
    num_tokens, k = route.shape
    device = route.device
    route = route.long()

    expert_rank = route // L
    expert_node = expert_rank // G
    local_expert = route % L
    direct = expert_node == node

    # One route slot represents each (token, remote node), independently of which
    # rank receives its pilot. same[t, j, i] groups slots by destination node.
    same = expert_node.unsqueeze(2) == expert_node.unsqueeze(1)
    pilot_slot = same.to(torch.uint8).argmax(dim=2)
    is_pilot = (pilot_slot == torch.arange(k, device=device)) & ~direct
    pilot_offset = None
    if pilot == "even":
        pilot_offset = _even_pilot_offsets(expert_node, is_pilot, world // G, ep_group)
    pilot_rank = _select_pilot_rank(
        expert_rank, expert_node, same, pilot_slot, rank, G, pilot, pilot_offset
    )

    # Pack all counts into one exchange so every split size can be read in one
    # host sync. Each destination has direct expert counts, a pilot count, then
    # stage 2 send counts by local rank and receive counts by (local rank, expert).
    pilot_column = L
    stage2_send_start = pilot_column + 1
    stage2_recv_start = stage2_send_start + G
    C = stage2_recv_start + G * L
    unused = world * C
    stage1 = torch.where(
        direct,
        expert_rank * C + local_expert,
        torch.where(is_pilot, pilot_rank * C + pilot_column, unused),
    )
    to_pilot = torch.where(~direct, pilot_rank * C + stage2_send_start + expert_rank % G, unused)
    to_expert = torch.where(
        ~direct,
        expert_rank * C + stage2_recv_start + (pilot_rank % G) * L + local_expert,
        unused,
    )
    slots = torch.cat([stage1.flatten(), to_pilot.flatten(), to_expert.flatten()])
    counts, recv_counts = _exchange_counts(slots, world, C, ep_group)

    # Every split size in one host sync.
    stage2_in = recv_counts[:, stage2_recv_start:].view(world, G, L).sum(dim=0)
    sizes = torch.cat(
        [
            counts[:, :stage2_send_start].sum(dim=1),
            recv_counts[:, :stage2_send_start].sum(dim=1),
            counts[:, pilot_column],
            recv_counts[:, pilot_column],
            recv_counts[:, stage2_send_start:stage2_recv_start].sum(dim=0),
            stage2_in.sum(dim=1),
        ]
    ).tolist()
    send1, recv1 = sizes[:world], sizes[world : 2 * world]
    pilot_send, pilot_recv = sizes[2 * world : 3 * world], sizes[3 * world : 4 * world]
    send2, recv2 = sizes[4 * world : 4 * world + G], sizes[4 * world + G :]

    # Stage 1 send order: by destination, then expert for direct rows. Copies that
    # a pilot covers sort to the end and aren't sent.
    num_send1 = sum(send1)
    stage1_key = torch.where(stage1 < unused, stage1 // C * (L + 1) + stage1 % C, unused)
    order = stage1_key.flatten().sort(stable=True).indices
    sent = order[:num_send1]

    # This node's ranks are contiguous, so its direct rows are one block of the send
    # order and of the receive buffer, with pilots on either side.
    lo, hi = node * G, (node + 1) * G
    send_lo = sum(send1[:lo])
    send_hi = send_lo + sum(send1[lo:hi])
    recv_lo = sum(recv1[:lo])
    num_direct = sum(recv1[lo:hi])

    pilot_copy = torch.cat([sent[:send_lo], sent[send_hi:]])
    copy_expert, merge_weight = _send_pilot_meta(
        pilot_copy,
        same.view(-1, k)[pilot_copy],
        (expert_rank % G) * L + local_expert,
        weights,
        pilot_send,
        pilot_recv,
        ep_group,
    )

    # Received pilots skip over the direct block in the stage 1 buffer.
    num_pilots = sum(pilot_recv)
    pilot_row = torch.arange(num_pilots, device=device)
    pilot_row = pilot_row + num_direct * (pilot_row >= recv_lo)

    # Stage 2 send order: by node-local expert, which sorts by destination and then
    # by expert, as the receivers' counts expect. Empty slots sort to the end.
    num_send2 = sum(send2)
    stage2_key = torch.where(copy_expert >= 0, copy_expert, G * L).flatten()
    order2 = stage2_key.sort(stable=True).indices[:num_send2]
    send2_row = pilot_row[order2 // k]
    merge_index = torch.full((num_pilots * k,), -1, dtype=torch.long, device=device)
    merge_index[order2] = torch.arange(num_send2, device=device)

    expert_ids = torch.cat(
        [
            _expert_ids(recv_counts[lo:hi, :L], rank * L, num_direct),
            _expert_ids(stage2_in, rank * L, sum(recv2)),
        ]
    )

    # Each copy's row in the returned stage 1 buffer, which keeps the send order.
    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=device)
    position = position.view(num_tokens, k)
    final_index = torch.where(position < num_send1, position, -1)
    final_weight = torch.where(direct, weights.float(), is_pilot.float())

    return _Plan(
        send1_token=sent // k,
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


def _dispatch(
    tokens: torch.Tensor,
    plan: _Plan,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Send tokens to their experts' ranks; returns direct rows, then stage 2 rows."""
    with record_function("autoep/rbd_stage1_dispatch"):
        received = _all2all(tokens[plan.send1_token], plan.send1, plan.recv1, ep_group)

    with record_function("autoep/rbd_stage2_dispatch"):
        start, end = plan.direct
        num_direct = end - start
        expert_inputs = tokens.new_empty((num_direct + sum(plan.recv2), tokens.shape[1]))
        # Stage 2 lands straight in its part of the input buffer. The direct rows
        # are copied in on the compute stream while it runs.
        _all2all(
            received[plan.send2_row],
            plan.send2,
            plan.recv2,
            local_group,
            out=expert_inputs[num_direct:],
        )
        expert_inputs[:num_direct].copy_(received[start:end])
    return expert_inputs


def _combine(
    outputs: torch.Tensor,
    plan: _Plan,
    num_tokens: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Reverse _dispatch(): return expert outputs and sum each token's, weighted."""
    start, end = plan.direct
    num_direct = end - start
    H = outputs.shape[1]
    outputs = outputs.contiguous()

    with record_function("autoep/rbd_stage2_combine"):
        returned = _all2all(outputs[num_direct:], plan.recv2, plan.send2, local_group)
        # Rebuild the stage 1 buffer: direct outputs in their block, and each
        # pilot's weighted sum written straight into its row.
        back = outputs.new_empty((sum(plan.recv1), H))
        back[start:end].copy_(outputs[:num_direct])
        _gather_sum(
            returned, plan.merge_index, plan.merge_weight, out=back, out_rows=plan.pilot_row
        )

    with record_function("autoep/rbd_stage1_combine"):
        returned = _all2all(back, plan.recv1, plan.send1, ep_group)

    with record_function("autoep/rbd_final_merge"):
        output = outputs.new_empty((num_tokens, H))
        return _gather_sum(returned, plan.final_index, plan.final_weight, out=output)


def RBD_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    E: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
    pilot: str = "rail",
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
            placed contiguously, E // world per rank of ep_group, so expert e
            lives on rank e // (E // world).
        E: Total number of experts across all ranks.
        gpus_per_node: Ranks per node. Ranks must be numbered contiguously by node.
        local_group: Process group of this rank's node.
        ep_group: Process group holding the experts. None means the global group.
        pilot: "rail", "even", "first" or "most", as described above.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    with record_function("autoep/rbd_plan"):
        plan = _make_plan(route, weights, E, gpus_per_node, ep_group, pilot)
    expert_inputs = _dispatch(tokens, plan, local_group, ep_group)
    with record_function("autoep/rbd_experts"):
        outputs = expert(expert_inputs, plan.expert_ids)
    return _combine(outputs, plan, tokens.shape[0], local_group, ep_group)
