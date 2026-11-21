"""Tutel, DeepSpeed's MOELayer and DeepEP behind the arguments of plain_combine_dispatch.

Each adapter dispatches the tokens to their experts' ranks, runs the experts, and
combines the weighted outputs on the source rank, using the library's own planning
and transfers. All take the recorded routing instead of choosing experts, and each
exposes the library's performance settings as keyword arguments.

Tutel (tutel_combine_dispatch) and DeepSpeed's MOELayer (ds_moe_combine_dispatch) both
follow GShard: every expert is padded to one capacity, the same on all ranks, and the
padded [num_experts, capacity, H] buffer goes in an all2all with equal splits. They
share one planner (Tutel's), so at the same capacity_factor they send the same rows
and differ only in how they fill and read the padded buffer. capacity_factor 0 pads
to the busiest expert over all ranks, so no copy is dropped and the output matches
plain. A positive capacity_factor f gives every expert f times its even share and
drops the copies past it, as both libraries do by default with f = 1; capacity_kept
tells which copies survive. Tutel's degree > 1 cuts capacity into that many chunks
and sends them with Tutel's own NCCL communicator while the experts run on the chunks
that have arrived. That path and Tutel's CUDA kernels need a GPU. DeepSpeed's
dense=True fills and reads the padded buffer with the einsums over a dense
[num_tokens, num_experts, capacity] matrix that MOELayer used before it got a sparse
encode, which is the DeepSpeed that Tutel was published against.

DeepEP (deepep_combine_dispatch) sends one row per (token, rank) through buffers
shared between the GPUs of one node. The expert rank expands each row to its local
experts and sums their weighted outputs, and DeepEP's combine adds those sums on the
source rank. This build has no inter-node path, so every rank must be on one node.

Usage:
    from benchmark.all2all.baselines import (
        capacity_kept, tutel_combine_dispatch, ds_moe_combine_dispatch,
        make_deepep_buffer, deepep_combine_dispatch,
    )
"""
from collections.abc import Callable
from functools import cache

import deep_ep
import torch
import torch.distributed as dist
from deepspeed.moe.sharded_moe import _route_slots, _sparse_decode, _sparse_encode, einsum
from torch.profiler import record_function
from tutel.impls import communicate as tutel_comm
from tutel.impls.fast_dispatch import extract_critical, fast_decode, fast_encode
from tutel.impls.overlap import a2a_ffn_overlap_forward

from spec_ep.ops import gather_sum

Expert = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _gshard_plan(
    weights: torch.Tensor,
    route: torch.Tensor,
    num_experts: int,
    capacity_factor: float,
    alignment: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple:
    """Plan one call with Tutel's planner. Collective over ep_group.

    Tutel picks experts itself, with a top-k over a dense [num_tokens, num_experts]
    score matrix. Scattering the weights into zeros makes it pick the recorded
    experts with the given weights. Weights must be positive and a token's experts
    distinct.

    Returns:
        plan: Tutel's (num_experts, expert ids, slots, gates, capacity, counts). The
            ids, slots and gates are lists of k [num_tokens] tensors, ordered by
            falling weight. A copy whose slot is at or past capacity is dropped.
            capacity is rounded up to a multiple of alignment.
    """
    scores = torch.zeros(route.shape[0], num_experts, device=route.device)
    scores.scatter_(1, route, weights.float())
    plan, _ = extract_critical(
        scores,
        top_k=route.shape[1],
        loss_fn=None,
        capacity_factor=capacity_factor,
        normalize_gate=False,
        alignment=alignment,
        group=ep_group,
    )
    return plan


def capacity_kept(
    weights: torch.Tensor,
    route: torch.Tensor,
    num_experts: int,
    capacity_factor: float = 0,
    degree: int = 1,
    ep_group: dist.ProcessGroup | None = None,
) -> tuple[int, torch.Tensor]:
    """Capacity of the GShard adapters on this routing, and the copies it keeps.

    Collective over ep_group. Every rank sends num_experts * capacity rows each way,
    whatever its real copies.

    Returns:
        capacity: Rows every expert is padded to, per source rank.
        kept: [num_tokens, k] bool, False where the copy route[t, j] is dropped.
    """
    _, experts, slots, _, capacity, _ = _gshard_plan(
        weights, route, num_experts, capacity_factor, degree, ep_group
    )
    kept = torch.zeros(route.shape[0], num_experts, dtype=torch.bool, device=route.device)
    kept.scatter_(1, torch.stack(experts, dim=1).long(), torch.stack(slots, dim=1) < capacity)
    return capacity, kept.gather(1, route)


def _run_local_experts(
    expert: Expert,
    rows: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Run this rank's experts on padded rows whose first axis cycles over them.

    Args:
        rows: [n * experts_per_rank, padded rows, H], where rows[i] belong to this
            rank's expert i % experts_per_rank. Padding rows are zero and run too.

    Returns:
        outputs: The expert outputs, shaped as rows.
    """
    world, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    experts_per_rank = num_experts // world
    local = torch.arange(rows.shape[0], device=rows.device) % experts_per_rank
    expert_ids = (rank * experts_per_rank + local).repeat_interleave(rows.shape[1])
    return expert(rows.reshape(-1, rows.shape[-1]), expert_ids).view(rows.shape)


def tutel_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Expert,
    route: torch.Tensor,
    num_experts: int,
    capacity_factor: float = 0,
    degree: int = 1,
    local_size: int = 0,
    half: bool = False,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with Tutel.

    Collective over ep_group. Takes the arguments of plain_combine_dispatch, then
    Tutel's settings. Tutel scatters and merges in float32 and sends in the dtype of
    tokens, unless that dtype is float16, the one its kernels handle directly.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to, as int64.
        num_experts: Total number of experts across all ranks.
        capacity_factor: 0 pads to the busiest expert and drops nothing. f > 0 gives
            every expert f times its even share and drops the rest; Tutel's default
            is 1.
        degree: Chunks Tutel pipelines the transfers and experts in. 1 sends
            everything in one all2all each way.
        local_size: Ranks per node for Tutel's 2D hierarchical all2all, which goes
            between nodes first and within the node second. 0 uses one flat all2all.
        half: If True, casts the tokens to float16 first, so Tutel skips float32 and
            the experts and the output are float16. Tutel caches its kernels for the
            first dtype it sees, so one process must use half on every Tutel call or
            on none.
        ep_group: Process group holding the experts. None means the global group.

    Returns:
        output: [num_tokens, H] weighted sum of each token's kept expert outputs, in
            input order.
    """
    if half:
        tokens = tokens.half()
    use_2dh = local_size > 0
    if use_2dh:
        tutel_comm.PrimAllToAll2D.LOCAL_SIZE = local_size
    run_experts = lambda rows: _run_local_experts(expert, rows, num_experts, ep_group)

    with record_function("tutel/plan"):
        plan = _gshard_plan(weights, route, num_experts, capacity_factor, degree, ep_group)
    with record_function("tutel/encode"):
        padded = fast_encode(tokens, plan, True)
    if degree > 1:
        with record_function("tutel/pipeline"):
            padded = a2a_ffn_overlap_forward(
                padded, run_experts, degree, use_2dh=use_2dh, group=ep_group
            )
    else:
        with record_function("tutel/dispatch"):
            padded = tutel_comm.all_to_all(padded, 1, 0, use_2dh=use_2dh, group=ep_group)
        with record_function("tutel/experts"):
            padded = run_experts(padded)
        with record_function("tutel/combine"):
            padded = tutel_comm.all_to_all(padded, 0, 1, use_2dh=use_2dh, group=ep_group)
    with record_function("tutel/decode"):
        return fast_decode(padded, plan, True)


def ds_moe_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Expert,
    route: torch.Tensor,
    num_experts: int,
    capacity_factor: float = 0,
    dense: bool = False,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch, run the experts and combine as DeepSpeed's MOELayer does.

    Collective over ep_group. Takes the arguments of plain_combine_dispatch, then
    capacity_factor as in tutel_combine_dispatch. This is the forward of
    deepspeed.moe.sharded_moe.MOELayer after its gate: its own encode into the padded
    buffer, an all2all with equal splits each way, and its own decode. The plan
    comes from Tutel's planner, which hands out slots as DeepSpeed's gate does, by a
    running count per expert.

    Args:
        dense: If True, encodes and decodes with MOELayer's earlier einsums over a
            dense [num_tokens, num_experts, capacity] matrix of combine weights. The
            matrix is filled with one scatter here; the earlier gate built it from
            one-hot products, so this is a lower bound on what that layer cost.

    Returns:
        output: [num_tokens, H] weighted sum of each token's kept expert outputs, in
            input order.
    """
    with record_function("ds_moe/plan"):
        _, experts, slots, gates, capacity, _ = _gshard_plan(
            weights, route, num_experts, capacity_factor, 1, ep_group
        )
        experts, slots, gates = torch.stack(experts), torch.stack(slots), torch.stack(gates)
        # DeepSpeed marks a dropped copy with a negative expert id.
        slots = _route_slots(torch.where(slots < capacity, experts, -1), slots,
                             num_experts, capacity)
        if dense:
            # Dropped copies land on the scratch slot past the buffer, cut off below.
            combine = torch.zeros(tokens.shape[0], num_experts * capacity + 1, device=tokens.device)
            combine.scatter_(1, slots.t(), gates.t().float())
            combine = combine[:, :-1].view(-1, num_experts, capacity).to(tokens.dtype)
    with record_function("ds_moe/encode"):
        if dense:
            padded = einsum("sec,sm->ecm", (combine != 0).to(tokens.dtype), tokens)
        else:
            padded = _sparse_encode(tokens, slots, num_experts, capacity)
    with record_function("ds_moe/dispatch"):
        received = torch.empty_like(padded)
        dist.all_to_all_single(received, padded, group=ep_group)
    with record_function("ds_moe/experts"):
        outputs = _run_local_experts(expert, received, num_experts, ep_group)
    with record_function("ds_moe/combine"):
        returned = torch.empty_like(outputs)
        dist.all_to_all_single(returned, outputs, group=ep_group)
    with record_function("ds_moe/decode"):
        if dense:
            return einsum("sec,ecm->sm", combine, returned)
        return _sparse_decode(returned.view(-1, tokens.shape[1]), slots, gates, tokens.shape[0])


def make_deepep_buffer(group: dist.ProcessGroup, num_nvl_bytes: int = int(2e9)) -> deep_ep.Buffer:
    """Build the DeepEP buffer that every call reuses. Collective over group.

    Args:
        group: Process group holding the experts. All its ranks must be on one node.
        num_nvl_bytes: Bytes each rank shares with the others. The default is what
            DeepEP's own intranode test uses, enough for every setting swept here.

    Returns:
        buffer: The buffer.
    """
    return deep_ep.Buffer(group, num_nvl_bytes, 0)


@cache
def _deepep_configs(
    num_ranks: int,
    num_sms: int,
    dispatch_chunk: int,
    combine_chunk: int,
) -> tuple[deep_ep.Config, deep_ep.Config]:
    """Build DeepEP's dispatch and combine configs for one setting.

    Returns:
        dispatch: Config of dispatch.
        combine: Config of combine.
    """
    deep_ep.Buffer.set_num_sms(num_sms)
    dispatch = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine = deep_ep.Buffer.get_combine_config(num_ranks)
    if dispatch_chunk:
        dispatch = deep_ep.Config(num_sms, dispatch_chunk, 256)
    if combine_chunk:
        combine = deep_ep.Config(num_sms, combine_chunk, 256)
    return dispatch, combine


def deepep_combine_dispatch(
    buffer: deep_ep.Buffer,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Expert,
    route: torch.Tensor,
    num_experts: int,
    num_sms: int = 20,
    dispatch_chunk: int = 0,
    combine_chunk: int = 0,
    fused: bool = False,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with DeepEP.

    Collective over the buffer's group. Takes the arguments of plain_combine_dispatch,
    with the buffer in place of ep_group, then DeepEP's settings.

    Args:
        buffer: From make_deepep_buffer().
        tokens: [num_tokens, H] bf16 tokens on this rank.
        weights: [num_tokens, k] combine weights.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to, as int64.
        num_experts: Total number of experts across all ranks.
        num_sms: Streaming multiprocessors DeepEP's kernels run on. Must be even;
            DeepEP's default is 20.
        dispatch_chunk: Tokens dispatch sends per chunk and channel. 0 takes DeepEP's
            recommendation for the group size.
        combine_chunk: The same for combine.
        fused: If True, sums each row's local expert outputs with gather_sum, as
            plain's fused merge does. If False, scales and adds with index_add_.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    dispatch_config, combine_config = _deepep_configs(
        buffer.group_size, num_sms, dispatch_chunk, combine_chunk
    )
    with record_function("deepep/plan"):
        weights = weights.float()
        tokens_per_rank, _, tokens_per_expert, token_in_rank, _ = buffer.get_dispatch_layout(
            route, num_experts
        )
    with record_function("deepep/dispatch"):
        recv_tokens, recv_route, recv_weights, _, handle, _ = buffer.dispatch(
            tokens,
            topk_idx=route,
            topk_weights=weights,
            num_tokens_per_rank=tokens_per_rank,
            is_token_in_rank=token_in_rank,
            num_tokens_per_expert=tokens_per_expert,
            config=dispatch_config,
        )

    with record_function("deepep/expand"):
        # recv_route holds this rank's local expert ids, and -1 for experts on other ranks.
        local = recv_route >= 0
        row, slot = local.nonzero(as_tuple=True)
        expert_ids = recv_route[row, slot] + buffer.rank * (num_experts // buffer.group_size)
        inputs = recv_tokens[row]
    with record_function("deepep/experts"):
        outputs = expert(inputs, expert_ids)
    with record_function("deepep/reduce"):
        if fused:
            position = torch.full_like(recv_route, -1)
            position[local] = torch.arange(row.numel(), device=row.device)
            partial = gather_sum(outputs, position, recv_weights)
        else:
            outputs = outputs.float() * recv_weights[row, slot].unsqueeze(1)
            partial = torch.zeros(recv_tokens.shape, device=tokens.device)
            partial = partial.index_add_(0, row, outputs).to(tokens.dtype)

    with record_function("deepep/combine"):
        combined, _, _ = buffer.combine(partial, handle, config=combine_config)
    return combined
