"""Tutel, DeepSpeed's MOELayer and AutoEP, and DeepEP behind the arguments of plain_combine_dispatch.

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

AutoEP (autoep_combine_dispatch) is the forward of DeepSpeed's AutoEPMoELayer after its
router, built from the layer's own functions: sort the copies by expert, exchange the
per-expert counts, one all2all with uneven splits each way, a regroup by local expert
around the experts, and the layer's weighted sum. Nothing is padded or dropped.

DeepEP (deepep_combine_dispatch) sends one row per (token, rank) through buffers
shared between the GPUs of one node. The expert rank expands each row to its local
experts and sums their weighted outputs, and DeepEP's combine adds those sums on the
source rank. The installed v1.2.1 has no inter-node path, so every rank must be on one
node. The multinode build (DeepEP 7705f53 over NVSHMEM's libfabric transport, with as
many ranks per node as the nodes have GPUs) also sends between nodes: a row for
another node goes by RDMA to the GPU with the same index there, which forwards it to
the row's ranks.

Every adapter marks its steps with record_function("<library>/<step>") and wraps the
library's transfer call in record_function("comm/both"), so spec_ep.phase_clock traces
that call as communication. Whatever the library packs or copies inside the call
counts as communication too, and what its planner exchanges counts as the plan step's
compute. Tutel's pipeline is one call that sends and runs the experts, so the whole
call is one comm event, and the expert callback inside it is the compute step
tutel/experts, which overlaps it.

Usage:
    from benchmark.all2all.baselines import (
        capacity_kept, tutel_combine_dispatch, ds_moe_combine_dispatch,
        autoep_combine_dispatch, make_deepep_buffer, deepep_combine_dispatch,
    )
"""
from collections.abc import Callable
from functools import cache

import deep_ep
import deep_ep_cpp
import deepspeed.comm as ds_comm
import torch
import torch.distributed as dist
from deepspeed.module_inject.auto_ep_layer import (
    _AllToAllV,
    combine_from_routed,
    compute_split_plan,
    permute_by_local_expert,
    unpermute_by_local_expert,
)
from deepspeed.moe.ep_count import count_tokens_per_expert
from deepspeed.moe.sharded_moe import _route_slots, _sparse_decode, _sparse_encode, einsum
from deepspeed.ops.triton_ops import autoep_fused_token_ops
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


def _tutel_pipeline_per_peer(
    padded: torch.Tensor,
    expert: Expert,
    degree: int,
    ep_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Run Tutel's pipeline on the padded buffer laid out as one block per peer and chunk.

    Tutel cuts the buffer's second axis into degree chunks and sends every entry of
    the first axis as one message. [E, capacity, H] therefore sends one message per
    expert. Here the buffer becomes [world, degree * experts_per_rank * chunk, H],
    with a peer's rows ordered by chunk first, so the same call sends one message per
    peer and chunk.

    Returns:
        padded: [E, capacity, H] expert outputs back on the source rank, as
            a2a_ffn_overlap_forward returns them.
    """
    world, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    num_experts, capacity, hidden = padded.shape
    per_rank, chunk = num_experts // world, capacity // degree

    def run_experts(rows: torch.Tensor) -> torch.Tensor:
        # rows is [1, world * per_rank * chunk, H]: by source rank, then expert, then slot.
        with record_function("tutel/experts"):
            row = torch.arange(rows.shape[1], device=rows.device)
            expert_ids = rank * per_rank + row // chunk % per_rank
            return expert(rows.view(-1, hidden), expert_ids).view(rows.shape)

    by_peer = padded.view(world, per_rank, degree, chunk, hidden).transpose(1, 2)
    by_peer = by_peer.reshape(world, -1, hidden)
    with record_function("comm/both"):
        by_peer = a2a_ffn_overlap_forward(
            by_peer, run_experts, degree, use_2dh=False, group=ep_group
        )
    by_peer = by_peer.view(world, degree, per_rank, chunk, hidden).transpose(1, 2)
    return by_peer.reshape(num_experts, capacity, hidden)


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
    per_peer: bool = False,
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
        per_peer: Only with degree > 1, and not Tutel as shipped. Tutel's pipeline
            sends every expert's rows of a chunk as a message of its own, so a peer
            gets experts-per-rank messages per chunk, and NCCL runs the messages to
            one peer one after another. If True, the buffer is laid out peer by peer
            and chunk by chunk first, so Tutel's same kernel sends one message per
            peer per chunk.
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

    def run_experts(rows: torch.Tensor) -> torch.Tensor:
        with record_function("tutel/experts"):
            return _run_local_experts(expert, rows, num_experts, ep_group)

    with record_function("tutel/plan"):
        plan = _gshard_plan(weights, route, num_experts, capacity_factor, degree, ep_group)
    with record_function("tutel/encode"):
        padded = fast_encode(tokens, plan, True)
    if degree > 1 and per_peer:
        with record_function("tutel/pipeline"):
            padded = _tutel_pipeline_per_peer(padded, expert, degree, ep_group)
    elif degree > 1:
        with record_function("tutel/pipeline"), record_function("comm/both"):
            padded = a2a_ffn_overlap_forward(
                padded, run_experts, degree, use_2dh=use_2dh, group=ep_group
            )
    else:
        with record_function("tutel/dispatch"), record_function("comm/both"):
            padded = tutel_comm.all_to_all(padded, 1, 0, use_2dh=use_2dh, group=ep_group)
        padded = run_experts(padded)
        with record_function("tutel/combine"), record_function("comm/both"):
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
        with record_function("comm/both"):
            dist.all_to_all_single(received, padded, group=ep_group)
    with record_function("ds_moe/experts"):
        outputs = _run_local_experts(expert, received, num_experts, ep_group)
    with record_function("ds_moe/combine"):
        returned = torch.empty_like(outputs)
        with record_function("comm/both"):
            dist.all_to_all_single(returned, outputs, group=ep_group)
    with record_function("ds_moe/decode"):
        if dense:
            return einsum("sec,ecm->sm", combine, returned)
        return _sparse_decode(returned.view(-1, tokens.shape[1]), slots, gates, tokens.shape[0])


def autoep_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Expert,
    route: torch.Tensor,
    num_experts: int,
    fused: bool = False,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch, run the experts and combine as DeepSpeed's AutoEPMoELayer does.

    Collective over ep_group. Takes the arguments of plain_combine_dispatch. This is
    the layer's forward after its router, with its default settings (scores applied
    after the experts), calling the layer's own functions in the layer's order.

    Args:
        fused: If True, merges with AutoEP's fused Triton kernel
            (combine_impl="fused_weighted_sum"). If False, with its default weighted
            sum over a dense [num_tokens, k, H] float32 tensor.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    if ds_comm.cdb is None:
        # The layer's functions send through deepspeed.comm, which wraps the process
        # group torch already has.
        ds_comm.init_distributed(dist_backend=dist.get_backend(), verbose=False)
    world, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    per_rank = num_experts // world
    num_tokens, k = route.shape

    with record_function("autoep/plan"):
        order = torch.argsort(route.view(-1), stable=True)
        counts = count_tokens_per_expert(route, num_experts)
        plan = compute_split_plan(route, num_experts, world, per_rank, ep_group, counts)
        rows = tokens[order // k]
    with record_function("autoep/dispatch"), record_function("comm/both"):
        rows = _AllToAllV.apply(ep_group, rows, plan.input_splits, plan.output_splits)
    with record_function("autoep/experts"):
        # The regroup pads every expert's rows to a multiple of 8, with zero rows.
        rows, permutation, padded_counts, received = permute_by_local_expert(
            rows, plan.local_counts_by_source
        )
        # Rows past the last expert's are padding too; they count for the last expert.
        row = torch.arange(rows.shape[0], device=rows.device)
        local = torch.bucketize(row, padded_counts.cumsum(0), right=True).clamp_max(per_rank - 1)
        outputs = expert(rows, rank * per_rank + local)
        outputs = unpermute_by_local_expert(outputs, permutation, received)
    with record_function("autoep/combine"), record_function("comm/both"):
        outputs = _AllToAllV.apply(ep_group, outputs, plan.output_splits, plan.input_splits)
    with record_function("autoep/merge"):
        shape = (1, num_tokens, tokens.shape[1])
        if fused:
            return autoep_fused_token_ops.fused_weighted_restore(
                outputs, top_scores=weights, token_indices_sorted=order, top_k=k, shape=shape
            )[0]
        return combine_from_routed(
            outputs, top_scores=weights, token_indices_sorted=order, top_k=k,
            score_apply="post", combine_impl="weighted_sum", shape=shape,
        )[0]


# Ranks DeepEP takes as one node. v1.2.1 has it fixed at 8; the multinode build reports its own.
DEEPEP_RANKS_PER_NODE = getattr(deep_ep_cpp, "NUM_MAX_NVL_PEERS", 8)


@cache
def make_deepep_buffer(
    group: dist.ProcessGroup,
    num_nvl_bytes: int = int(2e9),
    num_rdma_bytes: int = int(1e9),
) -> deep_ep.Buffer:
    """Build the DeepEP buffer that every call reuses. Collective over group.

    Built once per group and shared by every DeepEP setting: across nodes the buffer
    sets up NVSHMEM, which a process can do only once.

    Args:
        group: Process group holding the experts, in blocks of DEEPEP_RANKS_PER_NODE
            ranks per node. More ranks than that need the multinode build.
        num_nvl_bytes: Bytes each rank shares with the ranks of its node. The default
            is what DeepEP's own intranode test uses, enough for every setting swept
            here.
        num_rdma_bytes: Bytes each rank registers for rows from other nodes, used only
            when the group spans nodes. On two nodes the default holds up to 96 SMs
            with 128-token RDMA chunks; DeepEP raises if a setting needs more.

    Returns:
        buffer: The buffer.
    """
    spans_nodes = group.size() > DEEPEP_RANKS_PER_NODE
    return deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes if spans_nodes else 0)


@cache
def _deepep_configs(
    num_ranks: int,
    num_sms: int,
    dispatch_chunk: int,
    combine_chunk: int,
    rdma_chunk: int,
) -> tuple[deep_ep.Config, deep_ep.Config]:
    """Build DeepEP's dispatch and combine configs for one setting.

    Across nodes a config also holds the tokens per RDMA put and the slots of the
    receive queues. A combine forwarder has 16 / nodes warps that take one token each,
    and combine hangs if an RDMA chunk gives one of them no token, so the chunk must be
    at least that many tokens.

    Returns:
        dispatch: Config of dispatch.
        combine: Config of combine.
    """
    deep_ep.Buffer.set_num_sms(num_sms)
    dispatch = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine = deep_ep.Buffer.get_combine_config(num_ranks)
    num_nodes = max(num_ranks // DEEPEP_RANKS_PER_NODE, 1)
    if num_nodes == 1:
        if dispatch_chunk:
            dispatch = deep_ep.Config(num_sms, dispatch_chunk, 256)
        if combine_chunk:
            combine = deep_ep.Config(num_sms, combine_chunk, 256)
    elif dispatch_chunk or combine_chunk or rdma_chunk:
        # DeepEP's two-node values stand in for the chunks not given.
        dispatch_rdma, combine_rdma = rdma_chunk or 20, rdma_chunk or 28
        if combine_rdma < 16 // num_nodes:
            raise ValueError(f"rdma_chunk {rdma_chunk} is less than the {16 // num_nodes} "
                             f"warps of a combine forwarder on {num_nodes} nodes.")
        # Each node's share of the same-node queue must hold more than one RDMA chunk.
        nvl_slots = max(288, 2 * num_nodes * combine_rdma)
        dispatch = deep_ep.Config(num_sms, dispatch_chunk or 16, nvl_slots,
                                  dispatch_rdma, max(128, 4 * dispatch_rdma))
        combine = deep_ep.Config(num_sms, combine_chunk or 2, nvl_slots,
                                 combine_rdma, max(128, 4 * combine_rdma))
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
    rdma_chunk: int = 0,
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
        dispatch_chunk: Tokens dispatch sends per chunk and channel to a rank of the
            same node. 0 takes DeepEP's recommendation for the group size.
        combine_chunk: The same for combine.
        rdma_chunk: Tokens per RDMA put to another node, for dispatch and combine. 0
            takes DeepEP's recommendation. Only used when the buffer spans nodes.
        fused: If True, sums each row's local expert outputs with gather_sum, as
            plain's fused merge does. If False, scales and adds with index_add_.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    dispatch_config, combine_config = _deepep_configs(
        buffer.group_size, num_sms, dispatch_chunk, combine_chunk, rdma_chunk
    )
    with record_function("deepep/plan"):
        weights = weights.float()
        # tokens_per_node is None when the buffer is within one node.
        tokens_per_rank, tokens_per_node, tokens_per_expert, token_in_rank, _ = (
            buffer.get_dispatch_layout(route, num_experts)
        )
    with record_function("deepep/dispatch"), record_function("comm/both"):
        recv_tokens, recv_route, recv_weights, _, handle, _ = buffer.dispatch(
            tokens,
            topk_idx=route,
            topk_weights=weights,
            num_tokens_per_rank=tokens_per_rank,
            num_tokens_per_rdma_rank=tokens_per_node,
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

    with record_function("deepep/combine"), record_function("comm/both"):
        combined, _, _ = buffer.combine(partial, handle, config=combine_config)
    return combined
