"""Tutel and DeepEP behind the arguments of plain_combine_dispatch, as baselines.

Each adapter dispatches the tokens to their experts' ranks, runs the experts, and
combines the weighted outputs on the source rank, using the library's own planning
and transfers. Both take the recorded routing instead of choosing experts.

Tutel (tutel_combine_dispatch) pads every expert to one capacity, the same on all
ranks, and sends the padded [num_experts, capacity, H] buffer in an all2all with
equal splits. Capacity is the busiest expert over all ranks, so no token is dropped
and the output matches plain. overlap_degree > 1 cuts capacity into that many chunks
and sends them with Tutel's own NCCL communicator while the experts run on the
chunks that have arrived. That path and Tutel's CUDA kernels need a GPU; the rest
also runs on CPU.

DeepEP (deepep_combine_dispatch) sends one row per (token, rank) through buffers
shared between the GPUs of one node. The expert rank expands each row to its local
experts and sums their weighted outputs, and DeepEP's combine adds those sums on the
source rank. This build has no inter-node path, so every rank must be on one node.

Usage:
    from benchmark.all2all.baselines import tutel_combine_dispatch, tutel_capacity
    from benchmark.all2all.baselines import make_deepep_buffer, deepep_combine_dispatch
"""
from collections.abc import Callable

import deep_ep
import torch
import torch.distributed as dist
from tutel.impls import communicate as tutel_comm
from tutel.impls.fast_dispatch import extract_critical, fast_decode, fast_encode
from tutel.impls.overlap import a2a_ffn_overlap_forward


def _tutel_plan(
    weights: torch.Tensor,
    route: torch.Tensor,
    num_experts: int,
    alignment: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple:
    """Plan one call with Tutel's own planner. Collective over ep_group.

    Tutel picks experts itself, with a top-k over a dense [num_tokens, num_experts]
    score matrix. Scattering the weights into zeros makes it pick the recorded
    experts with the given weights. Weights must be positive and a token's experts
    distinct.

    Returns:
        plan: Tutel's (num_experts, expert ids, slots, gates, capacity, counts).
            capacity is the rows of the busiest expert over all ranks, rounded up to
            a multiple of alignment.
    """
    scores = torch.zeros(route.shape[0], num_experts, device=route.device)
    scores.scatter_(1, route, weights.float())
    plan, _ = extract_critical(
        scores,
        top_k=route.shape[1],
        loss_fn=None,
        capacity_factor=0,
        normalize_gate=False,
        alignment=alignment,
        group=ep_group,
    )
    return plan


def tutel_capacity(
    weights: torch.Tensor,
    route: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None = None,
) -> int:
    """Rows Tutel pads every expert to on this routing. Collective over ep_group.

    Every rank sends num_experts * capacity rows each way, whatever its real copies.

    Returns:
        capacity: The most rows any one rank sends to any one expert.
    """
    return _tutel_plan(weights, route, num_experts, 1, ep_group)[4]


def tutel_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
    overlap_degree: int = 1,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with Tutel.

    Collective over ep_group. Takes the arguments of plain_combine_dispatch. The
    experts also run on the padding rows, which are zero. Tutel scatters and merges
    in float32 and sends in the dtype of tokens.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to, as int64.
        num_experts: Total number of experts across all ranks.
        overlap_degree: Chunks Tutel pipelines the transfers and experts in. 1 sends
            everything in one all2all each way.
        ep_group: Process group holding the experts. None means the global group.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    world, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    experts_per_rank = num_experts // world
    plan = _tutel_plan(weights, route, num_experts, overlap_degree, ep_group)

    def run_experts(rows: torch.Tensor) -> torch.Tensor:
        """Run this rank's experts on [experts_per_rank, padded rows, H]."""
        first = rank * experts_per_rank
        expert_ids = torch.arange(first, first + experts_per_rank, device=rows.device)
        expert_ids = expert_ids.repeat_interleave(rows.shape[1])
        return expert(rows.reshape(-1, rows.shape[-1]), expert_ids).view(rows.shape)

    padded = fast_encode(tokens, plan, True)
    if overlap_degree > 1:
        padded = a2a_ffn_overlap_forward(
            padded, run_experts, overlap_degree, use_2dh=False, group=ep_group
        )
    else:
        padded = tutel_comm.all_to_all(padded, 1, 0, group=ep_group)
        padded = run_experts(padded)
        padded = tutel_comm.all_to_all(padded, 0, 1, group=ep_group)
    return fast_decode(padded, plan, True)


def make_deepep_buffer(group: dist.ProcessGroup, hidden: int) -> deep_ep.Buffer:
    """Build the DeepEP buffer that every call reuses. Collective over group.

    Args:
        group: Process group holding the experts. All its ranks must be on one node.
        hidden: H, the width of the bf16 tokens.

    Returns:
        buffer: The buffer, sized by DeepEP's recommended configs for this group.
    """
    num_ranks, hidden_bytes = group.size(), hidden * 2
    configs = (
        deep_ep.Buffer.get_dispatch_config(num_ranks),
        deep_ep.Buffer.get_combine_config(num_ranks),
    )
    num_nvl_bytes = max(c.get_nvl_buffer_size_hint(hidden_bytes, num_ranks) for c in configs)
    return deep_ep.Buffer(group, num_nvl_bytes, 0)


def deepep_combine_dispatch(
    buffer: deep_ep.Buffer,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with DeepEP.

    Collective over the buffer's group. Takes the arguments of plain_combine_dispatch,
    with the buffer in place of ep_group.

    Args:
        buffer: From make_deepep_buffer().
        tokens: [num_tokens, H] bf16 tokens on this rank.
        weights: [num_tokens, k] combine weights.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to, as int64.
        num_experts: Total number of experts across all ranks.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    weights = weights.float()
    tokens_per_rank, _, tokens_per_expert, token_in_rank, _ = buffer.get_dispatch_layout(
        route, num_experts
    )
    recv_tokens, recv_route, recv_weights, _, handle, _ = buffer.dispatch(
        tokens,
        topk_idx=route,
        topk_weights=weights,
        num_tokens_per_rank=tokens_per_rank,
        is_token_in_rank=token_in_rank,
        num_tokens_per_expert=tokens_per_expert,
    )

    # recv_route holds this rank's local expert ids, and -1 for experts on other ranks.
    row, slot = (recv_route >= 0).nonzero(as_tuple=True)
    expert_ids = recv_route[row, slot] + buffer.rank * (num_experts // buffer.group_size)
    outputs = expert(recv_tokens[row], expert_ids).float()
    outputs = outputs * recv_weights[row, slot].unsqueeze(1)
    partial = torch.zeros(recv_tokens.shape, device=tokens.device).index_add_(0, row, outputs)

    combined, _, _ = buffer.combine(partial.to(tokens.dtype), handle)
    return combined
