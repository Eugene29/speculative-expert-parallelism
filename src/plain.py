"""Plain all2all dispatch and combine: the reference RBD is checked and timed against.

Sends one row per (token, expert) pair in a single all2all each way, with no notion
of nodes and no deduplication across experts on the same node.

Usage:
    from spec_ep.plain import plain_combine_dispatch
"""
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.profiler import record_function

from spec_ep.ops import all2all, gather_sum

__all__ = ["plain_combine_dispatch"]


def plain_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
    fused_merge: bool = False,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with plain all2alls.

    Collective over ep_group. Takes the same arguments as rbd_combine_dispatch, minus
    the node layout it doesn't need.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights, where weights[t, j] scales the
            output of expert route[t, j] for token t.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to. Experts must be placed
            contiguously, num_experts // world per rank of ep_group.
        num_experts: Total number of experts across all ranks.
        fused_merge: If True, merges with gather_sum, as RBD does. If False, scales
            and adds with index_add_.
        ep_group: Process group holding the experts. None means the global group.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    world = dist.get_world_size(ep_group)
    experts_per_rank = num_experts // world
    num_tokens, k = route.shape

    with record_function("plain/plan"):
        copy_token = torch.arange(num_tokens, device=route.device).repeat_interleave(k)
        copy_expert = route.flatten()
        copy_rank = copy_expert // experts_per_rank
        order = copy_rank.argsort(stable=True)
        copy_token, copy_expert = copy_token[order], copy_expert[order]
        copy_weight = weights.flatten()[order]

        send = copy_rank.bincount(minlength=world)
        recv = torch.empty_like(send)
        with record_function("comm/both"):
            dist.all_to_all_single(recv, send, group=ep_group)
        send, recv = send.tolist(), recv.tolist()
        expert_ids = all2all(copy_expert, send, recv, ep_group)

    with record_function("plain/dispatch"):
        inputs = all2all(tokens[copy_token], send, recv, ep_group)

    with record_function("plain/experts"):
        outputs = expert(inputs, expert_ids)

    with record_function("plain/combine"):
        returned = all2all(outputs, recv, send, ep_group)

    with record_function("plain/merge"):
        if fused_merge:
            position = torch.empty_like(order)
            position[order] = torch.arange(order.numel(), device=order.device)
            return gather_sum(returned, position.view(num_tokens, k), weights)
        weight = copy_weight.to(returned.dtype).unsqueeze(-1)
        output = returned.new_zeros((num_tokens, returned.shape[1]))
        return output.index_add_(0, copy_token, returned * weight)
