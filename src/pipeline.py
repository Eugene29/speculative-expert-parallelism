"""Pipelined dispatch and combine: experts run while the next slice is on the network.

Based on Tutel's adaptive pipelining (arXiv 2206.03382). In one blocking round trip
the GPU idles while rows cross the network, and the network idles while the experts
run. The pipeline cuts this rank's tokens into `degree` slices and gives each slice
its own round trip, so the experts of slice c run while slice c + 1 is on the network.

Each function makes three passes over the slices:
    1. Plan the slices and start every slice's dispatch all2all, without waiting.
    2. Per slice: wait for its rows, run the experts on them, and start its combine
       all2all without waiting.
    3. Per slice: wait for its returned rows and merge them into its tokens.

Two versions:
    pipelined_rbd_combine_dispatch: RBD's stages on slices planned together
        (spec_ep.rbd.start_plans): one exchange, one host sync and one sort for all of
        them. The dispatches start as soon as the stage 1 rows are known, and a slice's
        receiving half of the plan (finish_plan) is queued behind its rows without
        waiting for them. Only the stage 1 all2alls are started ahead. A slice's stage 2
        over local_group and its experts run under the next slice's stage 1. With
        lanes=2 every slice's stage 1 goes in the two lanes of spec_ep.two_lane.
    pipelined_plain_combine_dispatch: one row per (token, expert). Each slice is
        planned on its own, as spec_ep.plain plans a layer, so planning costs `degree`
        times its exchanges and host syncs.

The profiler label of an all2all started without waiting covers both its start and
its wait. The event clock (spec_ep.phase_clock) sees such an all2all as one comm event
from its start to its real end, over whatever the stream does meanwhile. Forward only,
like spec_ep.rbd.

Usage:
    from spec_ep.pipeline import pipelined_rbd_combine_dispatch
    python -m pytest tests/test_pipeline.py -v
"""
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

from spec_ep.ops import all2all, all2all_async, gather_sum
from spec_ep.rbd import (
    finish_plan,
    stage2_combine,
    stage2_dispatch,
    start_plans,
    token_slices,
    with_meta,
)
from spec_ep.two_lane import two_lane_all2all_async

__all__ = ["pipelined_rbd_combine_dispatch", "pipelined_plain_combine_dispatch"]


def pipelined_rbd_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    degree: int,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    lanes: int = 1,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine through RBD, pipelined in slices.

    Collective over ep_group and local_group. Every rank must call it, with the same
    degree. Takes the arguments of rbd_combine_dispatch(), plus degree. Each slice
    goes through the stages of rbd_combine_dispatch(); with degree 1 the output is
    the same.

    Args:
        expert: As in rbd_combine_dispatch(). Called once per slice.
        degree: Slices to cut this rank's tokens into.
        lanes: 1 sends a slice's stage 1 rows in one all2all over ep_group. 2 sends
            them as spec_ep.two_lane does: pilots over ep_group, direct rows over
            local_group.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    with record_function("autoep/rbd_plan"):
        starts = start_plans(
            route, weights, num_experts, gpus_per_node, ep_group, pilot_rule, degree, lanes
        )
    start_all2all = partial(all2all_async, group=ep_group)
    if lanes == 2:
        start_all2all = partial(
            two_lane_all2all_async,
            gpus_per_node=gpus_per_node, local_group=local_group, ep_group=ep_group,
        )
    with record_function("autoep/rbd_stage1_dispatch"):
        rows = with_meta(tokens, starts[0].meta)
        arriving = [
            start_all2all(rows[start.send1_token], start.send1, start.recv1) for start in starts
        ]

    plans, returning = [], []
    for start, wait in zip(starts, arriving):
        with record_function("autoep/rbd_stage1_dispatch"):
            received = wait()
        with record_function("autoep/rbd_plan"):
            received, plan = finish_plan(start, received)
        plans.append(plan)
        expert_inputs = stage2_dispatch(received, plan, local_group)
        with record_function("autoep/rbd_experts"):
            outputs = expert(expert_inputs, plan.expert_ids)
        back = stage2_combine(outputs, plan, local_group)
        with record_function("autoep/rbd_stage1_combine"):
            returning.append(start_all2all(back, plan.recv1, plan.send1))

    output = tokens.new_empty(tokens.shape)
    for s, plan, wait in zip(token_slices(tokens.shape[0], degree), plans, returning):
        with record_function("autoep/rbd_stage1_combine"):
            returned = wait()
        with record_function("autoep/rbd_final_merge"):
            gather_sum(returned, plan.final_index, plan.final_weight, out=output[s])
    return output


class _PlainPlan(NamedTuple):
    """Where every row of one slice goes in a plain all2all over ep_group.

    send_token is the [sum(send)] token each sent row carries. send and recv are the
    rows sent to and received from each rank. expert_ids is the [sum(recv)] expert of
    each received row. final_index is the [num_tokens, k] returned row of each copy.
    """

    send_token: torch.Tensor
    send: list[int]
    recv: list[int]
    expert_ids: torch.Tensor
    final_index: torch.Tensor


def _make_plain_plan(
    route: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup | None,
) -> _PlainPlan:
    """Plan one round trip as plain_combine_dispatch() does. Collective over ep_group.

    Returns:
        plan: The indices and split sizes of the round trip.
    """
    world = dist.get_world_size(ep_group)
    num_tokens, k = route.shape
    copy_expert = route.flatten()
    copy_rank = copy_expert // (num_experts // world)
    order = copy_rank.argsort(stable=True)

    send = copy_rank.bincount(minlength=world)
    recv = torch.empty_like(send)
    with record_function("comm/both"):
        dist.all_to_all_single(recv, send, group=ep_group)
    send, recv = send.tolist(), recv.tolist()
    expert_ids = all2all(copy_expert[order], send, recv, ep_group)

    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=order.device)
    return _PlainPlan(order // k, send, recv, expert_ids, position.view(num_tokens, k))


def pipelined_plain_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
    degree: int,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine with plain all2alls, pipelined in slices.

    Collective over ep_group. Takes the arguments of pipelined_rbd_combine_dispatch(),
    minus the node layout it doesn't need. With degree 1 it does what
    plain_combine_dispatch() does with fused_merge=True.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    slices = token_slices(tokens.shape[0], degree)
    plans, arriving = [], []
    for s in slices:
        with record_function("plain/plan"):
            plan = _make_plain_plan(route[s], num_experts, ep_group)
        with record_function("plain/dispatch"):
            rows = tokens[s][plan.send_token]
            arriving.append(all2all_async(rows, plan.send, plan.recv, ep_group))
        plans.append(plan)

    returning = []
    for plan, wait in zip(plans, arriving):
        with record_function("plain/dispatch"):
            inputs = wait()
        with record_function("plain/experts"):
            outputs = expert(inputs, plan.expert_ids)
        with record_function("plain/combine"):
            returning.append(all2all_async(outputs, plan.recv, plan.send, ep_group))

    output = tokens.new_empty(tokens.shape)
    for s, plan, wait in zip(slices, plans, returning):
        with record_function("plain/combine"):
            returned = wait()
        with record_function("plain/merge"):
            gather_sum(returned, plan.final_index, weights[s], out=output[s])
    return output
