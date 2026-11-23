"""Two-lane RBD: stage 1 of RBD split into a network lane and a local lane.

In RBD (spec_ep.rbd), stage 1 is one all2all over ep_group that carries both the
pilots, which cross the network, and the direct rows, which stay on their node and
travel over NVLink. Two-lane RBD sends them as two all2alls that run at the same time:
    network lane, over ep_group: the pilots. Ranks on the same node exchange nothing.
    local lane, over local_group: the direct rows.
The direct rows then move at the speed of a node-local all2all instead of sharing the
multi-node one. The same split applies on the way back. make_plan(lanes=2) orders the
stage 1 rows so each lane is one block of the sent and of the received buffer, pilots
and then direct rows, and no row is copied to split or join the lanes. Stage 2 and the
final merge are RBD's own.

The lanes also free the experts from waiting for the network. The direct rows arrive
long before the pilots, and their outputs are not needed before the pilot sums are
back, so with `early` the experts run on the direct rows while the network lane is
busy. No transfer is added: only the order changes.

Each lane's split sizes are read from the plan's send1 and recv1 lists, so planning
costs what it costs RBD. A call makes six data all2alls where RBD makes four.

Assumes, as RBD does, that ranks are numbered contiguously by node, and also that
local_group holds this node's ranks in ep_group order. Forward only, as RBD is.

Usage:
    from spec_ep.two_lane import two_lane_rbd_combine_dispatch
    python -m pytest tests/test_two_lane.py -v
"""
import torch
import torch.distributed as dist
from torch.profiler import record_function

from spec_ep.ops import all2all, all2all_async
from spec_ep.rbd import (
    Expert,
    final_merge,
    make_plan,
    run_experts,
    stage2_combine,
    stage2_dispatch,
    with_meta,
)

__all__ = ["two_lane_rbd_combine_dispatch"]


def two_lane_rbd_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Expert,
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    early: float | None = None,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs, with stage 1 in two lanes.

    Collective over ep_group and local_group. Every rank must call it, including
    ranks with no tokens. Takes the arguments of rbd_combine_dispatch() and gives the
    same output. local_group must hold this node's ranks in ep_group order.

    Args:
        early: None runs the experts on the direct rows once both lanes have arrived.
            A share between 0 and 1 runs them while the network lane is busy: on that
            share of the direct rows while the pilots travel out, and on the rest
            while the pilot sums travel back. The expert function is then called up
            to three times.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    with record_function("autoep/rbd_plan"):
        plan = make_plan(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule, lanes=2)
    lo = dist.get_rank(group=ep_group) // gpus_per_node * gpus_per_node
    hi, idle = lo + gpus_per_node, [0] * gpus_per_node
    network_send = plan.send1[:lo] + idle + plan.send1[hi:]
    network_recv = plan.recv1[:lo] + idle + plan.recv1[hi:]
    H, num_sent, (num_pilots, end) = tokens.shape[1], sum(network_send), plan.direct
    num_direct = end - num_pilots
    cut = num_direct if early is None else int(early * num_direct)

    # Out: the local lane starts first and the experts take its rows as they arrive.
    with record_function("autoep/two_lane_dispatch"):
        rows = with_meta(tokens, plan.meta)[plan.send1_token]
        received = rows.new_empty((sum(plan.recv1), rows.shape[1]))
        wait_local = all2all_async(
            rows[num_sent:], plan.send1[lo:hi], plan.recv1[lo:hi], local_group,
            out=received[num_pilots:], link="intra",
        )
        wait_network = all2all_async(
            rows[:num_sent], network_send, network_recv, ep_group, out=received[:num_pilots],
            link="inter",
        )
        wait_local()
        if early is None:
            wait_network()
    direct, ids = received[plan.direct_order, :H], plan.expert_ids
    outputs = run_experts(expert, direct[:cut], ids[:cut])
    with record_function("autoep/two_lane_dispatch"):
        wait_network()

    # The pilots' copies on this node, as in RBD.
    back = tokens.new_empty((sum(plan.recv1), H))
    stage2_rows, merge = stage2_dispatch(received, plan, local_group)
    stage2_outputs = run_experts(expert, stage2_rows, ids[num_direct:])
    stage2_combine(stage2_outputs, plan, merge, local_group, back)

    # Back: the pilot sums leave first and the rest of the direct rows run under them.
    with record_function("autoep/two_lane_combine"):
        returned = tokens.new_empty((sum(plan.send1), H))
        wait_network = all2all_async(
            back[:num_pilots], network_recv, network_send, ep_group, out=returned[:num_sent],
            link="inter",
        )
    if cut < num_direct:
        rest = run_experts(expert, direct[cut:], ids[cut:num_direct])
        outputs = torch.cat([outputs, rest]) if cut else rest
    with record_function("autoep/two_lane_combine"):
        torch.index_select(outputs, 0, plan.direct_place, out=back[num_pilots:])
        all2all(
            back[num_pilots:], plan.recv1[lo:hi], plan.send1[lo:hi], local_group,
            out=returned[num_sent:], link="intra",
        )
        wait_network()
    return final_merge(returned, plan)
