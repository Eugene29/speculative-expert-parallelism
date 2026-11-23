"""Two-lane RBD: stage 1 of RBD split into a network lane and a local lane.

In RBD (spec_ep.rbd), stage 1 is one all2all over ep_group that carries both the
pilots, which cross the network, and the direct rows, which stay on their node and
travel over NVLink. Two-lane RBD sends them as two all2alls that run at the same time:
    network lane, over ep_group: the pilots. Ranks on the same node exchange nothing.
    local lane, over local_group: the direct rows.
The aim is for direct rows to move at the speed of a node-local all2all instead of
sharing the multi-node one. The same split applies on the way back in stage 1
combine. Stage 2, the experts and the final merge are RBD's own.

two_lane_rbd_combine_dispatch() is rbd_combine_dispatch() with three changes:
    1. make_plan(lanes=2) orders the stage 1 rows so each lane is one block of the
       sent and of the received buffer, pilots and then direct rows, and no row is
       copied to split or join the lanes.
    2. two_lane_dispatch replaces stage1_dispatch.
    3. two_lane_combine replaces stage1_combine.
Only the row order of the plan changes, so stage2_dispatch, stage2_combine and
final_merge run unchanged.

Each lane's split sizes are read from the plan's send1 and recv1 lists, so planning
costs what it costs RBD, with no added communication or host sync. A call makes six
data all2alls where RBD makes four.

Assumes, as RBD does, that ranks are numbered contiguously by node, and also that
local_group holds this node's ranks in ep_group order. Forward only, as RBD is.

Usage:
    from spec_ep.two_lane import two_lane_rbd_combine_dispatch
    python -m pytest tests/test_two_lane.py -v
"""
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.profiler import record_function

from spec_ep.ops import all2all, all2all_async
from spec_ep.rbd import (
    Plan,
    final_merge,
    make_plan,
    stage2_combine,
    stage2_dispatch,
    with_meta,
)

__all__ = [
    "two_lane_all2all_async",
    "two_lane_dispatch",
    "two_lane_combine",
    "two_lane_rbd_combine_dispatch",
]


def _node_ranks(gpus_per_node: int, ep_group: dist.ProcessGroup | None) -> tuple[int, int]:
    """Ranks of ep_group on this rank's node.

    Returns:
        lo: First rank of the node.
        hi: One past its last rank.
    """
    lo = dist.get_rank(group=ep_group) // gpus_per_node * gpus_per_node
    return lo, lo + gpus_per_node


def two_lane_all2all_async(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
    block: bool = False,
) -> Callable[[], torch.Tensor]:
    """Start an all2all of stage 1 rows in two lanes. Collective over both groups.

    rows holds the pilots, then the direct rows. send and recv are per rank of
    ep_group, as in all2all. The pilots go over ep_group and the direct rows over
    local_group. The local lane is started first and waited for last, so the pilots
    cross the network while it is in flight.

    Args:
        block: Finish the network lane before returning, which leaves only the local
            lane to wait for.

    Returns:
        wait: Blocks until both lanes have arrived, then returns the [sum(recv), H]
            received pilots, then direct rows, each by source rank.
    """
    lo, hi = _node_ranks(gpus_per_node, ep_group)
    idle = [0] * gpus_per_node
    network_send = send[:lo] + idle + send[hi:]
    network_recv = recv[:lo] + idle + recv[hi:]
    num_sent, num_recv = sum(network_send), sum(network_recv)

    out = rows.new_empty((sum(recv), rows.shape[1]))
    wait_local = all2all_async(
        rows[num_sent:], send[lo:hi], recv[lo:hi], local_group, out=out[num_recv:], link="intra"
    )
    network = all2all if block else all2all_async
    wait_network = network(
        rows[:num_sent], network_send, network_recv, ep_group, out=out[:num_recv], link="inter"
    )

    def wait() -> torch.Tensor:
        if not block:
            wait_network()
        wait_local()
        return out

    return wait


def _two_lane_all2all(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """two_lane_all2all_async() with the network lane blocking, waited for.

    Returns:
        out: [sum(recv), H] received pilots, then direct rows, each by source rank.
    """
    return two_lane_all2all_async(
        rows, send, recv, gpus_per_node, local_group, ep_group, block=True
    )()


def two_lane_dispatch(
    tokens: torch.Tensor,
    plan: Plan,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Send pilots over ep_group and direct rows over local_group at the same time.

    Replaces stage1_dispatch. Collective over both groups. plan is from
    make_plan(lanes=2).

    Returns:
        received: [sum(recv1), H + M] pilots, then this node's direct rows, with
            their metadata columns. Each lane is ordered by source rank.
    """
    with record_function("autoep/two_lane_dispatch"):
        rows = with_meta(tokens, plan.meta)[plan.send1_token]
        return _two_lane_all2all(
            rows, plan.send1, plan.recv1, gpus_per_node, local_group, ep_group
        )


def two_lane_combine(
    back: torch.Tensor,
    plan: Plan,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return pilot sums over ep_group and direct outputs over local_group at the same time.

    Replaces stage1_combine. Collective over both groups. back is stage2_combine's
    output, so the local lane starts after the stage 2 return on local_group has
    finished and does not queue behind it.

    Returns:
        returned: [sum(send1), H] rows in the two-lane stage 1 send order.
    """
    with record_function("autoep/two_lane_combine"):
        return _two_lane_all2all(
            back, plan.recv1, plan.send1, gpus_per_node, local_group, ep_group
        )


def _overlapped_round_trip(
    tokens: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    plan: Plan,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None,
    early: float,
) -> torch.Tensor:
    """Run a planned layer with the direct rows' experts under the network lane.

    The direct rows arrive over NVLink long before the pilots arrive over the network,
    and their outputs are not needed before the pilot sums are back. So the experts
    run on the first `early` of the direct rows while the pilots are on their way out,
    and on the rest while the pilot sums are on their way back. The pilots' copies
    run in between, as in RBD. No transfer is added: only the order changes, and the
    expert function is called up to three times.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs.
    """
    lo, hi = _node_ranks(gpus_per_node, ep_group)
    idle = [0] * gpus_per_node
    network_send = plan.send1[:lo] + idle + plan.send1[hi:]
    network_recv = plan.recv1[:lo] + idle + plan.recv1[hi:]
    H, num_sent, (num_pilots, end) = tokens.shape[1], sum(network_send), plan.direct
    num_direct = end - num_pilots
    cut = int(early * num_direct)

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
    back = tokens.new_empty((sum(plan.recv1), H))
    direct = received[num_pilots:, :H]

    def run_direct(first: int, last: int) -> None:
        """Run the experts on direct rows first to last, into their rows of back."""
        if last > first:
            with record_function("autoep/rbd_experts"):
                back[num_pilots + first : num_pilots + last] = expert(
                    direct[first:last], plan.expert_ids[first:last]
                )

    run_direct(0, cut)
    with record_function("autoep/two_lane_dispatch"):
        wait_network()
    outputs, merge = stage2_dispatch(received, plan, local_group)
    if outputs.shape[0]:
        with record_function("autoep/rbd_experts"):
            outputs = expert(outputs, plan.expert_ids[num_direct:])
    stage2_combine(outputs, plan, merge, local_group, back)

    with record_function("autoep/two_lane_combine"):
        returned = tokens.new_empty((sum(plan.send1), H))
        wait_network = all2all_async(
            back[:num_pilots], network_recv, network_send, ep_group, out=returned[:num_sent],
            link="inter",
        )
    run_direct(cut, num_direct)
    with record_function("autoep/two_lane_combine"):
        all2all(
            back[num_pilots:], plan.recv1[lo:hi], plan.send1[lo:hi], local_group,
            out=returned[num_sent:], link="intra",
        )
        wait_network()
    return final_merge(returned, plan)


def two_lane_rbd_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        early: None runs the experts once, after every row has arrived. A share
            between 0 and 1 runs them on the direct rows while the network lane is
            busy: on that share while the pilots travel out, on the rest while the
            pilot sums travel back.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    with record_function("autoep/rbd_plan"):
        plan = make_plan(route, weights, num_experts, gpus_per_node, ep_group, pilot_rule, lanes=2)
    if early is not None:
        return _overlapped_round_trip(
            tokens, expert, plan, gpus_per_node, local_group, ep_group, early
        )
    H, (start, end) = tokens.shape[1], plan.direct
    num_direct = end - start
    received = two_lane_dispatch(tokens, plan, gpus_per_node, local_group, ep_group)
    inputs = tokens.new_empty((num_direct + sum(plan.recv2), H))
    _, merge = stage2_dispatch(received, plan, local_group, out=inputs[num_direct:])
    inputs[:num_direct].copy_(received[start:end, :H])
    with record_function("autoep/rbd_experts"):
        outputs = expert(inputs, plan.expert_ids)
    back = tokens.new_empty((sum(plan.recv1), H))
    back[start:end].copy_(outputs[:num_direct])
    stage2_combine(outputs[num_direct:], plan, merge, local_group, back)
    returned = two_lane_combine(back, plan, gpus_per_node, local_group, ep_group)
    return final_merge(returned, plan)
