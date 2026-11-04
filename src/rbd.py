"""Redundancy-Bypassing Dispatch (RBD) for expert-parallel MoE communication.

Rewrote from scratch based on my understanding of X-MoE (arXiv 2508.13337). In
a plain all2all, a token routed to several experts on one node crosses the network once per expert, and so does each output on
the way back. RBD sends one row per (token, destination node) between nodes: one copy
chosen by a deterministic rule, the pilot, travels between nodes and the other copies,
the replicas, stay within the destination node. Tokens routed to experts on their own
node skip the pilot step and go straight to those experts.

Dispatch:
    1. Intra-node all2all: send this rank's tokens to experts on its node.
    2. Inter-node all2all: send pilots to other nodes (overlappable with step 1).
    3. Intra-node all2all: copy received pilots and send them to their experts.

Combine:
    4. Intra-node all2all: return replica outputs to their pilot's rank, then scale
       and merge them per pilot.
    5. Intra-node all2all: return local outputs to their source rank.
    6. Inter-node all2all: return merged rows to their source rank. Can overlap with
       step 5.
    7. Scale and merge local outputs per token, and add the returned rows.

RBD_combine_dispatch runs every step and the experts in order.

TODO: Support backprop. The all2alls don't carry gradients, so tokens and expert
parameters get no gradient through RBD_combine_dispatch.

Usage:
    python -m pytest tests/test_rbd.py -v
"""
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.distributed as dist


class InterPlan(NamedTuple):
    """Pilot traffic between nodes.

    Attributes:
        pilot_token: [num_sent_pilots] token index of each pilot this rank sent, in
            send order.
        pilot_send: Pilots sent to each global rank; 0 for ranks on this rank's node.
        pilot_recv: Pilots received from each global rank; 0 for ranks on this rank's
            node.
    """

    pilot_token: torch.Tensor
    pilot_send: list[int]
    pilot_recv: list[int]


class IntraPlan(NamedTuple):
    """Copy traffic within a node, for either local tokens or received pilots.

    Attributes:
        copy_row: [num_copies] input row each copy comes from, in send order. Indexes
            tokens in the local plan and pilots in the replica plan.
        copy_weight: [num_copies] combine weight of each copy, in the same order.
        send: Copies sent to each rank on this node, including this rank.
        recv: Copies received from each rank on this node, including this rank.
    """

    copy_row: torch.Tensor
    copy_weight: torch.Tensor
    send: list[int]
    recv: list[int]


def _start_all2all(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, dist.Work]:
    """Start a variable-size all2all of rows without waiting for it to finish.

    Args:
        rows: [sum(send), ...] rows to send, grouped by destination rank.
        send: Rows to send to each rank in the group.
        recv: Rows to receive from each rank in the group.
        group: Process group to use. None means the global group.

    Returns:
        received: [sum(recv), ...] buffer ordered by source rank, valid only after
            work.wait().
        work: Handle for the running all2all.
    """
    received = rows.new_empty((sum(recv), *rows.shape[1:]))
    work = dist.all_to_all_single(received, rows, recv, send, group=group, async_op=True)
    return received, work


def _all2all(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Run a variable-size all2all of rows to completion.

    Args:
        rows: [sum(send), ...] rows to send, grouped by destination rank.
        send: Rows to send to each rank in the group.
        recv: Rows to receive from each rank in the group.
        group: Process group to use. None means the global group.

    Returns:
        received: [sum(recv), ...] rows ordered by source rank.
    """
    received = rows.new_empty((sum(recv), *rows.shape[1:]))
    dist.all_to_all_single(received, rows, recv, send, group=group)
    return received


def _exchange_counts(
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Tell each rank how many rows to expect from this rank.

    Args:
        counts: [group_size, c] rows this rank will send to each rank, for each of c
            separate all2alls.
        group: Process group to use. None means the global group.

    Returns:
        received: [group_size, c] rows this rank will receive from each rank.
    """
    received = counts.new_empty(counts.shape)
    dist.all_to_all_single(received, counts, group=group)
    return received


def _start_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Find where each rank's block starts in rows grouped by rank.

    Args:
        counts: [group_size] rows in each rank's block.

    Returns:
        offsets: [group_size] index of the first row of each rank's block.
    """
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])


def _sort_copies(
    copy_row: torch.Tensor,
    copy_weight: torch.Tensor,
    copy_expert: torch.Tensor,
    experts_per_rank: int,
    gpus_per_node: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group copies by the local rank that holds their expert.

    Every expert must be on this rank's node.

    Args:
        copy_row: [num_copies] input row each copy comes from.
        copy_weight: [num_copies] combine weight of each copy.
        copy_expert: [num_copies] expert id of each copy.
        experts_per_rank: Experts held by each rank.
        gpus_per_node: Ranks per node.

    Returns:
        copy_row: [num_copies] input rows, grouped by destination local rank.
        copy_weight: [num_copies] combine weights, in the same order.
        copy_expert: [num_copies] expert ids, in the same order.
        send: [gpus_per_node] copies going to each local rank.
    """
    local_rank = (copy_expert // experts_per_rank) % gpus_per_node
    order = local_rank.argsort(stable=True)
    send = local_rank.bincount(minlength=gpus_per_node)
    return copy_row[order], copy_weight[order], copy_expert[order], send


def make_plan(
    route: torch.Tensor,
    weights: torch.Tensor,
    E: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
) -> tuple[InterPlan, IntraPlan, IntraPlan, torch.Tensor]:
    """Decide where tokens go and tell the receiving ranks. Sends no token rows.

    Collective. Every rank must call it, including ranks with no tokens. Exchanges
    metadata over the global group and local_group, so the dispatch and combine
    functions only have to move token rows.

    A copy is one (token, expert) pair. Each copy takes one of two paths, based on
    where its expert lives:
        Same node: the source rank sends it straight to the expert's rank.
        Other node: one copy per (token, node) becomes the pilot and travels to its
            expert's rank. That rank then sends the token's other copies on that node.

    The pilot is the token's first copy on that node, in route order.

    Args:
        route: [num_tokens, k] experts each token is routed to. Experts must be placed
            contiguously, E // world per rank, so expert e lives on rank
            e // (E // world).
        weights: [num_tokens, k] combine weights, where weights[t, j] scales the
            output of expert route[t, j] for token t.
        E: Total number of experts across all ranks.
        gpus_per_node: Ranks per node. Ranks must be numbered contiguously by node.
        local_group: Process group of this rank's node.

    Returns:
        inter: Pilot traffic between nodes.
        local: Traffic for this rank's own tokens within the node.
        replica: Traffic for copies of received pilots within the node.
        expert_ids: [num_inputs] expert id of each input row: local rows, then replica
            rows.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    experts_per_rank = E // world
    num_tokens, k = route.shape
    device = route.device

    # Flatten to one entry per copy, in route order within each token.
    copy_token = torch.arange(num_tokens, device=device).repeat_interleave(k)
    copy_expert = route.flatten()
    copy_weight = weights.flatten()
    copy_rank = copy_expert // experts_per_rank
    copy_node = copy_rank // gpus_per_node
    on_this_node = copy_node == rank // gpus_per_node

    # Same-node copies skip the pilot step, so the source rank sends them directly.
    local_row, local_weight, local_expert, local_send = _sort_copies(
        copy_token[on_this_node],
        copy_weight[on_this_node],
        copy_expert[on_this_node],
        experts_per_rank,
        gpus_per_node,
    )

    # Other-node copies: each distinct (token, node) pair gets one pilot.
    remote_token = copy_token[~on_this_node]
    remote_expert = copy_expert[~on_this_node]
    remote_weight = copy_weight[~on_this_node]
    remote_rank = copy_rank[~on_this_node]
    num_nodes = world // gpus_per_node
    token_node = remote_token * num_nodes + copy_node[~on_this_node]
    unique_pairs, pilot_of_copy = token_node.unique(return_inverse=True)

    # The first copy of each pair becomes its pilot and decides where the pilot goes.
    positions = torch.arange(remote_token.numel(), device=device)
    first_copy = positions.new_full(
        (unique_pairs.numel(),), remote_token.numel()
    ).scatter_reduce(0, pilot_of_copy, positions, "amin")
    pilot_rank = remote_rank[first_copy]

    # Group pilots by destination rank, as the all2all requires.
    pilot_order = pilot_rank.argsort(stable=True)
    pilot_token = remote_token[first_copy][pilot_order]
    pilot_send = pilot_rank.bincount(minlength=world)

    # Find each pilot's position among the pilots going to the same rank. The receiver
    # adds the source's offset to this to get the row in its received pilots.
    send_position = torch.empty_like(pilot_order)
    send_position[pilot_order] = torch.arange(pilot_order.numel(), device=device)
    position_at_rank = send_position - _start_offsets(pilot_send)[pilot_rank]

    # Each other-node copy's metadata goes to its pilot's rank.
    meta_rank = pilot_rank[pilot_of_copy]
    meta_order = meta_rank.argsort(stable=True)
    meta_send = meta_rank.bincount(minlength=world)
    meta_ids = torch.stack([position_at_rank[pilot_of_copy], remote_expert], dim=1)

    pilot_recv, meta_recv = _exchange_counts(
        torch.stack([pilot_send, meta_send], dim=1)
    ).T
    meta_send_list, meta_recv_list = meta_send.tolist(), meta_recv.tolist()
    recv_ids = _all2all(meta_ids[meta_order], meta_send_list, meta_recv_list)
    recv_weight = _all2all(remote_weight[meta_order], meta_send_list, meta_recv_list)

    # Convert per-source pilot positions into rows of the received pilots.
    source_rank = torch.arange(world, device=device).repeat_interleave(meta_recv)
    recv_pilot = recv_ids[:, 0] + _start_offsets(pilot_recv)[source_rank]

    # Received copies include each pilot's own copy, which this rank sends to itself.
    replica_row, replica_weight, replica_expert, replica_send = _sort_copies(
        recv_pilot, recv_weight, recv_ids[:, 1], experts_per_rank, gpus_per_node
    )

    # Tell each rank on this node how many copies to expect and which experts they use.
    local_recv, replica_recv = _exchange_counts(
        torch.stack([local_send, replica_send], dim=1), local_group
    ).T.tolist()
    local_send_list, replica_send_list = local_send.tolist(), replica_send.tolist()
    expert_ids = torch.cat(
        [
            _all2all(local_expert, local_send_list, local_recv, local_group),
            _all2all(replica_expert, replica_send_list, replica_recv, local_group),
        ]
    )

    inter = InterPlan(pilot_token, pilot_send.tolist(), pilot_recv.tolist())
    local = IntraPlan(local_row, local_weight, local_send_list, local_recv)
    replica = IntraPlan(replica_row, replica_weight, replica_send_list, replica_recv)
    return inter, local, replica, expert_ids


def inter_node_dispatch(tokens: torch.Tensor, inter: InterPlan) -> torch.Tensor:
    """Send pilots to other nodes.

    Collective over the global group.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        inter: From make_plan().

    Returns:
        pilots: [num_recv_pilots, H] pilots this rank received.
    """
    return _all2all(tokens[inter.pilot_token], inter.pilot_send, inter.pilot_recv)


def intra_node_dispatch(
    rows: torch.Tensor,
    plan: IntraPlan,
    local_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, dist.Work]:
    """Start sending copies of rows to the experts' ranks on this node.

    Collective over local_group. Returns without waiting for the all2all to finish.
    
    Used both for dispatching tokens to experts on source node and dispatching replicas 
    originating from a different node. 

    Args:
        rows: Rows that plan.copy_row indexes: tokens for the local plan or pilots for
            the replica plan, with shape [num_rows, H].
        plan: The local or replica plan from make_plan().
        local_group: Process group of this rank's node.

    Returns:
        expert_inputs: [sum(plan.recv), H] buffer, valid only after work.wait().
        work: Handle for the running all2all.
    """
    return _start_all2all(rows[plan.copy_row], plan.send, plan.recv, local_group)


def intra_node_combine(
    outputs: torch.Tensor,
    plan: IntraPlan,
    local_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, dist.Work]:
    """Start returning expert outputs within the node, reversing intra_node_dispatch.

    Collective over local_group. Returns without waiting for the all2all to finish.

    Used both for combining tokens on source node and combining replicas originating
    from a different node. 

    Args:
        outputs: [sum(plan.recv), H] outputs, in the order of the matching
            expert_inputs.
        plan: The plan the matching intra_node_dispatch() call used.
        local_group: Process group of this rank's node.

    Returns:
        copy_outputs: [sum(plan.send), H] buffer in the order of plan.copy_row, valid
            only after work.wait().
        work: Handle for the running all2all.
    """
    return _start_all2all(outputs, plan.recv, plan.send, local_group)


def merge_outputs(
    copy_outputs: torch.Tensor,
    plan: IntraPlan,
    num_rows: int,
) -> torch.Tensor:
    """Scale each copy's output and add it to its row.

    Local, with no communication.

    Args:
        copy_outputs: [sum(plan.send), H] from intra_node_combine(), after its wait().
        plan: The plan the matching intra_node_combine() call used.
        num_rows: Rows in the merged result, which is the number of rows passed to the
            matching intra_node_dispatch().

    Returns:
        merged: [num_rows, H] weighted sum of each row's copy outputs.
    """
    weight = plan.copy_weight.to(copy_outputs.dtype).unsqueeze(-1)
    merged = copy_outputs.new_zeros((num_rows, *copy_outputs.shape[1:]))
    return merged.index_add_(0, plan.copy_row, copy_outputs * weight)


def inter_node_combine(
    merged: torch.Tensor,
    inter: InterPlan,
    num_tokens: int,
) -> torch.Tensor:
    """Return merged rows to their source ranks and add them per token.

    Collective over the global group. Reverses inter_node_dispatch().

    Args:
        merged: [num_recv_pilots, H] merged rows, in the order of the received pilots.
        inter: From make_plan().
        num_tokens: Number of tokens on this rank.

    Returns:
        remote_output: [num_tokens, H] each token's results from other nodes.
    """
    # Rows return in the order their pilots were sent, so pilot_token still lines up.
    returned = _all2all(merged, inter.pilot_recv, inter.pilot_send)
    remote_output = merged.new_zeros((num_tokens, *merged.shape[1:]))
    return remote_output.index_add_(0, inter.pilot_token, returned)


def RBD_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    E: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    concurrent: bool = False,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs.

    Collective.

    Each token reaches its experts by one of two paths, and returns the same way:
        Same node: source rank -> expert rank. Intra-node only.
        Other node: source rank -> pilot rank -> expert rank. One row per (token, node)
            crosses between nodes, and the pilot rank sends the other copies from there.

    Order of communication, with `|` marking all2alls that overlap when concurrent is
    True:
        Plan:     exchange routing metadata (make_plan)
        Dispatch: local tokens to experts | pilots to other nodes
                  replicas to experts
        Experts:  run local rows and replica rows together
        Combine:  replica outputs to pilot ranks, then merge per pilot
                  local outputs to source ranks | merged rows to source ranks
        Merge:    scale local outputs per token and add the returned rows

    Weights are applied on the rank that gathers a token's copies: the pilot rank for
    other-node copies and the source rank for same-node copies. Merged rows therefore
    cross between nodes already scaled.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights, where weights[t, j] scales the
            output of expert route[t, j] for token t.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to.
        E: Total number of experts across all ranks.
        gpus_per_node: Ranks per node. Ranks must be numbered contiguously by node.
        local_group: Process group of this rank's node.
        concurrent: If True, overlaps the local intra-node all2alls with the inter-node
            all2alls. If False, runs every all2all in sequence.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    inter, local, replica, expert_ids = make_plan(
        route, weights, E, gpus_per_node, local_group
    )
    num_tokens = tokens.shape[0]

    # Dispatch: local tokens don't depend on pilots, so these two can overlap.
    local_inputs, local_work = intra_node_dispatch(tokens, local, local_group)
    if not concurrent:
        local_work.wait()
    pilots = inter_node_dispatch(tokens, inter)
    if concurrent:
        local_work.wait()

    # Replicas are copied from pilots, so they start once the pilots have arrived.
    replica_inputs, replica_work = intra_node_dispatch(pilots, replica, local_group)
    replica_work.wait()

    # Experts: expert_ids lists local rows first, then replica rows, matching this cat.
    outputs = expert(torch.cat([local_inputs, replica_inputs]), expert_ids)
    local_outputs, replica_outputs = outputs.split(
        [local_inputs.shape[0], replica_inputs.shape[0]]
    )

    # Combine: replica outputs must merge into pilot rows before those rows can return.
    replica_copies, replica_work = intra_node_combine(
        replica_outputs, replica, local_group
    )
    replica_work.wait()
    merged = merge_outputs(replica_copies, replica, pilots.shape[0])

    # Local outputs don't depend on merged rows, so these two can overlap.
    local_copies, local_work = intra_node_combine(local_outputs, local, local_group)
    if not concurrent:
        local_work.wait()
    remote_output = inter_node_combine(merged, inter, num_tokens)
    if concurrent:
        local_work.wait()

    # Merged rows arrived scaled, so only the local outputs still need their weights.
    return merge_outputs(local_copies, local, num_tokens) + remote_output
