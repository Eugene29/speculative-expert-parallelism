"""Integration tests for guest experts (spec_ep.guests) under RBD, two-lane and the pipeline.

Every rank takes the layer's own routing as its demand, chooses guests, and runs RBD on
the relabeled routing, through the check of tests/checks.py. The experts scale a row
by its original expert id, so a copy served by the wrong guest changes the output. Each
run also checks what the placement promises: unique guests are distinct within a node,
and a copy only moves to a guest of its own node. The pool that holds the guests'
weights is checked on one GPU.

Usage:
    python -m pytest tests/test_guests.py -m smoke   # 2_nodes only, for development
    python -m pytest tests/test_guests.py -v         # every layout
"""
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from checks import LAYOUTS, Layout, run_check
from spec_ep.guests import (
    GuestPool,
    choose_guests,
    choose_guests_by_pilots,
    home_ids,
    pilot_credit,
    remote_demand,
)
from spec_ep.pipeline import pipelined_rbd_combine_dispatch
from spec_ep.rbd import rbd_combine_dispatch
from spec_ep.two_lane import two_lane_rbd_combine_dispatch

SLOTS = 3


def _with_guests(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None,
    pilot_rule: str,
    unique: bool,
    dispatch: Callable[..., torch.Tensor],
    by_pilots: bool = False,
) -> torch.Tensor:
    """An RBD variant with guests chosen from the routing itself.

    Returns:
        output: [num_tokens, H] combined output, to compare with the local reference.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    G, L = gpus_per_node, num_experts // world
    node = rank // G
    demand = remote_demand(route, num_experts, node, world // G)
    node_group = local_group if unique else None
    if by_pilots:
        guests = choose_guests_by_pilots(route, num_experts, SLOTS, G, node_group, ep_group, rounds=2)
    else:
        guests = choose_guests(demand, SLOTS, G, node_group, ep_group)

    held = guests.slot_expert[guests.slot_expert >= 0]
    assert (held // (L * G) != node).all(), "a guest is an expert of this rank's own node"
    served_by = guests.grown[route] // (L + SLOTS)
    moved = served_by != route // L
    assert (served_by[moved] // G == node).all(), "a copy moved to a guest of another node"
    if unique:
        node_slots = torch.empty(G * SLOTS, dtype=torch.long)
        dist.all_gather_into_tensor(node_slots, guests.slot_expert, group=local_group)
        node_held = node_slots[node_slots >= 0]
        assert node_held.unique().numel() == node_held.numel(), "two ranks hold one guest"
        assert (held % G == rank % G).all(), "a guest from another rank's stripe"
    else:
        assert (served_by[moved] == rank).all(), "a copy moved to another rank's guest"
        assert moved.any() == (demand.sum() > 0), "demand for other nodes but no copy moved"

    run = lambda rows, ids: expert(rows, home_ids(ids, guests, ep_group))
    return dispatch(
        tokens, weights, run, guests.grown[route], guests.num_experts, G, local_group,
        ep_group=ep_group, pilot_rule=pilot_rule,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_guests_match_local_reference(tmp_path: Path, layout: Layout) -> None:
    """Every RBD variant with unique or redundant guests gives the locally computed answer."""
    run_check(tmp_path, layout, {
        "unique guests": partial(_with_guests, unique=True, dispatch=rbd_combine_dispatch),
        "redundant guests": partial(_with_guests, unique=False, dispatch=rbd_combine_dispatch),
        "unique guests, two-lane": partial(
            _with_guests, unique=True, dispatch=two_lane_rbd_combine_dispatch
        ),
        "redundant guests, two-lane": partial(
            _with_guests, unique=False, dispatch=two_lane_rbd_combine_dispatch
        ),
        "unique guests, pipeline": partial(
            _with_guests, unique=True, dispatch=partial(pipelined_rbd_combine_dispatch, degree=2)
        ),
        "unique guests by pilots, two-lane pipeline": partial(
            _with_guests, unique=True, by_pilots=True,
            dispatch=partial(pipelined_rbd_combine_dispatch, degree=2, lanes=2),
        ),
        "redundant guests by pilots": partial(
            _with_guests, unique=False, by_pilots=True, dispatch=rbd_combine_dispatch
        ),
    })


def test_pilot_credit_prefers_the_expert_that_frees_pilots() -> None:
    """Credit adds up to the pilots and ranks an expert by the pilots a guest of it frees.

    Node 0 holds experts 0 to 7. Three tokens need expert 8 and one more expert of node
    1 each, so a guest of 8 frees no pilot. Two tokens need only expert 13 there.
    """
    route = torch.tensor([[8, 9], [8, 10], [8, 11], [0, 13], [0, 13]])
    demand = remote_demand(route, 16, node=0, num_nodes=2)
    credit = pilot_credit(route, 16, node=0, num_nodes=2)
    assert demand.argmax() == 8 and credit.argmax() == 13
    assert credit.sum() == 5, "one pilot per token"
    taken = torch.arange(16) == 13
    assert pilot_credit(route, 16, 0, 2, taken).sum() == 3, "a guest of 13 frees two pilots"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pool lives on a GPU")
def test_pool_loads_rows_and_keeps_weights_in_use() -> None:
    """A load fills the slots it is given, and waits for work queued on the old weights."""
    device = torch.device("cuda")
    sources = [torch.randn(6, 4, 3).pin_memory(), torch.randn(6, 5).pin_memory()]
    pool = GuestPool([torch.empty(3, 4, 3, device=device), torch.empty(3, 5, device=device)])

    pool.load(sources, [4, -1, 0])
    pool.wait()
    for weight, source in zip(pool.weights, sources):
        torch.testing.assert_close(weight[[0, 2]].cpu(), source[[4, 0]])

    # Work queued before the next load still reads the old weights, however long it
    # takes to get to them.
    busy = torch.randn(4096, 4096, device=device)
    for _ in range(20):
        busy = busy @ busy.T / 4096
    before = pool.weights[0][0] * 1.0
    pool.load(sources, [1, 2, 3])
    pool.wait()
    torch.testing.assert_close(before.cpu(), sources[0][4])
    torch.testing.assert_close(pool.weights[0].cpu(), sources[0][1:4])
    torch.cuda.synchronize()
    assert pool.started.elapsed_time(pool.loaded) >= 0
