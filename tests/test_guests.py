"""Integration tests for guest experts (spec_ep.guests) under RBD and two-lane.

Every rank takes the layer's own routing as its demand, chooses guests, and runs RBD on
the relabeled routing, through the check of tests/checks.py. The experts scale a row
by its original expert id, so a copy served by the wrong guest changes the output. Each
run also checks what the placement promises: unique guests are distinct within a node,
and a copy only moves to a guest of its own node. The pool that pulls the guests'
weights from their owners is checked over the same ranks.

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

import torch.multiprocessing as mp

from checks import EXPERTS_PER_RANK, GPUS_PER_NODE, LAYOUTS, Layout, run_check
from spec_ep.guests import (
    GuestPool,
    choose_guests,
    choose_guests_by_pilots,
    exchange_requests,
    home_ids,
    pilot_credit,
    remote_demand,
)
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
    balance: float = 0.0,
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
        guests = choose_guests_by_pilots(
            route, num_experts, SLOTS, G, node_group, ep_group, rounds=2, balance=balance
        )
    else:
        guests = choose_guests(demand, SLOTS, G, node_group, ep_group)

    held = guests.slot_expert[guests.slot_expert >= 0]
    assert (held.diff() > 0).all() and (guests.slot_expert[held.numel() :] < 0).all(), \
        "slots are not in expert order with the empty ones last"
    assert (held // (L * G) != node).all(), "a guest is an expert of this rank's own node"
    served_by = guests.grown[route] // (L + SLOTS)
    moved = served_by != route // L
    assert (served_by[moved] // G == node).all(), "a copy moved to a guest of another node"
    if unique:
        node_slots = [torch.empty_like(guests.slot_expert) for _ in range(G)]
        dist.all_gather(node_slots, guests.slot_expert, group=local_group)
        node_slots = torch.cat(node_slots)
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
        "unique guests by pilots, two-lane": partial(
            _with_guests, unique=True, by_pilots=True, dispatch=two_lane_rbd_combine_dispatch
        ),
        "unique guests by pilots, balanced": partial(
            _with_guests, unique=True, by_pilots=True, balance=2.0, dispatch=rbd_combine_dispatch
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


def _check_pool(rank: int, init_file: Path, world: int) -> None:
    """Pull random guests twice on one rank and check every slot against its owner's weights.

    Expert e's two weight tensors are filled with e and with -e, so a slot says whose
    weights it holds. The second fetch goes to the other set, and must not disturb
    the one in use.
    """
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        L, E, device = EXPERTS_PER_RANK, EXPERTS_PER_RANK * world, torch.device("cuda")
        own = torch.arange(rank * L, (rank + 1) * L, device=device, dtype=torch.float32)
        sources = [own.view(L, 1, 1).expand(L, 4, 3).contiguous(), -own.view(L, 1).repeat(1, 5)]
        pool = GuestPool(sources, SLOTS, dist.group.WORLD)

        held = []
        for fetch in range(2):
            gen = torch.Generator().manual_seed(100 * fetch + rank)
            others = torch.arange(E)[torch.arange(E) // L != rank]
            want = others[torch.randperm(others.numel(), generator=gen)[: SLOTS - fetch]].sort().values
            want = torch.cat([want, torch.full((fetch,), -1)]).to(device)  # the second leaves a slot empty
            pool.fetch(sources, exchange_requests(want, L))
            if fetch:
                torch.testing.assert_close(pool.sets[pool.current][0][:, 0, 0], held[0].float())
            weights = pool.take()
            pool.wait()
            filled = want[want >= 0].float()
            torch.testing.assert_close(weights[0][: filled.numel(), 0, 0], filled)
            torch.testing.assert_close(weights[1][: filled.numel(), 0], -filled)
            held.append(want)
        torch.cuda.synchronize()
        started, arrived, needed = pool.times[pool.current]
        assert started.elapsed_time(arrived) >= 0 and needed is not None
    finally:
        dist.destroy_process_group()


def test_pool_pulls_guests_from_their_owners(tmp_path: Path) -> None:
    """Every rank gets the weights of the experts it asked for, in the order of its slots."""
    mp.spawn(_check_pool, args=(tmp_path / "init", 2 * GPUS_PER_NODE), nprocs=2 * GPUS_PER_NODE)
