"""Integration tests for Redundancy-Bypassing Dispatch (spec_ep.rbd).

Each test spawns one CPU process per rank over gloo and runs RBD_combine_dispatch
end to end. Layouts follow Perlmutter, 4 GPUs per node. Each layout spawns once and
loops over every pilot rule and routing inside the workers, since starting the
processes costs far more than one RBD call.

Each expert scales its rows by 1 + expert id, so a row sent to the wrong expert,
merged into the wrong token or given the wrong weight changes the output. The
expected output needs no communication, so every rank checks its own tokens.
Everything runs in float64, so the match is exact up to rounding.

Usage:
    python -m pytest tests/test_rbd.py -m smoke   # 2_nodes only, for development
    python -m pytest tests/test_rbd.py -v         # every layout
"""
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spec_ep.rbd import RBD_combine_dispatch

GPUS_PER_NODE = 4
EXPERTS_PER_RANK = 4
NUM_TOKENS = 16
HIDDEN = 4
TOP_K = 4
PILOTS = ["rail", "first", "most", "even"]


class Layout(NamedTuple):
    """Global ranks, split into world // ep_size EP groups that each run RBD."""

    world: int
    ep_size: int


# Run only the smoke layout during development with -m smoke.
LAYOUTS = [
    pytest.param(Layout(world=8, ep_size=8), id="2_nodes", marks=pytest.mark.smoke),
    pytest.param(Layout(world=4, ep_size=4), id="1_node"),
    pytest.param(Layout(world=12, ep_size=12), id="3_nodes"),
    # 4 nodes as 2 EP groups of 2 nodes, as with autoep_size=8: the second group's
    # EP ranks differ from its global ranks.
    pytest.param(Layout(world=16, ep_size=8), id="2_ep_groups"),
]


# Each routing maps (generator, EP rank, E) to [num_tokens, k] experts per token.
def _random(gen: torch.Generator, ep_rank: int, E: int) -> torch.Tensor:
    return torch.rand(NUM_TOKENS, E, generator=gen).topk(TOP_K, dim=-1).indices


def _top1(gen: torch.Generator, ep_rank: int, E: int) -> torch.Tensor:
    return torch.randint(E, (NUM_TOKENS, 1), generator=gen)


def _hot_experts(gen: torch.Generator, ep_rank: int, E: int) -> torch.Tensor:
    """Every token on every rank goes to the same experts, all on EP rank 0."""
    return torch.arange(TOP_K).repeat(NUM_TOKENS, 1)


def _node_experts(gen: torch.Generator, node: int) -> torch.Tensor:
    """TOP_K distinct random experts per token, all on the given node."""
    per_node = GPUS_PER_NODE * EXPERTS_PER_RANK
    local = torch.rand(NUM_TOKENS, per_node, generator=gen).topk(TOP_K, dim=-1).indices
    return node * per_node + local


def _same_node(gen: torch.Generator, ep_rank: int, E: int) -> torch.Tensor:
    """Only experts on the token's own node, so no pilots are sent."""
    return _node_experts(gen, ep_rank // GPUS_PER_NODE)


def _next_node(gen: torch.Generator, ep_rank: int, E: int) -> torch.Tensor:
    """All experts on the next node, so one pilot fans out to every copy."""
    num_nodes = E // (GPUS_PER_NODE * EXPERTS_PER_RANK)
    return _node_experts(gen, (ep_rank // GPUS_PER_NODE + 1) % num_nodes)


def _empty_rank(gen: torch.Generator, ep_rank: int, E: int) -> torch.Tensor:
    """EP rank 1 has no tokens but must still join every collective."""
    route = _random(gen, ep_rank, E)
    return route[:0] if ep_rank == 1 else route


ROUTINGS: dict[str, Callable[[torch.Generator, int, int], torch.Tensor]] = {
    "random": _random,
    "top1": _top1,
    "hot_experts": _hot_experts,
    "same_node": _same_node,
    "next_node": _next_node,
    "empty_rank": _empty_rank,
}


def _scale_by_expert(rows: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    return rows * (1.0 + expert_ids.double()).unsqueeze(-1)


def _reference(tokens: torch.Tensor, weights: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    """Expected RBD output with _scale_by_expert experts, computed locally."""
    scale = (weights.double() * (1.0 + route.double())).sum(dim=1, keepdim=True)
    return tokens * scale


def _check_rbd(rank: int, init_file: Path, layout: Layout) -> None:
    """Run RBD on one rank for every pilot rule and routing, and check its output.

    Runs RBD_combine_dispatch once per pilot rule in PILOTS and routing in
    ROUTINGS, with experts that scale their rows by 1 + expert id, and compares the
    output with _reference(). A mismatch names the rank, pilot rule and routing.

    What each case covers:
        Layouts: 1 node (no pilots), 2 nodes (the smoke layout), 3 nodes (pilots
            arrive from both sides of the direct block) and 2 EP groups of 2 nodes
            (total 4 nodes).
        Routings: random top-k, top-1 (no stage 2 copies), hot experts (all
            traffic to one rank), same node (no pilots), next node (ring topo) and an 
            empty rank (no tokens, still a participant).

    Args:
        rank: Global rank of this process.
        init_file: Shared file the ranks use to find each other.
        layout: Ranks of this run.
    """
    # The tensors are tiny; one thread per rank avoids 16 ranks x all cores fighting.
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=layout.world
    )
    try:
        local_group, _ = dist.new_subgroups(group_size=GPUS_PER_NODE)
        ep_group = None
        if layout.ep_size < layout.world:
            ep_group, _ = dist.new_subgroups(group_size=layout.ep_size)
        ep_rank = dist.get_rank(group=ep_group)
        E = EXPERTS_PER_RANK * layout.ep_size

        for p, pilot in enumerate(PILOTS):
            for r, (name, make_route) in enumerate(ROUTINGS.items()):
                gen = torch.Generator().manual_seed(1000 * rank + 10 * p + r)
                route = make_route(gen, ep_rank, E)
                tokens = torch.randn(route.shape[0], HIDDEN, generator=gen, dtype=torch.float64)
                weights = torch.rand(route.shape, generator=gen)

                output = RBD_combine_dispatch(
                    tokens,
                    weights,
                    _scale_by_expert,
                    route,
                    E=E,
                    gpus_per_node=GPUS_PER_NODE,
                    local_group=local_group,
                    ep_group=ep_group,
                    pilot=pilot,
                )
                torch.testing.assert_close(
                    output,
                    _reference(tokens, weights, route),
                    msg=lambda m: f"rank={rank} pilot={pilot} routing={name}: {m}",
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rbd_matches_local_reference(tmp_path: Path, layout: Layout) -> None:
    """Spawn one process per rank of layout and run _check_rbd on each."""
    mp.spawn(_check_rbd, args=(tmp_path / "init", layout), nprocs=layout.world)
