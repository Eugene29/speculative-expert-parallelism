"""Shared check for the combine-dispatch tests: layouts, routings and the expected answer.

A test spawns one CPU process per rank over gloo and runs a combine-dispatch function
end to end. Layouts follow Perlmutter, 4 GPUs per node. Each layout spawns once and
loops over every variant, pilot rule and routing inside the workers, since starting
the processes costs far more than one call.

Each expert scales its rows by 1 + expert id, so a row sent to the wrong expert,
merged into the wrong token or given the wrong weight changes the output. The
expected output needs no communication, so every rank checks its own tokens.
Everything runs in float64, so the match is exact up to rounding.

Usage:
    from checks import LAYOUTS, Layout, run_check
"""
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spec_ep.rbd import PILOT_RULES

GPUS_PER_NODE = 4
EXPERTS_PER_RANK = 4
NUM_TOKENS = 16
HIDDEN = 4
TOP_K = 4

CombineDispatch = Callable[..., torch.Tensor]


class Layout(NamedTuple):
    """Global ranks, split into world // ep_size EP groups that each run the function."""

    world: int
    ep_size: int


LAYOUTS = [
    pytest.param(Layout(world=8, ep_size=8), id="2_nodes", marks=pytest.mark.smoke),
    pytest.param(Layout(world=4, ep_size=4), id="1_node"),
    pytest.param(Layout(world=12, ep_size=12), id="3_nodes"),
    pytest.param(Layout(world=16, ep_size=8), id="2_ep_groups"),
]


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
    """Expected output with _scale_by_expert experts, computed locally."""
    scale = (weights.double() * (1.0 + route.double())).sum(dim=1, keepdim=True)
    return tokens * scale


def _check_rank(
    rank: int,
    init_file: Path,
    layout: Layout,
    variants: dict[str, CombineDispatch],
) -> None:
    """Run every variant on one rank for every pilot rule and routing, and check its output.

    The output is compared with _reference(). A mismatch names the rank, variant,
    pilot rule and routing.

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
        variants: Named functions to check, as in run_check().
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

        for variant, combine_dispatch in variants.items():
            for p, pilot_rule in enumerate(PILOT_RULES):
                for r, (name, make_route) in enumerate(ROUTINGS.items()):
                    gen = torch.Generator().manual_seed(1000 * rank + 10 * p + r)
                    route = make_route(gen, ep_rank, E)
                    tokens = torch.randn(
                        route.shape[0], HIDDEN, generator=gen, dtype=torch.float64
                    )
                    weights = torch.rand(route.shape, generator=gen)

                    output = combine_dispatch(
                        tokens,
                        weights,
                        _scale_by_expert,
                        route,
                        num_experts=E,
                        gpus_per_node=GPUS_PER_NODE,
                        local_group=local_group,
                        ep_group=ep_group,
                        pilot_rule=pilot_rule,
                    )
                    torch.testing.assert_close(
                        output,
                        _reference(tokens, weights, route),
                        msg=lambda m: (
                            f"rank={rank} variant={variant} pilot_rule={pilot_rule} "
                            f"routing={name}: {m}"
                        ),
                    )
    finally:
        dist.destroy_process_group()


def run_check(tmp_path: Path, layout: Layout, variants: dict[str, CombineDispatch]) -> None:
    """Spawn one process per rank of layout and check every variant on each.

    Args:
        tmp_path: Directory for the file the ranks use to find each other.
        layout: Ranks of this run.
        variants: Named functions to check. Each takes the arguments of
            rbd_combine_dispatch and must be picklable, so a module-level function or
            a functools.partial of one.
    """
    mp.spawn(_check_rank, args=(tmp_path / "init", layout, variants), nprocs=layout.world)
