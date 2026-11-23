"""Tests of replica choice and routing (spec_ep.replicas), on CPU.

Usage:
    python -m pytest tests/test_replicas.py -v
"""
import pytest
import torch

from spec_ep.replicas import (
    RULES,
    choose_replicas,
    inter_node_rows,
    relabel,
    replica_ids,
    serving_node,
)

NODES, GPUS_PER_NODE, EXPERTS, TOP_K, TOKENS = 4, 2, 64, 4, 512


def _routing(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random routing skewed toward low expert ids, with tokens spread over the nodes."""
    gen = torch.Generator().manual_seed(seed)
    popularity = torch.arange(EXPERTS, 0, -1).float()
    route = torch.multinomial(popularity.expand(TOKENS, -1), TOP_K, generator=gen)
    return route, torch.arange(TOKENS) * NODES // TOKENS


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("cover", [False, True])
def test_every_copy_is_served_by_a_node_that_holds_its_expert(rule: str, cover: bool) -> None:
    """Whatever the rule and routing, a copy lands on a holder and stays home when it can."""
    route, source = _routing()
    held = choose_replicas(route, source, EXPERTS, NODES, 8, rule)
    dest = serving_node(route, source, held, cover)

    assert (held.sum(dim=1) == EXPERTS // NODES + 8).all()
    assert held[dest, route].all()
    own = held[source.unsqueeze(1).expand_as(route), route]
    assert (dest == source.unsqueeze(1))[own].all()


def test_cover_routing_never_sends_more_pilots_than_source_routing() -> None:
    """On the same layout, the smallest cover uses at most the nodes the homes would."""
    route, source = _routing()
    for rule in RULES:
        held = choose_replicas(route, source, EXPERTS, NODES, 8, rule)
        by_home = inter_node_rows(serving_node(route, source, held), source, NODES)[0]
        by_cover = inter_node_rows(serving_node(route, source, held, cover=True), source, NODES)[0]
        assert (by_cover <= by_home).all()


def test_cover_routing_shares_ties_equally() -> None:
    """Tokens that could use any of three nodes equally well are split evenly over them."""
    held = torch.zeros(NODES, EXPERTS, dtype=torch.bool)
    held[1:, 5] = True
    route = torch.full((1200, 1), 5)
    dest = serving_node(route, 0, held, cover=True)
    assert dest.flatten().bincount(minlength=NODES).tolist() == [0, 400, 400, 400]


def test_pilots_rule_keeps_a_pair_together_where_copies_rule_splits_it() -> None:
    """A pilot goes only when all of a token's experts on that node are replicated.

    Node 0's tokens need, on node 1, either the pair (32, 33) or expert 34 with two
    rare partners. Expert 34 is the busiest, so with two replicas the copies rule
    takes 34 and one of the pair and removes no pilot, while the pilots rule takes
    the pair and removes the six pilots that need only those two.
    """
    pair = torch.tensor([[0, 32, 33]]).repeat(6, 1)
    partners = torch.arange(35, 49).view(7, 2)
    hot = torch.cat([torch.full((7, 1), 34), partners], dim=1)
    route, source = torch.cat([pair, hot]), torch.zeros(13, dtype=torch.long)

    pilots = {}
    for rule in ("copies", "pilots"):
        held = choose_replicas(route, source, EXPERTS, 2, 2, rule)
        pilots[rule] = int(inter_node_rows(serving_node(route, source, held), source, 2)[0].sum())
    assert pilots == {"copies": 13, "pilots": 7}


def test_ring_with_a_full_budget_caps_four_nodes_at_one_pilot_per_token() -> None:
    """Each node then holds half the experts, and the node two ahead holds the other half."""
    route, source = _routing()
    held = choose_replicas(route, source, EXPERTS, NODES, EXPERTS // NODES, "ring")
    dest = serving_node(route, source, held, cover=True)
    visited = (dest != source.unsqueeze(1)).long() * (1 + dest)
    assert all(len(set(row) - {0}) <= 1 for row in visited.tolist())


def test_relabel_sends_each_copy_to_the_node_that_serves_it() -> None:
    """Home copies keep their rank, replicas get new ids on the serving node, none collide."""
    route, source = _routing()
    held = choose_replicas(route, source, EXPERTS, NODES, 8, "cover")
    dest = serving_node(route, source, held, cover=True)
    ids, grown = replica_ids(held, GPUS_PER_NODE)
    new = relabel(route, dest, ids)

    world = NODES * GPUS_PER_NODE
    per_rank, grown_per_rank = EXPERTS // world, grown // world
    assert grown_per_rank == per_rank + 8 // GPUS_PER_NODE
    assert (new // grown_per_rank // GPUS_PER_NODE == dest).all()
    home = dest == route // (EXPERTS // NODES)
    assert (new // grown_per_rank == route // per_rank)[home].all()
    assert (new % grown_per_rank >= per_rank)[~home].all()
    # Two copies share an id exactly when they are the same expert served on the same node.
    assert torch.equal(new.flatten().unique().numel() * torch.ones(1),
                       (dest * EXPERTS + route).flatten().unique().numel() * torch.ones(1))
