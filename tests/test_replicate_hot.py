"""Test of hot-expert replication (benchmark.utils.replicate_hot), on CPU.

Usage:
    python -m pytest tests/test_replicate_hot.py -v
"""
import torch

from benchmark.utils import replicate_hot

WORLD, GPUS_PER_NODE, EXPERTS, TOP_K = 8, 4, 64, 4


def _node(route: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Node that holds each expert of route, with num_experts spread over WORLD ranks."""
    return route // (num_experts // WORLD) // GPUS_PER_NODE


def test_hot_copies_move_to_the_tokens_node_and_the_rest_stay() -> None:
    """A remote copy of a hot expert lands on the token's node; every other copy keeps its rank."""
    gen = torch.Generator().manual_seed(0)
    hot = torch.tensor([1, 17, 33, 40, 62, 63])
    for rank in range(WORLD):
        route = torch.rand(32, EXPERTS, generator=gen).topk(TOP_K).indices
        new, grown = replicate_hot(route, hot, rank, WORLD, EXPERTS, GPUS_PER_NODE)

        moved = torch.isin(route, hot) & (_node(route, EXPERTS) != rank // GPUS_PER_NODE)
        assert moved.any()
        assert grown == WORLD * (EXPERTS // WORLD + 2)
        assert (_node(new, grown)[moved] == rank // GPUS_PER_NODE).all()
        assert (new // (grown // WORLD) == route // (EXPERTS // WORLD))[~moved].all()
        assert all(len(set(row)) == TOP_K for row in new.tolist())


def test_each_hot_expert_gets_one_replica_per_node() -> None:
    """Two ranks of a node send a hot expert to the same replica, and two hot experts never share one."""
    hot = torch.arange(32, 38)
    route = hot.unsqueeze(0)
    replicas = [replicate_hot(route, hot, rank, WORLD, EXPERTS, GPUS_PER_NODE)[0] for rank in (0, 3)]
    assert torch.equal(replicas[0], replicas[1])
    assert replicas[0].unique().numel() == hot.numel()
