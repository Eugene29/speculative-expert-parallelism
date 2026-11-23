"""Replicas of experts on other nodes, chosen to cut the rows RBD sends between nodes.

RBD sends one pilot per (token, other node the token needs). A replica is a second
copy of an expert, held by a node that is not its home. A pilot disappears only when
every expert the token needs on that node can be served somewhere the token goes
anyway, so the replicas that help RBD are not simply the busiest experts.

A layout is `held`, a [num_nodes, E] bool table: held[n, e] says node n holds expert
e, at home or as a replica. choose_replicas() fills it under a budget of replicas per
node, and serving_node() says which node serves each copy of a token:

    source routing: a copy stays on the token's node when that node holds the expert,
        and goes to the expert's home otherwise. A replica only helps the tokens that
        start on its node.
    cover routing: a token picks the fewest other nodes that together hold all its
        experts. A replica then also helps tokens of other nodes that visit its node
        anyway, which merges two pilots into one.

Rules of choose_replicas():
    copies: the busiest experts of other nodes, by copies from the node's own tokens.
        This minimizes the rows a plain all2all sends between nodes.
    pilots: greedy on the pilots the node's own tokens send, under source routing.
    cover: greedy on the pilots of all tokens, under cover routing.
    ring: node n replicates the home experts of node n + 1. It needs no routing
        statistics, and with a budget of a whole node's experts and cover routing it
        caps every token at ceil((num_nodes - 2) / 2) pilots.

replica_ids() and relabel() turn a layout into expert ids the dispatch functions
understand: every rank's block grows by the same number of replica slots, so experts
stay contiguous and plain, RBD and two-lane run unchanged on the relabeled routing.
Per call, a rank pays serving_node() and one lookup.

Usage:
    from spec_ep.replicas import choose_replicas, replica_ids, serving_node, relabel
    held = choose_replicas(route, source_node, num_experts, num_nodes, 64, "cover")
    ids, num_experts = replica_ids(held, gpus_per_node)   # once per layout
    route = relabel(route, serving_node(route, source_node, held, cover=True), ids)
"""
from functools import cache

import torch

__all__ = ["RULES", "choose_replicas", "serving_node", "replica_ids", "relabel", "inter_node_rows"]

RULES = ("copies", "pilots", "cover", "ring")


@cache
def _node_sets(num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Size and members of every set of nodes, a set being a bitmask.

    Returns:
        size: [2^num_nodes] number of nodes in each set.
        member: [2^num_nodes, num_nodes] the nodes of each set in rising order, so
            member[s, j] is the j-th node of set s; entries past the set's size are 0.
    """
    sets = [[n for n in range(num_nodes) if s >> n & 1] for s in range(1 << num_nodes)]
    size = torch.tensor([len(nodes) for nodes in sets], device=device)
    member = torch.tensor([nodes + [0] * (num_nodes - len(nodes)) for nodes in sets], device=device)
    return size, member


def _home(num_experts: int, num_nodes: int, device: torch.device) -> torch.Tensor:
    """[E] home node of every expert: contiguous blocks of E // num_nodes."""
    return torch.arange(num_experts, device=device) // (num_experts // num_nodes)


def serving_node(
    route: torch.Tensor,
    source_node: torch.Tensor | int,
    held: torch.Tensor,
    cover: bool = False,
) -> torch.Tensor:
    """Pick the node that serves each copy of each token.

    A copy always stays on the token's node when that node holds its expert. With
    cover=False every other copy goes to its expert's home. With cover=True the token
    takes the smallest set of other nodes that holds all its remaining experts, found
    by trying every set; among equally small sets the tokens take turns, so each set
    gets an equal share. A copy then goes to its expert's home if the set includes
    it, and otherwise to one of the set's nodes that hold the expert, again in turn.

    Args:
        route: [num_tokens, k] experts each token is routed to.
        source_node: [num_tokens] node each token starts on, or one node for all.
        held: [num_nodes, E] bool layout.
        cover: Route by the smallest cover instead of by home.

    Returns:
        dest: [num_tokens, k] node serving each copy.
    """
    num_nodes, num_experts = held.shape
    num_tokens = route.shape[0]
    device = route.device
    source = torch.as_tensor(source_node, device=device).expand(num_tokens).unsqueeze(1)
    home = _home(num_experts, num_nodes, device)[route]
    if not cover:
        return torch.where(held[source.expand_as(route), route], source, home)

    # Bit n of holders[t, j] says node n holds expert route[t, j].
    node_bit = 1 << torch.arange(num_nodes, device=device)
    holders = (held.long() * node_bit.unsqueeze(1)).sum(dim=0)[route]
    source_bit = 1 << source

    # Smallest node set: score all 2^num_nodes sets at once. A set must include the
    # token's node and hold every expert. Among the smallest, token i takes the
    # (i mod count)-th, so every one of them gets an equal share.
    sets = torch.arange(1 << num_nodes, device=device)
    set_size, member = _node_sets(num_nodes, device)
    token = torch.arange(num_tokens, device=device).unsqueeze(1)
    valid = ((holders.unsqueeze(2) & sets) != 0).all(dim=1) & ((sets & source_bit) != 0)
    smallest = valid & (set_size == torch.where(valid, set_size, num_nodes + 1).amin(dim=1, keepdim=True))
    turn = token % smallest.sum(dim=1, keepdim=True)
    best_set = ((smallest.cumsum(dim=1) == turn + 1) & smallest).long().argmax(dim=1, keepdim=True)

    # One node of the set per copy: own node, else home, else one of the set's holders,
    # again taken in turn by token index.
    usable = holders & best_set
    other = member[usable, (token % set_size[usable].clamp_min(1)).expand_as(route)]
    dest = torch.where((usable >> home) & 1 == 1, home, other)
    return torch.where((holders & source_bit) != 0, source, dest)


def inter_node_rows(
    dest: torch.Tensor,
    source_node: torch.Tensor | int,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Count the rows that cross between nodes on dispatch for a serving_node() result.

    Returns:
        pilots_out: [num_nodes] pilots each node sends: (token, other node) pairs.
        pilots_in: [num_nodes] pilots each node receives.
        copies: Copies served on another node; a plain all2all sends each one.
    """
    num_tokens = dest.shape[0]
    source = torch.as_tensor(source_node, device=dest.device).expand(num_tokens)
    remote = dest != source.unsqueeze(1)
    visits = torch.zeros(num_tokens, num_nodes, dtype=torch.bool, device=dest.device)
    visits.scatter_(1, dest, remote)
    pilots_out = torch.zeros(num_nodes, dtype=torch.long, device=dest.device)
    pilots_out.index_add_(0, source, visits.sum(dim=1))
    return pilots_out, visits.sum(dim=0), int(remote.sum())


def choose_replicas(
    route: torch.Tensor,
    source_node: torch.Tensor,
    num_experts: int,
    num_nodes: int,
    budget: int,
    rule: str = "cover",
) -> torch.Tensor:
    """Choose budget replicas for every node from one sample of routing.

    The greedy rules (pilots, cover) add one replica per node per round. A replica of
    expert e on node n earns credit from every copy of e it could take over: copies of
    tokens that start on n and, for cover, of tokens that visit n anyway. A copy
    counts 1 / r^2, where r is the number of the token's copies its current node
    serves, because a pilot only disappears once all r have moved. Squaring favors
    pilots that are close to disappearing.

    Args:
        route: [num_tokens, k] experts each sampled token is routed to.
        source_node: [num_tokens] node each sampled token starts on.
        num_experts: Total number of experts, split evenly over the nodes.
        num_nodes: Number of nodes.
        budget: Replicas per node.
        rule: One of RULES.

    Returns:
        held: [num_nodes, E] bool layout: every node's home experts plus its replicas.
    """
    device = route.device
    home = _home(num_experts, num_nodes, device)
    nodes = torch.arange(num_nodes, device=device)
    held = home.unsqueeze(0) == nodes.unsqueeze(1)
    if budget == 0:
        return held

    if rule == "ring":
        first = ((nodes + 1) % num_nodes) * (num_experts // num_nodes)
        held.scatter_(1, first.unsqueeze(1) + torch.arange(budget, device=device), True)
        return held

    if rule == "copies":
        for n in range(num_nodes):
            counts = route[source_node == n].flatten().bincount(minlength=num_experts)
            counts[home == n] = -1
            held[n, counts.topk(budget).indices] = True
        return held

    if rule not in ("pilots", "cover"):
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}.")
    num_tokens = route.shape[0]
    for _ in range(budget):
        dest = serving_node(route, source_node, held, cover=rule == "cover")
        remote = dest != source_node.unsqueeze(1)
        served = torch.zeros(num_tokens, num_nodes, device=device)
        served.scatter_add_(1, dest, remote.float())
        credit = remote / served.gather(1, dest).clamp_min(1) ** 2
        for n in range(num_nodes):
            takes_over = source_node == n
            if rule == "cover":
                takes_over = takes_over | (served[:, n] > 0)
            copy_credit = credit * (takes_over.unsqueeze(1) & (dest != n))
            gain = torch.zeros(num_experts, device=device)
            gain.scatter_add_(0, route.flatten(), copy_credit.flatten())
            gain[held[n]] = -1
            held[n, gain.argmax()] = True
    return held


def replica_ids(held: torch.Tensor, gpus_per_node: int) -> tuple[torch.Tensor, int]:
    """Number every expert a node holds, so the dispatch functions can address replicas.

    A node's replicas are dealt over its ranks in turn. A replica is a new expert id
    after its rank's own experts, so every rank still holds one contiguous block of
    the same size and the dispatch functions run unchanged. Built once per layout.

    Args:
        held: [num_nodes, E] bool layout.
        gpus_per_node: Ranks per node.

    Returns:
        ids: [num_nodes, E] id of expert e as node n serves it, in the grown
            numbering. Meaningless where the node does not hold the expert.
        num_experts: Experts plus replica slots across all ranks.
    """
    num_nodes, num_experts = held.shape
    device = held.device
    per_rank = num_experts // (num_nodes * gpus_per_node)
    experts = torch.arange(num_experts, device=device)
    is_home = _home(num_experts, num_nodes, device) == torch.arange(num_nodes, device=device).unsqueeze(1)
    replica = held & ~is_home
    grown = per_rank + -(-int(replica.sum(dim=1).max()) // gpus_per_node)

    index = replica.cumsum(dim=1) - 1
    first_rank = torch.arange(num_nodes, device=device).unsqueeze(1) * gpus_per_node
    replica_id = (first_rank + index % gpus_per_node) * grown + per_rank + index // gpus_per_node
    home_id = experts // per_rank * grown + experts % per_rank
    return torch.where(is_home, home_id, replica_id), num_nodes * gpus_per_node * grown


def relabel(route: torch.Tensor, dest: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Rewrite a routing so every copy goes to the node serving_node() picked.

    Args:
        route: [num_tokens, k] experts each token is routed to.
        dest: [num_tokens, k] node serving each copy, from serving_node().
        ids: [num_nodes, E] from replica_ids().

    Returns:
        route: [num_tokens, k] routes in the grown numbering.
    """
    return ids[dest, route]
