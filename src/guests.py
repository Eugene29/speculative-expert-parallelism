"""Guest experts: a rank runs hot experts of other nodes from weights it pulls from their owners.

RBD sends a pilot for every (token, other node) pair. A guest is an expert of another
node whose weights a rank holds for the length of one layer. A copy routed to a guest
stays on the token's node, so guests of the experts a node asks other nodes for most
remove inter-node rows.

A rank has `slots` guest slots on its GPU, and every layer reuses them. While one layer
runs, the next layer's guests travel GPU to GPU from the ranks that own them, on a
communicator of their own, so the transfer overlaps the layer's all2alls and experts.

Four steps, each one function or class here:

    1. remote_demand(): how often this rank's tokens ask for each expert of another
       node. The routing can be the layer's own, a recording, or a prediction.
    2. choose_guests(): the experts with the most demand become guests, which removes
       the most copies. choose_guests_by_pilots() takes the experts that keep the most
       pilots alive instead, and with balance > 0 it first relieves the node this one
       sends the most pilots to. Two placements:
           unique: the ranks of a node add up their demand, and the node's rank q
               takes the slots hottest experts e with e % G == q. No two ranks of a
               node pull the same expert, and a token uses any guest on its node.
           redundant: every rank takes its own slots hottest experts and only its own
               tokens use them. No collective, and a guest row never leaves its rank,
               but two ranks of a node may pull the same expert.
    3. exchange_requests(): every rank tells the others which guests it wants, and
       learns which of its own experts to send to whom.
    4. GuestPool: two sets of slots. fetch() fills the idle set from the owners while
       the other is in use, and take() swaps them.

Guests reuse the numbering of spec_ep.replicas: every rank's block of expert ids grows
from L to L + slots, a guest being a new id after the rank's own experts. Experts stay
contiguous, so rbd_combine_dispatch() runs unchanged on Guests.grown[route] with
world * (L + slots) experts. home_ids() turns the ids the experts are called with back
into the original ones.

Usage:
    from spec_ep.guests import GuestPool, choose_guests, exchange_requests, remote_demand
    demand = remote_demand(route, num_experts, node, num_nodes)
    guests = choose_guests(demand, slots, gpus_per_node, local_group)
    pool.fetch(own_weights, exchange_requests(guests.slot_expert, L, fetch_group))
    output = rbd_combine_dispatch(tokens, weights, expert, guests.grown[route],
                                  guests.num_experts, gpus_per_node, local_group)
    python -m pytest tests/test_guests.py -v
"""
from typing import NamedTuple

import torch
import torch.distributed as dist

from spec_ep.phase_clock import in_flight

__all__ = [
    "Guests", "Request", "remote_demand", "choose_guests", "pilot_credit",
    "choose_guests_by_pilots", "no_guests", "home_ids", "exchange_requests", "GuestPool",
]


class Guests(NamedTuple):
    """The guests one rank works with for one layer."""

    slot_expert: torch.Tensor  # [slots] expert in each of this rank's guest slots, ascending; -1 last
    grown: torch.Tensor  # [E] id that serves expert e for this rank's tokens, grown numbering
    num_experts: int  # experts in the grown numbering: world * (L + slots)


class Request(NamedTuple):
    """One all2all of guest weights, as one rank sees it."""

    rows: torch.Tensor  # [sum(send)] this rank's own experts to send, by the rank that wants them
    send: list[int]  # experts sent to each rank of the group
    recv: list[int]  # experts received from each rank of the group


def remote_demand(
    route: torch.Tensor,
    num_experts: int,
    node: int,
    num_nodes: int,
) -> torch.Tensor:
    """Count the copies of a routing per expert that lives on another node.

    Args:
        route: [num_tokens, k] experts each token is routed to.
        num_experts: Total number of experts, E.
        node: This rank's node.
        num_nodes: Nodes holding the experts, in contiguous blocks of E // num_nodes.

    Returns:
        demand: [E] copies per expert; 0 for the experts of this node.
    """
    flat = route.flatten().long()
    demand = torch.zeros(num_experts, dtype=torch.long, device=route.device)
    demand.scatter_add_(0, flat, torch.ones_like(flat))
    per_node = num_experts // num_nodes
    demand[node * per_node : (node + 1) * per_node] = 0
    return demand


def _home_ids(num_experts: int, world: int, slots: int, device: torch.device) -> torch.Tensor:
    """[E] id of every expert on its home rank, in the numbering grown by slots."""
    L = num_experts // world
    expert = torch.arange(num_experts, device=device)
    return expert // L * (L + slots) + expert % L


def no_guests(
    num_experts: int,
    slots: int,
    world: int,
    device: torch.device,
) -> Guests:
    """Guests with every slot empty: the grown numbering, every copy at its home.

    Returns:
        guests: What choose_guests() returns when nothing is in demand.
    """
    empty = torch.full((slots,), -1, dtype=torch.long, device=device)
    return Guests(empty, _home_ids(num_experts, world, slots, device), world * (num_experts // world + slots))


def choose_guests(
    demand: torch.Tensor,
    slots: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup | None = None,
    ep_group: dist.ProcessGroup | None = None,
) -> Guests:
    """Make the experts in most demand this rank's guests.

    With local_group the placement is unique and the call is collective over it: the
    node's demand is the sum over its ranks, and the node's rank q takes the slots
    hottest experts e with e % G == q. Without it the placement is redundant and the
    call is local: this rank takes its own slots hottest experts. An expert nobody
    asks for is never a guest, so slots can stay empty. Ties go to the lower expert id,
    which keeps the ranks of a node in agreement.

    Args:
        demand: [E] copies per expert, from remote_demand().
        slots: Guests per rank.
        gpus_per_node: Ranks per node, G.
        local_group: Process group of this rank's node, or None for redundant guests.
        ep_group: Process group holding the experts. None means the global group.

    Returns:
        guests: This rank's slots and the id that serves every expert for its tokens.
    """
    if local_group is not None:
        demand = demand.clone()
        dist.all_reduce(demand, group=local_group)
    hot = _top(demand, slots, gpus_per_node if local_group is not None else 1)
    hot = torch.where(demand[hot] > 0, hot, -1)
    return _deal(hot, demand.numel(), slots, gpus_per_node, ep_group)


def _top(score: torch.Tensor, count: int, stripes: int) -> torch.Tensor:
    """Take the count best experts of every stripe, a stripe being the experts e % stripes == q.

    A stable sort sends ties to the lower expert id on every rank.

    Returns:
        best: [count * stripes] experts; entry i * stripes + q is the i-th best of
            stripe q, the stripe _deal() gives to the node's rank q.
    """
    top = score.view(-1, stripes).argsort(dim=0, descending=True, stable=True)[:count]
    return (top * stripes + torch.arange(stripes, device=score.device)).flatten()


def _deal(
    hot: torch.Tensor,
    num_experts: int,
    slots: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None,
) -> Guests:
    """Seat chosen experts in guest slots: a node's list over its ranks in turn, or a rank's own.

    A rank's slots are in expert order, which is the order their weights arrive in.

    Args:
        hot: [slots * G] experts chosen for this rank's node, the same on all its
            ranks, or [slots] experts chosen for this rank alone; -1 leaves a slot empty.

    Returns:
        guests: This rank's slots and the id that serves every expert for its tokens.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    E, L, stripes = num_experts, num_experts // world, hot.numel() // slots
    first = rank // gpus_per_node * gpus_per_node if stripes > 1 else rank
    hot = torch.where(hot >= 0, hot, E).view(slots, stripes).sort(dim=0).values.flatten()
    seat = torch.arange(hot.numel(), device=hot.device)
    seat_id = (first + seat % stripes) * (L + slots) + L + seat // stripes

    grown = torch.cat([_home_ids(E, world, slots, hot.device), seat_id.new_empty(1)])
    grown[hot] = seat_id
    return Guests(torch.where(hot < E, hot, -1)[rank - first :: stripes], grown[:E], world * (L + slots))


def pilot_credit(
    route: torch.Tensor,
    num_experts: int,
    node: int,
    num_nodes: int,
    taken: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score every expert of another node by the pilots a guest of it would help remove.

    A copy that leaves the node counts 1 / r for its expert, where r is how many of its
    token's copies leave for the same node. All r must become guests before that pilot
    disappears, so a pilot's credit is split over the experts that keep it alive, and
    the credits of a routing add up to its pilots.

    Args:
        route: [num_tokens, k] experts each token is routed to.
        num_experts: Total number of experts, E.
        node: This rank's node.
        num_nodes: Nodes holding the experts, in contiguous blocks of E // num_nodes.
        taken: [E] bool experts that are guests already, whose copies stay on the node.

    Returns:
        credit: [E] float; 0 for the experts of this node and the taken ones.
    """
    route = route.long()
    dest = route // (num_experts // num_nodes)
    leaving = dest != node
    if taken is not None:
        leaving = leaving & ~taken[route]
    together = (dest.unsqueeze(2) == dest.unsqueeze(1)) & leaving.unsqueeze(1)
    credit = torch.zeros(num_experts, device=route.device)
    credit.scatter_add_(0, route.flatten(), (leaving / together.sum(dim=2).clamp_min(1)).flatten())
    return credit


def choose_guests_by_pilots(
    route: torch.Tensor,
    num_experts: int,
    slots: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup | None = None,
    ep_group: dist.ProcessGroup | None = None,
    rounds: int = 4,
    balance: float = 0.0,
) -> Guests:
    """Make guests of the experts whose copies keep the most pilots alive.

    choose_guests() takes the experts in most demand, which removes the most copies.
    RBD sends pilots, not copies, and a pilot goes only when every copy behind it has a
    guest. This takes the guests in `rounds` equal batches by pilot_credit(), counting
    the guests of earlier batches as taken, so later batches finish the pilots that
    earlier ones opened. Placements and collectives are those of choose_guests(), with
    one all_reduce over local_group per round.

    An all2all is as slow as its busiest link, so removing a pilot helps most on the
    node that gets the most. With balance > 0 an expert's credit is multiplied by
    (pilots to its node / pilots to the busiest node) ** balance in every round, which
    takes guests from the busiest node first and evens out what the others receive.

    Args:
        route: [num_tokens, k] experts each of this rank's tokens is routed to.
        num_experts: Total number of experts, E.
        slots: Guests per rank.
        gpus_per_node: Ranks per node, G.
        local_group: Process group of this rank's node, or None for redundant guests.
        ep_group: Process group holding the experts. None means the global group.
        rounds: Batches to take the guests in.
        balance: How strongly to prefer the busiest node. 0 only counts pilots.

    Returns:
        guests: This rank's slots and the id that serves every expert for its tokens.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    E, G = num_experts, gpus_per_node
    num_nodes = world // G
    stripes = G if local_group is not None else 1
    taken = torch.zeros(E + 1, dtype=torch.bool, device=route.device)
    hot = []
    for i in range(rounds):
        credit = pilot_credit(route, E, rank // G, num_nodes, taken[:E])
        if local_group is not None:
            dist.all_reduce(credit, group=local_group)
        score = credit
        if balance:
            to_node = credit.view(num_nodes, -1).sum(dim=1, keepdim=True)
            share = (to_node / to_node.max().clamp_min(1e-9)) ** balance
            score = (credit.view(num_nodes, -1) * share).flatten()
        best = _top(score, slots * (i + 1) // rounds - slots * i // rounds, stripes)
        best = torch.where(credit[best] > 0, best, -1)
        taken[torch.where(best >= 0, best, E)] = True
        hot.append(best)
    return _deal(torch.cat(hot), E, slots, G, ep_group)


def home_ids(
    grown_ids: torch.Tensor,
    guests: Guests,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Turn the grown ids this rank's experts are called with into original expert ids.

    Returns:
        expert_ids: The expert behind every id: this rank's own or one of its guests.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    slots = guests.slot_expert.numel()
    L = guests.num_experts // world - slots
    local = grown_ids - rank * (L + slots)
    return torch.where(local < L, rank * L + local, guests.slot_expert[(local - L).clamp_min(0)])


def exchange_requests(
    slot_expert: torch.Tensor,
    experts_per_rank: int,
    group: dist.ProcessGroup | None = None,
) -> Request:
    """Tell every rank which guests this rank wants, and learn what it has to send.

    Collective over group, with one host sync: NCCL needs the split sizes. Experts
    travel in expert order, so a rank's guests arrive in the order of its slots.

    Args:
        slot_expert: [slots] this rank's guests, from Guests.slot_expert.
        experts_per_rank: Experts each rank of group owns, L.
        group: Process group holding the experts. None means the global group.

    Returns:
        request: The all2all that brings every rank its guests, as this rank sees it.
    """
    world, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
    wanted = [torch.empty_like(slot_expert) for _ in range(world)]
    dist.all_gather(wanted, slot_expert, group=group)
    table = torch.stack(wanted).tolist()
    # An empty slot is -1, which no rank owns.
    owners = [[e // experts_per_rank for e in want] for want in table]
    mine = [e % experts_per_rank for want, owner in zip(table, owners)
            for e, r in zip(want, owner) if r == rank]
    return Request(
        rows=torch.tensor(mine, dtype=torch.long, device=slot_expert.device),
        send=[owner.count(rank) for owner in owners],
        recv=[owners[rank].count(r) for r in range(world)],
    )


class GuestPool:
    """Two sets of `slots` experts' weights on the GPU, filled from the ranks that own them.

    One set is in use while fetch() fills the other, and take() swaps them. fetch()
    returns at once: the transfer runs on the pool's stream and communicator, under
    whatever the main stream does, and starts only after the work that still reads the
    idle set. wait() makes the current stream wait for the set in use, which costs
    nothing when its transfer is already over.

    `times` holds three CUDA events per set for its last fetch: when the transfer
    started, when the weights had arrived, and when wait() first needed them.
    started.elapsed_time(arrived) is how long the transfer took, and
    arrived.elapsed_time(needed) how long the weights sat ready (negative when the
    stream had to wait for them).
    """

    def __init__(
        self,
        like: list[torch.Tensor],
        slots: int,
        group: dist.ProcessGroup,
    ) -> None:
        """Allocate both sets.

        Args:
            like: One [L, ...] tensor per weight tensor of an expert, such as a layer's
                own weights. Only their shapes, dtype and device are used.
            slots: Guests per rank.
            group: Process group holding the experts, used for nothing else, so a
                fetch does not queue behind a layer's all2alls.
        """
        self.sets = [[w.new_empty((slots, *w.shape[1:])) for w in like] for _ in range(2)]
        self.group, self.stream = group, torch.cuda.Stream(like[0].device)
        self.times: list[list[torch.cuda.Event | None] | None] = [None, None]
        self.current = 0
        self._sending = None

    def fetch(self, sources: list[torch.Tensor], request: Request) -> None:
        """Start filling the idle set with the guests every rank asked for. Collective over group.

        Args:
            sources: This rank's own experts of the layer the guests are for, one
                [L, ...] tensor per weight tensor.
            request: What exchange_requests() returned for those guests.
        """
        idle = 1 - self.current
        free, started, arrived = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        free.record()  # everything queued so far may still read the idle set
        self.times[idle] = [started, arrived, None]
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(free)
            started.record()
            # NCCL reads these until the transfer ends, which is before the next fetch.
            self._sending = [source[request.rows] for source in sources]
        num = sum(request.recv)
        for rows, guests in zip(self._sending, self.sets[idle]):
            # One block per transfer: the event clock leaves another stream current.
            with torch.cuda.stream(self.stream):
                work = dist.all_to_all_single(
                    guests[:num], rows, request.recv, request.send, group=self.group, async_op=True
                )
                work.wait()  # only makes the pool's stream wait for the transfer
                in_flight("inter", work)
        with torch.cuda.stream(self.stream):
            arrived.record()

    def take(self) -> list[torch.Tensor]:
        """Swap the sets: the one the last fetch filled is now in use.

        Returns:
            weights: The set in use, one [slots, ...] tensor per weight tensor. Call
                wait() before reading them.
        """
        self.current = 1 - self.current
        return self.sets[self.current]

    def wait(self) -> None:
        """Make the current stream wait until the set in use has arrived."""
        times = self.times[self.current]
        if times is None:
            return
        if times[2] is None:
            times[2] = torch.cuda.Event(enable_timing=True)
            times[2].record()
        torch.cuda.current_stream().wait_event(times[1])
