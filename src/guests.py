"""Guest experts: a rank runs hot experts of other nodes from weights it loads per layer.

RBD sends a pilot for every (token, other node) pair. A guest is an expert of another
node whose weights a rank holds for the length of one layer, loaded from host memory.
A copy routed to a guest stays on the token's node, so guests of the experts a node
asks other nodes for most remove inter-node rows.

Three steps, each one function or class here:

    1. remote_demand(): how often this rank's tokens ask for each expert of another
       node. The routing can be the layer's own, a recording, or a prediction.
    2. choose_guests(): the experts with the most demand become guests.
       choose_guests_by_pilots() takes the experts that keep the most pilots alive
       instead. Two placements:
           unique: the ranks of a node add up their demand and deal the node's
               slots * G hottest experts over themselves, so no two ranks of a node
               load the same expert, and a token uses any guest on its node.
           redundant: every rank loads its own slots hottest experts and only its own
               tokens use them. No collective, and a guest row never leaves its rank,
               but two ranks of a node may load the same expert.
    3. GuestPool: slots experts' weights on the GPU, refilled from pinned host memory
       on a side stream, so a load runs under whatever the main stream does meanwhile.
       Two pools that take turns hold the guests of the running layer and of the next
       one, which bounds the extra GPU memory at 2 * slots experts whatever the depth.

Guests reuse the numbering of spec_ep.replicas: every rank's block of expert ids grows
from L to L + slots, a guest being a new id after the rank's own experts. Experts stay
contiguous, so rbd_combine_dispatch() runs unchanged on Guests.grown[route] with
world * (L + slots) experts. home_ids() turns the ids the experts are called with back
into the original ones.

Usage:
    from spec_ep.guests import GuestPool, choose_guests, home_ids, remote_demand
    demand = remote_demand(route, num_experts, node, num_nodes)
    guests = choose_guests(demand, slots, gpus_per_node, local_group)
    output = rbd_combine_dispatch(tokens, weights, expert, guests.grown[route],
                                  guests.num_experts, gpus_per_node, local_group)
    python -m pytest tests/test_guests.py -v
"""
from typing import NamedTuple

import torch
import torch.distributed as dist

__all__ = [
    "Guests", "remote_demand", "choose_guests", "pilot_credit", "choose_guests_by_pilots",
    "no_guests", "home_ids", "GuestPool",
]


class Guests(NamedTuple):
    """The guests one rank works with for one layer."""

    slot_expert: torch.Tensor  # [slots] expert in each of this rank's guest slots, or -1
    grown: torch.Tensor  # [E] id that serves expert e for this rank's tokens, grown numbering
    num_experts: int  # experts in the grown numbering: world * (L + slots)


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
    node's demand is the sum over its ranks, and the node's slots * G hottest experts
    are dealt over the ranks in turn. Without it the placement is redundant and the
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
    E, seats = demand.numel(), slots
    if local_group is not None:
        demand = demand.clone()
        dist.all_reduce(demand, group=local_group)
        seats = slots * gpus_per_node

    # Hottest first; E - 1 - e breaks ties the same way on every rank.
    key = demand * E + (E - 1 - torch.arange(E, device=demand.device))
    hot = key.topk(seats).indices
    hot = torch.where(demand[hot] > 0, hot, -1)
    return _deal(hot, E, slots, gpus_per_node, ep_group)


def _deal(
    hot: torch.Tensor,
    num_experts: int,
    slots: int,
    gpus_per_node: int,
    ep_group: dist.ProcessGroup | None,
) -> Guests:
    """Seat chosen experts in guest slots: a node's list over its ranks in turn, or a rank's own.

    Args:
        hot: [slots * G] experts chosen for this rank's node, the same on all its
            ranks, or [slots] experts chosen for this rank alone; -1 leaves a slot empty.

    Returns:
        guests: This rank's slots and the id that serves every expert for its tokens.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    E, L, seats = num_experts, num_experts // world, hot.numel()
    first = rank // gpus_per_node * gpus_per_node if seats > slots else rank
    seat = torch.arange(seats, device=hot.device)
    holder = first + seat % (seats // slots)
    seat_id = holder * (L + slots) + L + seat // (seats // slots)

    grown = torch.cat([_home_ids(E, world, slots, hot.device), seat_id.new_empty(1)])
    grown[torch.where(hot >= 0, hot, E)] = seat_id
    return Guests(hot[rank - first :: seats // slots], grown[:E], world * (L + slots))


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
) -> Guests:
    """Make guests of the experts whose copies keep the most pilots alive.

    choose_guests() takes the experts in most demand, which removes the most copies.
    RBD sends pilots, not copies, and a pilot goes only when every copy behind it has a
    guest. This takes the guests in `rounds` equal batches by pilot_credit(), counting
    the guests of earlier batches as taken, so later batches finish the pilots that
    earlier ones opened. One round is a single weighted count; slots * G rounds add one
    guest at a time. Placements and collectives are those of choose_guests(), with one
    all_reduce over local_group per round.

    Args:
        route: [num_tokens, k] experts each of this rank's tokens is routed to.
        num_experts: Total number of experts, E.
        slots: Guests per rank.
        gpus_per_node: Ranks per node, G.
        local_group: Process group of this rank's node, or None for redundant guests.
        ep_group: Process group holding the experts. None means the global group.
        rounds: Batches to take the guests in.

    Returns:
        guests: This rank's slots and the id that serves every expert for its tokens.
    """
    world, rank = dist.get_world_size(group=ep_group), dist.get_rank(group=ep_group)
    E, G = num_experts, gpus_per_node
    seats = slots * G if local_group is not None else slots
    taken = torch.zeros(E + 1, dtype=torch.bool, device=route.device)
    hot = []
    for i in range(rounds):
        credit = pilot_credit(route, E, rank // G, world // G, taken[:E])
        if local_group is not None:
            dist.all_reduce(credit, group=local_group)
        # A stable sort breaks ties the same way on every rank of the node.
        best = credit.argsort(descending=True, stable=True)
        best = best[: seats * (i + 1) // rounds - seats * i // rounds]
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


class GuestPool:
    """The weights of `slots` experts on the GPU, refilled from pinned host memory.

    load() returns at once: the copies run on the pool's own stream, under whatever
    the main stream does, and start only after the work that still reads the pool's
    old contents. wait() makes the current stream wait for the last load, which costs
    nothing when the load is already over.

    After a load, `started`, `loaded` and `needed` are CUDA events that time it:
    started.elapsed_time(loaded) is how long the copies took, and
    loaded.elapsed_time(needed) how long the weights sat ready before the first wait()
    (negative when the stream had to wait for them).
    """

    def __init__(
        self,
        shapes: list[torch.Size],
        slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate the pool.

        Args:
            shapes: Shape of one expert's part of each weight tensor.
            slots: Experts the pool holds.
        """
        self.weights = [torch.empty((slots, *shape), dtype=dtype, device=device) for shape in shapes]
        self.stream = torch.cuda.Stream(device)
        self.started = self.loaded = self.needed = None

    def load(
        self,
        sources: list[torch.Tensor],
        rows: list[int],
        free: torch.cuda.Event | None = None,
    ) -> None:
        """Start copying one expert into every slot.

        Args:
            sources: Pinned host tensors, one per weight tensor, each [num_rows, *shape].
            rows: Row of the sources to load into each slot; a negative row leaves the
                slot as it is.
            free: Event after which nothing reads the pool's old contents. None means
                now: everything queued on the current stream so far may read them.
        """
        if free is None:
            free = torch.cuda.Event()
            free.record()
        self.started, self.loaded = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        self.needed = None
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(free)
            self.started.record()
            for slot, row in enumerate(rows):
                if row >= 0:
                    for weight, source in zip(self.weights, sources):
                        weight[slot].copy_(source[row], non_blocking=True)
            self.loaded.record()

    def wait(self) -> None:
        """Make the current stream wait until the last load has finished."""
        if self.loaded is None:
            return
        if self.needed is None:
            self.needed = torch.cuda.Event(enable_timing=True)
            self.needed.record()
        torch.cuda.current_stream().wait_event(self.loaded)
