"""Guest experts in the Qwen model: RBD with hot experts of other nodes loaded from host memory.

spec_ep.guests chooses guests and holds their weights; this file connects it to the
AutoEP layers of scripts/ds_moe.py:

    keep_remote_experts(): while DeepSpeed builds the AutoEP layers, every rank also
        keeps the experts of the other nodes in pinned host memory, the store.
    check_store(): every stored expert against the rank that owns it, once at setup.
    GuestDispatch: the mode rbd_guests. Every layer's weight tensors get `slots` rows
        after the rank's own experts, its guest rows, so one expert call runs a rank's
        experts and its guests with no host sync. Guests therefore take layers * slots
        experts of GPU memory. Every layer runs RBD on its routing relabeled to its
        guests: lanes=2 (default) is two-lane RBD, which sends the same-node rows that
        guests create over the node's own communicator; lanes=1 is RBD in one all2all.
        degree > 1 is the RBD pipeline on those lanes.

Where a layer's guests come from (source), and when they are loaded:
    speculative: layer i + 1's router applied to the tokens entering layer i's MoE.
        The experts it picks are the predicted routing; no recording is needed. In
        MoE layer i every rank chooses the guests of layer i + 1 and starts loading
        them from the store into that layer's guest rows, on a side stream, so they
        load under layer i. The choice reaches the host on a side stream and is read
        after the layer's own host sync, so choosing adds no sync of its own.
    static: a recording of another batch, so the same guests for every batch. They
        are loaded once, when the mode is set, and stay.
    oracle: a recording of this batch, the best any prediction could do, at no cost.
        Loaded once, like static.

GuestDispatch.measure() runs one forward that scores the guests against the oracle's
on the real routing and times every load; see its docstring for the numbers.

Usage:
    # From scripts/ds_moe.py, for modes such as
    #   rbd_guests:slots=16
    #   rbd_guests:slots=16,source=static,placement=redundant,lanes=1,degree=2
    with keep_remote_experts(gpus_per_node) as store:
        engine = init_deepspeed(model, config)
    check_store(layers, store, gpus_per_node)
    mode = GuestDispatch(layers, store, gpus_per_node, local_group, slots=16)
    mode.acquire()
    layers[i].combine_dispatch = mode.for_layer(i)
"""
import contextlib
import ctypes
import os
import statistics
from collections.abc import Callable, Iterator
from functools import partial

import deepspeed.module_inject.auto_ep_layer as auto_ep_layer
import torch
import torch.distributed as dist

from spec_ep.guests import (
    GuestPool,
    Guests,
    choose_guests,
    choose_guests_by_pilots,
    no_guests,
    remote_demand,
)
from spec_ep.pipeline import pipelined_rbd_combine_dispatch
from spec_ep.rbd import rbd_combine_dispatch
from spec_ep.two_lane import two_lane_rbd_combine_dispatch

SOURCES = ("speculative", "static", "oracle")
PLACEMENTS = ("unique", "redundant")
WEIGHTS = ("w1", "w2", "w3")  # an expert's weight tensors in a layer's GroupedExperts
_LIBC, _MADV_DONTNEED = ctypes.CDLL("libc.so.6"), 4
Store = list[list[torch.Tensor]]  # per MoE layer: w1, w2, w3 of the other nodes' experts


class _PinnedArena:
    """Page-locked host memory, handed out as views of large chunks.

    torch rounds every pinned allocation up to a power of two: a 192 MiB tensor takes
    256 MiB. A chunk is a power of two that holds about 16 tensors, so little is lost.
    """

    def __init__(self) -> None:
        self.chunk, self.used = torch.empty(0, dtype=torch.uint8), 0

    def put(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a tensor into page-locked memory.

        Returns:
            pinned: The copy, a view of the current chunk.
        """
        size = tensor.numel() * tensor.element_size()
        if self.used + size > self.chunk.numel():
            self.chunk = torch.empty(1 << (16 * size - 1).bit_length(), dtype=torch.uint8,
                                     pin_memory=True)
            self.used = 0
        pinned = self.chunk[self.used : self.used + size].view(tensor.dtype).view(tensor.shape)
        self.used += size
        return pinned.copy_(tensor)


def _release_pages(tensor: torch.Tensor) -> None:
    """Tell the kernel the whole pages under a tensor need not stay in memory.

    The model's weights are mapped from its file. An expert that was read once to be
    copied would otherwise stay resident next to its copy.
    """
    page = os.sysconf("SC_PAGE_SIZE")
    start = -(-tensor.data_ptr() // page) * page
    length = (tensor.data_ptr() + tensor.numel() * tensor.element_size() - start) // page * page
    if length > 0:
        _LIBC.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), _MADV_DONTNEED)


@contextlib.contextmanager
def keep_remote_experts(gpus_per_node: int, stripe: int = 1) -> Iterator[Store]:
    """Keep the other nodes' experts in pinned host memory while AutoEP builds its layers.

    AutoEP copies a rank's own experts out of the source model and drops the rest. For
    the duration, the function it does that with also copies the experts of every other
    node, in expert order, into page-locked memory, which a GPU loads from several
    times faster than from ordinary memory.

    Args:
        stripe: Keep one expert in `stripe`: those with e % stripe == rank % stripe.
            Unique guests need stripe = gpus_per_node, since a rank hosts no others,
            and the store is that many times smaller. Redundant guests need 1.

    Yields:
        store: Filled as the layers are built: store[i] holds w1, w2 and w3 of MoE
            layer i, each [(E - experts per node) / stripe, ...].
    """
    repack = auto_ep_layer.repack_expert_weights
    store: Store = []
    arena = _PinnedArena()

    def repack_and_keep(experts_source, spec, ep_rank: int, ep_size: int) -> tuple[torch.Tensor, ...]:
        # One expert per call, so a rank reads only the experts it keeps: reading a
        # whole node's experts of a 122B model put 136 GiB in every rank's memory.
        E = spec.num_experts
        per_node = E // (ep_size // gpus_per_node)
        remote = [
            e for e in range(E)
            if e // per_node != ep_rank // gpus_per_node and e % stripe == ep_rank % stripe
        ]
        kept = [repack(experts_source, spec, e, E) for e in remote]
        store.append([arena.put(torch.cat(parts)) for parts in zip(*kept)])
        if spec.expert_storage == "fused_3d":
            names = (spec.expert_w1_name, spec.expert_w2_name, spec.expert_w3_name)
            for source in (getattr(experts_source, name).data for name in names if name):
                for e in remote:
                    _release_pages(source[e])
        return repack(experts_source, spec, ep_rank, ep_size)

    auto_ep_layer.repack_expert_weights = repack_and_keep
    try:
        yield store
    finally:
        auto_ep_layer.repack_expert_weights = repack


@torch.no_grad()
def check_store(
    layers: list[auto_ep_layer.AutoEPMoELayer],
    store: Store,
    gpus_per_node: int,
) -> None:
    """Fail unless every stored expert is the one its owner holds. Collective over the EP group.

    Every rank takes the sum of squares of each of its own experts' three weight tensors
    in every layer, the ranks exchange them, and each compares them with those of its
    store's rows. A store row that holds another expert, or another layer's, does not
    match: two experts' sums of squares differ far more than rounding does.
    """
    group = layers[0].ep_group
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    energy = lambda w: w.float().square().sum(dim=(1, 2), dtype=torch.float64)
    own = torch.stack([
        torch.cat([energy(w) for w in (layer.experts.w1, layer.experts.w2, layer.experts.w3)])
        for layer in layers
    ])  # [num_layers, 3 * L]: the experts' w1, then w2, then w3
    owners = own.new_empty((world * own.shape[0], own.shape[1]))
    dist.all_gather_into_tensor(owners, own, group=group)
    truth = owners.view(world, own.shape[0], 3, -1).permute(1, 2, 0, 3).flatten(2).cpu()
    per_node = truth.shape[2] // (world // gpus_per_node)
    remote = torch.ones(truth.shape[2], dtype=torch.bool)
    remote[rank // gpus_per_node * per_node :][:per_node] = False
    kept = torch.stack([torch.stack([energy(w) for w in weights]) for weights in store])
    stripe = int(remote.sum()) // kept.shape[2]
    remote &= torch.arange(remote.numel()) % stripe == rank % stripe
    wrong = ((kept - truth[:, :, remote]).abs() > 1e-5 * kept).any(dim=1).sum().item()
    if wrong:
        raise RuntimeError(f"rank {rank}: {wrong} stored experts differ from their owner's weights.")


def _pilots(served: torch.Tensor, num_nodes: int, node: int) -> torch.Tensor:
    """Count the (token, other node) pairs of a [num_tokens, k] table of serving nodes."""
    visits = torch.zeros(served.shape[0], num_nodes, device=served.device)
    visits.scatter_(1, served, 1.0)
    return visits.sum() - visits[:, node].sum()


class GuestDispatch:
    """The mode rbd_guests: RBD on every MoE layer, with guests in the layer's own weights."""

    def __init__(
        self,
        layers: list[auto_ep_layer.AutoEPMoELayer],
        store: Store,
        gpus_per_node: int,
        local_group: dist.ProcessGroup,
        slots: int = 16,
        source: str = "speculative",
        placement: str = "unique",
        lanes: int = 2,
        degree: int = 1,
        choice: str = "demand",
        rounds: int = 4,
        routing: torch.Tensor | None = None,
    ) -> None:
        """Set the mode up. Collective over local_group for recorded unique guests.

        Args:
            layers: The model's AutoEP layers, in forward order.
            store: The other nodes' experts, from keep_remote_experts().
            gpus_per_node: Ranks per node.
            local_group: Process group of this rank's node.
            slots: Guests per rank and layer.
            source: "speculative", "static" or "oracle"; see the module docstring.
            placement: "unique" or "redundant"; see spec_ep.guests.
            lanes: 2 for two-lane RBD, 1 for RBD in one all2all.
            degree: Slices of the RBD pipeline; 1 is no pipeline.
            choice: "demand" takes the experts with the most copies, "pilots" the ones
                that keep the most pilots alive (spec_ep.guests.choose_guests_by_pilots).
            rounds: Batches the pilots choice takes its guests in.
            routing: [num_layers, num_tokens, k] recorded experts of this rank, for the
                static and oracle sources.
        """
        if source not in SOURCES or placement not in PLACEMENTS:
            raise ValueError(f"source must be one of {SOURCES} and placement one of {PLACEMENTS}.")
        self.layers, self.store, self.slots, self.source = layers, store, slots, source
        self.by_pilots, self.rounds = choice == "pilots", rounds
        self.G, self.ep_group = gpus_per_node, layers[0].ep_group
        self.node_group = local_group if placement == "unique" else None
        self.world, self.rank = dist.get_world_size(self.ep_group), dist.get_rank(self.ep_group)
        self.E, self.L = layers[0].num_experts, layers[0].num_local_experts
        self.device = layers[0].experts.w1.device
        self.none = no_guests(self.E, slots, self.world, self.device)
        rbd = dict(gpus_per_node=gpus_per_node, local_group=local_group, ep_group=self.ep_group)
        self.inner = partial(two_lane_rbd_combine_dispatch if lanes == 2 else rbd_combine_dispatch, **rbd)
        if degree > 1:
            self.inner = partial(pipelined_rbd_combine_dispatch, degree=degree, lanes=lanes, **rbd)
        # A predicted choice comes to the host through this stream and buffer.
        self.reader = torch.cuda.Stream(self.device)
        self.slot_host = torch.empty(slots, dtype=torch.long).pin_memory()

        # Recorded sources know every layer's guests now; a prediction is made per call.
        self.recorded = None
        if source != "speculative":
            guests = [self._choose(layer_routing.to(self.device)) for layer_routing in routing]
            self.recorded = [(g, self._rows(g.slot_expert)) for g in guests]
        self.pools = self.guests = self.predicted = self.stats = None

    def acquire(self) -> None:
        """Grow every layer's weights by its guest rows and load the recorded guests.

        Call before the mode's first forward. A layer's weight tensors become
        [L + slots, ...]: the rank's experts, then its guests.
        """
        stream = torch.cuda.Stream(self.device)
        self.pools = []
        for i, layer in enumerate(self.layers):
            for name in WEIGHTS:
                own = getattr(layer.experts, name)
                grown = own.new_empty((self.L + self.slots, *own.shape[1:]))
                grown[: self.L] = own
                setattr(layer.experts, name, torch.nn.Parameter(grown, requires_grad=False))
            rows = [getattr(layer.experts, name)[self.L :] for name in WEIGHTS]
            self.pools.append(GuestPool(rows, stream))
            if self.recorded is not None:
                self.pools[i].load(self.store[i], self.recorded[i][1])

    def release(self) -> None:
        """Shrink every layer's weights to the rank's own experts, for the next mode."""
        for layer in self.layers:
            for name in WEIGHTS:
                own = getattr(layer.experts, name)[: self.L].clone()
                setattr(layer.experts, name, torch.nn.Parameter(own, requires_grad=False))
        self.pools = None

    def for_layer(self, i: int) -> Callable[..., torch.Tensor]:
        """The combine_dispatch function of MoE layer i.

        Returns:
            combine_dispatch: Maps (tokens, weights, expert, route) to the layer's output.
        """
        return partial(self._dispatch, i)

    def _choose(self, route: torch.Tensor) -> Guests:
        """Choose guests for a layer from a routing of this rank's tokens. Collective if unique."""
        if self.by_pilots:
            return choose_guests_by_pilots(
                route, self.E, self.slots, self.G, self.node_group, self.ep_group, self.rounds
            )
        node, num_nodes = self.rank // self.G, self.world // self.G
        demand = remote_demand(route, self.E, node, num_nodes)
        return choose_guests(demand, self.slots, self.G, self.node_group, self.ep_group)

    def _rows(self, slot_expert: torch.Tensor) -> list[int]:
        """Row of the store for every slot's expert, -1 for an empty slot.

        Syncs with the host if slot_expert is on the GPU.
        """
        per_node = self.L * self.G
        first = self.rank // self.G * per_node
        stripe = (self.E - per_node) // self.store[0][0].shape[0]
        return [(e if e < first else e - per_node) // stripe for e in slot_expert.tolist()]

    def _dispatch(
        self,
        i: int,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        route: torch.Tensor,
    ) -> torch.Tensor:
        """Run layer i with its guests, after starting to load layer i + 1's if predicted.

        Returns:
            output: [num_tokens, H] the layer's combined output.
        """
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
        guests, predicted = (self.guests, self.predicted) if i else (self.none, None)
        read = None
        if self.recorded is not None:
            guests = self.recorded[i][0]
        elif i + 1 < len(self.layers):
            following = self.layers[i + 1]
            self.predicted = following.router.forward(tokens, following.expert_bias)[1]
            self.guests = self._choose(self.predicted)
            chosen, read = torch.cuda.Event(), torch.cuda.Event()
            chosen.record()
            with torch.cuda.stream(self.reader):
                self.reader.wait_event(chosen)
                self.slot_host.copy_(self.guests.slot_expert, non_blocking=True)
                read.record()

        run = partial(self._run_experts, self.layers[i], self.pools[i])
        output = self.inner(tokens, weights, run, guests.grown[route], guests.num_experts)
        if read is not None:
            # The layer's own syncs came after the choice, so this returns at once.
            read.synchronize()
            self.pools[i + 1].load(self.store[i + 1], self._rows(self.slot_host), begin)
        if self.stats is not None and i:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._score(route, guests, predicted, self.pools[i], begin, end)
        return output

    def _run_experts(
        self,
        layer: auto_ep_layer.AutoEPMoELayer,
        pool: GuestPool,
        rows: torch.Tensor,
        grown_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run rows through this rank's experts and its guests, by their grown ids.

        The layer's weights hold both, the guests after the rank's own, so this is the
        layer's expert call over L + slots experts, once the guests have loaded.

        Returns:
            outputs: [num_inputs, H] expert outputs, in the order of rows.
        """
        local = grown_ids - self.rank * (self.L + self.slots)
        order = local.argsort(stable=True)
        counts = torch.zeros(self.L + self.slots, dtype=torch.int32, device=rows.device)
        counts.scatter_add_(0, local, torch.ones_like(local, dtype=torch.int32))
        pool.wait()
        outputs = rows.new_empty(rows.shape)
        outputs[order] = layer.experts(rows[order], counts)
        return outputs

    def _score(
        self,
        route: torch.Tensor,
        guests: Guests,
        predicted: torch.Tensor | None,
        pool: GuestPool,
        begin: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        """Score one layer's guests against the oracle's on the real routing. Collective if unique."""
        node, num_nodes = self.rank // self.G, self.world // self.G
        best = self._choose(route)
        home = self.none.grown
        served, best_served = guests.grown[route], best.grown[route]
        is_guest, is_best = guests.grown != home, best.grown != home
        per_rank = self.L + self.slots
        recall = route.new_zeros((), dtype=torch.float)
        if predicted is not None:
            recall = (predicted.unsqueeze(2) == route.unsqueeze(1)).any(dim=2).float().mean()
        self.stats.append((torch.stack([
            remote_demand(route, self.E, node, num_nodes).sum(),  # copies for other nodes
            (served != home[route]).sum(),                        # of those, served by a guest
            (best_served != home[route]).sum(),                   # and by the oracle's guests
            (is_guest & is_best).sum(),                           # guests the oracle also picks
            is_best.sum(),                                        # the oracle's guests
            _pilots(route // self.L // self.G, num_nodes, node),  # pilots without guests
            _pilots(served // per_rank // self.G, num_nodes, node),
            _pilots(best_served // per_rank // self.G, num_nodes, node),
            recall,
        ]).double(), (pool.started, pool.loaded, pool.needed, begin, end)))

    def measure(self, forward: Callable[[], object], print0: Callable[[str], None], name: str) -> None:
        """Run one forward and print how good its guests were and what loading them took.

        Collective. Layer 0 is left out: it has no predicted guests. Printed, over all
        ranks and the layers 1 and up:
            pilots: inter-node rows RBD sends per direction, with these guests, with
                none, and with the oracle's guests on the same routing.
            covered: share of the copies for other nodes that a guest serves.
            hot-set recall: share of the oracle's guests that are also guests here.
            token recall: share of a token's real experts that the prediction also
                picked (speculative only), and its range over the layers.
            load: milliseconds the copies of one layer's guests took, how long that
                MoE layer itself then took, in how many layers the load was the
                shorter of the two, and how long before their first use the guests
                were ready (a negative time means the stream waited for them). A
                predicted layer's copies run under the previous layer and the
                attention between the two, so the layer's time is a yardstick for one
                MoE layer and not the span the load ran in. Recorded guests were
                loaded when the mode was set.

        Args:
            forward: Runs one forward of the model on this rank.
            print0: Prints a line on rank 0.
            name: The mode's name, to start the lines with.
        """
        self.stats = []
        forward()
        torch.cuda.synchronize(self.device)
        counts = torch.stack([row for row, _ in self.stats])
        times = torch.tensor([
            [started.elapsed_time(loaded), begin.elapsed_time(end),
             loaded.elapsed_time(needed) if needed is not None else float("nan")]
            for _, (started, loaded, needed, begin, end) in self.stats
        ], dtype=torch.double, device=self.device)
        self.stats = None
        slack = times[:, 2].nan_to_num(nan=float("inf"))
        slowest = times[:, :2].clone()
        dist.all_reduce(counts, group=self.ep_group)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX, group=self.ep_group)
        dist.all_reduce(slack, op=dist.ReduceOp.MIN, group=self.ep_group)

        total = counts.sum(dim=0).tolist()
        remote, served, best_served, shared, best, pilots_none, pilots, pilots_best, _ = total
        recall = (counts[:, 8] / self.world).tolist()
        load, layer = slowest[:, 0].tolist(), slowest[:, 1].tolist()
        slack = [s for s in slack.tolist() if s != float("inf")]
        print0(f"{name}: guests | pilots {pilots:.0f} against {pilots_none:.0f} with none "
               f"({pilots / pilots_none:.1%}) and {pilots_best:.0f} with the oracle's | covered "
               f"{served / remote:.1%} of the copies for other nodes (oracle {best_served / remote:.1%}) "
               f"| hot-set recall {shared / max(best, 1):.1%}")
        if self.source == "speculative":
            print0(f"{name}: token recall {statistics.mean(recall):.1%} "
                   f"(layers {min(recall):.1%} to {max(recall):.1%}) | per layer "
                   + " ".join(f"{r:.2f}" for r in recall))
        print0(f"{name}: load median {statistics.median(load):.2f} ms (max {max(load):.2f}) against "
               f"a MoE layer of {statistics.median(layer):.2f} ms, slowest rank per layer | shorter "
               f"than the layer it is for in {sum(a <= b for a, b in zip(load, layer))} of {len(load)} layers "
               f"| ready {min(slack, default=float('nan')):.2f} ms before first use at worst, "
               f"{statistics.median(slack) if slack else float('nan'):.2f} median")
