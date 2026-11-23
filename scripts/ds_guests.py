"""Guest experts in the Qwen model: RBD with hot experts of other nodes loaded from host memory.

spec_ep.guests chooses guests and holds their weights; this file connects it to the
AutoEP layers of scripts/ds_moe.py:

    keep_remote_experts(): while DeepSpeed builds the AutoEP layers, every rank also
        keeps the experts of the other nodes in pinned host memory, the store.
    GuestDispatch: the mode rbd_guests. In MoE layer i every rank
          1. chooses the guests of layer i + 1 and starts loading them from the store
             into the pool that layer i is not using, on a side stream;
          2. runs RBD (or its pipeline, degree > 1) for layer i on the routing
             relabeled to the guests chosen during layer i - 1, running a row of a
             guest with the pool's weights.
        So a layer's guests load under the previous layer's dispatch, experts and
        combine, and two pools of `slots` experts are all the GPU memory guests take.

Where the guests of layer i + 1 come from (source):
    speculative: layer i + 1's router applied to the tokens entering layer i's MoE.
        The experts it picks are the predicted routing; no recording is needed.
    static: a recording of another batch, so the same guests for every batch.
    oracle: a recording of this batch, the best any prediction could do, at no cost.

GuestDispatch.measure() runs one forward that scores the guests against the oracle's
on the real routing and times every load; see its docstring for the numbers.

Usage:
    # From scripts/ds_moe.py, for modes such as
    #   rbd_guests:slots=16
    #   rbd_guests:slots=16,source=static,placement=redundant,degree=2
    with keep_remote_experts(gpus_per_node) as store:
        engine = init_deepspeed(model, config)
    mode = GuestDispatch(layers, store, gpus_per_node, local_group, slots=16)
    mode.acquire()
    layers[i].combine_dispatch = mode.for_layer(i)
"""
import contextlib
import statistics
from collections.abc import Callable, Iterator
from functools import partial

import deepspeed.module_inject.auto_ep_layer as auto_ep_layer
import torch
import torch.distributed as dist
from deepspeed.moe.ep_experts import GroupedExperts

from spec_ep.guests import GuestPool, Guests, choose_guests, no_guests, remote_demand
from spec_ep.pipeline import pipelined_rbd_combine_dispatch
from spec_ep.rbd import rbd_combine_dispatch

SOURCES = ("speculative", "static", "oracle")
PLACEMENTS = ("unique", "redundant")
Store = list[list[torch.Tensor]]  # per MoE layer: w1, w2, w3 of the other nodes' experts


@contextlib.contextmanager
def keep_remote_experts(gpus_per_node: int) -> Iterator[Store]:
    """Keep the other nodes' experts in pinned host memory while AutoEP builds its layers.

    AutoEP copies a rank's own experts out of the source model and drops the rest. For
    the duration, the function it does that with also copies the experts of every other
    node, in expert order, into page-locked memory, which a GPU loads from several
    times faster than from ordinary memory.

    Yields:
        store: Filled as the layers are built: store[i] holds w1, w2 and w3 of MoE
            layer i, each [E - experts per node, ...].
    """
    repack = auto_ep_layer.repack_expert_weights
    store: Store = []

    def repack_and_keep(experts_source, spec, ep_rank: int, ep_size: int) -> tuple[torch.Tensor, ...]:
        node, num_nodes = ep_rank // gpus_per_node, ep_size // gpus_per_node
        others = [repack(experts_source, spec, n, num_nodes) for n in range(num_nodes) if n != node]
        store.append([torch.cat(parts).pin_memory() for parts in zip(*others)])
        return repack(experts_source, spec, ep_rank, ep_size)

    auto_ep_layer.repack_expert_weights = repack_and_keep
    try:
        yield store
    finally:
        auto_ep_layer.repack_expert_weights = repack


def _pilots(served: torch.Tensor, num_nodes: int, node: int) -> torch.Tensor:
    """Count the (token, other node) pairs of a [num_tokens, k] table of serving nodes."""
    visits = torch.zeros(served.shape[0], num_nodes, device=served.device)
    visits.scatter_(1, served, 1.0)
    return visits.sum() - visits[:, node].sum()


class GuestDispatch:
    """The mode rbd_guests: RBD on every MoE layer, with guests loaded one layer ahead."""

    def __init__(
        self,
        layers: list[auto_ep_layer.AutoEPMoELayer],
        store: Store,
        gpus_per_node: int,
        local_group: dist.ProcessGroup,
        slots: int = 16,
        source: str = "speculative",
        placement: str = "unique",
        degree: int = 1,
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
            degree: Slices of the RBD pipeline; 1 is plain RBD.
            routing: [num_layers, num_tokens, k] recorded experts of this rank, for the
                static and oracle sources.
        """
        if source not in SOURCES or placement not in PLACEMENTS:
            raise ValueError(f"source must be one of {SOURCES} and placement one of {PLACEMENTS}.")
        self.layers, self.store, self.slots, self.source = layers, store, slots, source
        self.G, self.ep_group = gpus_per_node, layers[0].ep_group
        self.node_group = local_group if placement == "unique" else None
        self.world, self.rank = dist.get_world_size(self.ep_group), dist.get_rank(self.ep_group)
        self.E, self.L = layers[0].num_experts, layers[0].num_local_experts
        self.device = layers[0].experts.w1.device
        self.none = no_guests(self.E, slots, self.world, self.device)
        rbd = dict(gpus_per_node=gpus_per_node, local_group=local_group, ep_group=self.ep_group)
        self.inner = partial(rbd_combine_dispatch, **rbd)
        if degree > 1:
            self.inner = partial(pipelined_rbd_combine_dispatch, degree=degree, **rbd)

        # Recorded sources know every layer's guests now; a prediction is made per call.
        self.recorded = None
        if source != "speculative":
            guests = [self._choose(layer_routing.to(self.device)) for layer_routing in routing]
            self.recorded = [(g, self._rows(g)) for g in guests]
        self.pools = self.pool_experts = self.guests = self.predicted = self.stats = None

    def acquire(self) -> None:
        """Allocate the two pools on the GPU. Call before the mode's first forward."""
        experts = self.layers[0].experts
        shapes = [w.shape[1:] for w in (experts.w1, experts.w2, experts.w3)]
        self.pools = [GuestPool(shapes, self.slots, experts.w1.dtype, self.device) for _ in range(2)]
        self.pool_experts = []
        for pool in self.pools:
            module = GroupedExperts(shapes[0][1], shapes[0][0], self.slots, experts.use_grouped_mm)
            module.use_triton_grouped_mm = experts.use_triton_grouped_mm
            module.w1, module.w2, module.w3 = (
                torch.nn.Parameter(w, requires_grad=False) for w in pool.weights
            )
            self.pool_experts.append(module)

    def release(self) -> None:
        """Free the pools, so another mode's peak GPU memory does not include them."""
        self.pools = self.pool_experts = None

    def for_layer(self, i: int) -> Callable[..., torch.Tensor]:
        """The combine_dispatch function of MoE layer i.

        Returns:
            combine_dispatch: Maps (tokens, weights, expert, route) to the layer's output.
        """
        return partial(self._dispatch, i)

    def _choose(self, route: torch.Tensor) -> Guests:
        """Choose guests for a layer from a routing of this rank's tokens. Collective if unique."""
        node, num_nodes = self.rank // self.G, self.world // self.G
        demand = remote_demand(route, self.E, node, num_nodes)
        return choose_guests(demand, self.slots, self.G, self.node_group, self.ep_group)

    def _rows(self, guests: Guests) -> list[int]:
        """Row of the store for every slot's expert, -1 for an empty slot. Syncs with the host."""
        per_node = self.L * self.G
        first = self.rank // self.G * per_node
        return [e if e < first else e - per_node for e in guests.slot_expert.tolist()]

    def _dispatch(
        self,
        i: int,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        route: torch.Tensor,
    ) -> torch.Tensor:
        """Start loading layer i + 1's guests, then run layer i with its own.

        Returns:
            output: [num_tokens, H] the layer's combined output.
        """
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
        guests, predicted = (self.guests, self.predicted) if i else (self.none, None)
        if i + 1 < len(self.layers):
            if self.recorded is not None:
                self.guests, rows = self.recorded[i + 1]
            else:
                following = self.layers[i + 1]
                self.predicted = following.router.forward(tokens, following.expert_bias)[1]
                self.guests = self._choose(self.predicted)
                rows = self._rows(self.guests)
            self.pools[(i + 1) % 2].load(self.store[i + 1], rows)

        run = partial(self._run_experts, self.layers[i], i % 2)
        output = self.inner(tokens, weights, run, guests.grown[route], guests.num_experts)
        if self.stats is not None and i:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._score(route, guests, predicted, self.pools[i % 2], begin, end)
        return output

    def _run_experts(
        self,
        layer: auto_ep_layer.AutoEPMoELayer,
        pool: int,
        rows: torch.Tensor,
        grown_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run rows through this rank's experts and its guests, by their grown ids.

        The grouped kernel wants rows sorted by expert, and the rank's own experts and
        the guests live in two weight tensors, so the sorted rows are cut where the
        guests begin. Reading that cut is the one host sync guests add to a layer.

        Returns:
            outputs: [num_inputs, H] expert outputs, in the order of rows.
        """
        local = grown_ids - self.rank * (self.L + self.slots)
        order = local.argsort(stable=True)
        counts = torch.zeros(self.L + self.slots, dtype=torch.int32, device=rows.device)
        counts.scatter_add_(0, local, torch.ones_like(local, dtype=torch.int32))
        num_home = int(counts[: self.L].sum())
        rows = rows[order]
        sorted_outputs = rows.new_empty(rows.shape)
        if num_home:
            sorted_outputs[:num_home] = layer.experts(rows[:num_home], counts[: self.L])
        if num_home < rows.shape[0]:
            self.pools[pool].wait()
            sorted_outputs[num_home:] = self.pool_experts[pool](rows[num_home:], counts[self.L :])
        outputs = rows.new_empty(rows.shape)
        outputs[order] = sorted_outputs
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

        Collective. Layer 0 has no guests and is left out. Printed, over all ranks and
        the layers 1 and up:
            pilots: inter-node rows RBD sends per direction, with these guests, with
                none, and with the oracle's guests on the same routing.
            covered: share of the copies for other nodes that a guest serves.
            hot-set recall: share of the oracle's guests that are also guests here.
            token recall: share of a token's real experts that the prediction also
                picked (speculative only), and its range over the layers.
            load: milliseconds the copies of one layer's guests took, the MoE layer
                they ran under, in how many layers the load was the shorter of the
                two, and how long before their first use the guests were ready (a
                negative time means the stream waited for them).

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
        print0(f"{name}: load median {statistics.median(load):.2f} ms (max {max(load):.2f}) under "
               f"a MoE layer of {statistics.median(layer):.2f} ms, slowest rank per layer | shorter "
               f"than its layer in {sum(a <= b for a, b in zip(load, layer))} of {len(load)} layers "
               f"| ready {min(slack, default=float('nan')):.2f} ms before first use at worst, "
               f"{statistics.median(slack) if slack else float('nan'):.2f} median")
