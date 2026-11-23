"""Guest experts in the Qwen model: RBD with hot experts of other nodes pulled from their owners.

spec_ep.guests chooses guests and moves their weights; this file connects it to the
AutoEP layers of scripts/ds_moe.py as the mode rbd_guests. Every rank has `slots` guest
slots that all layers share. In MoE layer i every rank starts pulling layer i + 1's
guests from the ranks that own them, GPU to GPU on a communicator of its own, so the
transfer runs under layer i. Layer i + 1 then runs RBD on its routing relabeled to its
guests: lanes=2 (default) is two-lane RBD, which sends the same-node rows that guests
create over the node's own communicator; lanes=1 is RBD in one all2all.

Where a layer's guests come from (source):
    speculative: layer i + 1's router applied to the tokens entering layer i's MoE.
        The experts it picks are the predicted routing; no recording is needed. Layer
        0 has no guests. Costs one more host sync per layer, for the split sizes of
        the transfer, right before the plan's own.
    static: a recording of another batch, so the same guests for every batch.
    oracle: a recording of this batch, the best any prediction could do.
    Recorded guests are known when the mode is built, so a layer adds no sync. Layer
    0's guests travel under the last layer of the forward before.

Which experts become guests (choice): "demand" takes the experts with the most copies,
"pilots" the ones that keep the most pilots alive, and balance > 0 makes "pilots"
relieve the node that receives the most first. See spec_ep.guests.

A rank's experts and its guests are in different tensors, and DeepSpeed's grouped GEMM
takes one, so _grouped_mm() launches DeepSpeed's kernel once per tensor into one output.

GuestDispatch.measure() runs one forward that scores the guests against the oracle's
on the real routing and times every transfer; see its docstring for the numbers.

Usage:
    # From scripts/ds_moe.py, for modes such as
    #   rbd_guests:slots=8
    #   rbd_guests:slots=8,source=static,placement=redundant,choice=pilots,balance=2
    mode = GuestDispatch(layers, gpus_per_node, local_group, fetch_group, slots=8)
    layers[i].combine_dispatch = mode.for_layer(i)
"""
import statistics
from collections.abc import Callable
from functools import partial

import deepspeed.module_inject.auto_ep_layer as auto_ep_layer
import torch
import torch.distributed as dist
import triton
from deepspeed.ops.triton_ops.group_gemm_triton import _group_gemm_kernel
from deepspeed.ops.triton_ops.swiglu_triton import swiglu

from spec_ep.guests import (
    GuestPool,
    Guests,
    Request,
    choose_guests,
    choose_guests_by_pilots,
    exchange_requests,
    no_guests,
    remote_demand,
)
from spec_ep.rbd import rbd_combine_dispatch
from spec_ep.two_lane import two_lane_rbd_combine_dispatch

SOURCES = ("speculative", "static", "oracle")
PLACEMENTS = ("unique", "redundant")
CHOICES = ("demand", "pilots")


def _grouped_mm(
    x: torch.Tensor,
    parts: list[torch.Tensor],
    starts: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Multiply every row by its group's weight, the groups' weights being in several tensors.

    Args:
        x: [M, K] rows, sorted by group.
        parts: Weight tensors [groups of the part, N, K]. Their groups follow one another.
        starts: [groups] int32 first row of each group.
        sizes: [groups] int32 rows of each group.

    Returns:
        out: [M, N] with out[m] = x[m] @ weight[group of m].T.
    """
    M, K = x.shape
    N = parts[0].shape[1]
    out = x.new_empty((M, N))
    first = 0
    for weight in parts if M else []:
        groups, b = weight.shape[0], weight.transpose(-2, -1)
        # As DeepSpeed sizes it: a group's last tile may be partial, and programs past
        # the last tile return at once.
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) + groups, triton.cdiv(N, meta["BLOCK_N"]))
        _group_gemm_kernel[grid](
            x, b, out, starts[first : first + groups], sizes[first : first + groups], M, K, N,
            x.stride(0), x.stride(1), b.stride(0), b.stride(1), b.stride(2),
            out.stride(0), out.stride(1), NUM_GROUPS=groups,
        )
        first += groups
    return out


def _weights(layer: auto_ep_layer.AutoEPMoELayer) -> list[torch.Tensor]:
    """A layer's expert weights, in the order the guest pool keeps them: gate, down, up."""
    return [layer.experts.w1, layer.experts.w2, layer.experts.w3]


def _pilots(served: torch.Tensor, num_nodes: int, node: int) -> torch.Tensor:
    """Count the (token, other node) pairs of a [num_tokens, k] table of serving nodes."""
    visits = torch.zeros(served.shape[0], num_nodes, device=served.device)
    visits.scatter_(1, served, 1.0)
    return visits.sum() - visits[:, node].sum()


class GuestDispatch:
    """The mode rbd_guests: RBD on every MoE layer, with guests in slots the layers share."""

    def __init__(
        self,
        layers: list[auto_ep_layer.AutoEPMoELayer],
        gpus_per_node: int,
        local_group: dist.ProcessGroup,
        fetch_group: dist.ProcessGroup,
        slots: int = 8,
        source: str = "speculative",
        placement: str = "unique",
        lanes: int = 2,
        choice: str = "demand",
        rounds: int = 4,
        balance: float = 0.0,
        routing: torch.Tensor | None = None,
    ) -> None:
        """Set the mode up. Collective for the recorded sources.

        Args:
            layers: The model's AutoEP layers, in forward order.
            gpus_per_node: Ranks per node.
            local_group: Process group of this rank's node.
            fetch_group: Process group of the EP ranks that only guest weights use.
            slots: Guests per rank.
            source: "speculative", "static" or "oracle"; see the module docstring.
            placement: "unique" or "redundant"; see spec_ep.guests.
            lanes: 2 for two-lane RBD, 1 for RBD in one all2all.
            choice: "demand" or "pilots"; see the module docstring.
            rounds: Batches the pilots choice takes its guests in.
            balance: How strongly the pilots choice prefers the busiest node.
            routing: [num_layers, num_tokens, k] recorded experts of this rank, for the
                static and oracle sources.
        """
        if source not in SOURCES or placement not in PLACEMENTS or choice not in CHOICES:
            raise ValueError(f"source must be one of {SOURCES}, placement one of {PLACEMENTS} "
                             f"and choice one of {CHOICES}.")
        self.layers, self.slots, self.source = layers, slots, source
        self.by_pilots, self.rounds, self.balance = choice == "pilots", rounds, balance
        self.G, self.ep_group, self.fetch_group = gpus_per_node, layers[0].ep_group, fetch_group
        self.node_group = local_group if placement == "unique" else None
        self.world, self.rank = dist.get_world_size(self.ep_group), dist.get_rank(self.ep_group)
        self.E, self.L = layers[0].num_experts, layers[0].num_local_experts
        self.device = layers[0].experts.w1.device
        self.none = no_guests(self.E, slots, self.world, self.device)
        rbd = two_lane_rbd_combine_dispatch if lanes == 2 else rbd_combine_dispatch
        self.inner = partial(rbd, gpus_per_node=gpus_per_node, local_group=local_group,
                             ep_group=self.ep_group)
        self.pool = GuestPool(_weights(layers[0]), slots, fetch_group)

        # Recorded sources know every layer's guests and transfers now; a prediction
        # is made per call.
        self.recorded: list[tuple[Guests, Request]] | None = None
        if source != "speculative":
            guests = [self._choose(layer_routing.to(self.device)) for layer_routing in routing]
            self.recorded = [
                (g, exchange_requests(g.slot_expert, self.L, fetch_group)) for g in guests
            ]
        self.fetched = self.guests = self.predicted = self.stats = None

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
                route, self.E, self.slots, self.G, self.node_group, self.ep_group, self.rounds,
                self.balance,
            )
        node, num_nodes = self.rank // self.G, self.world // self.G
        demand = remote_demand(route, self.E, node, num_nodes)
        return choose_guests(demand, self.slots, self.G, self.node_group, self.ep_group)

    def _fetch(self, i: int, request: Request) -> None:
        """Start pulling layer i's guests into the idle slots. Collective over fetch_group."""
        self.pool.fetch(_weights(self.layers[i]), request)
        self.fetched = i

    def _dispatch(
        self,
        i: int,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        route: torch.Tensor,
    ) -> torch.Tensor:
        """Run layer i with its guests, after starting to pull the next layer's.

        Returns:
            output: [num_tokens, H] the layer's combined output.
        """
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
        guests, predicted = self.none, None
        if self.recorded is not None:
            guests, request = self.recorded[i]
            if self.fetched != i:  # only in the mode's first forward
                self._fetch(i, request)
        elif i:
            guests, predicted = self.guests, self.predicted
        held, times = self.pool.take(), self.pool.times[self.pool.current]

        # Start pulling the next layer's guests. They travel under this layer.
        if self.recorded is not None:
            following = (i + 1) % len(self.layers)
            self._fetch(following, self.recorded[following][1])
        elif i + 1 < len(self.layers):
            following = self.layers[i + 1]
            self.predicted = following.router.forward(tokens, following.expert_bias)[1]
            self.guests = self._choose(self.predicted)
            self._fetch(i + 1, exchange_requests(self.guests.slot_expert, self.L, self.fetch_group))

        run = partial(self._run_experts, self.layers[i], held)
        output = self.inner(tokens, weights, run, guests.grown[route], guests.num_experts)
        if self.stats is not None and i:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._score(route, guests, predicted, times, begin, end)
        return output

    def _run_experts(
        self,
        layer: auto_ep_layer.AutoEPMoELayer,
        held: list[torch.Tensor],
        rows: torch.Tensor,
        grown_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run rows through this rank's experts and its guests, by their grown ids.

        The layer's expert call over L + slots experts, the last slots of them in the
        pool's tensors, once the guests have arrived. No host sync.

        Returns:
            outputs: [num_inputs, H] expert outputs, in the order of rows.
        """
        local = grown_ids - self.rank * (self.L + self.slots)
        order = local.argsort(stable=True)
        sizes = torch.zeros(self.L + self.slots, dtype=torch.int32, device=rows.device)
        sizes.scatter_add_(0, local, torch.ones_like(local, dtype=torch.int32))
        starts = (sizes.cumsum(0) - sizes).int()
        x = rows[order]
        self.pool.wait()
        gate, down, up = ([own, guest] for own, guest in zip(_weights(layer), held))
        hidden = swiglu(_grouped_mm(x, gate, starts, sizes), _grouped_mm(x, up, starts, sizes))
        outputs = rows.new_empty(rows.shape)
        outputs[order] = _grouped_mm(hidden, down, starts, sizes)
        return outputs

    def _score(
        self,
        route: torch.Tensor,
        guests: Guests,
        predicted: torch.Tensor | None,
        times: list[torch.cuda.Event | None] | None,
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
        visits = torch.zeros(route.shape[0], num_nodes, device=route.device)
        to_node = visits.scatter_(1, served // per_rank // self.G, 1.0).sum(dim=0)
        to_node[node] = 0  # pilots this rank sends to each other node
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
        ]).double(), to_node.double(), (*times, begin, end)))

    def measure(self, forward: Callable[[], object], print0: Callable[[str], None], name: str) -> None:
        """Run one forward and print how good its guests were and what pulling them took.

        Collective. Layer 0 is left out: a prediction gives it no guests. Printed, over
        all ranks and the layers 1 and up:
            pilots: inter-node rows RBD sends per direction, with these guests, with
                none, and with the oracle's guests on the same routing.
            busiest node: the pilots the node that receives the most gets, as a
                multiple of what a node gets on average, mean over the layers. 1 is
                an even all2all.
            covered: share of the copies for other nodes that a guest serves.
            hot-set recall: share of the oracle's guests that are also guests here.
            token recall: share of a token's real experts that the prediction also
                picked (speculative only), and its range over the layers.
            transfer: milliseconds one layer's guests took to arrive, how long the MoE
                layer they are for then took, in how many layers the transfer was the
                shorter of the two, and how long before their first use the guests
                had arrived (a negative time means the stream waited for them, so the
                transfer was not fully hidden). All for the slowest rank per layer.

        Args:
            forward: Runs one forward of the model on this rank.
            print0: Prints a line on rank 0.
            name: The mode's name, to start the lines with.
        """
        self.stats = []
        forward()
        torch.cuda.synchronize(self.device)
        counts = torch.stack([row for row, _, _ in self.stats])
        to_node = torch.stack([row for _, row, _ in self.stats])
        times = torch.tensor([
            [started.elapsed_time(arrived), begin.elapsed_time(end),
             arrived.elapsed_time(needed) if needed is not None else float("nan")]
            for _, _, (started, arrived, needed, begin, end) in self.stats
        ], dtype=torch.double, device=self.device)
        self.stats = None
        slack = times[:, 2].nan_to_num(nan=float("inf"))
        slowest = times[:, :2].clone()
        dist.all_reduce(counts, group=self.ep_group)
        dist.all_reduce(to_node, group=self.ep_group)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX, group=self.ep_group)
        dist.all_reduce(slack, op=dist.ReduceOp.MIN, group=self.ep_group)

        total = counts.sum(dim=0).tolist()
        remote, served, best_served, shared, best, pilots_none, pilots, pilots_best, _ = total
        busiest = (to_node.max(dim=1).values / to_node.mean(dim=1).clamp_min(1)).mean().item()
        recall = (counts[:, 8] / self.world).tolist()
        load, layer = slowest[:, 0].tolist(), slowest[:, 1].tolist()
        slack = [s for s in slack.tolist() if s != float("inf")]
        print0(f"{name}: guests | pilots {pilots:.0f} against {pilots_none:.0f} with none "
               f"({pilots / pilots_none:.1%}) and {pilots_best:.0f} with the oracle's | busiest node "
               f"{busiest:.3f}x the mean | covered {served / remote:.1%} of the copies for other "
               f"nodes (oracle {best_served / remote:.1%}) | hot-set recall {shared / max(best, 1):.1%}")
        if self.source == "speculative":
            print0(f"{name}: token recall {statistics.mean(recall):.1%} "
                   f"(layers {min(recall):.1%} to {max(recall):.1%}) | per layer "
                   + " ".join(f"{r:.2f}" for r in recall))
        print0(f"{name}: transfer median {statistics.median(load):.2f} ms (max {max(load):.2f}) against "
               f"a MoE layer of {statistics.median(layer):.2f} ms, slowest rank per layer | shorter "
               f"than the layer it is for in {sum(a <= b for a, b in zip(load, layer))} of {len(load)} layers "
               f"| arrived {min(slack, default=float('nan')):.2f} ms before first use at worst, "
               f"{statistics.median(slack) if slack else float('nan'):.2f} median")
