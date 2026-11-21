"""Benchmark every dispatch-and-combine version on the same recorded routing.

Every mode takes the same tokens, routing and expert function. It dispatches the
tokens to their experts' ranks, runs the experts, and combines the weighted outputs
back on the source rank.

Modes (--modes; plain always runs, since every mode is checked and reported against it):
    plain: spec_ep.plain. One row per (token, expert) in a single all2all each way.
    plain_fused: plain with RBD's merge kernel (spec_ep.ops.gather_sum) in place of
        index_add_, which separates the gain from fewer inter-node rows from the gain
        from a faster merge.
    rbd: spec_ep.rbd. One row per (token, destination node) between nodes, copied to
        the other experts within the node.
    two_lane: spec_ep.two_lane. RBD with the inter-node rows and the node-local rows
        in two all2alls that run at the same time.
    plain_pipe, rbd_pipe: spec_ep.pipeline. plain and rbd on --degree slices of the
        tokens, so the experts of one slice run while the next is on the network.
    tutel: Tutel. Every expert is padded to one capacity, the most rows any rank sends
        to any expert, and the padded buffer goes in an all2all with equal splits.
    tutel_pipe: Tutel's own pipelining, on --degree chunks of capacity (rounded up to
        a multiple of --degree), sent with a NCCL communicator that Tutel builds on
        first use.
    deepep: DeepEP. One row per (token, rank), through buffers shared by the GPUs of
        one node.

Node layout: rbd, two_lane and rbd_pipe need at least two nodes, since RBD saves
nothing on one. deepep needs every rank on one node. The other modes run on any
layout.

Routing is recorded but the tokens and weights are random. RUN_DIR/experts_rank{r}.pt
only holds a [num_layers, num_tokens, k] tensor of expert ids for recorded rank r, and
LAYER picks one layer. Thus, hidden dimension and number of experts are
hyper-parameters, set by flags and defaulting to Qwen3.6-35B-A3B. The recordings are
split or joined to fit the ranks, so the global workload stays the same.

The script first runs every mode with an expert that scales each row by its expert id
and checks its output against plain. It then times a full forward pass of every mode
with the --expert experts, from routing to combined output. The timed region includes each
mode's own planning: index building and the count exchange for plain, make_plan for
the RBD modes, either one once per slice in the pipes, the dense score matrix, top-k
and capacity all_reduce for Tutel, and the dispatch layout for DeepEP. For Tutel it
also includes scattering the recorded weights into that score matrix, and for DeepEP
expanding the received rows to their local experts; both adapt the library to the
shared arguments (benchmark/all2all/baselines.py). The DeepEP buffer is built once,
outside it.

Timing runs --repeats rounds. Each round times every mode in turn, with --warmup
untimed and --iterations timed calls, so the modes alternate and drift over the run
favors none. Each timed call counts as its slowest rank. A mode's median, min and max
are over all its timed calls of all rounds.

Experts in the timed runs (--expert):
    identity (default): returns its rows, so only communication is timed.
    ffn: one gated feed-forward block as in Qwen3.6-35B-A3B,
        down(silu(gate(x)) * up(x)) with --intermediate columns in between, applied to
        every row whatever its expert id. The experts then cost real GPU time, which
        is what the pipelined modes overlap with the network.
plain's time with ffn minus its time with identity is the expert time. Therefore
T_pipe(identity) + expert time - T_pipe(ffn) is the expert time a pipelined mode hid
under communication.

--trace DIR runs every mode once more under torch.profiler, with the --expert experts,
and writes DIR/{mode}_rank{r}.json on every rank. In a chrome trace viewer they show
where the expert kernels overlap the NCCL kernels.

Tutel compiles its CUDA kernels on first use. USE_NVRTC=1 makes it compile them in
process; without it every rank forks nvcc from CUDA_HOME.

Usage:
    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 [--pilot rail]

    USE_NVRTC=1 srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 --degree 2 \
        --modes plain plain_fused rbd two_lane plain_pipe rbd_pipe tutel tutel_pipe

    # One node. -w keeps rank 0 on MASTER_ADDR, where the other ranks look for it.
    srun -N 1 -w "$MASTER_ADDR" --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 --modes plain deepep

    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 --modes plain plain_pipe rbd rbd_pipe \
        --expert ffn [--intermediate 512] --trace $SCRATCH/traces
"""
import argparse
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from benchmark.utils import check_node_layout, init_dist, load_routing, make_inputs, time_ms
from spec_ep.pipeline import pipelined_plain_combine_dispatch, pipelined_rbd_combine_dispatch
from spec_ep.plain import plain_combine_dispatch
from spec_ep.rbd import PILOT_RULES, rbd_combine_dispatch
from spec_ep.two_lane import two_lane_rbd_combine_dispatch

MODES = (
    "plain", "plain_fused", "rbd", "two_lane", "plain_pipe", "rbd_pipe",
    "tutel", "tutel_pipe", "deepep",
)
NEED_TWO_NODES = {"rbd", "two_lane", "rbd_pipe"}
BASELINES = {"tutel", "tutel_pipe", "deepep"}


def inter_node_rows(
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
) -> tuple[int, int, int]:
    """Count the rows that cross between nodes on dispatch, over all ranks. Collective.

    Combine sends the same number back.

    Returns:
        plain_rows: Copies whose expert is on another node; plain sends each one.
        rbd_rows: (token, other node) pairs; RBD sends one pilot for each.
        copies: All (token, expert) copies.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    num_nodes = world // gpus_per_node
    route_node = route // (num_experts // world) // gpus_per_node
    remote = route_node != rank // gpus_per_node
    token = torch.arange(route.shape[0], device=route.device).unsqueeze(1).expand_as(route)
    pilots = (token * num_nodes + route_node)[remote].unique().numel()
    rows = torch.tensor([remote.sum().item(), pilots, route.numel()], device=route.device)
    dist.all_reduce(rows)
    return tuple(rows.tolist())


def make_ffn(
    hidden: int,
    intermediate: int,
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Make an expert that runs one gated feed-forward block on every row.

    The weights are random bf16 with the same seed on every rank, and shared by all
    experts: a row costs the flops it would with per-expert weights.

    Returns:
        ffn: Maps [num_inputs, hidden] bf16 rows and their expert ids, which it
            ignores, to [num_inputs, hidden] bf16 outputs.
    """
    gen = torch.Generator(device=device).manual_seed(0)
    shapes = ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
    gate, up, down = (
        torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16) / shape[1] ** 0.5
        for shape in shapes
    )

    def ffn(x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    return ffn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layer", type=int, help="Layer whose routing to replay")
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=["plain", "plain_fused", "rbd", "two_lane"])
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--pilot", choices=PILOT_RULES, default="rail", help="RBD pilot rule")
    parser.add_argument("--degree", type=int, default=2,
                        help="Slices of plain_pipe, rbd_pipe and tutel_pipe")
    parser.add_argument("--expert", choices=("identity", "ffn"), default="identity",
                        help="Expert of the timed runs")
    parser.add_argument("--intermediate", type=int, default=512, help="I, width inside ffn")
    parser.add_argument("--trace", metavar="DIR", help="Write a profiler trace per mode and rank")
    parser.add_argument("--repeats", type=int, default=5, help="Rounds over the modes")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed calls per round")
    parser.add_argument("--iterations", type=int, default=1, help="Timed calls per round")
    args = parser.parse_args()
    E, G = args.num_experts, args.gpus_per_node
    selected = {"plain", *args.modes}

    rank, world, device = init_dist()
    hosts = check_node_layout(G, min_nodes=1)
    num_nodes = world // G
    if E % world != 0:
        raise RuntimeError(f"{E} experts don't split evenly across {world} ranks.")
    if num_nodes < 2 and selected & NEED_TWO_NODES:
        raise RuntimeError(f"Modes {sorted(selected & NEED_TWO_NODES)} need at least 2 nodes, "
                           f"got 1. Use srun -N 2 or drop them from --modes.")
    if num_nodes > 1 and "deepep" in selected:
        raise RuntimeError(f"Mode deepep needs every rank on one node, got {num_nodes} "
                           f"nodes. Use srun -N 1 or drop it from --modes.")
    local_group, _ = dist.new_subgroups(group_size=G)

    route = load_routing(args.run_dir, args.layer, rank, world, E).to(device)
    tokens, weights = make_inputs(route, args.hidden, device, seed=rank)
    modes = {
        "plain": lambda expert: plain_combine_dispatch(tokens, weights, expert, route, E),
        "plain_fused": lambda expert: plain_combine_dispatch(
            tokens, weights, expert, route, E, fused_merge=True
        ),
        "rbd": lambda expert: rbd_combine_dispatch(
            tokens, weights, expert, route, E, G, local_group, pilot_rule=args.pilot
        ),
        "two_lane": lambda expert: two_lane_rbd_combine_dispatch(
            tokens, weights, expert, route, E, G, local_group, pilot_rule=args.pilot
        ),
        "plain_pipe": lambda expert: pipelined_plain_combine_dispatch(
            tokens, weights, expert, route, E, args.degree
        ),
        "rbd_pipe": lambda expert: pipelined_rbd_combine_dispatch(
            tokens, weights, expert, route, E, G, local_group, args.degree,
            pilot_rule=args.pilot,
        ),
    }
    if selected & BASELINES:
        from benchmark.all2all import baselines

        modes["tutel"] = lambda expert: baselines.tutel_combine_dispatch(
            tokens, weights, expert, route, E
        )
        modes["tutel_pipe"] = lambda expert: baselines.tutel_combine_dispatch(
            tokens, weights, expert, route, E, overlap_degree=args.degree
        )
    if "deepep" in selected:
        buffer = baselines.make_deepep_buffer(dist.group.WORLD, args.hidden)
        modes["deepep"] = lambda expert: baselines.deepep_combine_dispatch(
            buffer, tokens, weights, expert, route, E
        )
    modes = {name: mode for name, mode in modes.items() if name in selected}

    plain_rows, rbd_rows, copies = inter_node_rows(route, E, G)
    if selected & {"tutel", "tutel_pipe"}:
        capacity = baselines.tutel_capacity(weights, route, E)

    if rank == 0:
        print(
            f"{num_nodes} nodes x {G} GPUs: "
            f"{', '.join(hosts[::G])} | layer {args.layer} | "
            f"E={E} H={args.hidden} k={route.shape[1]} | "
            f"{route.shape[0]} tokens per rank | expert {args.expert}",
            flush=True,
        )
        print(
            f"inter-node rows per direction: plain {plain_rows} "
            f"({plain_rows / copies:.1%} of copies), rbd {rbd_rows} "
            f"({plain_rows / max(rbd_rows, 1):.2f}x fewer)",
            flush=True,
        )
        if selected & {"tutel", "tutel_pipe"}:
            padded = world * E * capacity
            print(
                f"tutel rows per direction: capacity {capacity} per expert and rank, "
                f"{padded} padded rows for {copies} copies ({padded / copies:.1f}x)",
                flush=True,
            )

    scale = lambda x, expert_ids: x * ((expert_ids + 1) / E).to(x.dtype).unsqueeze(-1)
    reference = modes["plain"](scale).float()
    for name, mode in modes.items():
        if name == "plain":
            continue
        if rank == 0:
            print(f"checking {name:>11} against plain: ", end="", flush=True)
        error = (mode(scale).float() - reference).abs().max()
        error = error / reference.abs().max().clamp_min(1e-12)
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if not error <= 2e-2:
            raise RuntimeError(f"Mode {name} differs from plain by up to {error:.2e} "
                               f"(relative).")
        if rank == 0:
            print(f"max relative error {error:.2e}", flush=True)

    expert = lambda x, expert_ids: x
    if args.expert == "ffn":
        expert = make_ffn(args.hidden, args.intermediate, device)
    calls = {name: [] for name in modes}
    for _ in range(args.repeats):
        for name, mode in modes.items():
            calls[name] += time_ms(lambda: mode(expert), args.warmup, args.iterations)
    medians = {name: statistics.median(ms) for name, ms in calls.items()}

    if rank == 0:
        print(f"{args.repeats} rounds x {args.iterations} timed calls per mode:", flush=True)
        for name, ms in calls.items():
            print(
                f"{name:>11}: median {medians[name]:8.3f} ms | min {min(ms):8.3f} | "
                f"max {max(ms):8.3f} | {medians['plain'] / medians[name]:.2f}x vs plain",
                flush=True,
            )

    if args.trace:
        os.makedirs(args.trace, exist_ok=True)
        for name, mode in modes.items():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                mode(expert)
                torch.cuda.synchronize()
            prof.export_chrome_trace(f"{args.trace}/{name}_rank{rank}.json")
        if rank == 0:
            print(f"traces: {args.trace}/{{mode}}_rank{{r}}.json", flush=True)

    dist.barrier()
    dist.destroy_process_group()
