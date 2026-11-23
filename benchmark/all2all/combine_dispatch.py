"""Benchmark every dispatch-and-combine version on the same recorded routing.

Every mode takes the same tokens, routing and expert function. It dispatches the
tokens to their experts' ranks, runs the experts, and combines the weighted outputs
back on the source rank.

Modes (--modes; plain always runs, since every mode is checked and reported against it)
are listed in benchmark/all2all/modes.py: plain, plain_fused, rbd, two_lane, autoep,
tutel, ds_moe, xmoe and deepep. A mode takes settings as
name:key=value,key=value, the keyword arguments of its function, so one run can time a
mode several times, as in tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2.

A mode that drops copies is checked against plain run on the copies it keeps.

Node layout: rbd and two_lane need at least two nodes, since RBD saves nothing on
one. deepep needs every rank on one node unless the multinode DeepEP build
is on PYTHONPATH (see benchmark/all2all/baselines.py). The other modes run on any
layout.

Routing is recorded but the tokens and weights are random. RUN_DIR/experts_rank{r}.pt
only holds a [num_layers, num_tokens, k] tensor of expert ids for recorded rank r, and
LAYER picks one layer. The token width and the number of experts are therefore set by
flags, which default to Qwen3.6-35B-A3B. The recordings are split or joined to fit
the ranks, so the global workload stays the same.

--replicas RULE gives every node --budget replicas of experts it does not hold, before
anything runs, chosen by spec_ep.replicas.choose_replicas: copies (the busiest, which
suits plain), pilots (the ones that remove the most of the node's own pilots), cover
(the ones that remove the most pilots overall when a token may use any node holding a
replica) or ring (the next node's experts, no statistics). --cover routes each token
to the fewest nodes that hold all its experts; without it a copy only uses a replica
on its own node. --choose-from DIR chooses from another recording, a static choice
made ahead; the default chooses from the routing that is timed, the best a loader
that knew the batch could do. The replicas are emulated by rerouting: a replica is a
new expert id on its node, and nothing is loaded. The time the routing to replicas
takes on the GPU is printed, and is not part of any mode's time.

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
        every row whatever its expert id. The experts then cost real GPU time.
plain's time with ffn minus its time with identity is the expert time.

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
        logs/2026_oct/2026-10-16_14-07-50 0 \
        --modes plain plain_fused rbd two_lane \
                tutel tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2 \
                ds_moe:capacity_factor=1

    # One node. -w keeps rank 0 on MASTER_ADDR, where the other ranks look for it.
    srun -N 1 -w "$MASTER_ADDR" --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 \
        --modes plain deepep deepep:num_sms=64,fused=True

    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 --modes plain rbd two_lane \
        --expert ffn [--intermediate 512] --trace $SCRATCH/traces

    # 64 replicas per node, chosen from another recording to remove the most pilots.
    srun -N 4 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.combine_dispatch \
        logs/2026_oct/2026-10-16_14-07-50 0 --replicas cover --budget 64 --cover \
        --choose-from logs/other_recording
"""
import argparse
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from benchmark.all2all.modes import NEED_TWO_NODES, PADDED, build_mode, parse_mode
from benchmark.utils import (
    check_node_layout,
    init_dist,
    load_routing,
    load_sample,
    make_inputs,
    time_ms,
)
from spec_ep.plain import plain_combine_dispatch
from spec_ep.rbd import PILOT_RULES
from spec_ep.replicas import RULES, choose_replicas, relabel, replica_ids, serving_node


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


def trace_aligned_call(fn: Callable[[], object], path: str) -> None:
    """Profile one call of fn and write its chrome trace to path. Collective.

    Ranks start their profilers at different times, so a barrier inside the profile
    lines them up before the call. Otherwise the first transfer of the trace would
    include waiting for the slower ranks.
    """
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        dist.barrier()
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layer", type=int, help="Layer whose routing to replay")
    parser.add_argument("--modes", nargs="+", metavar="MODE[:KEY=VALUE,...]",
                        default=["plain", "plain_fused", "rbd", "two_lane"])
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--pilot", choices=PILOT_RULES, default="rail", help="RBD pilot rule")
    parser.add_argument("--replicas", choices=RULES, help="Rule that chooses the replicas")
    parser.add_argument("--budget", type=int, default=64, help="Replicas per node")
    parser.add_argument("--cover", action="store_true",
                        help="Route each token to the fewest nodes that hold its experts")
    parser.add_argument("--choose-from", metavar="DIR",
                        help="Recording the replicas are chosen from; default RUN_DIR")
    parser.add_argument("--expert", choices=("identity", "ffn"), default="identity",
                        help="Expert of the timed runs")
    parser.add_argument("--intermediate", type=int, default=512, help="I, width inside ffn")
    parser.add_argument("--trace", metavar="DIR", help="Write a profiler trace per mode and rank")
    parser.add_argument("--repeats", type=int, default=5, help="Rounds over the modes")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed calls per round")
    parser.add_argument("--iterations", type=int, default=1, help="Timed calls per round")
    args = parser.parse_args()
    E, G = args.num_experts, args.gpus_per_node
    specs = dict.fromkeys(["plain", *args.modes])
    parsed = {spec: parse_mode(spec) for spec in specs}
    selected = {name for name, _ in parsed.values()}

    rank, world, device = init_dist()
    hosts = check_node_layout(G, min_nodes=1)
    num_nodes = world // G
    if E % world != 0:
        raise RuntimeError(f"{E} experts don't split evenly across {world} ranks.")
    if num_nodes < 2 and selected & NEED_TWO_NODES:
        raise RuntimeError(f"Modes {sorted(selected & NEED_TWO_NODES)} need at least 2 nodes, "
                           f"got 1. Use srun -N 2 or drop them from --modes.")
    if num_nodes > 1 and "deepep" in selected:
        import deep_ep_cpp

        # v1.2.1 has no inter-node path and takes 8 ranks as one node.
        if getattr(deep_ep_cpp, "NUM_MAX_NVL_PEERS", 8) != G:
            raise RuntimeError(f"Mode deepep on {num_nodes} nodes needs the multinode DeepEP "
                               f"build for {G} ranks per node. Use srun -N 1 or drop it "
                               f"from --modes.")
    local_group, _ = dist.new_subgroups(group_size=G)

    route = load_routing(args.run_dir, args.layer, rank, world, E).to(device)
    if args.replicas:
        # Rank 0 chooses for all nodes, so every rank routes by the same layout.
        held = torch.zeros(num_nodes, E, dtype=torch.long, device=device)
        if rank == 0:
            sample, sample_node = load_sample(args.choose_from or args.run_dir, args.layer, num_nodes)
            held = choose_replicas(
                sample.to(device), sample_node.to(device), E, num_nodes, args.budget, args.replicas
            ).long()
        dist.broadcast(held, src=0)
        held = held.bool()
        ids, E = replica_ids(held, G)
        recorded = route
        to_replicas = lambda: relabel(recorded, serving_node(recorded, rank // G, held, args.cover), ids)
        route = to_replicas()
        routing_ms = statistics.median(time_ms(to_replicas, warmup=2, iterations=10))
    tokens, weights = make_inputs(route, args.hidden, device, seed=rank)

    modes = {
        spec: build_mode(spec, E, G, local_group, pilot_rule=args.pilot)
        for spec in specs
    }
    run = lambda mode, expert: mode(tokens, weights, expert, route)

    plain_rows, rbd_rows, copies = inter_node_rows(route, E, G)
    capacity, kept, dropped = {}, {}, {}
    for spec, (name, settings) in parsed.items():
        if name in PADDED:
            from benchmark.all2all.baselines import capacity_kept

            capacity[spec], kept[spec] = capacity_kept(
                weights, route, E, settings.get("capacity_factor", 0), settings.get("degree", 1)
            )
        elif name == "xmoe" and settings.get("capacity_factor", 0):
            from benchmark.all2all.xmoe import xmoe_kept

            kept[spec] = xmoe_kept(
                weights, route, E, settings["capacity_factor"], settings.get("min_capacity", 4)
            )
        if spec in kept:
            count = (~kept[spec]).sum()
            dist.all_reduce(count)
            dropped[spec] = count.item()

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
        if args.replicas:
            print(
                f"replicas: {args.budget} per node by rule {args.replicas}, chosen from "
                f"{args.choose_from or 'the timed routing'}, "
                f"{'cover' if args.cover else 'source'} routing; E grows to {E}; "
                f"routing to replicas takes {routing_ms:.3f} ms per call",
                flush=True,
            )
        for spec in dropped.keys() - capacity.keys():
            print(f"{spec} drops {dropped[spec]} copies ({dropped[spec] / copies:.1%})", flush=True)
        for spec in capacity:
            rows = world * E * capacity[spec]
            print(
                f"{spec} rows per direction: capacity {capacity[spec]} per expert and rank, "
                f"{rows} padded rows for {copies} copies ({rows / copies:.1f}x), "
                f"drops {dropped[spec]} copies ({dropped[spec] / copies:.1%})",
                flush=True,
            )

    scale = lambda x, expert_ids: x * ((expert_ids + 1) / E).to(x.dtype).unsqueeze(-1)
    reference = run(modes["plain"], scale).float()
    width = max(map(len, modes))
    for name, mode in modes.items():
        if name == "plain":
            continue
        if rank == 0:
            print(f"checking {name:>{width}} against plain: ", end="", flush=True)
        expected = reference
        if dropped.get(name):
            expected = plain_combine_dispatch(
                tokens, weights * kept[name], scale, route, E
            ).float()
        error = (run(mode, scale).float() - expected).abs().max()
        error = error / expected.abs().max().clamp_min(1e-12)
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
            calls[name] += time_ms(lambda: run(mode, expert), args.warmup, args.iterations)
    medians = {name: statistics.median(ms) for name, ms in calls.items()}

    if rank == 0:
        print(f"{args.repeats} rounds x {args.iterations} timed calls per mode:", flush=True)
        for name, ms in calls.items():
            print(
                f"{name:>{width}}: median {medians[name]:8.3f} ms | min {min(ms):8.3f} | "
                f"max {max(ms):8.3f} | {medians['plain'] / medians[name]:.2f}x vs plain",
                flush=True,
            )

    if args.trace:
        os.makedirs(args.trace, exist_ok=True)
        for name, mode in modes.items():
            trace_aligned_call(lambda: run(mode, expert), f"{args.trace}/{name}_rank{rank}.json")
        if rank == 0:
            print(f"traces: {args.trace}/{{mode}}_rank{{r}}.json", flush=True)

    dist.barrier()
    dist.destroy_process_group()
