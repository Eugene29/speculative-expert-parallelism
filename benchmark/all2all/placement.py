"""Benchmark expert placements chosen ahead of time, for plain all2all and RBD.

Assumes the routing is known before the step (oracle) and moving experts is free.
A placement is applied by relabeling the routing, route -> slots[route], so tokens
stay where they are and only the experts' ranks change.

With --placements, the candidates come from a file written by
benchmark.all2all.build_placements, built offline so no node time goes to searching;
its build times, swap counts and token imbalance are printed with the results.
Otherwise rank 0 gathers every rank's routing, builds these candidates, and
broadcasts them:
    contiguous:     expert e in slot e, the default layout.
    even:           spec_ep.placement.even_placement, balances expert token counts
                    across all ranks with the same number of experts per rank.
    colocate:       spec_ep.placement.colocate, single best pilot-cutting swaps,
                    balance ignored.
    partition:      spec_ep.placement.partition_placement, Kernighan-Lin
                    partitioning of the token hypergraph (tokens' source nodes as
                    fixed vertices) for fewest pilots, then ranks balanced per node.
    greedy <rule>:  spec_ep.placement.greedy_placement, searched for one rule's cost
                    model: plain, first, rail, most or even.
    greedy+partition <rule>: the same search, also started from partition.
Rank 0's wall time for each search is printed with the results.

Every rule (plain and RBD with each pilot rule) is timed end to end, with identity
experts like rbd_vs_plain, under every candidate placement. The table reports each
combination and, per rule, the fastest placement. Each combination is first checked
against plain with the contiguous placement, using an expert that scales rows by its
original expert id; the error column reports the worst. bf16 rounding alone gives
about 1e-2.

Then each rule is profiled with torch.profiler under the contiguous placement, its
fastest placement, and any placements named by --profile. Per phase (the
autoep/rbd_* and plain/* labels), each rank's CPU span, GPU span and time in NCCL
kernels are averaged over the profiled calls; the mean and max over ranks, and the
slowest rank, are reported. With --trace-dir, the first local rank of each node
also writes a Chrome trace of each profile of one case (--trace-case, by default the
first scaling and layer).

With --scaling strong (the default) the recorded tokens are split across ranks, as
rbd_vs_plain does. With weak, every rank replays a whole recorded rank's tokens,
rank r taking recording r % (number of recordings), so tokens per rank stay fixed as
nodes are added. Both can run in one launch.

With --out, rank 0 writes every timing and profile as JSON.

Usage:
    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.placement \
        logs/2026_oct/2026-10-16_14-07-50 0 20 39 [--scaling strong weak]
    srun -N 4 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.placement \
        logs/2026_oct/2026-10-16_14-07-50 0 20 39 --scaling strong weak \
        --placements /pscratch/sd/e/eku/placement/4n/placements.pt \
        --profile partition mtk+kl --out results.json --trace-dir traces
"""
import argparse
import json
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from benchmark.all2all.rbd_vs_plain import (load_routing, make_inputs,
                                             plain_combine_dispatch, time_ms)
from benchmark.utils import check_placement, init_dist
from spec_ep.placement import colocate, even_placement, greedy_placement, partition_placement
from spec_ep.rbd import RBD_combine_dispatch

# rule: (rbd, pilot)
RULES = {
    "plain": (False, "rail"),
    "rbd first": (True, "first"),
    "rbd rail": (True, "rail"),
    "rbd most": (True, "most"),
    "rbd even": (True, "even"),
}

# Profiled phases, in call order. Each rule has only its own.
PHASES = {
    False: ["plain/plan", "plain/dispatch", "plain/experts", "plain/combine", "plain/merge"],
    True: ["autoep/rbd_plan", "autoep/rbd_stage1_dispatch", "autoep/rbd_stage2_dispatch",
           "autoep/rbd_experts", "autoep/rbd_stage2_combine", "autoep/rbd_stage1_combine",
           "autoep/rbd_final_merge"],
}


def shared_placements(
    route: torch.Tensor,
    E: int,
    gpus_per_node: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build every candidate placement on rank 0 from all ranks' routing and share them.

    Collective.

    Args:
        route: [num_tokens, k] this rank's routing. Same shape on every rank.
        E: Total number of experts.
        gpus_per_node: Ranks per node.

    Returns:
        placements: Name -> [E] slot of each expert, the same on every rank.
        seconds: Name -> rank 0's wall time to build it.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    tags = [rule.split()[-1] for rule in RULES]
    names = (["contiguous", "colocate", "partition", "even"] + [f"greedy {t}" for t in tags]
             + [f"greedy+partition {t}" for t in tags])
    everyone = route.new_empty((world, *route.shape))
    dist.all_gather_into_tensor(everyone, route.contiguous())
    contiguous = torch.arange(E, device=route.device)
    stacked = contiguous.repeat(len(names), 1)
    seconds = torch.zeros(len(names), device=route.device, dtype=torch.float64)
    if rank == 0:
        routes = everyone.flatten(0, 1)
        source = torch.arange(world, device=route.device).repeat_interleave(route.shape[0])
        L = E // world
        builders = [lambda: contiguous,
                    lambda: colocate(routes, source, contiguous, L, gpus_per_node),
                    lambda: partition_placement(routes, source, E, world, gpus_per_node),
                    lambda: even_placement(routes, E, world)]
        builders += [lambda r=r, p=p: greedy_placement(routes, source, E, world, gpus_per_node,
                                                       rbd=r, pilot=p)[0]
                     for r, p in RULES.values()]
        builders += [lambda r=r, p=p: greedy_placement(routes, source, E, world, gpus_per_node,
                                                       rbd=r, pilot=p, start=stacked[2])[0]
                     for r, p in RULES.values()]
        for i, build in enumerate(builders):
            torch.cuda.synchronize()
            start = time.perf_counter()
            stacked[i] = build()
            torch.cuda.synchronize()
            seconds[i] = time.perf_counter() - start
    dist.broadcast(stacked, 0)
    dist.broadcast(seconds, 0)
    return dict(zip(names, stacked)), dict(zip(names, seconds.tolist()))


def run_rule(
    rule: str,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert,
    route: torch.Tensor,
    slots: torch.Tensor,
    E: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run one rule with the routing relabeled by a placement.

    Collective.

    Args:
        rule: Key of RULES.
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights.
        expert: Expert function, given slot ids.
        route: [num_tokens, k] expert ids of this rank's tokens.
        slots: [E] slot of each expert.
        E: Total number of experts.
        gpus_per_node: Ranks per node.
        local_group: Process group of this rank's node.

    Returns:
        output: [num_tokens, H] combined output.
    """
    rbd, pilot = RULES[rule]
    placed = slots[route]
    if rbd:
        return RBD_combine_dispatch(tokens, weights, expert, placed, E, gpus_per_node,
                                    local_group, pilot=pilot)
    return plain_combine_dispatch(tokens, weights, expert, placed, E)


def phase_profile(
    fn: Callable[[], torch.Tensor],
    phases: list[str],
    iterations: int,
    trace_path: str | None = None,
) -> torch.Tensor:
    """Profile fn and measure each labeled phase on this rank.

    Collective. Each call is preceded by a barrier, as in time_ms. A GPU activity
    (kernel, memcpy or memset, on any stream, NCCL's included) belongs to the phase
    whose CPU range holds the CPU op that launched it (its linked correlation id), or,
    for launches with no CPU op (Triton's), whose compute-stream span it ran in.

    Args:
        fn: Function to profile, the same on every rank.
        phases: record_function labels to measure.
        iterations: Profiled calls to average over.
        trace_path: If given, where to write this rank's Chrome trace.

    Returns:
        ms: [len(phases), 4] per call: CPU span; GPU span, from the phase's first GPU
            activity start to its last one's end; time in NCCL kernels; time in its
            other GPU activities.
    """
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            dist.barrier()
            fn()
        torch.cuda.synchronize()
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    events = prof.profiler.kineto_results.events()
    index = {name: i for i, name in enumerate(phases)}
    ranges = []  # (start, end, phase) of every CPU range of a phase
    launch = {}  # correlation id of each CPU event -> its start, where GPU work links
    gpu_ranges = []  # (start, end, CPU range) of each phase's span on the compute stream
    for e in events:
        if e.device_type() != DeviceType.CPU:
            continue
        launch[e.correlation_id()] = e.start_ns()
        if e.name() in index:
            ranges.append((e.start_ns(), e.end_ns(), index[e.name()]))
    for e in events:
        if e.device_type() == DeviceType.CUDA and e.is_user_annotation() and e.name() in index:
            t = launch.get(e.correlation_id())
            owner = next((r for r in ranges if r[0] <= t <= r[1]), None) if t is not None else None
            if owner is not None:
                gpu_ranges.append((e.start_ns(), e.end_ns(), owner))
    ms = torch.zeros(len(phases), 4, dtype=torch.float64)
    first, last = {}, {}  # per CPU range: GPU start and end
    for e in events:
        if e.device_type() != DeviceType.CUDA or e.is_user_annotation():
            continue
        t = launch.get(e.linked_correlation_id())
        owner = next((r for r in ranges if r[0] <= t <= r[1]), None) if t is not None else None
        if owner is None:
            # Driver-API launches (Triton kernels) link to no CPU op; place them by the
            # compute-stream span of the phase they ran in.
            owner = next((r for a, b, r in gpu_ranges if a <= e.start_ns() < b), None)
        if owner is None:
            continue
        ms[owner[2], 2 if e.name().startswith("ncclDevKernel") else 3] += e.duration_ns()
        first[owner] = min(first.get(owner, e.start_ns()), e.start_ns())
        last[owner] = max(last.get(owner, e.end_ns()), e.end_ns())
    for r in ranges:
        ms[r[2], 0] += r[1] - r[0]
        if r in first:
            ms[r[2], 1] += last[r] - first[r]
    return ms / iterations / 1e6


def gather_rows(local: torch.Tensor) -> torch.Tensor:
    """Stack every rank's tensor on every rank.

    Collective.

    Args:
        local: This rank's tensor, the same shape on every rank.

    Returns:
        everyone: [world, *local.shape] on the CPU.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    everyone = torch.empty((dist.get_world_size(), *local.shape), dtype=local.dtype,
                           device=device)
    dist.all_gather_into_tensor(everyone, local.to(device).contiguous())
    return everyone.cpu()


def file_tag(text: str) -> str:
    """Text made safe for a file name.

    Returns:
        tag: text with spaces as _ and + as -.
    """
    return text.replace(" ", "_").replace("+", "-")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layers", type=int, nargs="+", help="Layers whose routing to replay")
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--scaling", nargs="+", choices=["strong", "weak"], default=["strong"],
                        help="weak gives every rank a whole recording instead of a share")
    parser.add_argument("--placements", help="Candidates from benchmark.all2all.build_placements")
    parser.add_argument("--profile", nargs="*", default=[],
                        help="Placements to profile for every rule, besides contiguous and best")
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--trace-dir", help="Write Chrome traces of each profile here")
    parser.add_argument("--trace-case", help='Case to trace, such as "strong 20"')
    parser.add_argument("--out", help="Write every timing and profile here as JSON")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    E, G = args.num_experts, args.gpus_per_node

    rank, world, device = init_dist()
    hosts = check_placement(G)
    local_group, _ = dist.new_subgroups(group_size=G)
    identity = lambda x, expert_ids: x
    prebuilt = torch.load(args.placements) if args.placements else None
    if prebuilt is not None and (prebuilt["E"], prebuilt["world"], prebuilt["gpus_per_node"]) != (E, world, G):
        raise RuntimeError(f"{args.placements} was built for E={prebuilt['E']}, "
                           f"world={prebuilt['world']}, gpus_per_node={prebuilt['gpus_per_node']}.")
    if args.trace_dir and rank == 0:
        os.makedirs(args.trace_dir, exist_ok=True)
    trace_case = args.trace_case or f"{args.scaling[0]} {args.layers[0]}"
    results = []

    for scale in args.scaling:
        for layer in args.layers:
            if scale == "weak":
                recorded = sum(f.startswith("experts_rank") and f.endswith(".pt")
                               for f in os.listdir(args.run_dir))
                route = load_routing(args.run_dir, layer, rank % recorded, recorded, E)
            else:
                route = load_routing(args.run_dir, layer, rank, world, E)
            route = route.to(device)
            tokens, weights = make_inputs(route, args.hidden, device, seed=rank)
            if prebuilt is None:
                placements, seconds = shared_placements(route, E, G)
                stats = None
            else:
                case = prebuilt["cases"][f"{scale} {layer}"]
                placements = {n: case["slots"][i].to(device) for i, n in enumerate(case["names"])}
                stats = case["stats"]
                seconds = {n: stats[n]["seconds"] for n in placements}
            names = list(placements)

            contiguous = placements["contiguous"]
            by_id = lambda inverse: (lambda x, ids: x * ((inverse[ids] + 1) / E)
                                     .to(x.dtype).unsqueeze(-1))
            reference = run_rule("plain", tokens, weights, by_id(contiguous), route, contiguous,
                                 E, G, local_group).float()

            ms, worst_err = {}, torch.zeros(1, device=device)
            for rule in RULES:
                for name, slots in placements.items():
                    inverse = torch.empty_like(slots)
                    inverse[slots] = contiguous
                    out = run_rule(rule, tokens, weights, by_id(inverse), route, slots, E, G,
                                   local_group).float()
                    err = (out - reference).abs().max() / reference.abs().max()
                    worst_err = torch.maximum(worst_err, err)
                    ms[rule, name] = time_ms(
                        lambda: run_rule(rule, tokens, weights, identity, route, slots, E, G,
                                         local_group),
                        args.warmup, args.iterations)
            dist.all_reduce(worst_err, op=dist.ReduceOp.MAX)

            # ms is the same on every rank (time_ms reduces it), so so is best.
            profiles = {}
            for rule, (rbd, _) in RULES.items():
                best = min(names, key=lambda n: ms[rule, n])
                for name in dict.fromkeys(["contiguous", best]
                                          + [n for n in args.profile if n in placements]):
                    trace = None
                    if args.trace_dir and rank % G == 0 and f"{scale} {layer}" == trace_case:
                        trace = (f"{args.trace_dir}/{scale}_layer{layer}_{file_tag(rule)}_"
                                 f"{file_tag(name)}_rank{rank}.json")
                    slots = placements[name]
                    local = phase_profile(
                        lambda: run_rule(rule, tokens, weights, identity, route, slots, E, G,
                                         local_group),
                        PHASES[rbd], args.profile_iterations, trace)
                    profiles[rule, name] = gather_rows(local)  # [world, phases, 4]

            if rank != 0:
                continue
            print(f"\n=== layer {layer}, {route.shape[0]} tokens/rank, {scale} scaling, oracle | "
                  f"{world // G} nodes x {G} GPUs: {', '.join(hosts[::G])} | "
                  f"max rel err {worst_err.item():.1e}", flush=True)
            if stats is None:
                print("  search seconds: " + ", ".join(f"{n} {seconds[n]:.1f}" for n in names),
                      flush=True)
            else:
                print("  placements (built offline): seconds, swaps, experts moved, pilots, "
                      "expert rows per rank and per node as max/mean", flush=True)
                for n in names:
                    st = stats[n]
                    swaps = st.get("greedy_swaps", st.get("best_run", {}).get("swaps", "-"))
                    print(f"    {n:>18}: {st['seconds']:6.1f} s, swaps {swaps!s:>4}, moved "
                          f"{st['experts_moved']:3d}, pilots {st['pilots']:6d}, rank "
                          f"{st['rank_rows']['max']:.3f}, node {st['node_rows']['max']:.3f}",
                          flush=True)
            base = ms["plain", "contiguous"]
            for rule in RULES:
                best = min(names, key=lambda n: ms[rule, n])
                print(f"  {rule:>10}: " + ", ".join(f"{n} {ms[rule, n]:.3f}" for n in names)
                      + f" | best {best} {ms[rule, best]:.3f} ms, "
                      f"{base / ms[rule, best]:.2f}x vs plain contiguous", flush=True)
            print("  phase profiles, ms per call: CPU span, GPU span, NCCL kernels, other GPU "
                  "work, as mean/max over ranks (slowest rank)", flush=True)
            for (rule, name), everyone in profiles.items():
                print(f"    {rule} | {name} | end to end {ms[rule, name]:.3f} ms", flush=True)
                for p, phase in enumerate(PHASES[RULES[rule][0]]):
                    cells = []
                    for c in range(4):
                        col = everyone[:, p, c]
                        cells.append(f"{col.mean():.3f}/{col.max():.3f} (r{col.argmax().item()})")
                    print(f"      {phase.split('/')[-1]:>16}: cpu {cells[0]}, gpu {cells[1]}, "
                          f"nccl {cells[2]}, other {cells[3]}", flush=True)

            results.append({
                "scaling": scale, "layer": layer, "tokens_per_rank": route.shape[0],
                "nodes": world // G, "hosts": hosts[::G], "max_rel_err": worst_err.item(),
                "ms": {f"{r} | {n}": v for (r, n), v in ms.items()},
                "search_seconds": seconds,
                "profiles": {f"{r} | {n}": {"phases": PHASES[RULES[r][0]],
                                            "per_rank_ms": v.tolist()}
                             for (r, n), v in profiles.items()},
            })
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(results, f)

    dist.barrier()
    dist.destroy_process_group()
