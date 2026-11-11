"""Build candidate expert placements offline, for benchmark.all2all.placement --placements.

Runs on one GPU (the login node's is enough), so no node time goes to searching.
For each scaling (strong, weak) and recorded layer, it rebuilds the routing every
rank of a WORLD-rank run would replay (as benchmark.all2all.placement does), then
builds every candidate placement, timing each on the wall clock:
    contiguous:      expert e in slot e.
    even:            spec_ep.placement.even_placement.
    partition:       spec_ep.placement.partition_placement (Kernighan-Lin).
    mtk:             spec_ep.placement.hypergraph_placement (Mt-KaHyPar, load
                     weights, 5% node load cap), the fewest pilots of several runs
                     over presets and seeds.
    mtk+kl:          mtk refined with spec_ep.placement.kl_refine.
    kahypar:         spec_ep.placement.kahypar_placement without load balance (the
                     exp/fixed-vertex-partition approach), fewest pilots of 3 seeds.
    kahypar+bal:     the same with node loads capped at 5% over the mean.
    greedy <rule>:   spec_ep.placement.greedy_placement for one rule's cost model.
    greedy+mtk <rule>: the same search, also started from mtk+kl.

Per placement it records the build seconds, swaps (accepted greedy swaps, or the
moves and swaps the Mt-KaHyPar or KaHyPar repair made), experts that changed rank from
contiguous, the total pilot (inter-node) rows, and each rank's and node's routed
expert rows (token imbalance) as max / mean and min / mean. Per (placement, rule) it
records every rank's network, stage 1 NVLink and stage 2 rows sent and received
(spec_ep.placement.rank_loads) and the fitted model's estimate.

Writes OUT (a .pt with slots and stats) and prints a summary.

KaHyPar needs its module and config:
    KAHYPAR_INI=/pscratch/sd/e/eku/pylibs/kahypar_config/km1_kKaHyPar_sea20.ini
    PYTHONPATH also holding /pscratch/sd/e/eku/pylibs/kahypar_only.

Usage:
    PYTHONPATH=/pscratch/sd/e/eku/pylibs:/pscratch/sd/e/eku/pylibs/kahypar_only \
    KAHYPAR_INI=/pscratch/sd/e/eku/pylibs/kahypar_config/km1_kKaHyPar_sea20.ini \
    python -u -m benchmark.all2all.build_placements \
        logs/2026_oct/2026-10-16_14-07-50 0 20 39 --world 16 \
        --out /pscratch/sd/e/eku/placement/4n/placements.pt
"""
import argparse
import os
import time
from collections.abc import Callable

import torch

from benchmark.all2all.rbd_vs_plain import load_routing
from spec_ep.placement import (even_placement, greedy_placement, hypergraph_placement,
                               kahypar_placement, kl_refine, partition_placement,
                               placement_cost, rank_loads)

# rule: (rbd, pilot), as in benchmark.all2all.placement.
RULES = {
    "plain": (False, "rail"),
    "rbd first": (True, "first"),
    "rbd rail": (True, "rail"),
    "rbd most": (True, "most"),
    "rbd even": (True, "even"),
}


def replayed_routing(
    run_dir: str,
    layer: int,
    world: int,
    E: int,
    weak: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing of every rank in a world-rank run, as benchmark.all2all.placement loads it.

    Args:
        run_dir: Directory holding experts_rank{r}.pt.
        layer: Recorded layer.
        world: Ranks in the run.
        E: Total number of experts.
        weak: Every rank replays a whole recording if True, else a share.
        device: Device for the result.

    Returns:
        route: [num_tokens, k] experts of every token, rank by rank.
        source_rank: [num_tokens] rank each token starts on.
    """
    recorded = sum(f.startswith("experts_rank") and f.endswith(".pt")
                   for f in os.listdir(run_dir))
    if weak:
        routes = [load_routing(run_dir, layer, r % recorded, recorded, E) for r in range(world)]
    else:
        routes = [load_routing(run_dir, layer, r, world, E) for r in range(world)]
    source = torch.cat([torch.full((x.shape[0],), r) for r, x in enumerate(routes)])
    return torch.cat(routes).to(device), source.to(device)


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    """Run fn and time it on the wall clock, with the GPU synchronized.

    Returns:
        result: What fn returned.
        seconds: Wall time of fn.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def ratio(x: torch.Tensor) -> dict[str, float]:
    """Max and min over the mean of x.

    Returns:
        stats: {"max": max / mean, "min": min / mean, "mean": mean}.
    """
    x = x.float()
    mean = x.mean().item()
    return {"max": x.max().item() / mean, "min": x.min().item() / mean, "mean": mean}


def build_case(
    route: torch.Tensor,
    source: torch.Tensor,
    E: int,
    world: int,
    G: int,
    mtk_runs: list[tuple[str, int]],
) -> tuple[list[str], torch.Tensor, dict[str, dict]]:
    """Build and describe every candidate placement for one routing.

    Args:
        route: [num_tokens, k] experts of every token.
        source: [num_tokens] rank each token starts on.
        E: Total number of experts.
        world: Ranks in the run.
        G: Ranks per node.
        mtk_runs: (preset, seed) of each Mt-KaHyPar run.

    Returns:
        names: Placement names.
        slots: [len(names), E] slot of each expert per placement.
        stats: Name -> its statistics, with per-rule entries under "rules".
    """
    L = E // world
    contiguous = torch.arange(E, device=route.device)
    pilots = lambda s: rank_loads(route, source, s[None], L, G).net_out.sum().item()
    built: dict[str, torch.Tensor] = {"contiguous": contiguous}
    info: dict[str, dict] = {"contiguous": {"seconds": 0.0}}

    built["even"], seconds = timed(lambda: even_placement(route, E, world))
    info["even"] = {"seconds": seconds}
    built["partition"], seconds = timed(lambda: partition_placement(route, source, E, world, G))
    info["partition"] = {"seconds": seconds}

    runs = []
    for preset, seed in mtk_runs:
        (slots, moves, swaps), seconds = timed(
            lambda: hypergraph_placement(route, source, E, world, G, "load", preset=preset,
                                         seed=seed))
        runs.append({"preset": preset, "seed": seed, "seconds": seconds, "moves": moves,
                     "swaps": swaps, "pilots": pilots(slots), "slots": slots})
    best = min(runs, key=lambda r: r["pilots"])
    built["mtk"] = best["slots"]
    info["mtk"] = {"seconds": sum(r["seconds"] for r in runs),
                   "best_run": {k: v for k, v in best.items() if k != "slots"},
                   "runs": [{k: v for k, v in r.items() if k != "slots"} for r in runs]}
    built["mtk+kl"], seconds = timed(lambda: kl_refine(route, source, best["slots"], L, G))
    info["mtk+kl"] = {"seconds": info["mtk"]["seconds"] + seconds, "kl_seconds": seconds}
    for name, balance in (("kahypar", False), ("kahypar+bal", True)):
        built[name], info[name] = build_kahypar(route, source, E, world, G, balance)

    for rule, (rbd, pilot) in RULES.items():
        tag = rule.split()[-1]
        for name, start in ((f"greedy {tag}", None), (f"greedy+mtk {tag}", built["mtk+kl"])):
            (slots, history), seconds = timed(
                lambda: greedy_placement(route, source, E, world, G, rbd=rbd, pilot=pilot,
                                         start=start))
            built[name] = slots
            info[name] = {"seconds": seconds, "greedy_swaps": len(history) - 1,
                          "model_start_ms": history[0], "model_end_ms": min(history)}

    describe(route, source, built, info, E, world, G)
    names = list(built)
    return names, torch.stack([built[n] for n in names]), info


def build_kahypar(
    route: torch.Tensor,
    source: torch.Tensor,
    E: int,
    world: int,
    G: int,
    balance: bool,
    seeds: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Run kahypar_placement over several seeds and keep the fewest pilots.

    Args:
        route: [num_tokens, k] experts of every token.
        source: [num_tokens] rank each token starts on.
        E: Total number of experts.
        world: Ranks in the run.
        G: Ranks per node.
        balance: Passed to kahypar_placement.
        seeds: Seeds to try.

    Returns:
        slots: [E] the kept placement.
        info: Total seconds, the kept run, and every run's seconds, repairs and pilots.
    """
    L = E // world
    runs = []
    for seed in range(seeds):
        (slots, moves, swaps, refine_swaps), seconds = timed(
            lambda: kahypar_placement(route, source, E, world, G, balance=balance, seed=seed))
        pilots = rank_loads(route, source, slots[None], L, G).net_out.sum().item()
        runs.append({"seed": seed, "seconds": seconds, "moves": moves, "swaps": swaps,
                     "refine_swaps": refine_swaps, "pilots": pilots, "slots": slots})
    best = min(runs, key=lambda r: r["pilots"])
    strip = lambda r: {k: v for k, v in r.items() if k != "slots"}
    return best["slots"], {"seconds": sum(r["seconds"] for r in runs), "best_run": strip(best),
                           "runs": [strip(r) for r in runs]}


def describe(
    route: torch.Tensor,
    source: torch.Tensor,
    built: dict[str, torch.Tensor],
    info: dict[str, dict],
    E: int,
    world: int,
    G: int,
) -> None:
    """Add pilots, experts moved, token imbalance and per-rule row counts to info.

    Args:
        route: [num_tokens, k] experts of every token.
        source: [num_tokens] rank each token starts on.
        built: Name -> [E] slot of each expert.
        info: Name -> statistics, updated in place.
        E: Total number of experts.
        world: Ranks in the run.
        G: Ranks per node.
    """
    L = E // world
    contiguous = torch.arange(E, device=route.device)
    pilots = lambda s: rank_loads(route, source, s[None], L, G).net_out.sum().item()
    for name, slots in built.items():
        rows = torch.bincount(slots[route].flatten() // L, minlength=world)
        info[name].update(
            pilots=pilots(slots),
            experts_moved=(slots // L != contiguous // L).sum().item(),
            rank_rows=ratio(rows), node_rows=ratio(rows.view(-1, G).sum(1)),
            rank_rows_values=rows.tolist(), rules={})
        for rule, (rbd, pilot) in RULES.items():
            ld = rank_loads(route, source, slots[None], L, G, rbd, pilot)
            info[name]["rules"][rule] = {
                field: getattr(ld, field)[0].tolist() for field in ld._fields}
            info[name]["rules"][rule]["model_ms"] = placement_cost(
                route, source, slots[None], L, G, rbd, pilot).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layers", type=int, nargs="+", help="Layers whose routing to replay")
    parser.add_argument("--world", type=int, default=16, help="Ranks of the timed run")
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--mtk-seeds", type=int, default=3, help="Mt-KaHyPar seeds per preset")
    parser.add_argument("--out", required=True, help="Output .pt file")
    args = parser.parse_args()
    E, G, W = args.num_experts, args.gpus_per_node, args.world
    device = torch.device("cuda")
    mtk_runs = [(p, s) for p in ("QUALITY", "HIGHEST_QUALITY") for s in range(args.mtk_seeds)]

    # Mt-KaHyPar's first call in a process starts its thread pool; keep that out of timings.
    warm_route, warm_source = replayed_routing(args.run_dir, args.layers[0], W, E, False, device)
    hypergraph_placement(warm_route, warm_source, E, W, G)

    cases = {}
    for scaling in ("strong", "weak"):
        for layer in args.layers:
            route, source = replayed_routing(args.run_dir, layer, W, E, scaling == "weak", device)
            names, slots, info = build_case(route, source, E, W, G, mtk_runs)
            cases[f"{scaling} {layer}"] = {"names": names, "slots": slots.cpu(), "stats": info,
                                           "tokens_per_rank": route.shape[0] // W}
            torch.save({"E": E, "world": W, "gpus_per_node": G, "cases": cases}, args.out)
            print(f"\n=== {scaling} layer {layer}, {route.shape[0] // W} tokens/rank", flush=True)
            for n in names:
                s = info[n]
                swaps = s.get("greedy_swaps", s.get("best_run", {}).get("swaps", "-"))
                print(f"  {n:>18}: {s['seconds']:6.1f} s, swaps {swaps!s:>4}, moved "
                      f"{s['experts_moved']:3d}, pilots {s['pilots']:6d}, rank rows "
                      f"max/mean {s['rank_rows']['max']:.3f}, node {s['node_rows']['max']:.3f}, "
                      f"rail model {s['rules']['rbd rail']['model_ms']:.2f} ms", flush=True)
