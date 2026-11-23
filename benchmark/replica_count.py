"""Count the rows that cross between nodes under every way of choosing replicas.

Reads recorded routing, RUN_DIR/experts_rank{r}.pt, lays it over --nodes nodes as the
benchmarks do, and for every replica rule of spec_ep.replicas, budget and routing
counts what one dispatch sends between nodes. No GPU nodes are needed, so this is
where a rule is judged before it is timed.

Each row is a mean over the layers, relative to the same routing with no replicas:

    pilots: (token, other node) pairs, the rows RBD sends between nodes.
    copies: copies served on another node, the rows a plain all2all sends.
    busiest: pilots sent plus received by the node that handles the most.
    in max/mean: pilots received by the busiest node over the mean node, the balance.

Choice says where the replicas were chosen from: "oracle" is the routing that is
counted, and "static" is another recording, --choose-from, chosen ahead.

Usage:
    python -m benchmark.replica_count logs/2026_oct/2026-10-16_14-07-50 --nodes 4 \
        --choose-from logs/2026-11-21_exp/routing --budgets 16 32 64 --layers 0 4 8
"""
import argparse

import torch

from benchmark.utils import load_sample
from spec_ep.replicas import RULES, choose_replicas, inter_node_rows, serving_node


def count(
    route: torch.Tensor,
    source_node: torch.Tensor,
    held: torch.Tensor,
    cover: bool,
) -> torch.Tensor:
    """Count one layout's inter-node rows under one routing.

    Returns:
        counts: [4] pilots, copies, pilots through the busiest node, and the busiest
            node's received pilots over the mean node's.
    """
    dest = serving_node(route, source_node, held, cover)
    pilots_out, pilots_in, copies = inter_node_rows(dest, source_node, held.shape[0])
    return torch.tensor([
        pilots_out.sum(), copies, (pilots_out + pilots_in).max(), pilots_in.max() / pilots_in.float().mean(),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--choose-from", metavar="DIR", help="Recording for the static choice")
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(0, 40, 4)))
    parser.add_argument("--rules", nargs="+", choices=RULES, default=RULES)
    args = parser.parse_args()
    N, E = args.nodes, args.num_experts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    totals = {}
    for layer in args.layers:
        route, source_node = (x.to(device) for x in load_sample(args.run_dir, layer, N))
        samples = {"oracle": (route, source_node)}
        if args.choose_from:
            samples["static"] = tuple(x.to(device) for x in load_sample(args.choose_from, layer, N))
        base = count(route, source_node, choose_replicas(route, source_node, E, N, 0), cover=False)
        base[3] = 1
        for budget in args.budgets:
            for choice, sample in samples.items():
                for rule in args.rules:
                    held = choose_replicas(*sample, E, N, budget, rule)
                    for cover in (False, True) if N > 2 else (False,):
                        key = (budget, choice, rule, "cover" if cover else "source")
                        totals[key] = totals.get(key, 0) + count(route, source_node, held, cover) / base

    print(f"{N} nodes, {E} experts, mean of {len(args.layers)} layers, relative to no replicas")
    print(f"{'budget':>6} {'choice':>6} {'rule':>6} {'routing':>7} | {'pilots':>6} {'copies':>6} "
          f"{'busiest':>7} {'in max/mean':>11}")
    for (budget, choice, rule, routing), total in totals.items():
        pilots, copies, busiest, balance = (total / len(args.layers)).tolist()
        print(f"{budget:>6} {choice:>6} {rule:>6} {routing:>7} | {pilots:6.3f} {copies:6.3f} "
              f"{busiest:7.3f} {balance:11.2f}")
