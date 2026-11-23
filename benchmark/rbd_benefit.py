"""Predict what RBD gains over a plain all2all from the routing and where the experts sit.

The metric is the saved rows: the rows a plain all2all sends between nodes minus the
rows RBD sends between nodes, per layer and direction, summed over the ranks.

    copies: (token, expert) pairs whose expert is on another node. Plain sends each.
    pilots: (token, other node) pairs. RBD sends one row for each.
    saved = copies - pilots

Both come from the routing and the placement alone: experts sit on ranks and nodes in
contiguous blocks, and --guests DIR counts the pilots that guests chosen from DIR's
routing leave (spec_ep.replicas, rule "copies"), so the same number scores a placement.

A model step of scripts/ds_moe.py --record leaves its routing in the directory named
after its output file. Given several such steps, the script fits

    gain per forward = a * saved - b        gain = seconds of --base - seconds of --mode

over them. a is what a saved row is worth and b what RBD costs whatever the batch, its
plan. saved = b / a is where RBD starts to pay. CPU only.

Usage:
    python -m benchmark.rbd_benefit logs/2026-11-23_merge/2n_b/{few_1x128,few_1x512,few_2x512,profile_ours}.out
    python -m benchmark.rbd_benefit STEP.out ... --base plain --mode two_lane --gpus-per-node 4
    python -m benchmark.rbd_benefit STEP.out --guests logs/2026-11-21_exp/routing --slots 24
"""
import argparse
import glob
import re

import torch

from spec_ep.replicas import choose_replicas, inter_node_rows, serving_node

TIMING = re.compile(r"^\s*(\S+): median\s+([\d.]+) s")


def saved_rows(
    route: torch.Tensor,
    gpus_per_node: int,
    num_experts: int,
    held: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Count one layer's inter-node rows per direction under plain and under RBD.

    Args:
        route: [ranks, tokens, k] experts each rank's tokens are routed to.
        gpus_per_node: Ranks per node.
        num_experts: Total number of experts, split evenly over the nodes.
        held: [num_nodes, E] bool layout of spec_ep.replicas, for a placement with
            guests. None means every expert only at its home.

    Returns:
        copies: Rows a plain all2all sends between nodes.
        pilots: Rows RBD sends between nodes.
    """
    ranks, tokens, _ = route.shape
    num_nodes = ranks // gpus_per_node
    source_node = torch.arange(ranks).repeat_interleave(tokens) // gpus_per_node
    flat = route.flatten(0, 1).long()
    home = choose_replicas(flat, source_node, num_experts, num_nodes, 0)
    _, _, copies = inter_node_rows(serving_node(flat, source_node, home), source_node, num_nodes)
    dest = serving_node(flat, source_node, home if held is None else held)
    pilots, _, _ = inter_node_rows(dest, source_node, num_nodes)
    return float(copies), float(pilots.sum())


def load_routing(run_dir: str) -> torch.Tensor:
    """Load every rank's recording of one run.

    Returns:
        route: [layers, ranks, tokens, k] expert ids.
    """
    files = sorted(glob.glob(f"{run_dir}/experts_rank*.pt"), key=lambda f: int(re.findall(r"\d+", f)[-1]))
    return torch.stack([torch.load(f, map_location="cpu") for f in files], dim=1)


def main() -> None:
    """Print each step's saved rows and gain, and the fit of one to the other."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="+", help="Output files of ds_moe.py --record steps")
    parser.add_argument("--base", default="plain")
    parser.add_argument("--mode", default="two_lane")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--guests", help="Routing directory to choose guests from")
    parser.add_argument("--slots", type=int, default=24, help="Guests per rank with --guests")
    args = parser.parse_args()

    rows = []
    for step in args.steps:
        route = load_routing(step.removesuffix(".out"))
        layers, ranks, tokens, _ = route.shape
        sample = load_routing(args.guests)[:, :ranks, :tokens] if args.guests else None
        counts = []
        for layer in range(layers):
            held = None
            if sample is not None:
                source_node = torch.arange(ranks).repeat_interleave(tokens) // args.gpus_per_node
                held = choose_replicas(
                    sample[layer].flatten(0, 1).long(), source_node, args.num_experts,
                    ranks // args.gpus_per_node, args.slots * args.gpus_per_node, "copies",
                )
            counts.append(saved_rows(route[layer], args.gpus_per_node, args.num_experts, held))
        copies, pilots = (sum(c) / layers for c in zip(*counts))
        seconds = dict(m.groups() for line in open(step) if (m := TIMING.match(line)))
        gain = 1e3 * (float(seconds[args.base]) - float(seconds[args.mode]))
        rows.append((step, tokens, layers, copies, pilots, gain))

    saved = torch.tensor([r[3] - r[4] for r in rows], dtype=torch.float64)
    gain = torch.tensor([r[5] for r in rows], dtype=torch.float64)
    a, b = 0.0, 0.0
    if len(rows) > 1:
        design = torch.stack([saved, -torch.ones_like(saved)], dim=1)
        a, b = torch.linalg.lstsq(design, gain.unsqueeze(1)).solution.flatten().tolist()
    print(f"| step | tokens per rank | copies | pilots | saved rows | {args.base} - {args.mode}, ms "
          f"| fit, ms |")
    print("|---|---|---|---|---|---|---|")
    for (step, tokens, _, copies, pilots, g), s in zip(rows, saved.tolist()):
        name = step.rsplit("/", 1)[-1].removesuffix(".out")
        print(f"| {name} | {tokens} | {copies:.0f} | {pilots:.0f} | {s:.0f} | {g:+.0f} | {a * s - b:+.0f} |")
    if len(rows) > 1:
        layers = rows[0][2]
        print(f"\ngain per forward = {1e3 * a:.2f} ms per 1000 saved rows - {b:.0f} ms, which is "
              f"{1e3 * a / layers * 1e3:.1f} us per 1000 saved rows and {b / layers:.2f} ms per layer. "
              f"RBD pays above {b / a:.0f} saved rows per layer.")


if __name__ == "__main__":
    main()
