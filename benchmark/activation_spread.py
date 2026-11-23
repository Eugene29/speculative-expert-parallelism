"""Compare how widely tokens spread over the experts at different batch sizes.

Reads recorded routing, RUN_DIR/experts_rank{r}.pt, each a [layers, tokens, k] tensor
of one sequence. A setting BxS gives every rank B sequences of S tokens: the first S
tokens of B recordings. A causal model routes the first S tokens of a sequence the
same whether or not more follow, so a prefix is the routing a short batch would get.
Experts sit on ranks and nodes in contiguous blocks, as AutoEP places them.

Per setting, as means over layers, the script prints

    active: experts with at least one copy, per rank's tokens and per node's tokens.
    effective: exp(entropy) of a rank's copies over the experts. E means even use.
    top-N: share of a rank's copies on its N busiest experts.
    pilots: inter-node rows RBD sends per token, with no guests.
    For every guest budget (--slots per rank, so slots * G per node), with guests
    chosen from the batch itself (oracle) and from another batch of the same size
    (static):
        covered: share of the copies for other nodes that a guest serves.
        pilots: pilots left, as a share of those with no guests, when the guests are
            the experts in most demand (spec_ep.guests) and when they are chosen
            greedily by the pilots they remove (spec_ep.replicas, rule "pilots").

Usage:
    python -m benchmark.activation_spread logs/2026-11-21_exp/routing \
        --settings 1x4096 1x512 2x512 1x128 --slots 8 24
    python -m benchmark.activation_spread RUN_DIR --ranks 8 --gpus-per-node 4 --layers 0 20 39
"""
import argparse

import torch

from spec_ep.replicas import choose_replicas, inter_node_rows, serving_node


def load_batch(
    run_dir: str,
    first: int,
    ranks: int,
    sequences: int,
    length: int,
    layers: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build one batch: every rank gets the first `length` tokens of `sequences` recordings.

    Returns:
        route: [layers, ranks, sequences * length, k] expert ids, as int64.
    """
    recordings = [
        torch.load(f"{run_dir}/experts_rank{first + i}.pt", map_location="cpu")[layers, :length]
        for i in range(ranks * sequences)
    ]
    route = torch.stack(recordings, dim=1).long().to(device)
    return route.view(len(layers), ranks, sequences * length, -1)


def left_for_network(
    route: torch.Tensor,
    source_node: torch.Tensor,
    held: torch.Tensor,
) -> tuple[float, float]:
    """Count what a layout of guests leaves for the network, under source routing.

    Returns:
        copies: Copies still served on another node.
        pilots: (token, other node) pairs still sent.
    """
    dest = serving_node(route, source_node, held)
    pilots, _, copies = inter_node_rows(dest, source_node, held.shape[0])
    return float(copies), float(pilots.sum())


def spread(
    batch: torch.Tensor,
    other: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    slots: list[int],
) -> dict[str, float]:
    """Score one setting on a batch, with static guests taken from another batch.

    Args:
        batch: [layers, ranks, tokens, k] routing to score.
        other: Routing of another batch of the same shape, for the static guests.
        slots: Guest budgets per rank.

    Returns:
        scores: The numbers of the module docstring, as means over layers.
    """
    num_layers, ranks, tokens, k = batch.shape
    num_nodes = ranks // gpus_per_node
    device = batch.device
    source_node = torch.arange(ranks, device=device).repeat_interleave(tokens) // gpus_per_node
    none = choose_replicas(batch[0].flatten(0, 1), source_node, num_experts, num_nodes, 0)
    scores = {}

    def add(name: str, value: float) -> None:
        scores[name] = scores.get(name, 0.0) + value / num_layers

    for layer in range(num_layers):
        route = batch[layer]
        counts = torch.zeros(ranks, num_experts, device=device)
        counts.scatter_add_(1, route.flatten(1), torch.ones(ranks, tokens * k, device=device))
        share = counts / counts.sum(dim=1, keepdim=True)
        node_counts = counts.view(num_nodes, gpus_per_node, -1).sum(dim=1)
        entropy = -(share * share.clamp_min(1e-12).log()).sum(dim=1)
        ranked = share.sort(dim=1, descending=True).values
        add("active per rank", float((counts > 0).sum(dim=1).float().mean()))
        add("active per node", float((node_counts > 0).sum(dim=1).float().mean()))
        add("effective per rank", float(entropy.exp().mean()))
        for n in (8, 32, 64):
            add(f"top-{n} share", float(ranked[:, :n].sum(dim=1).mean()))

        flat = route.flatten(0, 1)
        copies_none, pilots_none = left_for_network(flat, source_node, none)
        add("pilots per token", pilots_none / (ranks * tokens))
        add("remote copies per token", copies_none / (ranks * tokens))
        for s in slots:
            for source, sample in (("oracle", flat), ("static", other[layer].flatten(0, 1))):
                for rule, name in (("copies", "demand"), ("pilots", "greedy")):
                    held = choose_replicas(
                        sample, source_node, num_experts, num_nodes, s * gpus_per_node, rule
                    )
                    copies, pilots = left_for_network(flat, source_node, held)
                    add(f"slots {s} {source} {name} covered", 1 - copies / copies_none)
                    add(f"slots {s} {source} {name} pilots", pilots / pilots_none)
    return scores


def main() -> None:
    """Print one Markdown table: a row per number, a column per setting."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory with experts_rank{r}.pt recordings")
    parser.add_argument("--settings", nargs="+", default=["1x4096", "1x512", "2x512", "1x128"],
                        metavar="BxS", help="Sequences per rank x tokens per sequence")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--slots", type=int, nargs="+", default=[8, 24], help="Guests per rank")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(0, 40, 4)))
    parser.add_argument("--static-from", help="Directory of the other batch, for the static "
                        "guests. Default: the recordings of run_dir after the ones scored")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    columns = {}
    for setting in args.settings:
        sequences, length = map(int, setting.split("x"))
        shape = (args.ranks, sequences, length, args.layers, device)
        batch = load_batch(args.run_dir, 0, *shape)
        other = load_batch(args.static_from or args.run_dir,
                           0 if args.static_from else args.ranks * sequences, *shape)
        columns[setting] = spread(batch, other, args.num_experts, args.gpus_per_node, args.slots)

    print(f"| mean over layers {args.layers} | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for name in next(iter(columns.values())):
        digits = 1 if name.startswith(("active", "effective")) else 3
        print(f"| {name} | " + " | ".join(f"{c[name]:.{digits}f}" for c in columns.values()) + " |")


if __name__ == "__main__":
    main()
