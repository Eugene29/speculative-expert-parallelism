"""Measure how stable the hot experts are from one batch of tokens to the next.

Reads recorded routing, RUN_DIR/experts_rank{r}.pt, and treats every recording as one
batch, in file order. The hot set of a batch is its N busiest experts in a layer.
Loading the hot experts next to the tokens only pays for the copies routed to them,
so the script scores a way of choosing the hot set by its coverage: the share of a
batch's copies that go to the chosen experts. Three ways are compared, each scored on
batches it did not see:

    static: the N busiest experts over all the other batches. Chosen once, ahead.
    previous: the N busiest experts of the batch before. A dynamic loader that
        reloads after every batch and uses no prediction.
    oracle: the N busiest experts of the batch itself. The most any dynamic loader
        could cover, and only reachable by predicting the routing.

It also prints the churn of the previous-batch loader, the experts per layer it has
to load for each new batch, and how far the hot set moves: the Jaccard overlap of
consecutive hot sets. All numbers are means over layers and batches.

--group G pools G consecutive recordings into one batch, for example 4 for the
tokens one node holds in the benchmarks.

Usage:
    python -m benchmark.hot_experts logs/2026_oct/2026-10-16_14-07-50
    python -m benchmark.hot_experts RUN_DIR --hot 8 16 32 64 --group 4 \
        --first 0 --count 24 --png hot.png
"""
import argparse

import torch

from benchmark.utils import expert_counts


def coverage(share: torch.Tensor, hot: torch.Tensor) -> torch.Tensor:
    """Share of each batch's copies that go to its hot set.

    Args:
        share: [batches, layers, experts] share of a batch's copies per expert.
        hot: [batches, layers, n] chosen experts.

    Returns:
        covered: [batches, layers] summed share of the chosen experts.
    """
    return share.gather(-1, hot).sum(-1)


def score(counts: torch.Tensor, n: int) -> dict[str, float]:
    """Score the three ways of choosing n hot experts on every batch but the first.

    Args:
        counts: [batches, layers, experts] copies per expert.

    Returns:
        scores: Mean coverage of static, previous and oracle, the mean Jaccard overlap
            of consecutive hot sets, and churn, the mean experts per layer that enter
            the hot set from one batch to the next.
    """
    share = counts / counts.sum(-1, keepdim=True)
    oracle = counts.topk(n).indices
    static = (counts.sum(0, keepdim=True) - counts).topk(n).indices
    is_hot = torch.zeros_like(counts, dtype=torch.bool).scatter_(-1, oracle, True)
    kept = (is_hot[1:] & is_hot[:-1]).sum(-1).float()
    return {
        "static": coverage(share, static)[1:].mean().item(),
        "previous": coverage(share[1:], oracle[:-1]).mean().item(),
        "oracle": coverage(share, oracle)[1:].mean().item(),
        "jaccard": (kept / (2 * n - kept)).mean().item(),
        "churn": (n - kept).mean().item(),
    }


def draw(scores: dict[int, dict[str, float]], num_experts: int, title: str, path: str) -> None:
    """Draw coverage against the size of the hot set, one line per way of choosing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    sizes = list(scores)
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=surface)
    ax.set_facecolor(surface)
    lines = (("oracle", "#2a78d6"), ("previous", "#eb6834"), ("static", "#1baf7a"))
    for name, color in lines:
        values = [scores[n][name] for n in sizes]
        ax.plot(sizes, values, color=color, linewidth=2, marker="o", markersize=5)
        ax.annotate(f"{name} {values[-1]:.0%}", (sizes[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=ink)
    even = [n / num_experts for n in sizes]
    ax.plot(sizes, even, color=muted, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.annotate(f"even load {even[-1]:.0%}", (sizes[-1], even[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8, color=ink)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_xlabel(f"hot experts loaded per layer, of {num_experts}", fontsize=8, color=muted)
    ax.set_ylabel("copies that go to a loaded expert", fontsize=8, color=muted)
    ax.set_title(title, loc="left", fontsize=9, color=ink)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.tick_params(length=0, labelsize=8, labelcolor=muted)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(path, dpi=170, facecolor=surface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hot", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="Sizes of the hot set")
    parser.add_argument("--group", type=int, default=1, help="Recordings pooled into one batch")
    parser.add_argument("--first", type=int, default=0, help="First recording to use")
    parser.add_argument("--count", type=int, help="Recordings to use; default all from --first")
    parser.add_argument("--per-layer", type=int, metavar="N",
                        help="Also print every layer's coverage for this hot set size")
    parser.add_argument("--png", help="Draw coverage against hot set size to this file")
    args = parser.parse_args()

    counts = expert_counts(args.run_dir, args.num_experts)[args.first:]
    counts = counts[: args.count] if args.count else counts
    batches = counts.shape[0] // args.group
    counts = counts[: batches * args.group].unflatten(0, (batches, args.group)).sum(1)
    tokens = counts[0, 0].sum().item()
    print(f"{args.run_dir}: {batches} batches of {tokens} copies, {counts.shape[1]} layers, "
          f"E={args.num_experts}")

    print(f"{'hot':>4} {'even':>6} {'static':>7} {'previous':>9} {'oracle':>7} "
          f"{'jaccard':>8} {'churn':>6}")
    scores = {n: score(counts, n) for n in args.hot}
    for n, s in scores.items():
        print(f"{n:>4} {n / args.num_experts:>6.1%} {s['static']:>7.1%} {s['previous']:>9.1%} "
              f"{s['oracle']:>7.1%} {s['jaccard']:>8.2f} {s['churn']:>6.1f}")

    if args.per_layer:
        print(f"\nper layer, {args.per_layer} hot experts: static | previous | oracle")
        for layer in range(counts.shape[1]):
            s = score(counts[:, layer : layer + 1], args.per_layer)
            print(f"  layer {layer:>2}: {s['static']:.1%} | {s['previous']:.1%} | {s['oracle']:.1%}")

    if args.png:
        title = f"Hot-expert coverage, {batches} batches of {tokens} copies"
        draw(scores, args.num_experts, title, args.png)
        print(f"chart: {args.png}")
