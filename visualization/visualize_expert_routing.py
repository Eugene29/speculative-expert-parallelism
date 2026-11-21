"""Plot a histogram of expert popularity across experts and GPUs.

Reads one or more `experts_rank{r}.pt` files written by
`scripts/ds_moe.py --record`. Each file holds a tensor of shape
[L, T, K]: for every MoE layer L and every token T on that rank, the ids of
the top-K experts it was routed to. Files are concatenated along the token
dim.

The figure has one row per layer and four columns:
    1. tokens per expert   -- histogram over expert ids (raw order).
    2. experts, sorted     -- same counts sorted high to low, which shows how
                              skewed the load is regardless of which experts
                              are hot.
    3. tokens per GPU      -- counts summed over each GPU's contiguous block
                              of `num_experts // num_gpus` experts, i.e. the
                              load each rank would receive under static
                              expert parallelism.
    4. GPUs, sorted        -- per-GPU counts sorted high to low.

A red dashed line in each panel marks the mean, the load every expert/GPU
would get under perfect balance. The PNG is saved next to
the first input file, with the same stem.

Usage:
    python visualization/visualize_expert_routing.py logs/2026_oct/2026-10-16_14-07-50/experts_rank*.pt

    python visualization/visualize_expert_routing.py $LOGS/experts_rank*.pt \\
        --num_experts 256 --num_gpus 8
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", type=Path, help="experts_rank*.pt, each [L, T, K]")
ap.add_argument("--num_experts", type=int, default=256)
ap.add_argument("--num_gpus", type=int, default=8)
args = ap.parse_args()

E = torch.cat([torch.load(f).long() for f in args.files], dim=1)
L = E.shape[0]
experts_per_gpu = args.num_experts // args.num_gpus

fig, axes = plt.subplots(
    L, 4, figsize=(20, 1.8 * L), squeeze=False,
    gridspec_kw={"width_ratios": [4, 4, 1.5, 1.5]},
)
col_titles = ["tokens per expert", "experts, sorted", "tokens per GPU", "GPUs, sorted"]

for l in range(L):
    ids = E[l].flatten()
    expert_hist = ids.bincount(minlength=args.num_experts)
    gpu_hist = (ids // experts_per_gpu).bincount(minlength=args.num_gpus)

    panels = [
        (expert_hist, args.num_experts, 1.0),
        (expert_hist.sort(descending=True).values, args.num_experts, 1.0),
        (gpu_hist, args.num_gpus, 0.8),
        (gpu_hist.sort(descending=True).values, args.num_gpus, 0.8),
    ]
    for c, (h, n, w) in enumerate(panels):
        ax = axes[l, c]
        ax.bar(range(n), h, width=w)
        ax.axhline(h.float().mean(), color="red", lw=0.8, ls="--")
        ax.tick_params(labelsize=6)
        if c >= 2:
            ax.set_xticks(range(n))
    for c0 in (0, 2):
        top = max(axes[l, c0].get_ylim()[1], axes[l, c0 + 1].get_ylim()[1])
        axes[l, c0].set_ylim(0, top)
        axes[l, c0 + 1].set_ylim(0, top)
    axes[l, 0].set_ylabel(f"L{l}", fontsize=8, rotation=0, labelpad=14, va="center")

for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=9)
axes[-1, 0].set_xlabel("expert id", fontsize=7)
axes[-1, 1].set_xlabel("rank by load", fontsize=7)
axes[-1, 2].set_xlabel("GPU", fontsize=7)
axes[-1, 3].set_xlabel("rank by load", fontsize=7)

fig.suptitle(f"Routing distribution ({len(args.files)} rank file(s), "
             f"{E.shape[1]} tokens, top-{E.shape[2]}); red dashed = balanced level", y=1.0)
fig.tight_layout()

out = args.files[0].with_suffix(".png")
fig.savefig(out, dpi=120)
print(f"saved {out}")