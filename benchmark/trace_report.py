"""Summarize profiler traces as a table of GPU time and a timeline picture.

Reads the chrome traces that combine_dispatch --trace writes, DIR/{mode}_rank{r}.json,
one forward of one mode on one rank. For every mode it prints where the GPU time of
that forward went, and --png draws the same events on one time axis per mode, so a
slow mode shows which kind of work fills its time.

GPU time is split four ways, by what the kernel does:
    transfer: NCCL's and DeepEP's kernels, which move rows between GPUs.
    experts: kernels launched inside a span whose name ends in "experts".
    copy and fill: memcpy and memset, such as zeroing a padded buffer.
    other: every other kernel, such as gathers, casts, sorts and merges.
Kernels on different streams overlap, so the four can add up to more than the wall
time between the first and the last GPU event, which is printed beside them.

Usage:
    python -m benchmark.trace_report $SCRATCH/traces
    python -m benchmark.trace_report $SCRATCH/traces --modes plain tutel deepep \
        --rank 0 --png traces.png
"""
import argparse
import glob
import json
import os
from typing import Literal, NamedTuple

KINDS = ("transfer", "experts", "copy and fill", "other")
COLORS = dict(zip(KINDS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
PHASE_COLOR, INK, MUTED, GRID, SURFACE = "#c3c2b7", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


class Event(NamedTuple):
    """One GPU kernel, memcpy or memset, or one CPU-side phase.

    start_ms counts from the start of the forward's first phase. A phase has
    gpu_stream 0.
    """

    name: str
    start_ms: float
    dur_ms: float
    kind: Literal["transfer", "experts", "copy and fill", "other", "phase"]
    gpu_stream: int


def load_events(path: str) -> list[Event]:
    """Read one trace into GPU events and CPU phases on a shared clock.

    A phase is a record_function span named "<mode>/<step>", timed on the CPU side:
    when the call that launched its work started and returned. The forward starts
    with its first phase, so GPU events before it, such as the barrier that lines the
    ranks up, are left out.

    Returns:
        events: Every GPU event and phase, by start time.
    """
    with open(path) as file:
        raw = [e for e in json.load(file)["traceEvents"] if e.get("ph") == "X"]
    gpu = [e for e in raw if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    phases = [e for e in raw if e.get("cat") == "user_annotation" and "/" in e["name"]]
    expert_spans = [
        (e["ts"], e["ts"] + e["dur"], e["tid"]) for e in raw
        if e.get("cat") == "gpu_user_annotation" and e["name"].endswith("experts")
    ]
    forward_start = min(e["ts"] for e in phases)
    gpu = [e for e in gpu if e["ts"] >= forward_start]

    def kind(event: dict) -> str:
        if event["cat"] != "kernel":
            return "copy and fill"
        if "ncclDevKernel" in event["name"] or "deep_ep" in event["name"]:
            return "transfer"
        inside = any(a <= event["ts"] < b and event["tid"] == tid for a, b, tid in expert_spans)
        return "experts" if inside else "other"

    streams = sorted({e["tid"] for e in gpu})
    events = [
        Event(e["name"], (e["ts"] - forward_start) / 1e3, e["dur"] / 1e3, kind(e), streams.index(e["tid"]))
        for e in gpu
    ]
    events += [
        Event(e["name"], (e["ts"] - forward_start) / 1e3, e["dur"] / 1e3, "phase", 0) for e in phases
    ]
    return sorted(events, key=lambda e: e.start_ms)


def short(name: str) -> str:
    """Shorten a C++ kernel name to the part that tells kernels apart.

    Returns:
        short: At most 60 characters.
    """
    for prefix in ("void ", "at::native::", "(anonymous namespace)::"):
        name = name.replace(prefix, "")
    return name[:60]


def report(mode: str, events: list[Event]) -> None:
    """Print one mode's GPU time by kind, its phases and its five longest kernels."""
    gpu = [e for e in events if e.kind != "phase"]
    wall = max(e.start_ms + e.dur_ms for e in gpu) - min(e.start_ms for e in gpu)
    by_kind = {k: sum(e.dur_ms for e in gpu if e.kind == k) for k in KINDS}
    print(f"\n{mode}: {wall:.2f} ms from first to last GPU event")
    print("  GPU time: " + " | ".join(f"{k} {by_kind[k]:.2f} ms" for k in KINDS))

    phases = {}
    for e in events:
        if e.kind == "phase":
            phases[e.name] = phases.get(e.name, 0) + e.dur_ms
    print("  CPU-side phases: " + " | ".join(f"{n.split('/')[1]} {d:.2f}" for n, d in phases.items()))

    by_name = {}
    for e in gpu:
        key = (e.kind, short(e.name))
        total, count = by_name.get(key, (0, 0))
        by_name[key] = (total + e.dur_ms, count + 1)
    for (kind, name), (total, count) in sorted(by_name.items(), key=lambda x: -x[1][0])[:5]:
        print(f"    {total:8.3f} ms  x{count:<4d} [{kind}] {name}")


def draw(traces: dict[str, list[Event]], path: str) -> None:
    """Draw every mode's phases and GPU events on a shared time axis, one panel each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lanes = {mode: 1 + max(e.gpu_stream for e in events if e.kind != "phase") for mode, events in traces.items()}
    end = max(e.start_ms + e.dur_ms for events in traces.values() for e in events)
    fig, axes = plt.subplots(
        len(traces), 1, sharex=True, squeeze=False, facecolor=SURFACE,
        figsize=(11, 0.45 * sum(lanes.values()) + 0.95 * len(traces) + 0.9),
        gridspec_kw={"height_ratios": [n + 1.2 for n in lanes.values()]},
    )
    for ax, (mode, events) in zip(axes[:, 0], traces.items()):
        ax.set_facecolor(SURFACE)
        for e in events:
            if e.kind == "phase":
                ax.barh(0, e.dur_ms, left=e.start_ms, height=0.7, color=PHASE_COLOR,
                        edgecolor=SURFACE, linewidth=1)
                if e.dur_ms > end / 28:
                    ax.text(e.start_ms + e.dur_ms / 2, 0, e.name.split("/")[1], ha="center",
                            va="center", fontsize=7, color=INK, clip_on=True)
            else:
                ax.barh(1 + e.gpu_stream, max(e.dur_ms, end / 2500), left=e.start_ms, height=0.7,
                        color=COLORS[e.kind], linewidth=0)
        ax.set_yticks(range(lanes[mode] + 1))
        ax.set_yticklabels(["CPU phase"] + [f"GPU stream {i}" for i in range(lanes[mode])],
                           fontsize=7, color=MUTED)
        ax.set_ylim(lanes[mode] + 0.5, -0.6)
        ax.set_title(mode, loc="left", fontsize=9, color=INK, pad=3)
        ax.set_xlim(0, end * 1.01)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=7, labelcolor=MUTED)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[-1, 0].set_xlabel("ms since the forward started", fontsize=8, color=MUTED)
    fig.legend(
        handles=[Patch(color=COLORS[k], label=k) for k in KINDS]
        + [Patch(color=PHASE_COLOR, label="CPU phase (name inside)")],
        loc="upper right", ncols=5, fontsize=7, frameon=False, labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace_dir", help="Directory holding {mode}_rank{r}.json")
    parser.add_argument("--modes", nargs="+", help="Modes to report; default every trace found")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--png", help="Also draw the timelines to this file")
    args = parser.parse_args()

    suffix = f"_rank{args.rank}.json"
    found = sorted(os.path.basename(p)[: -len(suffix)]
                   for p in glob.glob(f"{glob.escape(args.trace_dir)}/*{suffix}"))
    traces = {
        mode: load_events(f"{args.trace_dir}/{mode}{suffix}") for mode in args.modes or found
    }
    for mode, events in traces.items():
        report(mode, events)
    if args.png:
        draw(traces, args.png)
        print(f"\ntimelines: {args.png}")
