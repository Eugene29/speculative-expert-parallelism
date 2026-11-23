"""Turn the outputs of combine_dispatch and ds_moe runs into Markdown tables.

Reads the text a run printed, one file per job step, and prints one table: a row per
mode and a column per file, or with --by file a row per file and a column per mode.
A cell is the median of the step's timed calls, with min and max in brackets when
--spread is given. With --by file every row also shows the inter-node rows the step
counted and, for a step with replicas, what the routing to replicas cost.

Where each mode's forward goes needs no table from here: scripts/ds_moe.py
--profile events prints one per mode, already in Markdown.

Usage:
    python -m benchmark.summarize logs/2026-11-22_exp/2n/micro_layer0.out \
        logs/2026-11-22_exp/4n/micro_layer0.out --spread
    python -m benchmark.summarize logs/2026-11-22_exp/4n/replicas_layer0_*.out --by file
"""
import argparse
import os
import re
from typing import NamedTuple

TIMING = re.compile(r"^\s*(\S+): median\s+([\d.]+) (ms|s) \| min\s+([\d.]+) \| max\s+([\d.]+)")
ROWS = re.compile(r"inter-node rows per direction: plain (\d+) .*, rbd (\d+)")
ROUTING = re.compile(r"routing to replicas takes ([\d.]+) ms")


class Step(NamedTuple):
    """What one job step printed."""

    name: str  # file name without .out
    times: dict[str, tuple[float, float, float]]  # mode -> median, min, max
    unit: str  # "ms" or "s"
    plain_rows: int | None  # copies plain sends between nodes, per direction
    rbd_rows: int | None  # pilots RBD sends between nodes, per direction
    routing_ms: float | None  # cost of routing to replicas per call


def read_step(path: str) -> Step:
    """Parse one step's output.

    Returns:
        step: The step's timings and counts; fields the file lacks are None or empty.
    """
    text = open(path).read()
    times, unit = {}, "ms"
    for line in text.splitlines():
        if match := TIMING.match(line):
            times[match[1]] = tuple(float(match[i]) for i in (2, 4, 5))
            unit = match[3]
    rows, routing = ROWS.search(text), ROUTING.search(text)
    return Step(
        os.path.basename(path).removesuffix(".out"), times, unit,
        int(rows[1]) if rows else None, int(rows[2]) if rows else None,
        float(routing[1]) if routing else None,
    )


def cell(step: Step, mode: str, spread: bool) -> str:
    """Format one mode's time in one step, or a dash if the step did not run it."""
    if mode not in step.times:
        return "-"
    median, low, high = step.times[mode]
    digits = 2 if step.unit == "ms" else 3
    text = f"{median:.{digits}f}"
    return f"{text} ({low:.{digits}f} to {high:.{digits}f})" if spread else text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", help="Step outputs, *.out")
    parser.add_argument("--by", choices=("mode", "file"), default="mode", help="What a row is")
    parser.add_argument("--spread", action="store_true", help="Add min and max to every cell")
    args = parser.parse_args()
    steps = [read_step(path) for path in args.files]
    modes = list(dict.fromkeys(mode for step in steps for mode in step.times))
    unit = steps[0].unit

    if args.by == "mode":
        print(f"| mode ({unit}) | " + " | ".join(step.name for step in steps) + " |")
        print("|---" * (len(steps) + 1) + "|")
        for mode in modes:
            print(f"| {mode} | " + " | ".join(cell(step, mode, args.spread) for step in steps) + " |")
    else:
        print(f"| step ({unit}) | plain rows | rbd rows | routing ms | " + " | ".join(modes) + " |")
        print("|---" * (len(modes) + 4) + "|")
        for step in steps:
            counts = [step.plain_rows, step.rbd_rows, step.routing_ms]
            print(f"| {step.name} | " + " | ".join("-" if c is None else str(c) for c in counts)
                  + " | " + " | ".join(cell(step, mode, args.spread) for mode in modes) + " |")
