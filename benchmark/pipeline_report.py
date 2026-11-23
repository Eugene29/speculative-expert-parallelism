"""What the RBD pipeline hides against the most it could, from a model run's event tables.

Reads the output of scripts/ds_moe.py --profile events for a run that timed rbd and
pipelined modes. From rbd's own events it takes, per forward:

    network: the two stage 1 all2alls over all ranks, which a pipeline runs slices under.
    work: what a slice does between its arrival and its return: stage 2 both ways and
        the experts.

and replays them as a pipeline of `degree` slices with nothing lost: slices of equal
size, dispatches back to back, a slice's work as soon as it has arrived and the
previous slice's work is done, its combine as soon as that work and the previous
combine are done. The time that saves over rbd is the ceiling for that degree. The
contention ceiling keeps only the share --hides of it, the share of the experts an
overlapped all2all hides when both share a GPU (benchmark/all2all/contention.py).

It then prints every mode's forward next to those ceilings, and each step's events
against rbd's. CPU only.

Usage:
    python -m benchmark.pipeline_report logs/2026-11-22_pipeline/2n_b/model.out
    python -m benchmark.pipeline_report MODEL.out --base rbd --hides 0.81 --steps
"""
import argparse
import re

Key = tuple[str, str, str]  # step, category, link
NETWORK = [("rbd_stage1_dispatch", "comm", "both"), ("rbd_stage1_combine", "comm", "both")]
WORK = ["rbd_stage2_dispatch", "rbd_experts", "rbd_stage2_combine"]


def read(path: str) -> tuple[dict[str, float], dict[str, dict[Key, tuple[int, float, float]]]]:
    """Read a model run's timed forwards and event tables.

    Returns:
        seconds: Median seconds per timed forward of every mode.
        tables: Per mode, every (step, category, link) row's events, ms and overlapped ms.
    """
    seconds, tables, mode = {}, {}, None
    for line in open(path):
        timed = re.match(r"^\s*(\S+): median\s+([\d.]+) s ", line)
        traced = re.match(r"^(\S+): traced ", line)
        if timed:
            seconds[timed.group(1)] = float(timed.group(2))
        elif traced:
            mode = traced.group(1)
            tables[mode] = {}
        elif mode and line.startswith(("| moe", "| outside")):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            step = cells[1].split("/")[-1]
            tables[mode][(step, cells[2], cells[3])] = (int(cells[4]), float(cells[5]), float(cells[6]))
    return seconds, tables


def pipeline_ms(network: float, work: float, degree: int) -> float:
    """Replay a layer as `degree` equal slices with nothing lost.

    Returns:
        ms: From the first dispatch's start to the last combine's end.
    """
    chunk, slice_work = network / 2 / degree, work / degree
    worked = combined = 0.0
    for c in range(degree):
        worked = max(worked, (c + 1) * chunk) + slice_work
        combined = max(combined, worked, degree * chunk) + chunk
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", help="Output of scripts/ds_moe.py --profile events")
    parser.add_argument("--base", default="rbd", help="The mode the pipelines are measured against")
    parser.add_argument("--hides", type=float, default=0.81,
                        help="Share of the experts an overlapped all2all hides on a shared GPU")
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--steps", action="store_true", help="Also print every step against the base")
    args = parser.parse_args()

    seconds, tables = read(args.path)
    base = tables[args.base]
    network = sum(base[key][1] for key in NETWORK)
    work = sum(ms for (step, _, _), (_, ms, _) in base.items() if step in WORK)
    forward = seconds[args.base] * 1e3
    print(f"{args.base}: {forward:.0f} ms per forward | stage 1 network {network:.1f} ms | "
          f"slice work (stage 2 and experts) {work:.1f} ms, {network / work:.2f}x less than the network")
    print("\n| Degree | Hidden at most, ms | Ceiling | Ceiling with contention |\n|---|---|---|---|")
    for degree in [*args.degrees, 1000]:
        hidden = network + work - pipeline_ms(network, work, degree)
        name = "every slice hidden" if degree == 1000 else str(degree)
        print(f"| {name} | {hidden:.1f} | {forward / (forward - hidden):.3f}x | "
              f"{forward / (forward - args.hides * hidden):.3f}x |")

    print(f"\n| Mode | s per forward | vs {args.base} | comm+compute overlapped, ms |\n|---|---|---|---|")
    for mode, table in tables.items():
        overlapped = sum(row[2] for (_, category, _), row in table.items() if category == "compute")
        print(f"| {mode} | {seconds[mode]:.3f} | {seconds[args.base] / seconds[mode]:.3f}x | {overlapped:.1f} |")

    if args.steps:
        for mode, table in tables.items():
            if mode == args.base:
                continue
            print(f"\n| Step | Category | Link | {args.base} ms | {mode} ms | Change |\n|---|---|---|---|---|---|")
            for key in dict.fromkeys([*base, *table]):
                a, b = base.get(key, (0, 0.0, 0.0))[1], table.get(key, (0, 0.0, 0.0))[1]
                print(f"| {key[0]} | {key[1]} | {key[2]} | {a:.1f} | {b:.1f} | {b - a:+.1f} |")
