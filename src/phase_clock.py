"""Trace a forward as events on the GPU's own clock, with no profiler.

Code marks what it does with record_function("<mode>/<step>"). While a PhaseClock is
active, every such span records a CUDA event on the current stream when it opens and
when it closes. A CUDA event completes once everything queued before it has run, so
the time between two marks is what the stream spent on the work the host queued
between them: kernels, the transfers the stream waited for, and the time the GPU sat
idle while the host prepared the next kernel. Every mark's time is read against the
clock's first mark, so all marks of one rank sit on one time axis whatever stream
recorded them.

The clock turns the marks into events. An Event is communication or computation, never
both. What a span gives depends on the part of its name before the slash (KINDS):

    comm/<link>    One comm event from the span's opening mark to its closing mark.
                   link says which ranks the transfer reaches: intra for this node's,
                   inter for other nodes', both for all. The stream's own time inside
                   the span is no event of its own. Spans nested in it give their own
                   events, which overlap it.
    wait/<name>    No event. The stream only waits for a transfer that an in-flight
                   event already covers.
    <mode>/<step>  Compute. One event per stretch of the stream that belongs to the
                   span, so a span with spans nested in it gives several. The time
                   outside every span is compute too, with the step OUTSIDE.

Compute therefore includes host time during which the GPU idles between kernels. A
span whose name has no slash is ignored.

A transfer that runs while the stream goes on has no span, because its launch returns
at once. in_flight(link, work), called right after the launch, makes it one comm event
from that point of the stream to the transfer's real end. The end is a CUDA event on a
side stream that waits for the transfer, so it is stamped when the transfer ends even
if the main stream gets to its own wait much later.

Overlap and totals are plain functions of the events and need no GPU. with_overlap()
gives every event the part of it during which another event was running. totals() adds
up one rank's events, so that wall = comm + compute - overlapped. write_trace() writes
the events as a Chrome trace for Perfetto.

On the login node the clock adds about 15 microseconds of host time to a span, 5 per
mark for the CUDA event itself, and an in-flight transfer costs about 20
(tests/test_phase_clock.py -k cost prints both). scripts/ds_moe.py --profile events
prints every traced forward against an untraced one.

Usage:
    clock = PhaseClock()
    with clock:
        engine(**batch)
    torch.cuda.synchronize()
    events = clock.events()                 # this rank's events, in the order they began
    print(totals(events))                   # comm, compute, overlapped and wall, in ms
    write_trace(Path("rank0.json"), events, rank=0)

    python -m pytest tests/test_phase_clock.py -v
"""
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function

__all__ = [
    "OUTSIDE",
    "KINDS",
    "Link",
    "Event",
    "Totals",
    "PhaseClock",
    "in_flight",
    "with_overlap",
    "totals",
    "write_trace",
]

OUTSIDE = "outside"
# What a span records, by the part of its name before the slash. Any other is compute.
KINDS = {"comm": "comm", "wait": "wait"}
Link = Literal["intra", "inter", "both"]


class Event(NamedTuple):
    """One stretch of one rank's time, either communication or computation."""

    step: str  # innermost "<mode>/<step>" span open that is not a comm or wait span, or OUTSIDE
    path: tuple[str, ...]  # every span open, outermost first
    category: Literal["comm", "compute"]
    link: Link | None  # comm only: the links that carry it
    start_ms: float  # since the clock opened, on the GPU clock
    dur_ms: float
    overlap_ms: float  # part of dur_ms during which another event was running


class Totals(NamedTuple):
    """One rank's events added up, in ms. wall_ms = comm_ms + compute_ms - overlapped_ms."""

    comm_ms: float  # intra_ms + inter_ms + both_ms
    intra_ms: float
    inter_ms: float
    both_ms: float
    compute_ms: float
    overlapped_ms: float  # time counted more than once: comm_comm_ms + comm_compute_ms
    comm_comm_ms: float  # comm events running at the same time as other comm events
    comm_compute_ms: float  # a comm event running at the same time as a compute event
    wall_ms: float  # time during which any event was running


_active: "PhaseClock | None" = None  # the clock that in_flight() reports to


def _kind(name: str) -> str:
    """What the span called name records: comm, wait or compute."""
    return KINDS.get(name.partition("/")[0], "compute")


class PhaseClock:
    """Marks the GPU stream at every span boundary while active. Reusable."""

    def __init__(self) -> None:
        self.marks: list[torch.cuda.Event] = []  # reused by every use of this clock
        self.side: dict[str, torch.cuda.Stream] = {}  # per link, where in-flight transfers end

    def __enter__(self) -> "PhaseClock":
        global _active
        self.stream = torch.cuda.current_stream()
        # One side stream per link: transfers of one link share a process group, which
        # runs them in order, so a stream never waits for one that ends after the next.
        self.side = self.side or {
            link: torch.cuda.Stream(self.stream.device) for link in get_args(Link)
        }
        self.used = 0  # marks recorded in this use
        self.open: list[str] = []  # spans open now, outermost first
        self.opened: list[int] = []  # the mark each of them opened at
        self.tiles: list[tuple[int, tuple[str, ...]]] = []  # (mark, spans open after it)
        self.comms: list[tuple[int, int, tuple[str, ...], str]] = []  # (start, end, path, link)
        clock, enter, leave = self, record_function.__enter__, record_function.__exit__
        self.unpatched = (enter, leave)

        def marked_enter(span: record_function) -> record_function:
            if "/" in span.name:
                clock.open.append(span.name)
                clock.opened.append(clock._tile())
            return enter(span)

        def marked_exit(span: record_function, *exc: object) -> None:
            leave(span, *exc)
            if "/" in span.name:
                name = clock.open.pop()
                start, end = clock.opened.pop(), clock._tile()
                if _kind(name) == "comm":
                    clock.comms.append((start, end, (*clock.open, name), name.partition("/")[2]))

        record_function.__enter__, record_function.__exit__ = marked_enter, marked_exit
        _active = self
        self._tile()
        return self

    def __exit__(self, *exc: object) -> None:
        global _active
        record_function.__enter__, record_function.__exit__ = self.unpatched
        _active = None
        self._tile()

    def _mark(self, stream: torch.cuda.Stream) -> int:
        """Record a CUDA event on stream.

        Returns:
            mark: Its index in self.marks.
        """
        if self.used == len(self.marks):
            self.marks.append(torch.cuda.Event(enable_timing=True))
        self.marks[self.used].record(stream)
        self.used += 1
        return self.used - 1

    def _tile(self) -> int:
        """Mark the main stream where the open spans change; the marks tile its time.

        Returns:
            mark: Its index in self.marks.
        """
        mark = self._mark(self.stream)
        self.tiles.append((mark, tuple(self.open)))
        return mark

    def events(self) -> list[Event]:
        """Turn the marks into events. Call after the clock closed and a synchronize.

        Returns:
            events: This rank's events in the order they began, with overlap_ms set.
        """
        at = [self.marks[0].elapsed_time(mark) for mark in self.marks[: self.used]]
        # The main stream between two marks is compute, unless a comm or wait span is innermost.
        spans = [
            (start, end, path, "compute", None)
            for (start, path), (end, _) in zip(self.tiles, self.tiles[1:])
            if not path or _kind(path[-1]) == "compute"
        ]
        spans += [(start, end, path, "comm", link) for start, end, path, link in self.comms]
        spans.sort(key=lambda span: span[0])  # every start is a mark of the main stream
        # An in-flight transfer can be over before the stream gets to its launch: an
        # empty one, or any under a backend that blocks the host. It has no length then.
        return with_overlap([
            Event(
                next((name for name in reversed(path) if _kind(name) == "compute"), OUTSIDE),
                path, category, link, at[start], max(at[end] - at[start], 0.0), 0.0,
            )
            for start, end, path, category, link in spans
        ])


def in_flight(link: Link, work: dist.Work) -> None:
    """Record a transfer that runs while the stream goes on as one comm event.

    Call it right after the launch. A no-op when no clock is active. The event runs
    from this point of the current stream to the transfer's end. work.wait() only
    makes the current stream wait for the transfer and returns at once, so calling it
    with a side stream current stamps that stream when the transfer really ends, even
    when the main stream gets to its own wait later. With a backend whose wait()
    blocks the host, such as gloo, the transfer is over when this returns.

    work.wait() also drops torch's hold on the transfer's tensors (WorkNCCL stashes
    them until the first wait), so the caller must keep them alive until the main
    stream has waited too, as spec_ep.ops.all2all_async does.

    Args:
        link: The links that carry the transfer.
        work: What the launch returned, such as all_to_all_single(..., async_op=True).
    """
    clock = _active
    if clock is None:
        return
    start, side = clock._mark(clock.stream), clock.side[link]
    torch.cuda.set_stream(side)
    work.wait()
    torch.cuda.set_stream(clock.stream)
    clock.comms.append((start, clock._mark(side), tuple(clock.open), link))


def _pieces(events: list[Event]) -> Iterator[tuple[float, set[int]]]:
    """Cut the time axis at every start and end of an event.

    Yields:
        length: Milliseconds between two neighboring cuts during which some event runs.
        running: Indices of the events running then. Valid until the next piece.
    """
    cuts = sorted(
        [(event.start_ms, 0, i) for i, event in enumerate(events)]
        + [(event.start_ms + event.dur_ms, 1, i) for i, event in enumerate(events)]
    )  # at one time, starts sort before ends, so an event of no length opens and closes
    running: set[int] = set()
    for (time, ends, i), (following, _, _) in zip(cuts, cuts[1:]):
        running.remove(i) if ends else running.add(i)
        if running and following > time:
            yield following - time, running


def with_overlap(events: list[Event]) -> list[Event]:
    """Set every event's overlap_ms: the part of it during which another event ran.

    Returns:
        events: The events in the order given, with overlap_ms set.
    """
    overlap = [0.0] * len(events)
    for length, running in _pieces(events):
        if len(running) > 1:
            for i in running:
                overlap[i] += length
    return [event._replace(overlap_ms=ms) for event, ms in zip(events, overlap)]


def totals(events: list[Event]) -> Totals:
    """Add up the events of one rank's forward, or of a part of it.

    comm_ms and compute_ms are sums of durations. wall_ms is the time during which any
    event ran, and overlapped_ms is what the sums count beyond it, split by what ran
    at the same time. Events that leave no gap make wall_ms the time from the first
    start to the last end.

    Returns:
        totals: The sums, in ms.
    """
    ms = dict.fromkeys(("compute", *get_args(Link)), 0.0)
    for event in events:
        ms[event.link or "compute"] += event.dur_ms
    wall = comm_comm = 0.0
    for length, running in _pieces(events):
        wall += length
        comm_comm += length * max(sum(events[i].category == "comm" for i in running) - 1, 0)
    comm = ms["intra"] + ms["inter"] + ms["both"]
    overlapped = comm + ms["compute"] - wall
    return Totals(
        comm, ms["intra"], ms["inter"], ms["both"], ms["compute"],
        overlapped, comm_comm, overlapped - comm_comm, wall,
    )


def write_trace(
    file: Path,
    events: list[Event],
    rank: int,
    extra: Callable[[Event], dict[str, object]] = lambda event: {},
) -> None:
    """Write one rank's events as a Chrome trace, which ui.perfetto.dev opens.

    Every event is a complete event ("ph": "X") whose cat is comm or compute, whose pid
    is the rank and whose tid is a lane: 0 for compute, which tiles the stream, and for
    comm the first lane from 1 on that no running comm event holds. Events that overlap
    therefore never share a lane. args holds extra(event), then step, link, overlap_ms
    and overlap_pct.

    Args:
        file: Where to write. Its directory is created.
        events: One rank's events, with overlap_ms set.
        rank: The rank that recorded them.
        extra: What the caller knows about an event, added to its args.
    """
    rows = [{"ph": "M", "name": "process_name", "pid": rank, "args": {"name": f"rank {rank}"}}]
    busy_until: list[float] = []  # end of the last comm event on lanes 1, 2, ...
    for event in sorted(events, key=lambda event: event.start_ms):
        lane, name = 0, event.step
        if event.category == "comm":
            free = (i for i, end in enumerate(busy_until) if end <= event.start_ms)
            lane, name = 1 + next(free, len(busy_until)), f"{event.step} ({event.link})"
            busy_until[lane - 1 : lane] = [event.start_ms + event.dur_ms]
        rows.append({
            "name": name, "cat": event.category, "ph": "X", "pid": rank, "tid": lane,
            "ts": 1e3 * event.start_ms, "dur": 1e3 * event.dur_ms,
            "args": extra(event) | {
                "step": event.step, "link": event.link, "overlap_ms": event.overlap_ms,
                "overlap_pct": 100 * event.overlap_ms / event.dur_ms if event.dur_ms else 0.0,
            },
        })
    rows += [
        {"ph": "M", "name": "thread_name", "pid": rank, "tid": lane,
         "args": {"name": f"comm {lane}" if lane else "compute"}}
        for lane in range(len(busy_until) + 1)
    ]
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"traceEvents": rows, "displayTimeUnit": "ms"}))
