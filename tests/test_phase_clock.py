"""Tests for the event clock (spec_ep.phase_clock).

The overlap, totals and trace file tests run on hand-built events and need no GPU. The
clock tests need one. The login node's GPU is shared, and another process's time slice
can delay a CUDA event by a few milliseconds, so they use phases of 100 ms and more and
allow 15 ms.

Usage:
    python -m pytest tests/test_phase_clock.py -v
    python -m pytest tests/test_phase_clock.py -s -k cost    # prints what a mark costs
"""
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch.profiler import record_function

from spec_ep.phase_clock import (
    OUTSIDE, Event, PhaseClock, Totals, in_flight, totals, with_overlap, write_trace,
)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MS = 15  # what a time slice of another process can add to a phase


def event(category: str, start_ms: float, dur_ms: float, link: str | None = None) -> Event:
    """Build an event of the step mode/step by hand, with no overlap set."""
    return Event("mode/step", ("mode/step",), category, link, start_ms, dur_ms, 0.0)


# Two compute stretches back to back, a transfer across both, a short one inside the
# second, and one after a gap of 20 ms in which nothing runs.
HAND_BUILT = [
    event("compute", 0, 100),
    event("comm", 60, 100, "both"),
    event("compute", 100, 100),
    event("comm", 150, 20, "intra"),
    event("comm", 220, 30, "inter"),
]


def test_overlap_is_the_part_of_an_event_that_other_events_cover() -> None:
    """Time covered by two other events counts once, and an event alone has none."""
    events = with_overlap(HAND_BUILT)
    assert [e.overlap_ms for e in events] == [40, 100, 70, 20, 0]
    assert [e._replace(overlap_ms=0.0) for e in events] == HAND_BUILT


def test_totals_split_the_overlap_by_what_runs_together() -> None:
    """Sums of durations, less the time counted twice, give the time anything ran."""
    t = totals(HAND_BUILT)
    assert t == Totals(
        comm_ms=150, intra_ms=20, inter_ms=30, both_ms=100, compute_ms=200,
        overlapped_ms=120, comm_comm_ms=10, comm_compute_ms=110, wall_ms=230,
    )
    assert t.comm_ms + t.compute_ms - t.overlapped_ms == t.wall_ms
    assert t.wall_ms == 250 - 20  # first start to last end, less the gap


def test_the_trace_file_keeps_overlapping_events_on_separate_lanes(tmp_path: Path) -> None:
    """Every event is in the file with its category, rank, and the caller's args."""
    events = with_overlap(HAND_BUILT)
    write_trace(tmp_path / "trace" / "rank3.json", events, 3, lambda e: {"scope": "moe"})
    rows = json.loads((tmp_path / "trace" / "rank3.json").read_text())["traceEvents"]
    rows = [row for row in rows if row["ph"] == "X"]

    assert [(row["cat"], row["ts"], row["dur"]) for row in rows] == [
        (e.category, 1e3 * e.start_ms, 1e3 * e.dur_ms) for e in events
    ]
    assert {row["pid"] for row in rows} == {3}
    for a in rows:
        for b in rows:
            if a is not b and a["ts"] < b["ts"] + b["dur"] and b["ts"] < a["ts"] + a["dur"]:
                assert a["tid"] != b["tid"]
    assert rows[3]["args"] == {
        "scope": "moe", "step": "mode/step", "link": "intra", "overlap_ms": 20, "overlap_pct": 100,
    }


def gpu_work(steps: int) -> None:
    """Queue steps matrix products on the current stream without waiting for them."""
    x = torch.ones(2048, 2048, device="cuda")
    for _ in range(steps):
        x = (x @ x).clamp(max=1)


@pytest.fixture(scope="module")
def steps_100ms() -> int:
    """Matrix products that keep the GPU busy for about 100 ms."""
    gpu_work(8)
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_work(32)
    torch.cuda.synchronize()
    return round(32 * 0.1 / (time.perf_counter() - start))


def traced(body: Callable[[], None], clock: PhaseClock | None = None) -> tuple[list[Event], float]:
    """Run body under the clock between two synchronizes.

    Returns:
        events: The clock's events.
        wall_ms: Host time from before the clock opened to after the GPU drained.
    """
    clock = clock or PhaseClock()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with clock:
        body()
    torch.cuda.synchronize()
    return clock.events(), 1e3 * (time.perf_counter() - start)


class Transfer:
    """Stands in for a NCCL transfer: GPU work on a stream of its own.

    Like WorkNCCL.wait(), wait() only makes the current stream wait for the end of the
    work, and returns at once.
    """

    def __init__(self, steps: int) -> None:
        self.stream = torch.cuda.Stream()
        self.begin, self.end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        self.begin.record(self.stream)
        with torch.cuda.stream(self.stream):
            gpu_work(steps)
        self.end.record(self.stream)

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.end)

    def ms(self) -> float:
        """How long the work really took. Call after a synchronize."""
        return self.begin.elapsed_time(self.end)


@needs_gpu
def test_compute_events_tile_the_run_and_add_up_to_the_wall_time(steps_100ms: int) -> None:
    """Nothing is counted twice or lost, and queued GPU work goes to the span that queued it."""
    def body() -> None:
        with record_function("mode/host"):
            time.sleep(0.2)
        with record_function("mode/gpu"):
            gpu_work(steps_100ms)
        with record_function("no slash, not a span"):
            pass

    events, wall_ms = traced(body)
    assert [e.step for e in events] == [OUTSIDE, "mode/host", OUTSIDE, "mode/gpu", OUTSIDE]
    assert all(e.category == "compute" and e.link is None and e.overlap_ms < 1e-6 for e in events)
    assert events[0].start_ms == 0
    for before, after in zip(events, events[1:]):
        assert after.start_ms == pytest.approx(before.start_ms + before.dur_ms, abs=1e-6)
    assert sum(e.dur_ms for e in events) == pytest.approx(wall_ms, abs=MS)
    assert events[1].dur_ms == pytest.approx(200, abs=MS)
    # The host left mode/gpu long before the GPU finished its work.
    assert events[3].dur_ms == pytest.approx(wall_ms - 200, abs=MS)
    assert events[3].dur_ms > 50


@needs_gpu
@pytest.mark.parametrize("link", ["intra", "inter", "both"])
def test_a_comm_span_is_one_comm_event_with_its_link(link: str) -> None:
    """The time inside the span is communication of its step, and not compute as well."""
    def body() -> None:
        with record_function("mode/dispatch"):
            time.sleep(0.1)
            with record_function(f"comm/{link}"):
                time.sleep(0.2)

    events, _ = traced(body)
    [comm] = [e for e in events if e.category == "comm"]
    assert (comm.step, comm.link) == ("mode/dispatch", link)
    assert comm.path == ("mode/dispatch", f"comm/{link}")
    assert comm.dur_ms == pytest.approx(200, abs=MS)
    assert sum(e.dur_ms for e in events if e.step == "mode/dispatch" and e.category == "compute") \
        == pytest.approx(100, abs=MS)
    assert all(e.overlap_ms < 1e-6 for e in events)


@needs_gpu
def test_compute_nested_in_a_comm_span_overlaps_it_by_its_own_duration() -> None:
    """A library's opaque pipeline is one comm event, and its expert callback is compute."""
    def body() -> None:
        with record_function("lib/pipeline"), record_function("comm/both"):
            time.sleep(0.1)
            with record_function("lib/experts"):
                time.sleep(0.1)
            time.sleep(0.1)

    events, _ = traced(body)
    [comm] = [e for e in events if e.category == "comm"]
    [experts] = [e for e in events if e.step == "lib/experts"]
    assert experts.category == "compute"
    assert experts.path == ("lib/pipeline", "comm/both", "lib/experts")
    assert experts.dur_ms == pytest.approx(100, abs=MS)
    assert experts.overlap_ms == pytest.approx(experts.dur_ms, abs=1e-6)
    assert comm.dur_ms == pytest.approx(300, abs=MS)
    assert comm.overlap_ms == pytest.approx(experts.dur_ms, abs=1e-6)


@needs_gpu
def test_a_wait_span_is_no_event() -> None:
    """The time the stream only waits in is left to the transfer it waits for."""
    def body() -> None:
        with record_function("mode/merge"):
            time.sleep(0.1)
            with record_function("wait/intra"):
                time.sleep(0.2)

    events, wall_ms = traced(body)
    assert {e.step for e in events} == {OUTSIDE, "mode/merge"}
    assert all(e.category == "compute" for e in events)
    assert sum(e.dur_ms for e in events) == pytest.approx(100, abs=MS)
    assert wall_ms == pytest.approx(300, abs=MS)


@needs_gpu
def test_an_in_flight_transfer_ends_when_its_work_ends(steps_100ms: int) -> None:
    """Its end is the work's, not the later time at which the main stream waits for it."""
    work = []

    def body() -> None:
        with record_function("mode/dispatch"):
            work.append(Transfer(steps_100ms))
            in_flight("intra", work[0])
            time.sleep(0.4)
            with record_function("wait/intra"):
                work[0].wait()

    events, wall_ms = traced(body)
    [comm] = [e for e in events if e.category == "comm"]
    assert (comm.step, comm.link, comm.path) == ("mode/dispatch", "intra", ("mode/dispatch",))
    assert comm.dur_ms == pytest.approx(work[0].ms(), abs=MS)
    assert comm.dur_ms < 300  # the main stream reached its wait at 400 ms
    # The host worked on under the whole transfer.
    assert comm.overlap_ms == pytest.approx(comm.dur_ms, abs=1e-6)
    assert totals(events).wall_ms == pytest.approx(wall_ms, abs=MS)


@needs_gpu
def test_an_in_flight_transfer_covers_the_wait_for_it(steps_100ms: int) -> None:
    """When the stream waits for the transfer, the transfer's event fills the wait."""
    work = []

    def body() -> None:
        with record_function("mode/dispatch"):
            work.append(Transfer(steps_100ms))
            in_flight("inter", work[0])
            with record_function("wait/inter"):
                work[0].wait()
        with record_function("mode/after"):
            time.sleep(0.1)

    events, wall_ms = traced(body)
    [comm] = [e for e in events if e.category == "comm"]
    [after] = [e for e in events if e.step == "mode/after"]
    assert comm.dur_ms == pytest.approx(work[0].ms(), abs=MS)
    assert after.start_ms == pytest.approx(comm.start_ms + comm.dur_ms, abs=MS)
    assert totals(events).wall_ms == pytest.approx(wall_ms, abs=MS)


@needs_gpu
def test_a_transfer_over_before_the_stream_reaches_its_launch_has_no_length(
    steps_100ms: int,
) -> None:
    """A backend that blocks the host ends a transfer while the stream still runs older work."""
    class Done:
        def wait(self) -> None:
            pass

    def body() -> None:
        with record_function("mode/dispatch"):
            gpu_work(steps_100ms)
            in_flight("both", Done())

    events, wall_ms = traced(body)
    [comm] = [e for e in events if e.category == "comm"]
    assert comm.dur_ms == 0
    assert comm.start_ms > 50  # where the stream launched it, after the older work
    assert totals(events).wall_ms == pytest.approx(wall_ms, abs=MS)


@needs_gpu
def test_the_totals_of_a_run_add_up_to_its_wall_time(steps_100ms: int) -> None:
    """wall = comm + compute - overlapped, and wall is the time the run took."""
    def body() -> None:
        with record_function("mode/plan"):
            time.sleep(0.1)
        with record_function("mode/dispatch"):
            work = Transfer(steps_100ms)
            in_flight("intra", work)
            with record_function("comm/inter"):
                time.sleep(0.2)
            with record_function("wait/intra"):
                work.wait()

    events, wall_ms = traced(body)
    t = totals(events)
    [intra] = [e for e in events if e.link == "intra"]
    assert t.comm_ms + t.compute_ms - t.overlapped_ms == pytest.approx(t.wall_ms)
    assert t.wall_ms == pytest.approx(events[-1].start_ms + events[-1].dur_ms, abs=MS)
    assert t.wall_ms == pytest.approx(wall_ms, abs=MS)
    assert t.inter_ms == pytest.approx(200, abs=MS)
    assert (t.intra_ms, t.both_ms) == (intra.dur_ms, 0)
    assert t.compute_ms == pytest.approx(100, abs=MS)
    assert t.comm_comm_ms == pytest.approx(intra.dur_ms, abs=MS)  # the two transfers ran together
    assert t.comm_compute_ms == pytest.approx(0, abs=MS)


@needs_gpu
def test_a_reused_clock_forgets_the_earlier_run() -> None:
    """A second, shorter run reports only its own events and its own time."""
    def long() -> None:
        for _ in range(8):
            with record_function("mode/long"), record_function("comm/both"):
                time.sleep(0.03)

    def short() -> None:
        with record_function("mode/short"):
            time.sleep(0.1)

    clock = PhaseClock()
    traced(long, clock)
    events, wall_ms = traced(short, clock)
    assert {e.step for e in events} == {OUTSIDE, "mode/short"}
    assert sum(e.dur_ms for e in events) == pytest.approx(wall_ms, abs=MS)


@needs_gpu
def test_the_clock_leaves_no_trace_once_closed() -> None:
    """record_function is the function it was before, and in_flight does nothing."""
    enter, leave = record_function.__enter__, record_function.__exit__
    traced(lambda: None)
    assert (record_function.__enter__, record_function.__exit__) == (enter, leave)
    in_flight("both", None)  # would fail on None.wait() if a clock were still active


@needs_gpu
def test_cost_of_a_span_and_of_an_in_flight_transfer() -> None:
    """Print what the clock adds to a span and what an in-flight transfer costs under it."""
    work = Transfer(0)
    torch.cuda.synchronize()

    def spans() -> None:
        for _ in range(2000):
            with record_function("mode/step"):
                pass

    def transfers() -> None:
        for _ in range(2000):
            in_flight("both", work)

    def host_us(body: Callable[[], None]) -> float:
        start = time.perf_counter()
        body()
        return 1e6 * (time.perf_counter() - start) / 2000

    bare = host_us(spans)
    with PhaseClock():
        span, transfer = host_us(spans) - bare, host_us(transfers)
    print(f"\nthe clock adds {span:.1f} us to a span that costs {bare:.1f} us without it, "
          f"and an in-flight transfer costs {transfer:.1f} us")
    assert span < 100 and transfer < 100
