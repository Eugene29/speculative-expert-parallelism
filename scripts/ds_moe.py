"""Time inference of the MoE model under every dispatch mode, and optionally record its routing.

Loads Qwen3.6-35B-A3B on CPU, lets DeepSpeed's AutoEP spread the experts over the
ranks, and runs a prefill forward on --sequences sequences of --seq-len tokens per rank, cut from
data/short_text.txt. The model is loaded once. --modes then picks how every MoE layer
dispatches and combines: autoep is the layer's own path, and any other mode of
benchmark/all2all/modes.py (rbd, two_lane, tutel:capacity_factor=1, ds_moe, xmoe, ...)
replaces it with that mode's function, the same one the layer benchmark times.
rbd_guests is built here (scripts/ds_guests.py): RBD in which every rank also runs the
hottest experts of other nodes, loaded from host memory one layer ahead. Its settings
are slots (guests per rank), source (speculative, static or oracle), placement (unique
or redundant), lanes (2 for two-lane RBD, 1 for RBD in one all2all) and degree (slices
of the one-lane RBD pipeline), as in rbd_guests:slots=16,source=static. After its
first forward it prints how its guests compare with the oracle's and what loading them
took (GuestDispatch.measure).

For every mode the script first runs one untimed forward. It prints how far the
forward's last logits are from the first mode's, on how many of the last tokens both
pick the same top token, and the forward's peak GPU memory. A mode that drops copies,
such as tutel at its default capacity, differs for that reason. Modes that keep every
copy still differ in bf16, because their merge kernels round differently and 40 layers
grow the difference. --dtype float32 shows whether they compute the same function.

The script then times --repeats rounds of one forward per mode, so the modes
alternate, and prints each mode's median, min and max. Python's garbage collector is
off inside every measured forward. A full collection takes a few hundred milliseconds
here and stalls every rank, because the others wait for the collecting rank in their
next transfer.

It also measures where the time goes, with each profiler of --profile:

    events (default): in every round, each mode's timed forward is followed by one
        forward traced with spec_ep.phase_clock, so drift over the run favors
        neither. The clock turns the steps a mode marks into events on the GPU's own
        clock, without a profiler. An event is communication or computation, never
        both. Communication is every transfer call, with the link that carries it:
        intra (ranks of one node), inter (other nodes) or both. A library's own
        packing inside its transfer call counts as communication. Computation is
        everything else the stream does, host time with an idle GPU included: the
        mode's steps, "moe/layer" for the part of a MoE layer no step covers, and
        "outside" for the model outside its MoE layers. Per mode the script prints
          1. one line: the traced forward against the timed one, with the median and
             range of both, which is what tracing costs. Then the forward's comm (by
             link), compute, overlapped (comm with comm, comm with compute) and
             wall = comm + compute - overlapped, and the same inside the MoE layers
             and outside them.
          2. a Markdown table with a row per (scope, step, category, link): events
             per forward, ms, overlapped ms and overlapped percent. A comm row has
             one event per transfer. A compute row has one per stretch of the
             stream, so a step with a transfer inside it has two per call.
        All in ms, as the mean over ranks and the median over rounds. Every rank
        writes the last round's events to LOGS/events/{mode}_rank{r}.json, a Chrome
        trace that ui.perfetto.dev opens: comm and compute on separate lanes, and
        scope, layer, step, link, overlap_ms and overlap_pct in every event's args.
    torch (default): one forward per mode under torch.profiler, after the rounds,
        timed like the others and printed against the timed median. Rank 0 and the
        ranks of --trace-ranks leave LOGS/traces/{mode}_rank{r}.json, which
        benchmark/trace_report.py turns into GPU time by kind and a timeline, and
        rank 0 leaves LOGS/traces/{mode}_key_averages.txt, the 30 ops with the most
        CUDA time. The profiler slows the host side, so read these for the kernels
        and the events for the times.
    flops: DeepSpeed's flops profiler. Leaves LOGS/{mode}/flops_profile_{r}.txt, with
        flops, MACs, parameters and latency per module.
    comm: DeepSpeed's comms logger. Prints the count, message size, latency and
        bandwidth of every communication op that goes through deepspeed.comm, which
        is only the autoep mode. The config must turn it on with
        "comms_logger": {"enabled": true}.

With --record it first saves the experts every token chose to
LOGS/experts_rank{r}.pt, which the benchmarks replay.

Usage:
    # In a SLURM allocation. run_ds.sh sets up NCCL, LOGS and the python env, then runs
    # srun --ntasks-per-node=4 python -u scripts/ds_moe.py --config-pth configs/qwen_ep4.json
    bash scripts/run_ds.sh

    # Several modes on two nodes, and record the routing.
    CONFIG=configs/qwen_ep8.json RECORD=1 \
        MODES="autoep rbd two_lane tutel:capacity_factor=1" bash scripts/run_ds.sh

    # Every profiler.
    PROFILE="events torch flops comm" bash scripts/run_ds.sh
"""
import argparse
import contextlib
import gc
import json
import os
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import deepspeed
import deepspeed.comm as dist
import torch
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from torch.profiler import record_function
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

from benchmark.all2all.modes import MODES, Mode, build_mode, parse_mode
from spec_ep.phase_clock import Event, PhaseClock, Totals, totals, write_trace

from ds_guests import GuestDispatch, check_store, keep_remote_experts
from ds_utils import (
    add_expert_statistics_hook,
    count_params,
    end_flops_profile,
    print0,
    rank,
    world_size,
    report_peak_mem,
    save_torch_profile,
    shutdown_comm,
    start_flops_profile,
    torch_profile_ctx,
    watch_full_collections,
)

logs = Path(os.environ["LOGS"])
LAYER = "moe/layer"  # the span around MoE layer i is called moe/layer{i}
GUESTS = "rbd_guests"  # the one mode built here, not by benchmark/all2all/modes.py
Row = tuple[str, str, str, str]  # scope, step, category and link of one row of the events table


def load_model(model_id: str, dtype: torch.dtype) -> Qwen3_5MoeForCausalLM:
    """Load the text backbone on CPU and print its parameter counts.

    Returns:
        model: The model, not yet on any GPU.
    """
    print0("Loading model and tokenizer...")
    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=None,
        attn_implementation="sdpa",
        output_loading_info=True,
    )

    total, expert = count_params(model)
    print0(f"Full model: {total/1e9:.2f}B params ({expert/1e9:.2f}B in experts, "
           f"{(total-expert)/1e9:.2f}B dense)")

    assert model.config.model_type == "qwen3_5_moe_text"
    model.config.use_cache = False

    if loading_info["missing_keys"]:
        raise RuntimeError(
            f"Missing model weights: {loading_info['missing_keys']}"
        )
    return model


def init_deepspeed(
    model: Qwen3_5MoeForCausalLM,
    ds_config_pth: str,
) -> deepspeed.DeepSpeedEngine:
    """Wrap the model in a DeepSpeed engine, which applies AutoEP from the config.

    Returns:
        engine: The engine in eval mode.
    """
    with open(ds_config_pth, 'r') as file:
        ds_config = json.load(file)

    print0("Applying autoEP...")
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    engine.eval()
    return engine


def set_mode(engine: deepspeed.DeepSpeedEngine, mode: Mode | GuestDispatch | None) -> None:
    """Make every AutoEP layer dispatch and combine through mode; None is its own path.

    A GuestDispatch gives every layer its own function and holds two pools of guest
    weights on the GPU, which it gets here and gives back when another mode is set.
    """
    layers = [m for m in engine.module.modules() if isinstance(m, AutoEPMoELayer)]
    previous = getattr(getattr(layers[0].combine_dispatch, "func", None), "__self__", None)
    if isinstance(previous, GuestDispatch):
        previous.release()
    if isinstance(mode, GuestDispatch):
        mode.acquire()
    for i, layer in enumerate(layers):
        layer.combine_dispatch = mode.for_layer(i) if isinstance(mode, GuestDispatch) else mode


def get_batch(
    model_id: str,
    dev: torch.device,
    seq_len: int,
    sequences: int = 1,
    text_path: str = "data/short_text.txt",
    skip: int = 0,
) -> dict[str, torch.Tensor]:
    """Cut this rank's sequences of seq_len tokens from a text file, after its first skip tokens.

    Rank r gets the sequences * seq_len tokens that start at token
    skip + r * sequences * seq_len, so no two ranks share any.

    Returns:
        batch: input_ids and attention_mask, each [sequences, seq_len].
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = Path(text_path).read_text(encoding="utf-8")
    text = " ".join(text.split())

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    needed = sequences * seq_len
    strt, end = skip + rank * needed, skip + (rank + 1) * needed

    if tokens.input_ids[:, strt:end].shape[1] < needed:
        raise ValueError(f"Provide more text: at least {needed} tokens are required. "
                         f"[{rank}] has {tokens.input_ids[:, strt:end].shape[1]} seq.")

    batch = {
        "input_ids": tokens.input_ids[0, strt:end].view(sequences, seq_len).to(dev),
        "attention_mask": tokens.attention_mask[0, strt:end].view(sequences, seq_len).to(dev),
    }

    return batch


def record_expert_routing(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run one forward and return the experts every token on this rank chose.

    The hooks come off afterwards, so later forwards are timed without them.

    Returns:
        routing: [num_layers, num_tokens, k] int16 expert ids, on CPU.
    """
    sent = {}
    handles = add_expert_statistics_hook(engine.module, sent)
    engine(**batch, use_cache=False)
    for handle in handles:
        handle.remove()
    return torch.stack(list(sent.values()))


def timed_forward(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
    clock: PhaseClock | None = None,
) -> tuple[float, torch.Tensor]:
    """Time one forward between barriers, under clock if given. Collective.

    The garbage collector is off during the forward; see the module docstring.

    Returns:
        seconds: Wall time of the forward, the same on every rank up to the barriers.
        tail: [64, vocab] float32 logits of this rank's last 64 tokens.
    """
    torch.cuda.synchronize(engine.device)
    dist.barrier()
    gc.disable()
    start = time.time()
    with clock or contextlib.nullcontext(), record_function("forward"):
        logits = engine(**batch, use_cache=False).logits
    torch.cuda.synchronize(engine.device)
    dist.barrier()
    seconds = time.time() - start
    gc.enable()
    return seconds, logits[0, -64:].float()


@contextlib.contextmanager
def layer_spans(engine: deepspeed.DeepSpeedEngine) -> Iterator[None]:
    """Put every MoE layer's forward in a span moe/layer{i} for the duration.

    The span tells a layer from the model around it and names the layer. The hooks
    come off afterwards, so the timed forwards run without them.
    """
    spans = []
    layers = [m for m in engine.module.modules() if isinstance(m, AutoEPMoELayer)]
    handles = [
        hook for i, layer in enumerate(layers) for hook in (
            layer.register_forward_pre_hook(
                lambda *_, name=f"{LAYER}{i}": spans.append(record_function(name).__enter__())),
            layer.register_forward_hook(lambda *_: spans.pop().__exit__(None, None, None)),
        )
    ]
    yield
    for handle in handles:
        handle.remove()


def traced_forward(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
    clock: PhaseClock,
) -> tuple[float, list[Event]]:
    """Time one forward under the event clock, with every MoE layer in its span. Collective.

    Returns:
        seconds: Wall time of the forward, measured as timed_forward does.
        events: This rank's events of the forward.
    """
    with layer_spans(engine):
        seconds, _ = timed_forward(engine, batch, clock)
    return seconds, clock.events()


def scope(event: Event) -> tuple[str, int | None]:
    """Tell where in the model an event ran, from the spans open around it.

    Returns:
        scope: "moe" inside a MoE layer, "outside" in the rest of the model.
        layer: Index of that MoE layer, or None outside the layers.
    """
    layer = next((int(name[len(LAYER):]) for name in event.path if name.startswith(LAYER)), None)
    return ("outside" if layer is None else "moe"), layer


def summarize(events: list[Event]) -> tuple[list[Row], list[float]]:
    """Reduce one rank's events of one forward to the numbers the script prints.

    Returns:
        rows: The (scope, step, category, link) of every table row, in the order they
            first ran. The same on every rank.
        numbers: The Totals of the forward, of its MoE layers and of the rest, then
            every row's events, ms and overlapped ms.
    """
    scopes: dict[str, list[Event]] = {"moe": [], "outside": []}
    table: dict[Row, list[float]] = {}
    for event in events:
        where, _ = scope(event)
        scopes[where].append(event)
        step = LAYER if event.step.startswith(LAYER) else event.step
        cell = table.setdefault((where, step, event.category, event.link or "-"), [0, 0.0, 0.0])
        cell[0] += 1
        cell[1] += event.dur_ms
        cell[2] += event.overlap_ms
    numbers = [*totals(events), *totals(scopes["moe"]), *totals(scopes["outside"])]
    return list(table), numbers + [number for cell in table.values() for number in cell]


def rank_mean(numbers: list[float], device: torch.device) -> list[float]:
    """Average one forward's numbers over all ranks. Collective; every rank passes as many.

    Returns:
        mean: The numbers, averaged over the ranks.
    """
    total = torch.tensor(numbers, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(total)
    return (total / world_size).tolist()


def print_events(
    spec: str,
    timed: list[float],
    traced: list[float],
    rows: list[Row],
    numbers: list[float],
) -> None:
    """Print one mode's traced forward against its timed one, its totals and its table.

    Args:
        timed: Seconds of the mode's timed forwards, one per round.
        traced: Seconds of its traced forwards, one per round.
        rows: The table's rows, from summarize().
        numbers: What summarize() returns with them, as the mean over ranks and the
            median over rounds.
    """
    def text(t: Totals) -> str:
        return (f"comm {t.comm_ms:.1f} (intra {t.intra_ms:.1f}, inter {t.inter_ms:.1f}, "
                f"both {t.both_ms:.1f}), compute {t.compute_ms:.1f}, "
                f"overlapped {t.overlapped_ms:.1f} (comm+comm {t.comm_comm_ms:.1f}, "
                f"comm+compute {t.comm_compute_ms:.1f}), wall {t.wall_ms:.1f}")

    n = len(Totals._fields)
    forward, moe, outside = (text(Totals(*numbers[i * n : (i + 1) * n])) for i in range(3))
    fast, slow = statistics.median(timed), statistics.median(traced)
    print0(f"{spec}: traced {slow:.3f} s ({min(traced):.3f} to {max(traced):.3f}) against timed "
           f"{fast:.3f} s ({min(timed):.3f} to {max(timed):.3f}), {slow / fast - 1:+.1%} | "
           f"ms: {forward} | in MoE layers: {moe} | outside them: {outside}")
    print0("\n| scope | step | category | link | events | ms | overlapped ms | overlapped % |")
    print0("|---|---|---|---|---|---|---|---|")
    cells = numbers[3 * n :]
    for row, count, ms, overlap in zip(rows, cells[0::3], cells[1::3], cells[2::3]):
        print0(f"| {' | '.join(row)} | {count:.0f} | {ms:.2f} | {overlap:.2f} | "
               f"{100 * overlap / ms if ms else 0:.0f} |")
    print0("")


def run_pytorch_profile(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
    spec: str,
    trace_ranks: list[int],
) -> float:
    """Run one forward under torch.profiler and write its trace under LOGS/traces. Collective.

    Rank 0 and trace_ranks write {spec}_rank{r}.json, and rank 0 also writes
    {spec}_key_averages.txt. The trace is written before the next profiler starts: a
    profile exported after another one has run loses its GPU timestamps.

    Returns:
        seconds: Wall time of the profiled forward, measured as timed_forward does.
    """
    with torch_profile_ctx() as tprof:
        seconds, _ = timed_forward(engine, batch)
    if rank == 0 or rank in trace_ranks:
        save_torch_profile(tprof, logs / "traces", spec)
    return seconds


def run_ds_flops_profile(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
    outdir: Path,
) -> None:
    """Run one forward under DeepSpeed's flops profiler and save the table under outdir."""
    prof = start_flops_profile(engine.module)
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    end_flops_profile(prof, outdir)


def run_ds_comm_profile(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
) -> None:
    """Run one forward and print DeepSpeed's comms logger summary of it. Collective.

    Prints nothing unless the config sets "comms_logger": {"enabled": true}.
    """
    dist.comms_logger.reset_data()
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    dist.log_summary(show_straggler=True)


def parse_args() -> argparse.Namespace:
    """Parse the command line, ignoring flags the DeepSpeed launcher adds, such as --local_rank.

    Returns:
        args: The parsed flags.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config-pth", type=str, default="configs/qwen_ep8.json",
                        help="Path to the DeepSpeed config (e.g. configs/qwen_ep4.json)")
    parser.add_argument("--model", type=str, default="/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B",
                        help="Model directory; must be a qwen3_5_moe checkpoint")
    parser.add_argument("--seq-len", type=int, default=4096, help="Tokens per sequence")
    parser.add_argument("--sequences", type=int, default=1, help="Sequences per rank")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                        help="float32 shows whether two modes compute the same function: bf16 "
                             "rounding differs between merge kernels and grows through the layers")
    parser.add_argument("--modes", nargs="+", default=["autoep"], metavar="MODE[:KEY=VALUE,...]",
                        help="autoep for the layer's own path, or modes of benchmark/all2all/modes.py")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3, help="Timed rounds over the modes")
    parser.add_argument("--history-text", default="data/long_text.txt",
                        help="Text of the batch that rbd_guests:source=static takes its guests from")
    parser.add_argument("--history-skip", type=int, default=600_000,
                        help="Tokens of --history-text to skip: it starts with the timed text")
    parser.add_argument("--record", action="store_true",
                        help="Save the expert routing to LOGS/experts_rank{r}.pt before timing")
    parser.add_argument("--profile", nargs="*", choices=["events", "torch", "flops", "comm"],
                        default=["events", "torch"],
                        help="Profilers to run: events traces one forward per mode right after "
                             "each timed one, the others run one forward per mode afterwards")
    parser.add_argument("--trace-ranks", type=int, nargs="*", default=[],
                        help="Ranks besides 0 that write their torch trace")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """Load the model once, then check, time, trace and profile a forward under every mode."""
    args = parse_args()
    model_id, seq_len = args.model, args.seq_len
    full_collections = watch_full_collections()

    known = (*MODES, GUESTS)
    guest_settings = {
        spec: parse_mode(spec, known)[1] for spec in args.modes if parse_mode(spec, known)[0] == GUESTS
    }
    sources = {settings.get("source", "speculative") for settings in guest_settings.values()}

    model = load_model(model_id, getattr(torch, args.dtype))
    # Guests load from host memory, so the experts of other nodes are kept there.
    with keep_remote_experts(args.gpus_per_node) if guest_settings else contextlib.nullcontext() as store:
        engine = init_deepspeed(model, args.config_pth)
    batch = get_batch(model_id, engine.device, seq_len, args.sequences)

    layers = [m for m in engine.module.modules() if isinstance(m, AutoEPMoELayer)]
    layer = layers[0]
    if guest_settings:
        check_store(layers, store, args.gpus_per_node)
        print0(f"guest store: {len(store)} layers x {store[0][0].shape[0]} experts of other nodes "
               f"per rank, each checked against its owner")
    local_group, _ = torch.distributed.new_subgroups(group_size=args.gpus_per_node)
    width = max(map(len, args.modes))

    with torch.inference_mode():
        # Recorded routing: this batch's for --record and the oracle guests, another
        # batch's for the static guests.
        routing = {}
        if args.record or "oracle" in sources:
            routing["oracle"] = record_expert_routing(engine, batch)
        if args.record:
            torch.save(routing["oracle"], logs / f"experts_rank{rank}.pt")
        if "static" in sources:
            history = get_batch(model_id, engine.device, seq_len, args.sequences,
                                args.history_text, args.history_skip)
            routing["static"] = record_expert_routing(engine, history)

        modes = {}
        for spec in args.modes:
            if spec in guest_settings:
                settings = guest_settings[spec]
                modes[spec] = GuestDispatch(
                    layers, store, args.gpus_per_node, local_group, **settings,
                    routing=routing.get(settings.get("source", "speculative")),
                )
            elif spec != "autoep":
                modes[spec] = build_mode(
                    spec, layer.num_experts, args.gpus_per_node, local_group, layer.ep_group
                )
            else:
                modes[spec] = None

        # One untimed forward per mode: warms it up and checks its logits.
        first, reference = args.modes[0], None
        for spec, mode in modes.items():
            set_mode(engine, mode)
            torch.cuda.reset_peak_memory_stats(engine.device)
            seconds, tail = timed_forward(engine, batch)
            peak = torch.tensor(torch.cuda.max_memory_allocated(engine.device) / 2**30,
                                device=engine.device)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            reference = tail if reference is None else reference
            error = (tail - reference).abs().max() / reference.abs().max()
            torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX)
            same_token = (tail.argmax(-1) == reference.argmax(-1)).sum()
            torch.distributed.all_reduce(same_token)
            print0(f"{spec:>{width}}: first forward {seconds:7.3f} s | last logits differ from "
                   f"{first} by up to {error:.2e} (relative), same top token on "
                   f"{same_token.item() / (tail.shape[0] * world_size):.1%} of the last tokens "
                   f"| peak GPU memory {peak.item():.2f} GiB on the fullest rank")
            if isinstance(mode, GuestDispatch):
                mode.measure(lambda: engine(**batch, use_cache=False), print0, spec)

        # Rounds of one timed forward per mode, each followed by a traced one of the
        # same mode, so drift over the run favors neither the modes nor the tracing.
        collections = [len(full_collections)]        # full collections by the end of each stage
        calls = {spec: [] for spec in modes}         # seconds of every timed forward
        traced = {spec: [] for spec in modes}        # seconds of every traced forward
        numbers = {spec: [] for spec in modes}       # its numbers, mean over ranks
        rows, clock = {}, PhaseClock()
        for round in range(args.repeats):
            for spec, mode in modes.items():
                set_mode(engine, mode)
                calls[spec].append(timed_forward(engine, batch)[0])
                if "events" in args.profile:
                    seconds, events = traced_forward(engine, batch, clock)
                    traced[spec].append(seconds)
                    rows[spec], summary = summarize(events)
                    numbers[spec].append(rank_mean(summary, engine.device))
                    if round == args.repeats - 1:
                        write_trace(logs / "events" / f"{spec}_rank{rank}.json", events, rank,
                                    lambda event: dict(zip(("scope", "layer"), scope(event))))
        medians = {spec: statistics.median(s) for spec, s in calls.items()}
        print0(f"{args.repeats} rounds x 1 timed forward per mode, {args.sequences} x {seq_len} tokens per rank:")
        for spec, s in calls.items():
            print0(f"{spec:>{width}}: median {medians[spec]:7.3f} s | min {min(s):7.3f} | "
                   f"max {max(s):7.3f} | {medians[first] / medians[spec]:.2f}x vs {first}")
        if "events" in args.profile:
            print0(f"{args.repeats} rounds x 1 traced forward per mode, each right after the timed "
                   f"one. Events on the GPU clock, in ms, mean over ranks and median over rounds "
                   f"of every number, so the medians add up only roughly. "
                   f"wall = comm + compute - overlapped.")
            for spec in modes:
                middle = [statistics.median(column) for column in zip(*numbers[spec])]
                print_events(spec, calls[spec], traced[spec], rows[spec], middle)

        collections.append(len(full_collections))
        for spec, mode in modes.items():
            set_mode(engine, mode)
            if "torch" in args.profile:
                seconds = run_pytorch_profile(engine, batch, spec, args.trace_ranks)
                print0(f"{spec:>{width}}: profiled {seconds:7.3f} s "
                       f"({seconds / medians[spec] - 1:+.1%} vs timed)")
            if "flops" in args.profile:
                (logs / spec).mkdir(parents=True, exist_ok=True)
                run_ds_flops_profile(engine, batch, logs / spec)
            if "comm" in args.profile:
                run_ds_comm_profile(engine, batch)

    collections = torch.tensor(collections + [len(full_collections)], device=engine.device)
    collecting = (collections[-1:] > 0).long()       # 1 on a rank that ran a full collection
    longest = torch.tensor(max(full_collections, default=0.0), device=engine.device)
    torch.distributed.all_reduce(collections)
    torch.distributed.all_reduce(collecting)
    torch.distributed.all_reduce(longest, op=torch.distributed.ReduceOp.MAX)
    checks, rounds, profiled = collections.diff(prepend=collections.new_zeros(1)).tolist()
    print0(f"full garbage collections over all ranks: {checks} up to the first forwards, "
           f"{rounds} in the timed and traced rounds, {profiled} in the profiles, on "
           f"{collecting.item()} of {world_size} ranks; longest {longest.item():.3f} s. None can "
           f"run inside a measured forward.")
    report_peak_mem(engine.device)
    shutdown_comm()
    # Tutel keeps its CUDA events in a static vector that is destroyed after CUDA has
    # shut down, which aborts the process at exit once its pipeline has run. Everything
    # is written by now, so leave without running static destructors.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
