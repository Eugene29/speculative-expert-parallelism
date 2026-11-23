"""How much an all2all and expert compute slow each other when they share a GPU.

Pipelining runs the experts of one slice while the next slice is on the network. That
hides the experts only if the two do not slow each other. NCCL moves rows with kernels
of its own on the same GPU, so they compete with the experts' GEMMs. This script times,
on every rank at once:

    comm: one all2all of --rows rows of --hidden bf16 values per rank, split evenly
        over all ranks, alone.
    experts: one gated feed-forward block on --expert-rows rows, alone.
    both: the all2all started without waiting, the experts on the main stream, then
        the wait. With no contention this takes max(comm, experts).

It prints, per --rows: the three times, the experts' own time inside `both`, the time
`both` hides (comm + experts - both) and that as a share of the most it could hide
(min(comm, experts)). Each time is the slowest rank of a call, median over calls.

Usage:
    srun -N 2 --ntasks-per-node=4 python -u -m benchmark.all2all.contention
    srun -N 2 --ntasks-per-node=4 python -u -m benchmark.all2all.contention \
        --rows 8192 16384 32768 --expert-rows 32768 --intermediate 512
"""
import argparse
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist

from benchmark.all2all.combine_dispatch import make_ffn
from benchmark.utils import check_node_layout, init_dist
from spec_ep.ops import all2all, all2all_async


def timed(
    fn: Callable[[], torch.cuda.Event | None],
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Time fn on the GPU clock of every rank. Collective.

    fn may return a CUDA event it recorded part way; the time up to it is reported too.

    Returns:
        total: Milliseconds of a call on its slowest rank, median over the calls.
        part: The same for the time up to fn's event, or nan if it returned none.
    """
    marks = []
    for i in range(warmup + iterations):
        dist.barrier()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        middle = fn()
        end.record()
        if i >= warmup:
            marks.append((start, middle, end))
    torch.cuda.synchronize()
    times = torch.tensor(
        [[s.elapsed_time(e), s.elapsed_time(m) if m is not None else float("nan")] for s, m, e in marks],
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    total, part = times.T.tolist()
    return statistics.median(total), statistics.median(part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[8192, 16384, 32768, 65536],
                        help="Rows per rank of the all2all")
    parser.add_argument("--expert-rows", type=int, default=32768, help="Rows the experts run on")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--intermediate", type=int, default=512, help="I, width inside the experts")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    rank, world, device = init_dist()
    hosts = check_node_layout(args.gpus_per_node, min_nodes=1)
    ffn = make_ffn(args.hidden, args.intermediate, device)
    inputs = torch.randn(args.expert_rows, args.hidden, device=device, dtype=torch.bfloat16)
    run = lambda fn: timed(fn, args.warmup, args.iterations)

    def experts() -> None:
        ffn(inputs, None)

    experts_ms, _ = run(experts)
    if rank == 0:
        print(f"{world // args.gpus_per_node} nodes x {args.gpus_per_node} GPUs: "
              f"{', '.join(hosts[::args.gpus_per_node])} | experts on {args.expert_rows} rows "
              f"alone: {experts_ms:.3f} ms | median of {args.iterations}, slowest rank", flush=True)

    for num_rows in args.rows:
        even = [num_rows // world] * world
        rows = torch.randn(sum(even), args.hidden, device=device, dtype=torch.bfloat16)
        received = torch.empty_like(rows)

        def both() -> torch.cuda.Event:
            wait = all2all_async(rows, even, even, out=received)
            ffn(inputs, None)
            experts_done = torch.cuda.Event(enable_timing=True)
            experts_done.record()
            wait()
            return experts_done

        def comm() -> None:
            all2all(rows, even, even, out=received)

        comm_ms, _ = run(comm)
        both_ms, experts_in_both = run(both)
        hidden = comm_ms + experts_ms - both_ms
        if rank == 0:
            print(f"all2all of {num_rows:>6} rows ({rows.numel() * 2 / 2**20:4.0f} MiB): comm "
                  f"{comm_ms:7.3f} ms ({comm_ms / experts_ms:.1f}x the experts) | both "
                  f"{both_ms:7.3f} ms | experts inside both {experts_in_both:6.3f} ms "
                  f"({experts_in_both / experts_ms:.2f}x alone) | hides {hidden:6.3f} ms, "
                  f"{hidden / min(comm_ms, experts_ms):.0%} of the most it could", flush=True)

    dist.barrier()
    dist.destroy_process_group()
