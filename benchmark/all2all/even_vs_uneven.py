"""Benchmark one all2all with the recorded uneven split sizes against even ones.

Replays one layer of recorded routing as a single all2all: one row per (token,
expert) pair, sent to the expert's rank. The even run sends the same number of rows
from each rank, split equally across the ranks. Row values are random, so only the
split sizes come from the recording.

Usage:
    srun -N 2 --ntasks-per-node=4 \
        python -u benchmark/all2all/even_vs_uneven.py logs/2026_oct/2026-10-16_14-07-50 0
"""
import sys

import torch
import torch.distributed as dist

from benchmark.utils import init_dist, load_routing, time_ms

E, H = 256, 2048
WARMUP = 3
ITERATIONS = 5
SCALE = 1


def recorded_splits(route: torch.Tensor, world: int) -> tuple[list[int], list[int]]:
    """Count the rows this rank exchanges with each rank under the recorded routing.

    Collective. Expert e lives on rank e // (E // world).

    Returns:
        send: rows sent to each rank.
        recv: rows received from each rank.
    """
    send = torch.bincount(route.flatten() // (E // world), minlength=world) * SCALE
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return send.tolist(), recv.tolist()


def benchmark(name: str, send: list[int], recv: list[int], device: torch.device) -> None:
    """Time one all2all with the given split sizes and print its throughput. Collective."""
    rank, world = dist.get_rank(), dist.get_world_size()
    x = torch.randn(sum(send), H, device=device, dtype=torch.bfloat16)
    y = torch.empty(sum(recv), H, device=device, dtype=x.dtype)
    times = time_ms(lambda: dist.all_to_all_single(y, x, recv, send), WARMUP, ITERATIONS)
    ms = sum(times) / len(times)

    if rank == 0:
        total = (sum(send) - send[rank]) * H * x.element_size() * world
        agg_gbs = total / (ms / 1000) / 1e9
        print(
            f"{name}: {ms:.3f} ms | "
            f"{total / 2**20:.1f} MiB off-rank | "
            f"{agg_gbs:.2f} GB/s aggregate | "
            f"{agg_gbs / world:.2f} GB/s per GPU"
        )


if __name__ == "__main__":
    run_dir, layer = sys.argv[1], int(sys.argv[2])
    rank, world, device = init_dist()

    route = load_routing(run_dir, layer, rank, world, E).to(device)
    benchmark("uneven", *recorded_splits(route, world), device)

    even = [route.numel() * SCALE // world] * world
    benchmark("even", even, even, device)

    dist.destroy_process_group()
