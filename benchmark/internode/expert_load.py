"""Benchmark of loading MoE expert weights onto a GPU: host memory against other GPUs.

A rank that wants experts it does not hold can copy them up from its node's host
memory (offloading) or receive them from a GPU that holds them (prefetching). Every
rank loads N experts at the same time, as a layer's ranks would, from one of:

    host_pageable: ordinary host memory of its own node.
    host_pinned: page-locked host memory of its own node.
    intra_node: the next GPU of its own node, over a NCCL communicator of the node.
    intra_node_world: the same, over the NCCL communicator of all ranks.
    inter_node: the GPU with the same local rank on the next node, over NCCL.

Each source is timed alone. With --busy-rows R it is timed again while an all2all of
R rows of --hidden bf16 values per rank runs across all ranks on a second NCCL
communicator, which is what a load competes with when it overlaps a layer's dispatch.
That run reports the time until both have finished, next to the all2all alone.

The expert architecture assumes a gated FFNN (e.g. SwiGLU), as in
Qwen3.6-35B-A3B: each expert holds gate and up projections [I, H] and a
down projection [H, I], so 3 * H * I parameters. Defaults are taken from
its config (hidden_size=2048, moe_intermediate_size=512), giving 6 MiB per
expert in bf16. Experts move as one flat buffer.

Usage:
    srun -N 2 --ntasks-per-node=4 python -u -m benchmark.internode.expert_load
    srun -N 2 --ntasks-per-node=4 python -u -m benchmark.internode.expert_load \
        --experts 1 8 32 --busy-rows 32768
"""
import argparse
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist

from benchmark.utils import check_node_layout, init_dist, time_ms
from spec_ep.ops import all2all_async

SOURCES = ("host_pageable", "host_pinned", "intra_node", "intra_node_world", "inter_node")


def make_load(
    source: str,
    numel: int,
    rank: int,
    world: int,
    gpus_per_node: int,
    device: torch.device,
    local_group: dist.ProcessGroup,
) -> Callable[[], None]:
    """Build the function that loads one flat buffer of numel bf16 values on this rank.

    For the two GPU sources every rank also sends its own buffer to the rank that
    loads from it, in a ring, so all ranks load at once.

    Returns:
        load: Loads the buffer and returns once it is on this rank's GPU.
    """
    target = torch.empty(numel, device=device, dtype=torch.bfloat16)
    if source.startswith("host"):
        held = torch.ones(numel, dtype=torch.bfloat16)
        if source == "host_pinned":
            held = held.pin_memory()

        def load() -> None:
            target.copy_(held, non_blocking=True)
            torch.cuda.synchronize()

        return load

    held = torch.randn(numel, device=device, dtype=torch.bfloat16)
    group = None
    if source == "inter_node":
        holder, loader = (rank + gpus_per_node) % world, (rank - gpus_per_node) % world
    else:
        group = local_group if source == "intra_node" else None
        node_first = rank // gpus_per_node * gpus_per_node
        holder = node_first + (rank + 1) % gpus_per_node
        loader = node_first + (rank - 1) % gpus_per_node
    ops = [
        dist.P2POp(dist.isend, held, loader, group),
        dist.P2POp(dist.irecv, target, holder, group),
    ]

    def load() -> None:
        for request in dist.batch_isend_irecv(ops):
            request.wait()

    return load


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 4, 16, 64],
                        help="Experts per load")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--hidden", type=int, default=2048, help="hidden_size")
    parser.add_argument("--intermediate", type=int, default=512, help="moe_intermediate_size")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--busy-rows", type=int, default=0,
                        help="Rows per rank of an all2all that runs during the load")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    rank, world, device = init_dist()
    G = args.gpus_per_node
    hosts = check_node_layout(G, min_nodes=1)
    sources = [s for s in args.sources if world > G or s != "inter_node"]
    local_group, _ = dist.new_subgroups(group_size=G)
    median = lambda fn: statistics.median(time_ms(fn, args.warmup, args.iterations))

    if rank == 0:
        print(f"{world // G} nodes x {G} GPUs: {', '.join(hosts[::G])} | every rank loads "
              f"at once | median of {args.iterations}, slowest rank", flush=True)

    if args.busy_rows:
        busy_group = dist.new_group(list(range(world)))
        even = [args.busy_rows // world] * world
        rows = torch.randn(sum(even), args.hidden, device=device, dtype=torch.bfloat16)
        received = torch.empty_like(rows)
        start_all2all = lambda: all2all_async(rows, even, even, busy_group, received)
        busy_ms = median(lambda: start_all2all()())
        if rank == 0:
            print(f"all2all alone: {rows.numel() * 2 / 2**20:.0f} MiB per rank in "
                  f"{busy_ms:.3f} ms", flush=True)

    for n in args.experts:
        numel = n * 3 * args.hidden * args.intermediate
        for source in sources:
            load = make_load(source, numel, rank, world, G, device, local_group)
            alone = median(load)
            line = (f"{n:>3} experts {numel * 2 / 2**20:6.0f} MiB | {source:>13}: "
                    f"{alone:8.3f} ms, {numel * 2 / alone / 1e6:6.2f} GB/s per rank")
            if args.busy_rows:
                def both() -> None:
                    wait = start_all2all()
                    load()
                    wait()

                together = median(both)
                line += (f" | with all2all {together:8.3f} ms, "
                         f"{together - busy_ms:+.3f} over the all2all alone")
            if rank == 0:
                print(line, flush=True)

    dist.barrier()
    dist.destroy_process_group()
