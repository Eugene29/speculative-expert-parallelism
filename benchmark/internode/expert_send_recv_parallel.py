"""Benchmark of parallel send/recv of MoE expert weights in a ring across nodes.

Each GPU sends N experts to the GPU with the same local rank on the next node
and receives N experts from the previous node (node i -> i+1, last -> 0). With
2 nodes the ring is a bidirectional swap between the two nodes.

This measures whether concurrent transfers raise the aggregate inter-node
throughput, or leave it the same because the ranks share the network.

The expert architecture assumes a gated FFNN (e.g. SwiGLU), as in
Qwen3.6-35B-A3B: each expert holds gate and up projections [I, H] and a
down projection [H, I], so 3 * H * I parameters. Defaults are taken from
its config (hidden_size=2048, moe_intermediate_size=512), giving 6 MiB per
expert in bf16. Experts are sent as one flat buffer.

Usage:
    # 2 nodes: each GPU swaps with its counterpart on the other node
    srun -N 2 --ntasks-per-node=4 python -u benchmark/internode/expert_send_recv_parallel.py

    # 4 nodes: each GPU sends to the next node and receives from the previous
    srun -N 4 --ntasks-per-node=4 python -u benchmark/internode/expert_send_recv_parallel.py
"""
import argparse
import os

import torch
import torch.distributed as dist

from benchmark.utils import check_node_layout, init_dist, time_ms


def get_ring_peers(rank: int, world: int, gpus_per_node: int) -> tuple[int, int]:
    """Pair this rank with the same local rank on the next and previous node.

    Returns:
        next_rank: rank to send to.
        prev_rank: rank to receive from.
    """
    next_rank = (rank + gpus_per_node) % world
    prev_rank = (rank - gpus_per_node) % world
    return next_rank, prev_rank


def benchmark(
    n: int,
    args: argparse.Namespace,
    next_rank: int,
    prev_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, list[float]]:
    """Every rank sends n experts to next_rank and receives n from prev_rank at once.

    Returns:
        nbytes: message size per rank in bytes.
        times: ms of each timed transfer, the slowest of all ranks.
    """
    numel = n * 3 * args.hidden * args.intermediate
    send_buf = torch.randn(numel, device=device, dtype=dtype)
    recv_buf = torch.empty(numel, device=device, dtype=dtype)

    # One batch puts the send and recv in one NCCL group, so the ring can't deadlock.
    ops = [
        dist.P2POp(dist.isend, send_buf, next_rank),
        dist.P2POp(dist.irecv, recv_buf, prev_rank),
    ]

    def transfer() -> None:
        """Send to next_rank, receive from prev_rank, and wait for both."""
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    times = time_ms(transfer, args.warmup, args.iterations)
    return send_buf.numel() * send_buf.element_size(), times


def main(args: argparse.Namespace) -> None:
    """Run the ring benchmark for each expert count and print results on rank 0."""
    rank, world, device = init_dist()
    gpus_per_node = world // int(os.environ["SLURM_NNODES"])
    hosts = check_node_layout(gpus_per_node)
    next_rank, prev_rank = get_ring_peers(rank, world, gpus_per_node)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        node_hosts = hosts[::gpus_per_node]
        print(
            f"ring of {len(node_hosts)} nodes x {gpus_per_node} GPUs: "
            f"{' -> '.join(node_hosts)} -> {node_hosts[0]} | {args.dtype}",
            flush=True,
        )

    for n in args.experts:
        nbytes, times = benchmark(n, args, next_rank, prev_rank, device, dtype)
        if rank == 0:
            avg = sum(times) / len(times)
            per_rank_gbs = nbytes / (avg / 1000) / 1e9
            print(
                f"{n} experts: {nbytes / 2**20:.1f} MiB per rank | "
                f"min {min(times):.3f} / avg {avg:.3f} / max {max(times):.3f} ms | "
                f"{per_rank_gbs:.2f} GB/s per rank | "
                f"{per_rank_gbs * world:.2f} GB/s aggregate",
                flush=True,
            )
            print(f"  all ms: {', '.join(f'{t:.3f}' for t in times)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Number of experts per message")
    parser.add_argument("--hidden", type=int, default=2048, help="hidden_size")
    parser.add_argument("--intermediate", type=int, default=512,
                        help="moe_intermediate_size")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    main(parser.parse_args())
