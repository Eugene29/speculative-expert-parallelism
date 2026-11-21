"""Benchmark of all-to-all of MoE expert weights across nodes.

Each GPU splits a buffer of N experts evenly across the other nodes and sends
one chunk to the GPU with the same local rank on each of them, receiving a
chunk from each in return. Every rank sends and receives the same N experts'
worth of bytes as the ring in expert_send_recv_parallel.py, so the two are
directly comparable. With 2 nodes the all-to-all is the same bidirectional
swap as the ring.

--balance tests whether an uneven all2all is slower than an even one between
nodes. The load is uneven either across nodes (uneven_internode) or across the
GPUs of every node (uneven_intranode).

The expert architecture assumes a gated FFNN (e.g. SwiGLU), as in
Qwen3.6-35B-A3B: each expert holds gate and up projections [I, H] and a
down projection [H, I], so 3 * H * I parameters. Defaults are taken from
its config (hidden_size=2048, moe_intermediate_size=512), giving 6 MiB per
expert in bf16. Experts are sent as one flat buffer, split by element, so a
chunk may hold part of an expert.

Usage:
    # 2 nodes: each GPU swaps with its counterpart on the other node
    srun -N 2 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py

    # 4 nodes: each GPU sends a third of its buffer to each of the other 3 nodes
    srun -N 4 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py

    # 4 nodes, uneven: nodes 0-1 send 1.5N experts, nodes 2-3 send 0.5N
    srun -N 4 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py --balance uneven_internode
"""
import argparse
import os

import torch
import torch.distributed as dist

from benchmark.utils import check_node_layout, init_dist, time_ms


def get_splits(
    numel: int,
    rank: int,
    gpus_per_node: int,
    balance: str,
    load_delta: float,
) -> tuple[list[int], list[int]]:
    """Elements this rank sends to and receives from every rank.

    Each rank sends numel scaled by 1 (even), or by 1 +/- load_delta for the
    first / second half of nodes (uneven_internode) or of local ranks on every
    node (uneven_intranode), split evenly across all ranks on other nodes.
    Nothing goes to ranks on the same node, so every byte crosses the network.

    Returns:
        send_splits: elements this rank sends to each rank.
        recv_splits: elements this rank receives from each rank.
    """
    world = dist.get_world_size()
    num_nodes = world // gpus_per_node
    num_remote = world - gpus_per_node
    ranks = range(world)

    if balance == "even":
        load_delta, is_heavy = 0.0, [True] * world
    elif balance == "uneven_internode":
        if num_nodes % 2 != 0:
            raise RuntimeError(f"uneven_internode needs an even number of nodes, got {num_nodes}.")
        is_heavy = [r // gpus_per_node < num_nodes // 2 for r in ranks]
    elif balance == "uneven_intranode":
        if gpus_per_node % 2 != 0:
            raise RuntimeError(f"uneven_intranode needs an even number of GPUs per node, "
                               f"got {gpus_per_node}.")
        is_heavy = [r % gpus_per_node < gpus_per_node // 2 for r in ranks]

    heavy_chunk = numel * (1 + load_delta) / num_remote
    light_chunk = numel * (1 - load_delta) / num_remote
    if heavy_chunk != int(heavy_chunk) or light_chunk != int(light_chunk):
        raise RuntimeError(f"numel={numel} with load_delta={load_delta} doesn't split "
                           f"evenly across {num_remote} ranks.")
    chunks = [int(heavy_chunk if heavy else light_chunk) for heavy in is_heavy]

    other_node = [r // gpus_per_node != rank // gpus_per_node for r in ranks]
    send_splits = [chunks[rank] if other else 0 for other in other_node]
    recv_splits = [chunks[r] if other else 0 for r, other in enumerate(other_node)]
    return send_splits, recv_splits


def benchmark(
    n: int,
    args: argparse.Namespace,
    rank: int,
    gpus_per_node: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, list[float]]:
    """Every rank sends its experts split across all ranks on other nodes at once.

    Returns:
        nbytes: average bytes each rank sends in total.
        times: ms of each timed all-to-all, the slowest of all ranks.
    """
    numel = n * 3 * args.hidden * args.intermediate
    send_splits, recv_splits = get_splits(numel, rank, gpus_per_node,
                                          args.balance, args.load_delta)

    send_buf = torch.randn(sum(send_splits), device=device, dtype=dtype)
    recv_buf = torch.empty(sum(recv_splits), device=device, dtype=dtype)

    times = time_ms(
        lambda: dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits),
        args.warmup,
        args.iterations,
    )
    return numel * send_buf.element_size(), times


def main(args: argparse.Namespace) -> None:
    """Run the all-to-all benchmark for each expert count and print results on rank 0."""
    rank, world, device = init_dist()
    gpus_per_node = world // int(os.environ["SLURM_NNODES"])
    hosts = check_node_layout(gpus_per_node)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        node_hosts = hosts[::gpus_per_node]
        print(
            f"all-to-all of {len(node_hosts)} nodes x {gpus_per_node} GPUs: "
            f"{', '.join(node_hosts)} | {args.dtype} | balance={args.balance}",
            flush=True,
        )

    for n in args.experts:
        nbytes, times = benchmark(n, args, rank, gpus_per_node, device, dtype)
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
    parser.add_argument("--balance", choices=["even", "uneven_internode", "uneven_intranode"],
                        default="even",
                        help="even: every rank sends n experts; uneven_internode: ranks on "
                             "the first half of nodes are heavy, the rest light; "
                             "uneven_intranode: the first half of local ranks on every "
                             "node are heavy, the rest light")
    parser.add_argument("--load-delta", type=float, default=0.5,
                        help="Heavy ranks send n * (1 + load_delta) experts, "
                             "light ranks n * (1 - load_delta)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    main(parser.parse_args())
