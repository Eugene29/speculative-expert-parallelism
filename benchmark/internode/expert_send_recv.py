"""Benchmark of send/recv of MoE expert weights between two ranks.

Rank 0 sends N experts to rank 1, intra-node (NVLink) or inter-node
(Slingshot), and reports the message size and time per transfer.

The expert architecture assumes a gated FFNN (e.g. SwiGLU), as in
Qwen3.6-35B-A3B: each expert holds gate and up projections [I, H] and a
down projection [H, I], so 3 * H * I parameters. Defaults are taken from
its config (hidden_size=2048, moe_intermediate_size=512), giving 6 MiB per
expert in bf16. Experts are sent as one flat buffer.

Usage:
    # intra-node: 2 GPUs on one node
    srun -N 1 --ntasks-per-node=2 python -u benchmark/internode/expert_send_recv.py --intra-node
    or
    srun -N 1 -w "$MASTER_ADDR" --ntasks-per-node=2 --gpus=4 python -u benchmark/internode/expert_send_recv.py --intra-node

    # inter-node: 1 GPU on each of two nodes
    srun -N 2 --ntasks-per-node=1 python -u benchmark/internode/expert_send_recv.py --inter-node
"""
import argparse
import socket

import torch
import torch.distributed as dist

from benchmark.utils import init_dist, time_ms


def check_placement(inter_node: bool) -> list[str]:
    """Check rank 0 and rank 1 are on different nodes if inter_node, else the same node.

    Assumes exactly 2 ranks.

    Returns:
        hosts: hostnames of rank 0 and rank 1.
    """
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"Expected 2 ranks, got {world}.")

    hosts = [None, None]
    dist.all_gather_object(hosts, socket.gethostname())

    if inter_node and hosts[0] == hosts[1]:
        raise RuntimeError(f"--inter-node needs ranks on two nodes, got {hosts}. Use srun -N 2.")
    if not inter_node and hosts[0] != hosts[1]:
        raise RuntimeError(f"--intra-node needs ranks on one node, got {hosts}. Use srun -N 1.")
    return hosts


def benchmark(
    n: int,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, list[float]]:
    """Send n experts from rank 0 to rank 1 in one flat buffer.

    Returns:
        nbytes: message size in bytes.
        times: ms of each timed transfer, the slower of the two ranks.
    """
    numel = n * 3 * args.hidden * args.intermediate
    buf = torch.randn(numel, device=device, dtype=dtype)

    def transfer() -> None:
        """Send on rank 0, receive on rank 1."""
        if rank == 0:
            dist.send(buf, dst=1)
        else:
            dist.recv(buf, src=0)

    times = time_ms(transfer, args.warmup, args.iterations)
    return buf.numel() * buf.element_size(), times


def main(args: argparse.Namespace) -> None:
    """Run the benchmark for each expert count and print results on rank 0."""
    rank, world, device = init_dist()
    hosts = check_placement(args.inter_node)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        placement = "inter-node" if args.inter_node else "intra-node"
        print(f"{placement}: {hosts[0]} -> {hosts[1]} | {args.dtype}", flush=True)

    for n in args.experts:
        nbytes, times = benchmark(n, args, rank, device, dtype)
        if rank == 0:
            avg = sum(times) / len(times)
            print(
                f"{n} experts: {nbytes / 2**20:.1f} MiB | "
                f"min {min(times):.3f} / avg {avg:.3f} / max {max(times):.3f} ms | "
                f"{nbytes / (avg / 1000) / 1e9:.2f} GB/s avg",
                flush=True,
            )
            print(f"  all ms: {', '.join(f'{t:.3f}' for t in times)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    placement = parser.add_mutually_exclusive_group(required=True)
    placement.add_argument("--intra-node", action="store_true",
                           help="Both ranks on one node")
    placement.add_argument("--inter-node", action="store_true",
                           help="Ranks on two different nodes")
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Number of experts per message")
    parser.add_argument("--hidden", type=int, default=2048, help="hidden_size")
    parser.add_argument("--intermediate", type=int, default=512,
                        help="moe_intermediate_size")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    main(parser.parse_args())
