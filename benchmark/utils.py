"""Shared helpers for the benchmarks: launch, recorded routing, inputs and timing.

Usage:
    from benchmark.utils import init_dist, check_node_layout, load_routing, make_inputs, time_ms
"""
import os
import socket
from collections.abc import Callable

import torch
import torch.distributed as dist


def init_dist() -> tuple[int, int, torch.device]:
    """Init NCCL from SLURM env.

    Returns:
        rank: this rank.
        world: number of ranks.
        device: this rank's GPU.
    """
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    return rank, world, device


def check_node_layout(gpus_per_node: int) -> list[str]:
    """Check ranks fill at least 2 nodes in blocks of gpus_per_node.

    Block order (ranks 0-3 on node 0, 4-7 on node 1, ...) is what makes
    rank + gpus_per_node the same local rank on a different node.

    Returns:
        hosts: hostname of every rank.
    """
    world = dist.get_world_size()
    if world % gpus_per_node != 0:
        raise RuntimeError(f"{world} ranks is not a multiple of {gpus_per_node} per node.")

    num_nodes = world // gpus_per_node
    if num_nodes < 2:
        raise RuntimeError(f"Expected at least 2 nodes, got {num_nodes}. Use srun -N 2 or more.")

    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())

    # Every node should be a different host.
    node_hosts = hosts[::gpus_per_node]
    if len(set(node_hosts)) != num_nodes:
        raise RuntimeError(f"Some nodes share a host: {node_hosts}.")
    return hosts


def load_routing(
    run_dir: str,
    layer: int,
    rank: int,
    world: int,
    num_experts: int,
) -> torch.Tensor:
    """Load this rank's share of one layer's recorded routing.

    run_dir/experts_rank{r}.pt holds a [num_layers, num_tokens, k] tensor of expert
    ids for recorded rank r, written by scripts/ds_moe.py. With more ranks than
    recordings, each recording's tokens are split across world // recordings ranks, in
    rank order, so the global workload stays the same.

    Returns:
        route: [num_tokens, k] expert ids of this rank's tokens, as int64.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    if recorded == 0 or world % recorded != 0:
        raise RuntimeError(f"{world} ranks is not a multiple of {recorded} recordings "
                           f"in {run_dir}.")

    shards = world // recorded
    route = torch.load(f"{run_dir}/experts_rank{rank // shards}.pt", map_location="cpu")
    route = route[layer].tensor_split(shards)[rank % shards].long()

    if route.numel() > 0 and (route.min() < 0 or route.max() >= num_experts):
        raise RuntimeError(f"Recorded expert ids span {route.min()}..{route.max()}, "
                           f"outside 0..{num_experts - 1}.")
    return route


def make_inputs(
    route: torch.Tensor,
    H: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make random tokens and weights for the given routing.

    The recording holds only expert ids, so the token values are made up; they don't
    affect communication cost.

    Returns:
        tokens: [num_tokens, H] bf16 tokens.
        weights: [num_tokens, k] weights, each row summing to 1.
    """
    num_tokens, k = route.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.randn(num_tokens, H, generator=gen, device=device, dtype=torch.bfloat16)
    # Softmax over each token's k experts, like a router renormalizing its top-k gates.
    # Kept in float32 so rounding doesn't make RBD and plain outputs differ.
    logits = torch.randn(num_tokens, k, generator=gen, device=device)
    weights = logits.softmax(dim=-1)
    return tokens, weights


def time_ms(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    """Time fn on every rank and report the slowest rank's average. Collective.

    Runs warmup untimed calls, then iterations timed ones.

    Returns:
        ms: Average milliseconds per call on the slowest rank.
    """
    # Warmup also builds the NCCL communicators each group creates on first use.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        # Line all ranks up so each call is timed on its own.
        dist.barrier()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    # A rank's call isn't done until its slowest peer finishes, so take the max rank.
    avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iterations
    ms = torch.tensor(avg, device=torch.cuda.current_device())
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return ms.item()
