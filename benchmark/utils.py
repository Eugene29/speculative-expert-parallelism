"""Shared helpers for the benchmarks: launch, recorded routing, inputs and timing.

Usage:
    from benchmark.utils import init_dist, check_node_layout, load_routing, make_inputs, time_ms
    from benchmark.utils import expert_counts, load_sample
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


def check_node_layout(gpus_per_node: int, min_nodes: int = 2) -> list[str]:
    """Check ranks fill at least min_nodes nodes in blocks of gpus_per_node.

    Block order (ranks 0-3 on node 0, 4-7 on node 1, ...) is what makes
    rank + gpus_per_node the same local rank on a different node.

    Returns:
        hosts: hostname of every rank.
    """
    world = dist.get_world_size()
    if world % gpus_per_node != 0:
        raise RuntimeError(f"{world} ranks is not a multiple of {gpus_per_node} per node.")

    num_nodes = world // gpus_per_node
    if num_nodes < min_nodes:
        raise RuntimeError(f"Expected at least {min_nodes} nodes, got {num_nodes}. "
                           f"Use srun -N {min_nodes} or more.")

    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())

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
    ids for recorded rank r, written by scripts/ds_moe.py --record. With more ranks than
    recordings, each recording's tokens are split across world // recordings ranks, in
    rank order. With fewer, each rank replays recordings // world recordings back to
    back. Either way the global workload stays the same.

    Returns:
        route: [num_tokens, k] expert ids of this rank's tokens, as int64.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    if recorded == 0 or (world % recorded != 0 and recorded % world != 0):
        raise RuntimeError(f"{world} ranks and {recorded} recordings in {run_dir} "
                           f"don't divide each other.")

    shards, merged = max(world // recorded, 1), max(recorded // world, 1)
    first = rank // shards * merged
    route = torch.cat([
        torch.load(f"{run_dir}/experts_rank{r}.pt", map_location="cpu")[layer]
        for r in range(first, first + merged)
    ])
    route = route.tensor_split(shards)[rank % shards].long()

    if route.numel() > 0 and (route.min() < 0 or route.max() >= num_experts):
        raise RuntimeError(f"Recorded expert ids span {route.min()}..{route.max()}, "
                           f"outside 0..{num_experts - 1}.")
    return route


def expert_counts(run_dir: str, num_experts: int) -> torch.Tensor:
    """Count the copies every expert gets in each recording and layer.

    A recording is one batch of tokens: run_dir/experts_rank{r}.pt, as load_routing
    reads it.

    Returns:
        counts: [num_recordings, num_layers, num_experts] copies routed to each
            expert, as int64.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    counts = []
    for r in range(recorded):
        route = torch.load(f"{run_dir}/experts_rank{r}.pt", map_location="cpu").long()
        layer = torch.arange(route.shape[0]).view(-1, 1, 1)
        flat = (layer * num_experts + route).flatten()
        counts.append(flat.bincount(minlength=route.shape[0] * num_experts).view(-1, num_experts))
    return torch.stack(counts)


def load_sample(
    run_dir: str,
    layer: int,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one layer's routing of every recording, with the node each token starts on.

    The recordings go to the nodes in blocks, in file order, as load_routing() hands
    them to the ranks. For the run a benchmark replays this is therefore the routing
    it times, and for any other recording it is a sample from other batches.

    Returns:
        route: [num_tokens, k] expert ids of all recordings, as int64.
        source_node: [num_tokens] node each token starts on.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    routes = [
        torch.load(f"{run_dir}/experts_rank{r}.pt", map_location="cpu")[layer].long()
        for r in range(recorded)
    ]
    source_node = torch.cat([
        torch.full((route.shape[0],), r * num_nodes // recorded)
        for r, route in enumerate(routes)
    ])
    return torch.cat(routes), source_node


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
    logits = torch.randn(num_tokens, k, generator=gen, device=device)
    weights = logits.softmax(dim=-1)
    return tokens, weights


def time_ms(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
) -> list[float]:
    """Time fn on every rank and report each call's slowest rank. Collective.

    Runs warmup untimed calls, then iterations timed ones. All ranks line up at a
    barrier before each timed call.

    Returns:
        times: Milliseconds of each timed call on the rank that took longest for it.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        dist.barrier()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(starts, ends)],
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    return times.tolist()
