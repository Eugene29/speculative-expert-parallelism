"""Benchmark RBD dispatch and combine against a plain all2all.

Both versions take the same tokens, routing, and expert function. Each one
dispatches the tokens to their experts' ranks, runs the experts, and combines the
weighted outputs back on the source rank. The plain version sends one row per
(token, expert) pair in a single all2all each way. RBD (spec_ep.rbd) sends one row
per (token, destination node) between nodes and copies it to the other experts
within the node.

Routing is recorded but the tokens and weights are random. RUN_DIR/experts_rank{r}.pt 
only holds a [num_layers, num_tokens, k] tensor of expert ids for recorded rank r, and 
LAYER picks one layer. Thus, hidden dimension and number of experts are hyper-parameters
With more ranks than recordings, each recording's tokens are split
across world // recorded ranks so the global workload stays the same. The recording
holds no model shape, so E and H come from flags, defaulting to Qwen3.6-35B-A3B.

The script first checks that both versions give the same output, then times a
full forward pass of each, from routing to combined output. plain_fused is plain with
RBD's merge kernel (spec_ep.rbd._gather_sum) in place of index_add_, which separates the
gain from fewer inter-node rows from the gain from a faster merge.

RBD needs experts on more than one node to bypass any inter-node traffic, so run on
at least two nodes.



Usage:
    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.rbd_vs_plain \
        logs/2026_oct/2026-10-16_14-07-50 0
"""
import argparse
import os
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.profiler import record_function

from benchmark.utils import check_placement, init_dist
from spec_ep.rbd import _gather_sum as gather_sum
from spec_ep.rbd import RBD_combine_dispatch


def load_routing(
    run_dir: str,
    layer: int,
    rank: int,
    world: int,
    E: int,
) -> torch.Tensor:
    """Load this rank's share of one layer's recorded routing.

    Args:
        run_dir: Directory holding experts_rank{r}.pt for each recorded rank.
        layer: Layer whose routing to load.
        rank: This rank.
        world: Number of ranks. Must be a multiple of the number of recordings.
        E: Total number of experts. Every recorded id must be below it.

    Returns:
        route: [num_tokens, k] expert ids of this rank's tokens, as int64.
    """
    recorded = sum(
        f.startswith("experts_rank") and f.endswith(".pt") for f in os.listdir(run_dir)
    )
    if recorded == 0 or world % recorded != 0:
        raise RuntimeError(f"{world} ranks is not a multiple of {recorded} recordings "
                           f"in {run_dir}.")

    # Split each recording's tokens across world // recorded ranks, in rank order, so
    # the global workload stays the same.
    shards = world // recorded
    route = torch.load(f"{run_dir}/experts_rank{rank // shards}.pt", map_location="cpu")
    route = route[layer].tensor_split(shards)[rank % shards].long()

    if route.numel() > 0 and (route.min() < 0 or route.max() >= E):
        raise RuntimeError(f"Recorded expert ids span {route.min()}..{route.max()}, "
                           f"outside 0..{E - 1}.")
    return route


def make_inputs(
    route: torch.Tensor,
    H: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make random tokens and weights for the given routing.

    Args:
        route: [num_tokens, k] expert ids of this rank's tokens.
        H: Hidden size of each token.
        device: Device to put the tensors on.
        seed: Seed for the random values, so each rank's inputs are reproducible.

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


def plain_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    route: torch.Tensor,
    E: int,
    fused_merge: bool = False,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with plain all2alls.

    Collective over ep_group. Sends one row per (token, expert) pair, with no
    deduplication across experts on the same node. Takes the same arguments as
    RBD_combine_dispatch, minus the node layout it doesn't need.

    Args:
        tokens: [num_tokens, H] tokens on this rank.
        weights: [num_tokens, k] combine weights, where weights[t, j] scales the
            output of expert route[t, j] for token t.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to. Experts must be placed
            contiguously, E // world per rank of ep_group.
        E: Total number of experts across all ranks.
        fused_merge: If True, merges with gather_sum, as RBD does. If False, scales
            and adds with index_add_.
        ep_group: Process group holding the experts. None means the global group.

    Returns:
        output: [num_tokens, H] weighted sum of each token's expert outputs, in input
            order.
    """
    world = dist.get_world_size(ep_group)
    experts_per_rank = E // world
    num_tokens, k = route.shape

    # Phases are labeled like RBD's (autoep/rbd_*), for profiles.
    with record_function("plain/plan"):
        # One copy per (token, expert), grouped by the rank holding the expert.
        copy_token = torch.arange(num_tokens, device=route.device).repeat_interleave(k)
        copy_expert = route.flatten()
        copy_rank = copy_expert // experts_per_rank
        order = copy_rank.argsort(stable=True)
        copy_token, copy_expert = copy_token[order], copy_expert[order]
        copy_weight = weights.flatten()[order]

        # Plan: tell each rank how many copies to expect, then which expert each is for.
        send = copy_rank.bincount(minlength=world)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=ep_group)
        send, recv = send.tolist(), recv.tolist()
        expert_ids = copy_expert.new_empty(sum(recv))
        dist.all_to_all_single(expert_ids, copy_expert, recv, send, group=ep_group)

    # Dispatch: every copy travels on its own, even if another copy shares its node.
    with record_function("plain/dispatch"):
        inputs = tokens.new_empty((sum(recv), tokens.shape[1]))
        dist.all_to_all_single(inputs, tokens[copy_token], recv, send, group=ep_group)

    with record_function("plain/experts"):
        outputs = expert(inputs, expert_ids)

    # Combine: reverse the dispatch, so rows return in the order of copy_token.
    with record_function("plain/combine"):
        returned = outputs.new_empty((sum(send), outputs.shape[1]))
        dist.all_to_all_single(returned, outputs, send, recv, group=ep_group)

    # Weights are applied on the source rank, so they never cross the network.
    with record_function("plain/merge"):
        if fused_merge:
            position = torch.empty_like(order)
            position[order] = torch.arange(order.numel(), device=order.device)
            return gather_sum(returned, position.view(num_tokens, k), weights)
        weight = copy_weight.to(returned.dtype).unsqueeze(-1)
        output = returned.new_zeros((num_tokens, returned.shape[1]))
        return output.index_add_(0, copy_token, returned * weight)


def time_ms(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> float:
    """Time fn on every rank and report the slowest rank's average.

    Collective. Runs warmup untimed calls, then iterations timed ones.

    Args:
        fn: Function to time, the same on every rank.
        warmup: Untimed calls before timing.
        iterations: Timed calls to average over.

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layer", type=int, help="Layer whose routing to replay")
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--pilot", choices=["rail", "first", "most", "even"], default="rail",
                        help="RBD pilot rank policy")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    E = args.num_experts

    rank, world, device = init_dist()
    gpus_per_node = args.gpus_per_node
    hosts = check_placement(gpus_per_node)
    if E % world != 0:
        raise RuntimeError(f"{E} experts don't split evenly across {world} ranks.")
    local_group, _ = dist.new_subgroups(group_size=gpus_per_node)

    route = load_routing(args.run_dir, args.layer, rank, world, E).to(device)
    tokens, weights = make_inputs(route, args.hidden, device, seed=rank)
    rbd = lambda expert: RBD_combine_dispatch(
        tokens, weights, expert, route, E, gpus_per_node, local_group, pilot=args.pilot
    )
    plain = lambda expert, fused=False: plain_combine_dispatch(
        tokens, weights, expert, route, E, fused
    )

    # Rows crossing between nodes on dispatch: plain sends every copy on another node,
    # RBD one pilot per (token, other node). Combine sends the same number back.
    experts_per_rank = E // world
    num_nodes = world // gpus_per_node
    route_node = route // experts_per_rank // gpus_per_node
    remote = route_node != rank // gpus_per_node
    token_ids = torch.arange(route.shape[0], device=device).unsqueeze(1).expand_as(route)
    pilots = (token_ids * num_nodes + route_node)[remote].unique().numel()
    rows = torch.tensor([remote.sum().item(), pilots, route.numel()], device=device)
    dist.all_reduce(rows)
    plain_rows, rbd_rows, copies = rows.tolist()

    # Check: an expert that scales by its id catches rows sent to the wrong expert. Every
    # term of a token's output has the same sign, so a relative error is well defined.
    scale = lambda x, expert_ids: x * ((expert_ids + 1) / E).to(x.dtype).unsqueeze(-1)
    reference = plain(scale).float()
    errors = []
    for output in (rbd(scale), plain(scale, fused=True)):
        diff = (output.float() - reference).abs().max()
        errors.append(diff / reference.abs().max().clamp_min(1e-12))
    errors = torch.stack(errors)
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    if errors.max() > 2e-2:
        raise RuntimeError(f"Output differs from plain by up to {errors.max():.2e} "
                           f"(relative).")

    # Time communication only: identity experts leave no compute in the measurement.
    identity = lambda x, expert_ids: x
    times = {
        "plain": time_ms(lambda: plain(identity), args.warmup, args.iterations),
        "plain_fused": time_ms(lambda: plain(identity, fused=True), args.warmup,
                               args.iterations),
        "rbd": time_ms(lambda: rbd(identity), args.warmup, args.iterations),
    }

    if rank == 0:
        print(
            f"{num_nodes} nodes x {gpus_per_node} GPUs: "
            f"{', '.join(hosts[::gpus_per_node])} | layer {args.layer} | "
            f"E={E} H={args.hidden} k={route.shape[1]} | "
            f"{route.shape[0]} tokens per rank",
            flush=True,
        )
        print(
            f"inter-node rows per direction: plain {plain_rows} "
            f"({plain_rows / copies:.1%} of copies), rbd {rbd_rows} "
            f"({plain_rows / max(rbd_rows, 1):.2f}x fewer)",
            flush=True,
        )
        print(f"max relative error vs plain: rbd {errors[0]:.2e}, "
              f"plain_fused {errors[1]:.2e}", flush=True)
        for mode, ms in times.items():
            print(f"{mode:>15}: {ms:8.3f} ms | {times['plain'] / ms:.2f}x vs plain",
                  flush=True)

    dist.barrier()
    dist.destroy_process_group()