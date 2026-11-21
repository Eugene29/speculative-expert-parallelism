"""Benchmark RBD dispatch and combine against a plain all2all.

Both versions take the same tokens, routing, and expert function. Each one
dispatches the tokens to their experts' ranks, runs the experts, and combines the
weighted outputs back on the source rank. The plain version (spec_ep.plain) sends one
row per (token, expert) pair in a single all2all each way. RBD (spec_ep.rbd) sends
one row per (token, destination node) between nodes and copies it to the other
experts within the node.

Routing is recorded but the tokens and weights are random. RUN_DIR/experts_rank{r}.pt
only holds a [num_layers, num_tokens, k] tensor of expert ids for recorded rank r, and
LAYER picks one layer. Thus, hidden dimension and number of experts are
hyper-parameters, set by flags and defaulting to Qwen3.6-35B-A3B. With more ranks than
recordings, each recording's tokens are split across world // recorded ranks so the
global workload stays the same.

The script first checks that both versions give the same output, then times a
full forward pass of each, from routing to combined output. plain_fused is plain with
RBD's merge kernel (spec_ep.ops.gather_sum) in place of index_add_, which separates
the gain from fewer inter-node rows from the gain from a faster merge.

RBD needs experts on more than one node to bypass any inter-node traffic, so run on
at least two nodes.

Usage:
    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.rbd_vs_plain \
        logs/2026_oct/2026-10-16_14-07-50 0 [--pilot rail]
"""
import argparse

import torch
import torch.distributed as dist

from benchmark.utils import check_node_layout, init_dist, load_routing, make_inputs, time_ms
from spec_ep.plain import plain_combine_dispatch
from spec_ep.rbd import PILOT_RULES, rbd_combine_dispatch


def inter_node_rows(
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
) -> tuple[int, int, int]:
    """Count the rows that cross between nodes on dispatch, over all ranks. Collective.

    Combine sends the same number back.

    Returns:
        plain_rows: Copies whose expert is on another node; plain sends each one.
        rbd_rows: (token, other node) pairs; RBD sends one pilot for each.
        copies: All (token, expert) copies.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    num_nodes = world // gpus_per_node
    route_node = route // (num_experts // world) // gpus_per_node
    remote = route_node != rank // gpus_per_node
    token = torch.arange(route.shape[0], device=route.device).unsqueeze(1).expand_as(route)
    pilots = (token * num_nodes + route_node)[remote].unique().numel()
    rows = torch.tensor([remote.sum().item(), pilots, route.numel()], device=route.device)
    dist.all_reduce(rows)
    return tuple(rows.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layer", type=int, help="Layer whose routing to replay")
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--pilot", choices=PILOT_RULES, default="rail", help="RBD pilot rule")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    E, G = args.num_experts, args.gpus_per_node

    rank, world, device = init_dist()
    hosts = check_node_layout(G)
    if E % world != 0:
        raise RuntimeError(f"{E} experts don't split evenly across {world} ranks.")
    local_group, _ = dist.new_subgroups(group_size=G)

    route = load_routing(args.run_dir, args.layer, rank, world, E).to(device)
    tokens, weights = make_inputs(route, args.hidden, device, seed=rank)
    modes = {
        "plain": lambda expert: plain_combine_dispatch(tokens, weights, expert, route, E),
        "plain_fused": lambda expert: plain_combine_dispatch(
            tokens, weights, expert, route, E, fused_merge=True
        ),
        "rbd": lambda expert: rbd_combine_dispatch(
            tokens, weights, expert, route, E, G, local_group, pilot_rule=args.pilot
        ),
    }

    plain_rows, rbd_rows, copies = inter_node_rows(route, E, G)

    scale = lambda x, expert_ids: x * ((expert_ids + 1) / E).to(x.dtype).unsqueeze(-1)
    reference = modes["plain"](scale).float()
    errors = []
    for name in ("rbd", "plain_fused"):
        diff = (modes[name](scale).float() - reference).abs().max()
        errors.append(diff / reference.abs().max().clamp_min(1e-12))
    errors = torch.stack(errors)
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    if errors.max() > 2e-2:
        raise RuntimeError(f"Output differs from plain by up to {errors.max():.2e} "
                           f"(relative).")

    identity = lambda x, expert_ids: x
    times = {name: time_ms(lambda: mode(identity), args.warmup, args.iterations)
             for name, mode in modes.items()}

    if rank == 0:
        print(
            f"{world // G} nodes x {G} GPUs: "
            f"{', '.join(hosts[::G])} | layer {args.layer} | "
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
        for name, ms in times.items():
            print(f"{name:>15}: {ms:8.3f} ms | {times['plain'] / ms:.2f}x vs plain",
                  flush=True)

    dist.barrier()
    dist.destroy_process_group()
