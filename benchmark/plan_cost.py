"""Compute cost of RBD's plan on one GPU, with no NCCL involved.

Emulates every rank of the EP group in one process, so the counts and the received
rows are the real ones: a first pass plans every rank and keeps what it sends, a second
pass times rank 0 with the counts and rows it would receive. Times the source half up
to its host sync and the pilot rank's half to the end of its GPU work, each with the
GPU drained before it. The defaults are a layer of Qwen3.5-397B-A17B on 2 nodes.

It also runs against the tree before the kernel plan (commit c730d7e, start_plans and
finish_plan), which is how the two were compared.

Usage:
    python -m benchmark.plan_cost [--world 16]
    PYTHONPATH=<old tree>/.deps python benchmark/plan_cost.py
"""
import argparse
import statistics
import time

import torch
import torch.distributed as dist

import spec_ep.rbd as rbd

parser = argparse.ArgumentParser()
parser.add_argument("--world", type=int, default=8)
parser.add_argument("--experts", type=int, default=512)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--tokens", type=int, default=4096)
parser.add_argument("--hidden", type=int, default=4096)
args = parser.parse_args()
W, G, E, K, T, H = args.world, 4, args.experts, args.k, args.tokens, args.hidden
dev = torch.device("cuda")
NEW = hasattr(rbd, "make_plan")
state = {"rank": 0, "sent": {}, "feed": None}


def exchange(output: torch.Tensor, input: torch.Tensor, *a, **kw) -> None:
    """Keep what this rank sends; hand rank 0 what the others sent it in the second pass."""
    state["sent"][state["rank"]] = input.clone()
    output.copy_(state["feed"] if state["feed"] is not None else torch.zeros_like(input))


dist.get_world_size = lambda group=None: W
dist.get_rank = lambda group=None: state["rank"]
dist.all_to_all_single = exchange


def source_half(route: torch.Tensor, weights: torch.Tensor, lanes: int):
    if NEW:
        return rbd.make_plan(route, weights, E, G, None, "rail", lanes)
    return rbd.start_plans(route, weights, E, G, None, "rail", 1, lanes)[0]


def receiving_half(plan, received: torch.Tensor) -> None:
    if NEW:
        rbd._fan_out(received, plan)
    else:
        rbd.finish_plan(plan, received)


gen = torch.Generator(device=dev).manual_seed(0)
routes = [torch.rand(T, E, device=dev, generator=gen).topk(K, dim=-1).indices for _ in range(W)]
weights = [torch.rand(T, K, device=dev, generator=gen) for _ in range(W)]
tokens = torch.randn(T, H, device=dev, dtype=torch.bfloat16)

for lanes in (1, 2):
    # Pass 1: every rank's counts, and the stage 1 rows it sends to rank 0.
    state["feed"], blocks = None, []
    for r in range(W):
        state["rank"] = r
        plan = source_half(routes[r], weights[r], lanes)
        rows = rbd.with_meta(tokens, plan.meta)[plan.send1_token]
        first = 0 if lanes == 1 or r >= G else sum(plan.send1[G:])
        blocks.append(rows[first : first + plan.send1[0]])
    counts = torch.stack([state["sent"][r].view(W, -1)[0] for r in range(W)])
    if lanes == 2:
        blocks = blocks[G:] + blocks[:G]
    received = torch.cat(blocks)
    state["rank"], state["feed"] = 0, counts.view(state["sent"][0].shape)

    source, receiving = [], []
    for i in range(230):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        plan = source_half(routes[0], weights[0], lanes)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        assert sum(plan.recv1) == received.shape[0]
        t2 = time.perf_counter()
        receiving_half(plan, received)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= 30:
            source.append((t1 - t0) * 1e6)
            receiving.append((t3 - t2) * 1e6)
    a, b = statistics.median(source), statistics.median(receiving)
    print(f"{'kernel' if NEW else 'old'} plan, world {W}, E {E}, k {K}, {T} tokens, lanes {lanes}: "
          f"source half {a:.0f} us, receiving half {b:.0f} us, both {a + b:.0f} us "
          f"(median of 200; {sum(plan.send1)} stage 1 rows, {sum(plan.send2)} stage 2 rows)")
