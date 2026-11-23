"""Every dispatch-and-combine version behind one call, for the layer benchmark and the model.

build_mode() turns a mode given as name:key=value,key=value into a function
mode(tokens, weights, expert, route) -> output. benchmark/all2all/combine_dispatch.py
times that function on recorded routing, and scripts/ds_moe.py hands the same function
to every MoE layer of the model, so a mode is the same code in both.

Modes:
    plain: spec_ep.plain. One row per (token, expert) in a single all2all each way.
    plain_fused: plain with RBD's merge kernel (spec_ep.ops.gather_sum) in place of
        index_add_, which separates the gain from fewer inter-node rows from the gain
        from a faster merge.
    rbd: spec_ep.rbd. One row per (token, destination node) between nodes, copied to
        the other experts within the node.
    two_lane: spec_ep.two_lane. RBD with the inter-node rows and the node-local rows
        in two all2alls that run at the same time.
    plain_pipe, rbd_pipe: spec_ep.pipeline. plain and rbd on degree slices of the
        tokens, so the experts of one slice run while the next is on the network.
    autoep: DeepSpeed's AutoEP layer after its router. One row per (token, expert),
        sorted by expert, in one all2all each way. fused=True merges with AutoEP's
        fused Triton kernel instead of its default weighted sum.
    tutel: Tutel. Every expert is padded to one capacity and the padded buffer goes in
        an all2all with equal splits. capacity_factor=0 (default) pads to the most rows
        any rank sends to any expert; capacity_factor=1, Tutel's own default, gives
        every expert its even share and drops the copies past it. degree=N pipelines N
        chunks of capacity with a NCCL communicator Tutel builds on first use.
        local_size=G uses Tutel's 2D hierarchical all2all for G ranks per node.
    ds_moe: DeepSpeed's MOELayer. The same padded buffer and capacity_factor as tutel,
        filled and read by DeepSpeed's own encode and decode.
    xmoe: X-MoE's own redundancy-bypassing dispatch (benchmark/all2all/xmoe.py), from a
        checkout with its two bugs fixed. capacity_factor=0 (default) drops nothing;
        capacity_factor=1, X-MoE's own default, drops the copies past each expert's
        even share. permute=False takes a token's lowest expert on a node as its pilot
        instead of a random one.
    deepep: DeepEP. One row per (token, rank), through buffers shared by the GPUs of
        one node, and by RDMA between nodes with the multinode build (patches/).
        num_sms, dispatch_chunk, combine_chunk and rdma_chunk are DeepEP's tuning
        knobs, and fused=True sums the local expert outputs with RBD's merge kernel.

The Tutel, DeepSpeed, X-MoE and DeepEP adapters are imported only when their mode is
built, so a run without them needs none of those libraries.

Usage:
    from benchmark.all2all.modes import MODES, NEED_TWO_NODES, PADDED, build_mode, parse_mode
    mode = build_mode("tutel:capacity_factor=1,degree=2", num_experts, gpus_per_node, local_group)
    output = mode(tokens, weights, expert, route)
"""
import ast
from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist

from spec_ep.pipeline import pipelined_plain_combine_dispatch, pipelined_rbd_combine_dispatch
from spec_ep.plain import plain_combine_dispatch
from spec_ep.rbd import rbd_combine_dispatch
from spec_ep.two_lane import two_lane_rbd_combine_dispatch

MODES = (
    "plain", "plain_fused", "rbd", "two_lane", "plain_pipe", "rbd_pipe",
    "autoep", "tutel", "ds_moe", "xmoe", "deepep",
)
NEED_TWO_NODES = {"rbd", "two_lane", "rbd_pipe", "xmoe"}
PADDED = {"tutel", "ds_moe"}

Expert = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Mode = Callable[[torch.Tensor, torch.Tensor, Expert, torch.Tensor], torch.Tensor]


def parse_mode(spec: str, known: tuple[str, ...] = MODES) -> tuple[str, dict[str, object]]:
    """Split a mode given as name:key=value,key=value.

    Args:
        spec: The mode.
        known: Names to accept. A script with modes of its own passes them too.

    Returns:
        name: One of known.
        settings: Keyword arguments for the mode's function. Values are Python
            literals where they parse as one (2, 0.5, True) and strings otherwise.
    """
    name, _, rest = spec.partition(":")
    if name not in known:
        raise ValueError(f"Unknown mode {name!r} in {spec!r}. Modes: {', '.join(known)}.")
    settings = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        try:
            settings[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            settings[key] = value
    return name, settings


def build_mode(
    spec: str,
    num_experts: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
    pilot_rule: str = "rail",
    degree: int = 2,
) -> Mode:
    """Build the function of one mode. Collective for deepep, which builds its buffer.

    Args:
        spec: The mode as name:key=value,key=value.
        num_experts: Total number of experts across all ranks of ep_group.
        gpus_per_node: Ranks per node.
        local_group: Process group of this rank's node.
        ep_group: Process group holding the experts. None means the global group.
        pilot_rule: Pilot rule of the RBD modes, unless the settings give one.
        degree: Slices of plain_pipe and rbd_pipe, unless the settings give one.

    Returns:
        mode: Maps (tokens, weights, expert, route) to the [num_tokens, H] combined
            output, with the arguments of plain_combine_dispatch().
    """
    name, settings = parse_mode(spec)
    E, G = num_experts, gpus_per_node
    rbd = {"ep_group": ep_group, "pilot_rule": pilot_rule}

    if name == "plain":
        fn = partial(plain_combine_dispatch, num_experts=E, ep_group=ep_group)
    elif name == "plain_fused":
        fn = partial(plain_combine_dispatch, num_experts=E, ep_group=ep_group, fused_merge=True)
    elif name == "rbd":
        fn = partial(rbd_combine_dispatch, num_experts=E, gpus_per_node=G,
                     local_group=local_group, **rbd)
    elif name == "two_lane":
        fn = partial(two_lane_rbd_combine_dispatch, num_experts=E, gpus_per_node=G,
                     local_group=local_group, **rbd)
    elif name == "plain_pipe":
        fn = partial(pipelined_plain_combine_dispatch, num_experts=E, degree=degree,
                     ep_group=ep_group)
    elif name == "rbd_pipe":
        fn = partial(pipelined_rbd_combine_dispatch, num_experts=E, gpus_per_node=G,
                     local_group=local_group, degree=degree, **rbd)
    elif name == "xmoe":
        from benchmark.all2all.xmoe import xmoe_combine_dispatch

        fn = partial(xmoe_combine_dispatch, num_experts=E, gpus_per_node=G,
                     local_group=local_group, ep_group=ep_group)
    else:
        from benchmark.all2all import baselines

        if name == "deepep":
            buffer = baselines.make_deepep_buffer(ep_group or dist.group.WORLD)
            fn = partial(baselines.deepep_combine_dispatch, buffer, num_experts=E)
        else:
            adapter = {
                "autoep": baselines.autoep_combine_dispatch,
                "tutel": baselines.tutel_combine_dispatch,
                "ds_moe": baselines.ds_moe_combine_dispatch,
            }[name]
            fn = partial(adapter, num_experts=E, ep_group=ep_group)

    fn = partial(fn, **settings)
    return lambda tokens, weights, expert, route: fn(tokens, weights, expert, route)
