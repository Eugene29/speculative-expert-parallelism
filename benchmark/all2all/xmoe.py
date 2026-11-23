"""X-MoE's redundancy-bypassing dispatch behind the arguments of rbd_combine_dispatch.

xmoe_combine_dispatch dispatches the tokens to their experts' ranks, runs the experts,
and combines the weighted outputs on the source rank with the planning and transfer
code of X-MoE (arXiv 2508.13337), the system spec_ep's RBD is based on. Like the
adapters in baselines.py, it takes the recorded routing instead of choosing experts.

What it runs is X-MoE's own code, from deepspeed/moe/v2opt of X-MoE's DeepSpeed fork:
    gating.py: indices_and_bins sorts this rank's copies by expert. With a capacity
        factor, indices_and_bins_and_drop of drop_gating.py sorts them and drops
        the copies past each expert's capacity.
    metadata.py: rbd_metadata plans one call. Of a token's copies on one node, one
        becomes the pilot and goes to the rank of its own expert, and the others
        become replicas. The token's own node is treated like any other. The ranks
        exchange row counts in three all2alls, and two more tell each pilot rank
        the expert and the pilot of every replica.
    rbd.py: RBDispatcher.forward gathers the pilots, sends them over ep_group,
        gathers the replicas out of the received pilots and sends them over
        local_group. RBCombiner.forward sends the weights of pilots and replicas to
        the pilot ranks over ep_group, returns the replica outputs over local_group,
        scales every output, adds the replica outputs to their pilot's, returns
        those sums over ep_group and scatters them into the tokens.
    a2a_single.py, rbd_gating.py and the Triton kernels of kernels.py, drop_kernel.py
        and rbd_kernels.py, which the code above calls.

X-MoE as published returns wrong outputs. XMOE_DIR must hold a checkout with its two
bugs fixed: RBDispatcher returns the replicas before the pilots while RBCombiner
splits the expert outputs in the other order, and RBCombiner scales the replica
outputs with weights still in the order they had before rbd_metadata sorted the
replicas by expert.

What this module adds around X-MoE's code:
    Loading. X-MoE's fork and the repo's DeepSpeed are both the package deepspeed, so
        only X-MoE's v2opt directory is loaded. It is mounted as deepspeed.moe.v2opt,
        the name its files import each other by, which the repo's DeepSpeed does not
        define. What those files import from the rest of deepspeed (comm,
        utils.groups, utils.bwc and moe.sharded_moe._AllToAll) then comes from the
        repo's DeepSpeed, where it is the same thin layer over torch.distributed as
        in X-MoE's fork. deepspeed.comm is started on the first call unless
        DeepSpeed has started it.
    Routing. X-MoE's gate (TopKGateRBD) picks the experts, then calls
        indices_and_bins and rbd_metadata with a random permutation of the copies.
        The adapter makes those calls on the recorded routing.
    Expert ids. X-MoE's Experts module splits the received rows by expert, which is
        only right with one expert per rank. The adapter hands the rows to the
        expert function in the order they arrive, with each row's expert id, which
        it reads from the rows per source rank and expert that X-MoE's plan holds.

Limits. The tokens must be on the current CUDA device, because X-MoE gathers and
scatters with Triton kernels. The kernels compile and tune themselves on the first
calls. X-MoE takes a rank's node from its global rank, so ranks must be numbered node
by node and local_group must be this rank's node. The adapter is checked only with
every rank in ep_group, and only forward. X-MoE scales in the dtype of the tokens, so
bf16 tokens get bf16 weights.

Usage:
    from benchmark.all2all.xmoe import xmoe_combine_dispatch, xmoe_kept

    # The checkout XMOE_DIR points at, by default /pscratch/sd/e/eku/X-MoE-fixed.
    git -C ../X-MoE worktree add --detach $SCRATCH/X-MoE-fixed
    git -C $SCRATCH/X-MoE-fixed apply \
        $SCRATCH/hot_experts/backup_before_cleanup/benchmark/all2all/xmoe_rbd_fixes.patch
"""
import math
import os
import sys
import types
from collections.abc import Callable
from functools import cache

import deepspeed.comm
import torch
import torch.distributed as dist
from torch.profiler import record_function

XMOE_DIR = os.environ.get("XMOE_DIR", "/pscratch/sd/e/eku/X-MoE-fixed")

# Mount X-MoE's v2opt directory, so the imports below are X-MoE's, not the repo's DeepSpeed.
_v2opt = types.ModuleType("deepspeed.moe.v2opt")
_v2opt.__path__ = [os.path.join(XMOE_DIR, "deepspeed", "moe", "v2opt")]
sys.modules["deepspeed.moe.v2opt"] = _v2opt

from deepspeed.moe.v2opt.drop_gating import indices_and_bins_and_drop  # noqa: E402
from deepspeed.moe.v2opt.gating import indices_and_bins  # noqa: E402
from deepspeed.moe.v2opt.metadata import rbd_metadata  # noqa: E402
from deepspeed.moe.v2opt.rbd import RBCombiner, RBDispatcher  # noqa: E402

Expert = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@cache
def _rbd_modules(
    num_experts: int,
    k: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None,
) -> tuple[RBDispatcher, RBCombiner]:
    """Build X-MoE's dispatcher and combiner once per setting, as its layer does.

    Returns:
        dispatcher: X-MoE's RBDispatcher.
        combiner: X-MoE's RBCombiner.
    """
    experts_per_rank = num_experts // dist.get_world_size(ep_group)
    args = (num_experts, k, experts_per_rank, gpus_per_node, ep_group, local_group)
    return RBDispatcher(*args), RBCombiner(*args)


def _sorted_copies(
    weights: torch.Tensor,
    route: torch.Tensor,
    num_experts: int,
    capacity_factor: float,
    min_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort this rank's copies by expert, less those past capacity, as X-MoE's gate does.

    Returns:
        indices: The kept copies as positions in route.flatten(), ordered by expert.
        bin_ids: Their experts.
    """
    if capacity_factor == 0:
        indices, bin_ids, _, _ = indices_and_bins(num_experts, route.flatten())
    else:
        capacity = max(math.ceil(route.numel() / num_experts * capacity_factor), min_capacity)
        indices, bin_ids, _, _ = indices_and_bins_and_drop(
            num_experts, route.flatten(), weights.flatten(), capacity
        )
    return indices, bin_ids


def xmoe_kept(
    weights: torch.Tensor,
    route: torch.Tensor,
    num_experts: int,
    capacity_factor: float = 0,
    min_capacity: int = 4,
) -> torch.Tensor:
    """Tell which copies xmoe_combine_dispatch keeps at a capacity factor.

    Returns:
        kept: [num_tokens, k] bool, False where the copy route[t, j] is dropped.
    """
    indices, _ = _sorted_copies(weights, route, num_experts, capacity_factor, min_capacity)
    kept = torch.zeros(route.numel(), dtype=torch.bool, device=route.device)
    kept[indices] = True
    return kept.view_as(route)


def xmoe_combine_dispatch(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    expert: Expert,
    route: torch.Tensor,
    num_experts: int,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup | None = None,
    capacity_factor: float = 0,
    min_capacity: int = 4,
    permute: bool = True,
) -> torch.Tensor:
    """Dispatch tokens, run the experts, and combine their outputs with X-MoE's RBD.

    Collective over ep_group and local_group. Takes the arguments of
    rbd_combine_dispatch, then X-MoE's settings.

    Args:
        tokens: [num_tokens, H] tokens on this rank, on its GPU.
        weights: [num_tokens, k] combine weights.
        expert: Maps [num_inputs, H] rows and their [num_inputs] expert ids to
            [num_inputs, H] outputs, in the same order.
        route: [num_tokens, k] experts each token is routed to.
        num_experts: Total number of experts across all ranks.
        gpus_per_node: Ranks per node, X-MoE's rbd_mesh_size.
        local_group: Process group of this rank's node.
        ep_group: Process group holding the experts. None means the global group.
        capacity_factor: 0 drops nothing, X-MoE's drop_tokens=False. f > 0 lets every
            expert take f times its even share of this rank's copies, those with the
            largest weights, and drops the rest; X-MoE's default is 1. xmoe_kept
            tells which copies survive.
        min_capacity: The least that share can be, in copies. X-MoE's default is 4.
        permute: If True, the pilot is a random one of the token's copies on a node,
            as in X-MoE's layer, which permutes the copies on every call. If False it
            is the copy with the lowest expert id, which is what rbd_metadata picks
            without a permutation.

    Returns:
        output: [num_tokens, H] weighted sum of each token's kept expert outputs, in
            input order.
    """
    if not deepspeed.comm.is_initialized():
        deepspeed.comm.init_distributed(dist_backend=dist.get_backend())
    num_tokens, k = route.shape
    world, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    experts_per_rank = num_experts // world
    dispatcher, combiner = _rbd_modules(num_experts, k, gpus_per_node, local_group, ep_group)

    with record_function("xmoe/plan"):
        indices, bin_ids = _sorted_copies(
            weights, route, num_experts, capacity_factor, min_capacity
        )
        permutation = None
        if permute:
            permutation = torch.randperm(indices.numel(), device=route.device, dtype=torch.int32)
        plan = rbd_metadata(
            indices, bin_ids, num_experts, k, gpus_per_node, experts_per_rank,
            ep_group, local_group, permutation,
        )

    # X-MoE's dispatcher and combiner each gather, send over both groups and scatter in
    # one call, so the whole call is the transfer the event clock sees.
    with record_function("xmoe/dispatch"), record_function("comm/both"):
        inputs = dispatcher(tokens, num_tokens, plan)

    with record_function("xmoe/expert_ids"):
        # Pilots arrive first, then replicas, each by source rank and then by expert.
        # The plan ends with the rows per (source rank, local expert) of either kind.
        counts = torch.cat(plan[-2:])
        local = torch.arange(counts.numel(), device=counts.device) % experts_per_rank
        local = local.repeat_interleave(counts, output_size=inputs.shape[0])
        expert_ids = rank * experts_per_rank + local

    with record_function("xmoe/experts"):
        outputs = expert(inputs, expert_ids)

    with record_function("xmoe/combine"), record_function("comm/both"):
        return combiner(outputs, weights.flatten(), num_tokens, plan)
