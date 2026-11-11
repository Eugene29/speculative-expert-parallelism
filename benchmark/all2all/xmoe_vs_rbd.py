"""Benchmark spec_ep's RBD against X-MoE's own RBD code and plain all2all.

X-MoE's RBD lives in its DeepSpeed fork (deepspeed/moe/v2opt), which only needs a
few pieces of DeepSpeed: deepspeed.comm, an expert parallel world size from
deepspeed.utils.groups, and _AllToAll from deepspeed.moe.sharded_moe. XMoERBD puts
small stand-ins for those in sys.modules and imports the v2opt files unmodified
from an X-MoE clone, so the installed DeepSpeed is not touched.

XMoERBD.combine_dispatch follows MOEv2LayerRBD.forward from X-MoE's moe_rbd.py: no
token drop, a random pilot permutation as TopKGateRBD uses, rbd_metadata, then
RBDispatcher's and RBCombiner's steps. Two fixes make its output correct:
    - RBDispatcher concatenates expert inputs as [stage 2, stage 1], but RBCombiner
      splits them as [stage 1, stage 2]; here both use [stage 1, stage 2].
    - rbd_metadata sorts the received stage 2 rows by expert, but RBCombiner applies
      the stage 2 weights in the unsorted order. Here the weights get the same
      permutation, one small gather.

Each call includes X-MoE's routing sort and metadata, as its gate does every step.
Every mode is checked against plain with an expert that scales rows by expert id,
then timed with identity experts. Host syncs per call are counted with
torch.cuda.set_sync_debug_mode.

Usage:
    srun -N 2 --ntasks-per-node=4 \
        python -u -m benchmark.all2all.xmoe_vs_rbd \
        logs/2026_oct/2026-10-16_14-07-50 0 20 39 [--xmoe-dir ../X-MoE]
"""
import argparse
import importlib
import os
import sys
import types
import warnings
from collections.abc import Callable

import torch
import torch.distributed as dist

from benchmark.all2all.rbd_vs_plain import (load_routing, make_inputs,
                                             plain_combine_dispatch, time_ms)
from benchmark.utils import check_placement, init_dist
from spec_ep.rbd import RBD_combine_dispatch

DEFAULT_XMOE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "X-MoE")


class _AllToAll(torch.autograd.Function):
    """Equal-split all2all, as in X-MoE's sharded_moe._AllToAll."""

    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, input: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output


def _install_stubs(xmoe_dir: str) -> None:
    """Put stand-ins for the DeepSpeed modules v2opt imports into sys.modules.

    Args:
        xmoe_dir: Root of the X-MoE clone.
    """
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.__path__ = []
    deepspeed.comm = dist
    utils = types.ModuleType("deepspeed.utils")
    utils.__path__ = []
    groups = types.ModuleType("deepspeed.utils.groups")
    groups._get_expert_model_parallel_world_size = lambda: 1
    groups.mpu = None
    bwc = types.ModuleType("deepspeed.utils.bwc")
    bwc.bwc_tensor_model_parallel_world_size = lambda mpu=None: 1
    utils.groups, utils.bwc = groups, bwc
    moe = types.ModuleType("deepspeed.moe")
    moe.__path__ = [os.path.join(xmoe_dir, "deepspeed", "moe")]
    sharded_moe = types.ModuleType("deepspeed.moe.sharded_moe")
    sharded_moe._AllToAll = _AllToAll
    sys.modules.update({
        "deepspeed": deepspeed,
        "deepspeed.comm": dist,
        "deepspeed.utils": utils,
        "deepspeed.utils.groups": groups,
        "deepspeed.utils.bwc": bwc,
        "deepspeed.moe": moe,
        "deepspeed.moe.sharded_moe": sharded_moe,
    })


def _expert_ids(counts: torch.Tensor, num_local: int, first: int) -> torch.Tensor:
    """Expert id of each row of a buffer grouped by source, then by expert.

    Args:
        counts: [num_sources * num_local] rows per (source, local expert).
        num_local: Experts per rank.
        first: Id of this rank's first expert.

    Returns:
        expert_ids: [counts.sum()] expert id of each row.
    """
    ids = torch.arange(first, first + num_local, device=counts.device)
    return ids.repeat(counts.numel() // num_local).repeat_interleave(counts.long())


class XMoERBD:
    """X-MoE's RBD code, imported from a clone through DeepSpeed stand-ins."""

    def __init__(self, xmoe_dir: str) -> None:
        """Install the stand-ins and import X-MoE's v2opt modules.

        Args:
            xmoe_dir: Root of the X-MoE clone.
        """
        _install_stubs(xmoe_dir)
        v2opt = "deepspeed.moe.v2opt"
        self.a2a = importlib.import_module(f"{v2opt}.a2a_single")._AllToAllSingle
        drop_gating = importlib.import_module(f"{v2opt}.drop_gating")
        self.gather = drop_gating.gather_with_token_drop
        self.scatter = drop_gating.scatter_with_token_drop
        self.indices_and_bins = importlib.import_module(f"{v2opt}.gating").indices_and_bins
        self.meta = importlib.import_module(f"{v2opt}.metadata")
        rbd_gating = importlib.import_module(f"{v2opt}.rbd_gating")
        self.s1_to_s2 = rbd_gating.s1_to_s2_gather
        self.s2_to_s1 = rbd_gating.s2_to_s1_scatter

    def metadata(
        self,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
        mesh_size: int,
        num_local_experts: int,
        ep_group: dist.ProcessGroup | None,
        local_group: dist.ProcessGroup,
        permutation: torch.Tensor,
    ) -> tuple[tuple, torch.Tensor]:
        """X-MoE's metadata.rbd_metadata, copied, also returning the stage 2 sort order.

        The only addition is returning bin_ids_s2_indices, which X-MoE computes but
        drops, so the combine can line the stage 2 weights up with the sorted rows.

        Args:
            indices: Flat copy index of each routed copy, sorted by expert.
            bin_ids: Expert of each copy, in the same order.
            num_experts: Total number of experts.
            top_k: Experts per token.
            mesh_size: Ranks per node.
            num_local_experts: Experts per rank.
            ep_group: Process group holding the experts.
            local_group: Process group of this rank's node.
            permutation: Random permutation of the copies, which picks the pilots.

        Returns:
            metadata: X-MoE's 15-tuple.
            s2_order: Permutation that sorted the received stage 2 rows by expert.
        """
        m, a2a = self.meta, self.a2a
        indices_s1, indices_s2, bin_ids_s1, bin_ids_s2_virtual, s1_to_s2_virtual, \
            tokens_per_experts_s2_virtual = m.map_dup_tokens(
                indices, bin_ids, num_experts, top_k, mesh_size, num_local_experts, permutation)
        bins_s1, s1_tokens_per_expert = m.get_bins(bin_ids_s1, num_experts)
        ins_s1, outs_s1, tokens_per_expert_s1_exp = m.get_a2a_splits(
            s1_tokens_per_expert, ep_group, num_local_experts)
        s1_to_s2_virtual = m.map_s1_to_s2_indices_to_relative(bins_s1, s1_to_s2_virtual)
        ins_s2_virtual, outs_s2_virtual, tokens_per_experts_s2 = m.get_a2a_splits(
            tokens_per_experts_s2_virtual, ep_group, num_local_experts)
        s1exp_to_s2 = a2a.apply(ep_group, s1_to_s2_virtual, ins_s2_virtual, outs_s2_virtual)
        bin_ids_s2 = a2a.apply(ep_group, bin_ids_s2_virtual, ins_s2_virtual, outs_s2_virtual)
        s1exp_to_s2 = m.map_s1_to_s2_indices_relative_to_abs(
            s1exp_to_s2, tokens_per_expert_s1_exp, tokens_per_experts_s2)
        bin_ids_s2, s2_order = torch.sort(bin_ids_s2)
        s1exp_to_s2 = s1exp_to_s2[s2_order]
        node_experts = mesh_size * num_local_experts
        first = ((dist.get_rank() // mesh_size) * node_experts) % num_experts
        tokens_per_experts_s2_ = torch.histc(bin_ids_s2, node_experts, first,
                                             first + node_experts - 1)
        bins_s2 = torch.cumsum(tokens_per_experts_s2_, 0).to(torch.int32)
        bins_s2 = bins_s2.view(1) if not len(bins_s2.size()) else bins_s2
        ins_s2, outs_s2, tokens_per_experts_s2_exp = m.get_a2a_splits(
            tokens_per_experts_s2_, local_group, num_local_experts)
        bin_ids_s2 = bin_ids_s2 - first
        metadata = (indices_s1, indices_s2, bin_ids_s1, bins_s1, ins_s1, outs_s1, ins_s2_virtual,
                    outs_s2_virtual, s1exp_to_s2, bin_ids_s2, bins_s2, ins_s2, outs_s2,
                    tokens_per_expert_s1_exp, tokens_per_experts_s2_exp)
        return metadata, s2_order

    def combine_dispatch(
        self,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        expert: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        route: torch.Tensor,
        E: int,
        gpus_per_node: int,
        local_group: dist.ProcessGroup,
        ep_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Dispatch, run the experts and combine with X-MoE's RBD code.

        Takes the same arguments as spec_ep.rbd.RBD_combine_dispatch, minus the pilot
        rule: X-MoE picks pilots by a random permutation.

        Returns:
            output: [num_tokens, H] weighted sum of each token's expert outputs.
        """
        a2a = self.a2a
        num_tokens, k = route.shape
        world, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
        num_local = E // world
        indices, bin_ids, _, _ = self.indices_and_bins(E, route.flatten())
        permutation = torch.randperm(indices.shape[0], device=route.device, dtype=torch.int32)
        metadata, s2_order = self.metadata(indices, bin_ids, E, k, gpus_per_node, num_local,
                                           ep_group, local_group, permutation)
        (indices_s1, indices_s2, bin_ids_s1, bins_s1, ins_s1, outs_s1, ins_s2_virtual,
         outs_s2_virtual, s1exp_to_s2, bin_ids_s2, bins_s2, ins_s2, outs_s2,
         per_expert_s1, per_expert_s2) = metadata

        # RBDispatcher.forward, with inputs concatenated as [stage 1, stage 2].
        dx_s1 = self.gather(tokens, indices_s1, bin_ids_s1, bins_s1, num_tokens, k)
        dx_s1_exp = a2a.apply(ep_group, dx_s1, ins_s1, outs_s1)
        dx_s2 = self.s1_to_s2(dx_s1_exp, s1exp_to_s2, bin_ids_s2, bins_s2)
        dx_s2_exp = a2a.apply(local_group, dx_s2, ins_s2, outs_s2)
        expert_inputs = torch.cat([dx_s1_exp, dx_s2_exp])

        expert_ids = torch.cat([_expert_ids(per_expert_s1, num_local, rank * num_local),
                                _expert_ids(per_expert_s2, num_local, rank * num_local)])
        outputs = expert(expert_inputs, expert_ids)

        # RBCombiner.forward, with the stage 2 weights put in sorted row order.
        flat_weights = weights.flatten().to(outputs.dtype)
        weights_s1_exp = a2a.apply(ep_group, flat_weights[indices_s1], ins_s1, outs_s1)
        weights_s2 = a2a.apply(ep_group, flat_weights[indices_s2], ins_s2_virtual,
                               outs_s2_virtual)[s2_order]
        dout_s1_exp, dout_s2_exp = torch.split(outputs, [sum(outs_s1), sum(outs_s2)])
        dout_s2 = a2a.apply(local_group, dout_s2_exp, outs_s2, ins_s2)
        dout_s1_exp = self.s2_to_s1(dout_s1_exp * weights_s1_exp[:, None],
                                    dout_s2 * weights_s2[:, None], s1exp_to_s2, bin_ids_s2,
                                    bins_s2)
        dout_s1 = a2a.apply(ep_group, dout_s1_exp, outs_s1, ins_s1)
        return self.scatter(dout_s1, indices_s1, bin_ids_s1, None, bins_s1, num_tokens, k)


def count_syncs(fn: Callable[[], torch.Tensor]) -> int:
    """Count host syncs in one call of fn.

    Args:
        fn: Function to run.

    Returns:
        syncs: Sync warnings raised during the call.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        fn()
        torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(w.message) for w in caught)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="Directory holding experts_rank{r}.pt")
    parser.add_argument("layers", type=int, nargs="+", help="Layers whose routing to replay")
    parser.add_argument("--xmoe-dir", default=DEFAULT_XMOE_DIR, help="Root of the X-MoE clone")
    parser.add_argument("--num-experts", type=int, default=256, help="E, total experts")
    parser.add_argument("--hidden", type=int, default=2048, help="H, token width")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    E, G = args.num_experts, args.gpus_per_node

    rank, world, device = init_dist()
    hosts = check_placement(G)
    local_group, _ = dist.new_subgroups(group_size=G)
    xmoe = XMoERBD(args.xmoe_dir)
    identity = lambda x, expert_ids: x
    scale = lambda x, expert_ids: x * ((expert_ids + 1) / E).to(x.dtype).unsqueeze(-1)

    for layer in args.layers:
        route = load_routing(args.run_dir, layer, rank, world, E).to(device)
        tokens, weights = make_inputs(route, args.hidden, device, seed=rank)
        modes = {
            "plain": lambda f: plain_combine_dispatch(tokens, weights, f, route, E),
            "plain_fused": lambda f: plain_combine_dispatch(tokens, weights, f, route, E, True),
            "xmoe_rbd": lambda f: xmoe.combine_dispatch(tokens, weights, f, route, E, G,
                                                        local_group),
            "spec_ep_rbd": lambda f: RBD_combine_dispatch(tokens, weights, f, route, E, G,
                                                          local_group),
        }

        reference = modes["plain"](scale).float()
        stats = torch.zeros(len(modes), 3, device=device, dtype=torch.float64)
        for i, (mode, fn) in enumerate(modes.items()):
            out = fn(scale).float()
            stats[i, 0] = (out - reference).abs().max() / reference.abs().max()
            stats[i, 1] = time_ms(lambda: fn(identity), args.warmup, args.iterations)
            stats[i, 2] = count_syncs(lambda: fn(identity))
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(f"\n=== layer {layer}, {route.shape[0]} tokens/rank | {world // G} nodes x "
                  f"{G} GPUs: {', '.join(hosts[::G])}", flush=True)
            base = stats[0, 1].item()
            for mode, (err, ms, syncs) in zip(modes, stats.tolist()):
                print(f"  {mode:>12}: {ms:7.3f} ms | {base / ms:5.2f}x vs plain | "
                      f"{int(syncs)} host syncs | rel err {err:.1e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
