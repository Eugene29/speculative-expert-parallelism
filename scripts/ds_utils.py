"""Helpers for scripts/ds_moe.py: rank bookkeeping, the routing hook and profiling.

Importing this module reads the rank from the launcher's environment, so import it
only from a script started by srun or the DeepSpeed launcher.

Usage:
    from ds_utils import rank, print0, count_params, add_expert_statistics_hook
"""
import os
from pathlib import Path

import torch
import torch.distributed as dist
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch.profiler import ProfilerActivity, profile
from torch.utils.hooks import RemovableHandle


def get_rank_and_world_size() -> tuple[int, int]:
    """Read this process's rank and the world size.

    Under plain srun it first copies the SLURM variables into RANK, WORLD_SIZE and
    LOCAL_RANK, which DeepSpeed needs to start its process group.

    Returns:
        rank: this rank.
        world_size: number of ranks.
    """
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return rank, world_size


rank, world_size = get_rank_and_world_size()


def print0(msg: str) -> None:
    """Print on rank 0 only."""
    if rank == 0:
        print(msg, flush=True)


def shutdown_comm() -> None:
    """Tear down NCCL so ranks don't race on rank 0's TCPStore at exit. Collective."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """Count the model's parameters.

    Returns:
        total: all parameters.
        expert: parameters whose name contains "experts".
    """
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for n, p in model.named_parameters() if "experts" in n)
    return total, expert


def add_expert_statistics_hook(
    model: torch.nn.Module,
    sent: dict[int, torch.Tensor],
) -> list[RemovableHandle]:
    """Hook every AutoEP router so a forward fills sent with the chosen experts.

    After a forward, sent[i] is a [num_tokens, k] int16 tensor on CPU: the expert ids
    each token on this rank chose in MoE layer i.

    Returns:
        handles: one per layer; remove() them to stop recording.
    """
    layers = [m for m in model.modules() if isinstance(m, AutoEPMoELayer)]
    handles = []
    for i, layer in enumerate(layers):
        def hook(_module: torch.nn.Module, _inputs: tuple, out: tuple, i: int = i) -> None:
            """Keep the router's second output, the expert ids."""
            sent[i] = out[1].to(torch.int16).cpu()
        handles.append(layer.router.register_forward_hook(hook))
    return handles


def report_peak_mem(dev: torch.device) -> None:
    """Print every rank's peak GPU memory, in rank order. Collective."""
    peak_alloc = torch.cuda.max_memory_allocated(dev) / 2**30
    peak_resv = torch.cuda.max_memory_reserved(dev) / 2**30

    for i in range(world_size):
        dist.barrier()
        if i == rank:
            print(f"rank {rank}: peak allocated {peak_alloc:.2f} GiB, "
                  f"peak reserved {peak_resv:.2f} GiB", flush=True)


def start_flops_profile(model: torch.nn.Module) -> FlopsProfiler:
    """Start DeepSpeed's flops profiler on the model.

    Returns:
        prof: the running profiler, to pass to end_flops_profile().
    """
    prof = FlopsProfiler(model)
    prof.start_profile()
    return prof


def end_flops_profile(prof: FlopsProfiler, outdir: Path) -> None:
    """Stop the flops profiler and write this rank's table to outdir/flops_profile_{r}.txt.

    The table lists flops, MACs, parameters and latency per module. Rank 0 also prints
    its parameter count.
    """
    prof.stop_profile()
    prof.print_model_profile(profile_step=1, module_depth=-1, top_modules=3, detailed=True,
                             output_file=str(outdir / f"flops_profile_{rank}.txt"))
    print0(f"{prof.get_total_params() / 1e9:.2f}B local params")
    prof.end_profile()


def torch_profile_ctx() -> profile:
    """Make a torch profiler of CPU and CUDA time, shapes, memory and flops.

    Returns:
        prof: the profiler, to use as a context manager.
    """
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,        # True gives source lines but makes traces much larger
        with_flops=True,
    )


def save_torch_profile(prof: profile, outdir: Path, verbose: bool = False) -> None:
    """Write this rank's trace to outdir/profile/trace_rank{r}.json.

    With verbose, rank 0 also writes the 30 ops with the most CUDA time to
    outdir/profile/key_averages.txt.
    """
    trace = outdir / "profile" / f"trace_rank{rank}.json"
    prof.export_chrome_trace(str(trace))

    events = prof.key_averages()

    if verbose and (rank == 0):
        table = events.table(sort_by="cuda_time_total", row_limit=30)
        (outdir / "profile" / "key_averages.txt").write_text(table)
