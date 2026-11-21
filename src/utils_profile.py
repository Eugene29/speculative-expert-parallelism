import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function
from deepspeed.profiling.flops_profiler import FlopsProfiler
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

from utils import get_rank_and_world_size, print0

rank, world_size = get_rank_and_world_size()

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    expert = sum(p.numel() for n, p in m.named_parameters() if "experts" in n)
    return total, expert

def add_expert_statistics_hook(model, sent):
    """sent[i] <- expert ids [T, K] chosen by each token on this rank in layer i.
    
    Assumption: 
        Experts ids are contiguous on each GPU.
    """
    for i, layer in enumerate(m for m in model.modules() if isinstance(m, AutoEPMoELayer)):
        def hook(_m, _in, out, i=i, layer=layer):
            sent[i] = out[1].to(torch.int16).cpu()
        layer.router.register_forward_hook(hook)
        
def report_peak_mem(dev):
    peak_alloc = torch.cuda.max_memory_allocated(dev) / 2**30
    peak_resv = torch.cuda.max_memory_reserved(dev) / 2**30

    for i in range(world_size):
        dist.barrier()
        if i == rank:
            print(f"rank {rank}: peak allocated {peak_alloc:.2f} GiB, "
                  f"peak reserved {peak_resv:.2f} GiB", flush=True)

def start_flops_profile(engine):
    prof = FlopsProfiler(engine.module)
    prof.start_profile()
    return prof
    
def end_flops_profile(prof, outdir):
    prof.stop_profile()
    prof.print_model_profile(profile_step=1, module_depth=-1, top_modules=3,
                                detailed=True, 
                                output_file=str(outdir / f"flops_profile_{rank}.txt"))
    params = prof.get_total_params()
    print0(f"{params/1e9:.2f}B local params")
    prof.end_profile()
    
def torch_profile_ctx():
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,        # True gives source lines but makes traces much larger
        with_flops=True,
    )

def save_torch_profile(prof, outdir, verbose=False):
    trace = outdir / "profile" / f"trace_rank{rank}.json"
    prof.export_chrome_trace(str(trace))

    events = prof.key_averages()

    if verbose and (rank == 0):
        table = events.table(sort_by="cuda_time_total", row_limit=30)
        (outdir / "profile" / "key_averages.txt").write_text(table)
