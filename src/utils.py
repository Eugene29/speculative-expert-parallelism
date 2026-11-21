import os

import torch


def print0(msg):
    if rank == 0:
        print(msg, flush=True)
        
def shutdown_comm():
    # Tear down NCCL cleanly so ranks don't race on rank 0's TCPStore at exit.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        
def get_rank_and_world_size():
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return rank, world_size
    
rank, world_size = get_rank_and_world_size()