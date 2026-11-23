"""Dry-run a benchmark module or a model script on the login node, over gloo on its one GPU.

Spawns --ranks processes that share GPU 0 and talk over gloo, pretending to be
--gpus-per-node ranks per node. A target that ends in .py runs as a script started by
srun (scripts/ds_moe.py); any other target is a module of benchmark/ whose
benchmark.utils.init_dist and check_placement are replaced. Outputs are real, so the
run checks that code runs and that modes agree. Timings mean nothing here.

Usage:
    export TRITON_CACHE_DIR=/tmp/triton_$USER PYTHONPATH=$PWD/.deps:$PWD
    # Every mode of the model on a tiny checkpoint, in float32, against autoep.
    LOGS=/tmp/dry_$USER python scripts/login_dry_run.py scripts/ds_moe.py \
        --model /dev/shm/tiny_qwen_eku --config-pth configs/qwen_ep4.json --gpus-per-node 2 \
        --seq-len 512 --dtype float32 --profile events --modes autoep rbd two_lane
    # One layer on recorded routing.
    python scripts/login_dry_run.py benchmark.all2all.combine_dispatch RUN_DIR 0 \
        --scaling weak --hidden 256 --gpus-per-node 2 --modes plain rbd
"""
import argparse
import os
import runpy
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(
    rank: int,
    world: int,
    gpus_per_node: int,
    init_file: str,
    target: str,
    argv: list[str],
) -> None:
    """Run the target as __main__ on one rank, with a gloo process group behind it."""
    os.environ.update(SLURM_PROCID=str(rank), SLURM_NTASKS=str(world), SLURM_LOCALID="0")
    sys.argv = [target, *argv]

    def init_gloo() -> None:
        torch.cuda.set_device(0)
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                                world_size=world)

    if target.endswith(".py"):
        # DeepSpeed uses the process group it finds.
        init_gloo()
        sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
        runpy.run_path(target, run_name="__main__")
        return

    from benchmark import utils

    def init_dist() -> tuple[int, int, torch.device]:
        init_gloo()
        return rank, world, torch.device("cuda", 0)

    utils.init_dist = init_dist
    utils.check_placement = lambda g: [f"fake{r // g}" for r in range(world)]
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--fake-gpus-per-node", type=int, default=2,
                        help="Ranks per pretended node; pass the target its own flag too")
    parser.add_argument("target", help="A script path ending in .py, or a module name")
    parser.add_argument("argv", nargs=argparse.REMAINDER, help="The target's arguments")
    args = parser.parse_args()
    init_file = os.path.join(tempfile.mkdtemp(), "init")
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(r, args.ranks, args.fake_gpus_per_node, init_file,
                                         args.target, args.argv))
        for r in range(args.ranks)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode for p in procs))
