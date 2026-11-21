#!/bin/bash
cd /global/cfs/cdirs/m5421/eku/speculative_expert_parallelism

# NCCL + RDMA libraries.
module load nccl/2.29.2-cu13 # provides libnccl-net.so (aws-ofi-nccl)
export NCCL_NET="AWS Libfabric" # fail loudly instead of silently falling back to Socket
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_DEBUG=NET
# export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT

export LOGS="logs/$(date +%Y-%m-%d_%H-%M-%S)"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

mkdir -p "$LOGS/profile"

# Load conda with import caching on all nodes
module load conda
source ~/conda_trick.sh

# # Scripts import `benchmark.*`, which needs the repo root on the path.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo Running benchmark...
# srun -N 2 --ntasks-per-node=4 python -u -m benchmark.all2all.combine_dispatch \
#   logs/2026_oct/2026-10-16_14-07-50 0

# Every mode but deepep. USE_NVRTC=1: Tutel compiles its kernels without forking nvcc.
# USE_NVRTC=1 srun -N 2 --ntasks-per-node=4 python -u -m benchmark.all2all.combine_dispatch \
#   logs/2026_oct/2026-10-16_14-07-50 0 --degree 2 \
#   --modes plain plain_fused rbd two_lane plain_pipe rbd_pipe tutel tutel_pipe

# DeepEP against plain. This DeepEP build needs every rank on one node, and -w keeps
# rank 0 on MASTER_ADDR, where the other ranks look for it.
# srun -N 1 -w "$MASTER_ADDR" --ntasks-per-node=4 python -u -m benchmark.all2all.combine_dispatch \
#   logs/2026_oct/2026-10-16_14-07-50 0 --modes plain deepep

srun -N 2 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py \
  --balance even

# echo \n\n\n
# srun -N 2 --ntasks-per-node=4 python -u benchmark/internode/expert_all2all.py \
  # --balance uneven_internode