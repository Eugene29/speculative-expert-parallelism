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
python="/dev/shm/spec-EP/bin/python"

mkdir -p "$LOGS/profile"

# Broadcast the env tarball to every node
module load conda
source ~/conda_trick.sh

# TODO: Logger info
# TODO: install tutel
# TODO: Try tutel and DeepEP and profile them

# CONFIG picks the DeepSpeed config (several, space separated, share one model load and only
# switch the comm backend), MODEL another qwen3_5_moe checkpoint, SEQ_LEN the tokens
# per rank, RECORD=1 also saves the expert routing to $LOGS/experts_rank{r}.pt,
# PROFILE picks the profilers (space separated: torch, flops, comm; default torch), and
# SRUN_ARGS narrows the step (e.g. "-N 1 -w <first node>" for EP4 on one node).
srun ${SRUN_ARGS:-} --ntasks-per-node=${TASKS_PER_NODE:-4} $python -u scripts/ds_moe.py \
    --config-pth ${CONFIG:-configs/qwen_ep4.json} \
    ${MODEL:+--model "$MODEL"} ${SEQ_LEN:+--seq-len "$SEQ_LEN"} ${RECORD:+--record} \
    ${PROFILE:+--profile $PROFILE} \
    2>&1 | tee "$LOGS/run.log"
# Q. need --gpu-bind=none?
    # --config-pth="configs/qwen_ep8.json" \