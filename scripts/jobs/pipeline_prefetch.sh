#!/bin/bash
# One 2-node allocation for the 2026-11-22 pipeline and prefetch experiments: what an
# all2all and the experts cost each other on one GPU, the RBD pipeline planned in one
# exchange against RBD on a layer and in the Qwen model, and RBD with guest experts
# loaded from host memory (scripts/ds_guests.py).
# Every step has a time limit and a failure only skips that step. STEPS picks the steps.
#
# Usage:
#   OUT=logs/2026-11-22_pipeline/2n salloc -N 2 -q interactive -t 00:20:00 -C gpu \
#       --gpus-per-node=4 -A m5421_g bash -l scripts/jobs/pipeline_prefetch.sh
#   STEPS=guests GUEST_MODES="rbd rbd_guests:slots=8" OUT=... salloc ... -C "gpu&hbm80g" \
#       bash -l scripts/jobs/pipeline_prefetch.sh
cd "${REPO:-$(dirname "$(readlink -f "$0")")/../..}"
REPO=$PWD
OUT=${OUT:-logs/$(date +%Y-%m-%d_%H-%M-%S)}
mkdir -p "$OUT"
# Run from a copy: bash reads a script as it goes, so an edit to this file during the
# job would break it, and the copy records what ran.
if [ -z "$JOB_COPY" ]; then
  cp "$REPO/scripts/jobs/pipeline_prefetch.sh" "$OUT/job.sh"
  REPO=$REPO OUT=$OUT JOB_COPY=1 exec bash -l "$OUT/job.sh"
fi
STEPS=${STEPS:-"contention pipe model guests"}
RUN=/global/cfs/cdirs/m5421/eku/speculative_expert_parallelism/logs/2026_oct/2026-10-16_14-07-50
N=$SLURM_JOB_NUM_NODES

# NCCL + RDMA settings, as in scripts/run_benchmark.sh.
module load nccl/2.29.2-cu13
export NCCL_NET="AWS Libfabric"
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
# .deps holds this checkout's spec_ep and DeepSpeed ahead of the env's installed ones.
export PYTHONPATH="$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export USE_NVRTC=1  # Tutel compiles its kernels in process
export TRITON_CACHE_DIR=/tmp/triton_$USER  # a cache on Lustre has hung ranks before

module load conda
source ~/conda_trick.sh
PY=/dev/shm/spec-EP/bin/python  # the env is only on the compute nodes

wants() { [[ " $STEPS " == *" $1 "* ]]; }
# step NAME SECONDS COMMAND...: run one step into $OUT/NAME.out with a time limit.
step() {
  local name=$1 limit=$2; shift 2
  echo "##### $name $(date +%T)"
  timeout "$limit" "$@" > "$OUT/$name.out" 2>&1 || echo "##### $name FAILED (exit $?)"
}
all="srun -N $N --ntasks-per-node=4 --kill-on-bad-exit=1"
bench="$PY -u -m benchmark.all2all.combine_dispatch $RUN"
model="$PY -u scripts/ds_moe.py --config-pth configs/qwen_ep$((4 * N)).json"

# What an all2all and the experts cost each other when they overlap.
if wants contention; then
  step contention 120 $all $PY -u -m benchmark.all2all.contention
  step contention_heavy 120 $all $PY -u -m benchmark.all2all.contention --intermediate 2048
fi

# The pipeline on one layer with real feed-forward experts, at the model's expert
# width and at four times it, where the experts are a larger share of the layer.
if wants pipe; then
  pipes="plain rbd two_lane rbd_pipe:degree=1 rbd_pipe:degree=2 rbd_pipe:degree=3 plain_pipe"
  step pipe_ffn 200 $all $bench 0 --repeats 5 --iterations 2 --expert ffn --modes $pipes
  step pipe_ffn_heavy 200 $all $bench 0 --repeats 5 --iterations 2 --expert ffn \
      --intermediate 2048 --modes $pipes
  step pipe_identity 150 $all $bench 0 --repeats 5 --iterations 2 --modes $pipes
fi

# The pipeline in the Qwen model, with the forward as comm and compute events.
if wants model; then
  MODEL_MODES=${MODEL_MODES:-"rbd rbd_pipe:degree=1 rbd_pipe:degree=2 rbd_pipe:degree=3 \
      two_lane tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2,per_peer=True"}
  LOGS=$OUT/model step model 600 $all $model --repeats 5 --profile events --modes $MODEL_MODES
fi

# Guest experts in the Qwen model: slots, placement, source, and one lane against two.
# The pinned store takes about 30 GiB of host memory per rank.
if wants guests; then
  GUEST_MODES=${GUEST_MODES:-"rbd two_lane rbd_guests:slots=4 rbd_guests:slots=8 \
      rbd_guests:slots=16 rbd_guests:slots=32 rbd_guests:slots=16,placement=redundant \
      rbd_guests:slots=16,source=static rbd_guests:slots=16,source=oracle \
      rbd_guests:slots=16,source=static,placement=redundant rbd_guests:slots=16,lanes=1 autoep"}
  LOGS=$OUT/guests step guests 900 $all $model --repeats 5 --profile events --modes $GUEST_MODES
fi
# More slot counts between the ones of the guests step, to find the largest whose load
# still ends before its MoE layer does.
if wants slots; then
  SLOT_MODES=${SLOT_MODES:-"rbd two_lane rbd_guests:slots=16 rbd_guests:slots=20 \
      rbd_guests:slots=24 rbd_guests:slots=28 rbd_guests:slots=32 \
      rbd_guests:slots=24,source=static rbd_guests:slots=24,placement=redundant"}
  LOGS=$OUT/slots step slots 900 $all $model --repeats 5 --profile --modes $SLOT_MODES
fi
# One table of the main modes from one process, with the guest store checked against
# the ranks that own its experts.
if wants final; then
  FINAL_MODES=${FINAL_MODES:-"autoep rbd two_lane rbd_pipe:degree=2 rbd_guests:slots=24 \
      rbd_guests:slots=24,source=static rbd_guests:slots=24,source=oracle \
      rbd_guests:slots=24,placement=redundant"}
  LOGS=$OUT/final step final 600 $all $model --repeats 5 --profile --modes $FINAL_MODES
fi
echo "##### done $(date +%T)"
