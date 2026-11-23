#!/bin/bash
# One allocation for the experiments on Qwen3.5-397B-A17B, the target workload: the
# plan's cost on one GPU, and the model under every mode with the forward as comm and
# compute events. The model needs 16 GPUs of 80 GB. Two nodes hold its first 30 layers
# (LAYERS=30, the default), four hold all 60 (LAYERS=).
# Every step has a time limit and a failure only skips that step. STEPS picks the steps.
# OLD names a checkout from before a change, to time both in one job.
#
# Usage:
#   OUT=logs/2026-11-23_gpu_prefetch/2n salloc -N 2 -q interactive -t 00:40:00 \
#       -C "gpu&hbm80g" --gpus-per-node=4 -A m5421_g bash -l scripts/jobs/gpu_prefetch.sh
#   LAYERS= STEPS=model MODES="autoep two_lane rbd_guests:slots=8,source=static" OUT=... \
#       salloc -N 4 ... bash -l scripts/jobs/gpu_prefetch.sh
cd "${REPO:-$(dirname "$(readlink -f "$0")")/../..}"
REPO=$PWD
OUT=${OUT:-logs/$(date +%Y-%m-%d_%H-%M-%S)}
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
# Run from a copy: bash reads a script as it goes, so an edit to this file during the
# job would break it, and the copy records what ran.
if [ -z "$JOB_COPY" ]; then
  cp "$REPO/scripts/jobs/gpu_prefetch.sh" "$OUT/job.sh"
  REPO=$REPO OUT=$OUT JOB_COPY=1 exec bash -l "$OUT/job.sh"
fi
STEPS=${STEPS:-"plan model"}
N=$SLURM_JOB_NUM_NODES
MODEL=${MODEL:-/pscratch/sd/e/eku/spec_ep/models/Qwen3.5-397B-A17B}
LAYERS=${LAYERS-30}
S=${SLOTS:-8}

# NCCL + RDMA settings, as in scripts/run_benchmark.sh.
module load nccl/2.29.2-cu13
export NCCL_NET="AWS Libfabric"
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export USE_NVRTC=1  # Tutel compiles its kernels in process
export TRITON_CACHE_DIR=/tmp/triton_$USER  # a cache on Lustre has hung ranks before

module load conda
source ~/conda_trick.sh
PY=/dev/shm/spec-EP/bin/python  # the env is only on the compute nodes

wants() { [[ " $STEPS " == *" $1 "* ]]; }
# step NAME SECONDS TREE COMMAND...: run one step from a checkout into $OUT/NAME.out,
# with a time limit. TREE/.deps holds that checkout's spec_ep and DeepSpeed.
step() {
  local name=$1 limit=$2 tree=$3; shift 3
  echo "##### $name $(date +%T)"
  (cd "$tree" && PYTHONPATH="$tree/.deps:$tree" timeout "$limit" "$@") > "$OUT/$name.out" 2>&1 \
      || echo "##### $name FAILED (exit $?)"
}
all="srun -N $N --ntasks-per-node=4 --kill-on-bad-exit=1"
one="srun -N 1 -n 1 --gpus=1 -w $MASTER_ADDR"
model="$PY -u scripts/ds_moe.py --config-pth configs/qwen_ep$((4 * N)).json --model $MODEL \
    ${LAYERS:+--layers $LAYERS} --repeats 5 --profile events"

# The plan's compute on one GPU, at the model's layer shape on 2 and on 4 nodes.
if wants plan; then
  for tree in ${OLD:+old:$OLD} new:$REPO; do
    for world in 8 16; do
      step plan_${tree%%:*}_$world 120 ${tree#*:} $one $PY -u benchmark/plan_cost.py --world $world
    done
  done
fi

# The checkout before the change, on the same nodes.
if wants old; then
  OLD_MODES=${OLD_MODES:-"rbd two_lane rbd_guests:slots=$S,source=static"}
  LOGS=$OUT/old step old 1500 $OLD $all $model --modes $OLD_MODES
fi

# The modes. Guests: how many fit under a layer, where they come from, the two
# placements, and the two choices.
if wants model; then
  MODES=${MODES:-"autoep plain rbd two_lane \
      rbd_guests:slots=4,source=static rbd_guests:slots=$S,source=static \
      rbd_guests:slots=16,source=static rbd_guests:slots=$S rbd_guests:slots=$S,source=oracle \
      rbd_guests:slots=$S,source=static,placement=redundant \
      rbd_guests:slots=$S,source=static,choice=pilots \
      rbd_guests:slots=$S,source=static,choice=pilots,balance=2"}
  LOGS=$OUT/model step model 1800 $REPO $all $model --record --modes $MODES
fi
echo "##### done $(date +%T)"
