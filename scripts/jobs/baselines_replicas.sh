#!/bin/bash
# One allocation of 2 or 4 nodes for the 2026-11-22 experiments: every dispatch library
# in the layer benchmark and in the Qwen model, with traces; Tutel's pipelining; and
# replicas chosen to cut inter-node rows; and NCCL's channels per peer. The node count
# comes from the allocation.
# Every step has a time limit and a failure only skips that step. STEPS picks the steps
# to run, default all but model_events, which traces the model's forward as events.
#
# Usage:
#   OUT=logs/2026-11-22_exp/2n salloc -N 2 -q interactive -t 00:25:00 -C gpu \
#       --gpus-per-node=4 -A m5421_g bash -l scripts/jobs/baselines_replicas.sh
#   OUT=logs/2026-11-22_exp/4n salloc -N 4 ... bash -l scripts/jobs/baselines_replicas.sh
#   STEPS="micro model" MODEL_MODES="autoep rbd xmoe" OUT=... salloc ... \
#       bash -l scripts/jobs/baselines_replicas.sh
#   STEPS=model_events OUT=logs/2026-11-22_events/2n salloc -N 2 ... \
#       bash -l scripts/jobs/baselines_replicas.sh
cd /global/cfs/cdirs/m5421/eku/speculative_expert_parallelism
OUT=${OUT:-logs/$(date +%Y-%m-%d_%H-%M-%S)}
mkdir -p "$OUT/traces" "$OUT/traces_ffn"
# Run from a copy: bash reads a script as it goes, so an edit to this file during the
# job would break it, and the copy records what ran.
if [ -z "$JOB_COPY" ]; then
  cp "$0" "$OUT/job.sh"
  OUT=$OUT JOB_COPY=1 exec bash -l "$OUT/job.sh"
fi
STEPS=${STEPS:-"micro xmoe deepep pipe tutel channels replicas model model_deepep model_tuned"}
RUN=logs/2026_oct/2026-10-16_14-07-50
REC=${REC:-logs/2026-11-21_exp/routing}  # other batches, for the static replica choice
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
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
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
timing="--repeats 5 --iterations 2"

# Every library on one layer's recorded routing, identity experts, with traces.
if wants micro; then
  libraries="plain plain_fused autoep autoep:fused=True ds_moe:capacity_factor=1 ds_moe \
      tutel:capacity_factor=1 tutel rbd two_lane"
  step micro_layer0 240 $all $bench 0 $timing --trace "$OUT/traces" --modes $libraries
  step micro_layer20 200 $all $bench 20 $timing --modes $libraries
fi
# X-MoE's own RBD, in its own step: its adapter loads a second DeepSpeed fork.
if wants xmoe; then
  xmoe="plain rbd rbd:pilot_rule=first two_lane xmoe xmoe:permute=False xmoe:capacity_factor=1"
  step xmoe_layer0 180 $all $bench 0 $timing --trace "$OUT/traces" --modes $xmoe
  step xmoe_layer20 150 $all $bench 20 $timing --modes $xmoe
fi

# DeepEP across nodes: a DeepEP build for 4 ranks per node over NVSHMEM's libfabric
# transport (/pscratch/sd/e/eku/deepep_multinode), first on the path, with the
# source-built NVSHMEM loaded before torch loads the pip wheel's. 32 CPUs per rank give
# NVSHMEM's proxy thread its own core. The tuned setting has its own step: a setting
# DeepEP rejects on this node count must not take the default's numbers with it.
DEEPEP=/pscratch/sd/e/eku/deepep_multinode/build
deepep_env="env PYTHONPATH=$(ls -d $DEEPEP/deepep-nvl4/lib.*):$PYTHONPATH \
    LD_PRELOAD=$DEEPEP/nvshmem-3.4.5-install/lib/libnvshmem_host.so.3 \
    NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=hsn DEEPEP_SKIP_NVLINK_CHECK=1"
wide="srun -N $N --ntasks-per-node=4 -c 32 --kill-on-bad-exit=1"
if wants deepep; then
  step deepep_layer0 150 $deepep_env $wide $bench 0 $timing --trace "$OUT/traces" --modes \
      plain plain_fused rbd two_lane deepep deepep:fused=True deepep:fused=True,num_sms=48
  step deepep_layer20 120 $deepep_env $wide $bench 20 $timing --modes \
      plain plain_fused rbd two_lane deepep:fused=True deepep:fused=True,num_sms=48
  step deepep_tuned_layer0 120 $deepep_env $wide $bench 0 $timing --trace "$OUT/traces" --modes \
      plain plain_fused two_lane deepep:fused=True,num_sms=48,rdma_chunk=128 \
      deepep:fused=True,num_sms=64,rdma_chunk=64
fi

# Pipelining with a real feed-forward expert: what each library hides under the network.
if wants pipe; then
  step pipe_ffn 200 $all $bench 0 $timing --expert ffn --trace "$OUT/traces_ffn" --modes \
      plain rbd tutel:capacity_factor=1 \
      tutel:capacity_factor=1,degree=2 tutel:capacity_factor=1,degree=4
fi
# Why Tutel's pipeline is slow: one all2all under each way of grouping its sends, then
# the pipeline with one message per peer instead of one per expert.
if wants tutel; then
  step tutel_a2a_chunks 110 $all $PY -u /pscratch/sd/e/eku/tutel_pipeline/a2a_chunks.py
  step tutel_per_peer 200 $all $bench 0 $timing --expert ffn --trace "$OUT/traces_ffn" --modes \
      plain tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2 \
      tutel:capacity_factor=1,degree=2,per_peer=True tutel:capacity_factor=1,degree=4,per_peer=True
fi

# NCCL gives every peer of a multi-node communicator the channel count of its network
# peers (2 here), so same-node rows get fewer channels than in the node's own
# communicator (8). NCCL_NCHANNELS_PER_NET_PEER raises it for all peers, and
# NCCL_MIN_P2P_NCHANNELS the communicator's total. channels SETTING runs a command
# under one setting: default, a per-peer count, or count+total.
channels() {
  local setting=$1; shift
  if [ "$setting" = default ]; then "$@"; return; fi
  (
    export NCCL_NCHANNELS_PER_NET_PEER=${setting%%+*}
    [[ $setting == *+* ]] && export NCCL_MIN_P2P_NCHANNELS=${setting#*+}
    "$@"
  )
}
best="pilots"; [ "$N" -gt 2 ] && best="cover --cover"
if wants channels; then
  modes="$timing --modes plain plain_fused autoep:fused=True tutel:capacity_factor=1 rbd two_lane"
  for c in default 4 8 8+32; do
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT channels $c step channels_${c}_init 60 \
        $all $bench 0 --repeats 1 --modes plain
    channels $c step channels_${c} 120 $all $bench 0 $modes
    channels $c step channels_${c}_replicas 120 $all $bench 0 $modes --replicas $best --budget 64
  done
fi

# Replicas: rules that choose them, source against cover routing, chosen from the timed
# batch (oracle) against chosen from other batches (static). All under TUNED channels
# per peer, so that rows kept on the node are not held back by the default of 2.
if wants replicas || wants model_tuned; then
  # TUNED defaults to the setting that made rbd fastest with replicas in the channels step.
  if [ -z "$TUNED" ]; then
    TUNED=$(for c in default 4 8 8+32; do
      ms=$(grep -m1 ' rbd: median' "$OUT/channels_${c}_replicas.out" 2>/dev/null | awk '{print $3}')
      [ -n "$ms" ] && echo "$ms $c"
    done | sort -n | head -1 | awk '{print $2}')
    TUNED=${TUNED:-8}
  fi
  echo "##### TUNED channels per peer: $TUNED"
fi
if wants replicas; then
  tuned="channels $TUNED"
  modes="$timing --modes plain plain_fused rbd two_lane"
  for layer in 0 20; do
    $tuned step replicas_layer${layer}_none 120 $all $bench $layer $modes
  done
  rules="copies pilots"; [ "$N" -gt 2 ] && rules="copies pilots cover:cover ring:cover copies:cover"
  for rule in $rules; do
    name=${rule/:/_}; flags="--replicas ${rule%%:*} --budget 64"
    [[ $rule == *:cover ]] && flags="$flags --cover"
    $tuned step replicas_layer0_${name}_oracle 120 $all $bench 0 $modes $flags
    [[ $rule == ring* ]] || $tuned step replicas_layer0_${name}_static 120 $all $bench 0 $modes $flags --choose-from "$REC"
  done
  $tuned step replicas_layer0_best_32_oracle 120 $all $bench 0 $modes --replicas $best --budget 32
  $tuned step replicas_layer20_best_64_oracle 120 $all $bench 20 $modes --replicas $best --budget 64
  $tuned step replicas_layer20_best_64_static 120 $all $bench 20 $modes --replicas $best --budget 64 --choose-from "$REC"
fi

# The Qwen model end to end, one load, every library as the MoE layers' dispatch.
# MODEL_MODES narrows the modes.
if wants model; then
  MODEL_MODES=${MODEL_MODES:-"autoep autoep:fused=True plain rbd two_lane \
      ds_moe:capacity_factor=1 tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2 \
      tutel:capacity_factor=1,degree=2,per_peer=True xmoe"}
  LOGS=$OUT/model step model 720 $all $PY -u scripts/ds_moe.py \
      --config-pth configs/qwen_ep$((4 * N)).json --repeats 3 --trace-ranks 4 --modes $MODEL_MODES
fi
# The model with DeepEP as the dispatch, in its own step and process environment.
if wants model_deepep; then
  LOGS=$OUT/model_deepep step model_deepep 600 $deepep_env $wide $PY -u scripts/ds_moe.py \
      --config-pth configs/qwen_ep$((4 * N)).json --repeats 3 --modes \
      ${DEEPEP_MODES:-autoep two_lane deepep:fused=True,num_sms=48}
fi
# The model again under TUNED channels per peer, for the modes it changes.
if wants model_tuned; then
  LOGS=$OUT/model_tuned channels $TUNED step model_tuned 600 $all $PY -u scripts/ds_moe.py \
      --config-pth configs/qwen_ep$((4 * N)).json --repeats 3 --modes \
      ${TUNED_MODES:-autoep plain plain_fused rbd two_lane xmoe}
fi
# Where the model's forward goes under each dispatch, as events on the GPU clock
# (scripts/ds_moe.py --profile events): every mode in one process, a traced forward
# after each timed one, a trace file per mode and rank under $OUT/model_events/events.
# Tutel has not run under the DeepEP environment before, so if that step does not
# reach its last line, the NCCL modes and DeepEP run as two steps.
if wants model_events; then
  events="$PY -u scripts/ds_moe.py --config-pth configs/qwen_ep$((4 * N)).json --repeats 5 \
      --profile events --modes"
  nccl_modes="rbd two_lane tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2"
  deepep_modes="deepep:fused=True,num_sms=48,rdma_chunk=128"
  LOGS=$OUT/model_events step model_events 480 $deepep_env $wide $events $nccl_modes $deepep_modes
  if ! grep -q "full garbage collections" "$OUT/model_events.out"; then
    LOGS=$OUT/model_events_nccl step model_events_nccl 420 $all $events $nccl_modes
    LOGS=$OUT/model_events_deepep step model_events_deepep 360 $deepep_env $wide $events $deepep_modes
  fi
fi
echo "##### DONE $(date +%T)"
