#!/bin/bash
# One 2-node allocation for three experiments: baseline settings (Tutel, DeepSpeed's
# MOELayer, DeepEP), hot-expert replication, and loading experts from host memory
# against other GPUs. Every step has a time limit and a failure only skips that step.
# STEPS picks the steps to run, default all.
#
# Usage:
#   OUT=logs/2026-11-21_exp salloc -N 2 -q interactive -t 00:28:00 -C gpu \
#       --gpus-per-node=4 -A m5421_g bash -l scripts/jobs/baselines_hot_load.sh
#   STEPS="tutel load" OUT=... salloc ... bash -l scripts/jobs/baselines_hot_load.sh
cd /global/cfs/cdirs/m5421/eku/speculative_expert_parallelism
OUT=${OUT:-logs/$(date +%Y-%m-%d_%H-%M-%S)}
STEPS=${STEPS:-"record deepep load1 tutel load deepep2 hot"}
RUN=logs/2026_oct/2026-10-16_14-07-50
REC=${REC:-$OUT/routing}  # set REC to reuse an earlier recording
mkdir -p "$OUT/traces_1node" "$OUT/traces_2node" "$REC"

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
NODE2=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n1)

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
one="srun -N 1 -w $MASTER_ADDR --ntasks-per-node=4"
two="srun -N 2 --ntasks-per-node=4"
bench="$PY -u -m benchmark.all2all.combine_dispatch $RUN"

# Routing of 59 more batches, on the second node while the first runs one-node steps.
# Batch 0 is recorded last and compared with the DeepSpeed recording of the same tokens.
if wants record; then
  LOGS=$REC step record 600 srun -N 1 -n 1 -c 32 -w "$NODE2" --gpus=4 \
      $PY -u scripts/record_routing.py \
      --text data/long_text.txt --batches $(seq 8 31) $(seq 40 8 312) 0 &
fi

if wants deepep; then
  step deepep 240 $one $bench 0 --repeats 5 --iterations 4 --trace "$OUT/traces_1node" --modes \
      plain plain_fused deepep deepep:fused=True \
      deepep:num_sms=48 \
      deepep:fused=True,num_sms=8 deepep:fused=True,num_sms=32 deepep:fused=True,num_sms=48 \
      deepep:fused=True,num_sms=64 deepep:fused=True,num_sms=96 \
      deepep:fused=True,dispatch_chunk=16 deepep:fused=True,dispatch_chunk=32 \
      deepep:fused=True,combine_chunk=4 deepep:fused=True,combine_chunk=16 \
      deepep:fused=True,num_sms=48,dispatch_chunk=32,combine_chunk=16 \
      deepep:fused=True,num_sms=96,dispatch_chunk=32,combine_chunk=16
fi
if wants load1; then
  step load_1node 150 $one $PY -u -m benchmark.internode.expert_load --busy-rows 65536
fi
wait
if wants record && [ -f "$REC/experts_rank59.pt" ]; then
  mv "$REC/experts_rank59.pt" "$REC/check_batch0.pt"
  /global/cfs/cdirs/m5421/eku/envs/spec-EP/bin/python - "$REC/check_batch0.pt" \
      "$RUN/experts_rank0.pt" > "$OUT/record_check.out" 2>&1 <<'PYEOF'
import sys
import torch
new, old = (torch.load(path).long() for path in sys.argv[1:])
chosen = [torch.zeros(*r.shape[:2], 256, dtype=torch.bool).scatter_(-1, r, True) for r in (new, old)]
shared = (chosen[0] & chosen[1]).sum(-1) / 8.0
print(f"copies that match the DeepSpeed recording, over all layers: {shared.mean():.2%}")
print(f"  by layer, every 4th: {[round(shared[l].mean().item(), 3) for l in range(0, 40, 4)]}")
print(f"tokens whose 8 experts all match: {(shared == 1).float().mean():.2%}")
hot = [c.sum(1).topk(32).indices for c in chosen]
overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(*hot)) / hot[0].numel()
print(f"32 busiest experts per layer that match: {overlap:.2%}")
PYEOF
fi

if wants tutel; then
  step tutel_layer0 240 $two $bench 0 --repeats 5 --iterations 2 --trace "$OUT/traces_2node" --modes \
      plain plain_fused rbd two_lane \
      tutel tutel:capacity_factor=1 tutel:capacity_factor=2 tutel:capacity_factor=4 \
      tutel:capacity_factor=1,degree=2 tutel:capacity_factor=1,degree=4 \
      ds_moe ds_moe:capacity_factor=1 ds_moe:capacity_factor=2 ds_moe:capacity_factor=4
  step tutel_dtype 180 $two $bench 0 --repeats 5 --iterations 2 --trace "$OUT/traces_2node" --modes \
      plain tutel:capacity_factor=1,half=True tutel:capacity_factor=4,half=True tutel:half=True
  step ds_dense 180 $two $bench 0 --repeats 5 --iterations 2 --trace "$OUT/traces_2node" --modes \
      plain tutel:capacity_factor=1 ds_moe:capacity_factor=1 \
      ds_moe:capacity_factor=1,dense=True ds_moe:capacity_factor=2,dense=True
  step tutel_ffn 180 $two $bench 0 --repeats 5 --iterations 2 --expert ffn --modes \
      plain plain_pipe tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2 \
      tutel:capacity_factor=1,degree=4 ds_moe:capacity_factor=1
  step tutel_2dh 150 $two $bench 0 --repeats 5 --iterations 2 --modes \
      plain tutel:capacity_factor=1 tutel:capacity_factor=1,local_size=4
  step tutel_layer20 240 $two $bench 20 --repeats 5 --iterations 2 --modes \
      plain plain_fused tutel tutel:capacity_factor=1 tutel:capacity_factor=4 \
      ds_moe:capacity_factor=1 ds_moe:capacity_factor=4
fi

if wants load; then
  step load_2node 180 $two $PY -u -m benchmark.internode.expert_load --busy-rows 32768
fi

# DeepEP across two nodes, once, to record how it fails.
if wants deepep2; then
  cat > "$OUT/deepep_two_nodes.py" <<'PYEOF'
import deep_ep
import torch
import torch.distributed as dist
from benchmark.utils import init_dist

rank, world, device = init_dist()
print(f"rank {rank}: building a DeepEP buffer over {world} ranks", flush=True)
buffer = deep_ep.Buffer(dist.group.WORLD, int(1e9), 0)
route = torch.randint(256, (1024, 8), device=device)
tokens = torch.randn(1024, 2048, device=device, dtype=torch.bfloat16)
per_rank, _, per_expert, in_rank, _ = buffer.get_dispatch_layout(route, 256)
received = buffer.dispatch(tokens, topk_idx=route, topk_weights=torch.rand(1024, 8, device=device),
                           num_tokens_per_rank=per_rank, is_token_in_rank=in_rank,
                           num_tokens_per_expert=per_expert)[0]
print(f"rank {rank}: dispatch returned {tuple(received.shape)}", flush=True)
PYEOF
  step deepep_two_nodes 120 $two $PY -u "$OUT/deepep_two_nodes.py"
fi

# The busiest experts replicated on every node: chosen ahead from the new recording
# (static) against chosen from the node's own tokens (the best a dynamic loader can do).
if wants hot; then
  HOT=$REC; [ -f "$REC/experts_rank0.pt" ] || HOT=$RUN
  hot_modes="--repeats 5 --iterations 2 --modes plain plain_fused rbd two_lane"
  step hot_layer0_0 120 $two $bench 0 $hot_modes --trace "$OUT/traces_hot0"
  for n in 16 32 64; do
    step hot_layer0_${n}_static 120 $two $bench 0 $hot_modes --hot $n --hot-from "$HOT"
    step hot_layer0_${n}_node 120 $two $bench 0 $hot_modes --hot $n --hot-from node \
        --trace "$OUT/traces_hot${n}"
  done
  step hot_layer20_0 120 $two $bench 20 $hot_modes
  step hot_layer20_32_static 120 $two $bench 20 $hot_modes --hot 32 --hot-from "$HOT"
  step hot_layer20_32_node 120 $two $bench 20 $hot_modes --hot 32 --hot-from node
fi
echo "##### DONE $(date +%T)"
