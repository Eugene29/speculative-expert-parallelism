#!/bin/bash
# Build the patched DeepEP (src/DeepEP-mn) for A100 with the portable NVSHMEM internode
# kernels, for a given number of ranks per node. The result goes to
# build/deepep-nvl<N>/lib.*; put that directory first on PYTHONPATH to use it.
#
# Usage:
#   bash scripts/build_deepep.sh 4    # Perlmutter GPU nodes: 4 ranks per node
#   bash scripts/build_deepep.sh 2    # login-node tests: 4 ranks stand in for 2 nodes of 2
set -euo pipefail
ROOT=/pscratch/sd/e/eku/deepep_multinode
N=${1:-4}
PY=${PY:-/global/cfs/cdirs/m5421/eku/envs/spec-EP/bin/python}
export CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2}
export CC=gcc CXX=g++
export NVSHMEM_DIR=${NVSHMEM_DIR:-$ROOT/build/nvshmem-3.4.5-install}
export DISABLE_SM90_FEATURES=1 EP_GENERIC_NVSHMEM=1 NUM_MAX_NVL_PEERS=$N

cd "$ROOT/src/DeepEP-mn"
"$PY" setup.py build --build-base "$ROOT/build/deepep-nvl$N${SUFFIX:-}"
ls -d "$ROOT/build/deepep-nvl$N${SUFFIX:-}"/lib.*
