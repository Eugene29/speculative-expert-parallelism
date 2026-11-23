#!/bin/bash
# Build NVSHMEM from source with only the libfabric transport, against Cray libfabric,
# for sm_80. The pip wheel's libfabric transport was linked against an old libfabric ABI
# and segfaults in fi_getinfo here.
#
# Usage:
#   bash scripts/build_nvshmem.sh                 # v3.4.5 source in src/nvshmem-3.4.5
#   SRC=src/nvshmem-3.8.0 bash scripts/build_nvshmem.sh
set -euo pipefail
ROOT=/pscratch/sd/e/eku/deepep_multinode
SRC=${SRC:-$ROOT/src/nvshmem-3.4.5}
NAME=$(basename "$SRC")
export CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2}
export LIBFABRIC_HOME=${LIBFABRIC_HOME:-/opt/cray/libfabric/2.3.1}
export NVSHMEM_PREFIX=$ROOT/build/$NAME-install
export NVSHMEM_LIBFABRIC_SUPPORT=1
export NVSHMEM_IBRC_SUPPORT=0 NVSHMEM_MPI_SUPPORT=0 NVSHMEM_USE_GDRCOPY=0 NVSHMEM_USE_MLX5DV=0
export NVSHMEM_BUILD_TESTS=0 NVSHMEM_BUILD_EXAMPLES=0 NVSHMEM_BUILD_PYTHON_LIB=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0 NVSHMEM_BUILD_TXZ_PACKAGE=0

cmake -S "$SRC" -B "$ROOT/build/$NAME-build" -DCUDA_ARCHITECTURES=80 \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build "$ROOT/build/$NAME-build" -j 16
cmake --install "$ROOT/build/$NAME-build"
ls "$NVSHMEM_PREFIX/lib"
