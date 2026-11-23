# Patches to third-party code the baselines need

The builds these produce live on scratch, which is purged. These files rebuild them.

| File | For | What it does |
|---|---|---|
| `deepep-7705f53-perlmutter-multinode.patch` | DeepEP commit 7705f53 | Lets DeepEP send between A100 nodes on Slingshot: NVSHMEM's portable device API in place of IBGDA, ranks per node as a build setting, Ampere fixes. See the DeepEP 01 page. |
| `build_nvshmem.sh` | NVSHMEM 3.4.5 | Builds NVSHMEM from source against Cray libfabric, libfabric transport only. The pip wheel's transport crashes here. |
| `build_deepep.sh` | the patched DeepEP | Builds it for a given number of ranks per node (4 on Perlmutter). |
| `xmoe_rbd_fixes.patch` | X-MoE's DeepSpeed fork | Fixes the two bugs that make X-MoE's RBD return wrong outputs. `benchmark/all2all/xmoe.py` loads the patched checkout named by `XMOE_DIR`. |

The build scripts assume the tree under `/pscratch/sd/e/eku/deepep_multinode` (`ROOT` at their top).
The `deepep` steps of `scripts/jobs/baselines_replicas.sh` use the result.
