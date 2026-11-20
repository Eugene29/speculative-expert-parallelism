# Benchmarks

Back to the [architecture map](../architecture.md).

Every benchmark is a standalone script launched with `srun`, one process per GPU.
`scripts/run_benchmark.sh` sets up NCCL and the environment.

| Script | What it measures |
|---|---|
| `benchmark/all2all/rbd_vs_plain.py` | Plain, plain with the fused merge, and RBD, end to end on recorded routing. |
| `benchmark/all2all/even_vs_uneven.py` | One all2all with the recorded (uneven) split sizes against even ones. |
| `benchmark/internode/expert_all2all.py` | All2all of expert weights across nodes. |
| `benchmark/internode/expert_send_recv.py` | Send/recv of expert weights between two ranks, within a node or between nodes. |
| `benchmark/internode/expert_send_recv_parallel.py` | The same transfers in parallel, as a ring across nodes. |
| `benchmark/compute.py` | GEMM and Flash Attention compute, no communication. |

## How `rbd_vs_plain` runs

1. **Record.** A model run (`scripts/ds_moe.py`) saves which experts every token
   chose: one file per recorded rank, `experts_rank{r}.pt`, holding a
   `[num_layers, num_tokens, k]` tensor of expert ids.
2. **Launch.** `init_dist` starts NCCL from the SLURM environment, and
   `check_node_layout` checks that ranks fill the nodes in blocks.
3. **Inputs.** `load_routing` gives this rank its share of one layer's recording, and
   `make_inputs` adds random tokens and weights. The routing is the only real data;
   token values do not affect communication cost.
4. **Count.** `inter_node_rows` counts the rows plain and RBD send between nodes.
5. **Check.** Every version runs once with an expert that scales each row by its
   expert id, and its output is compared with plain. A row sent to the wrong expert,
   merged into the wrong token or given the wrong weight changes the output.
6. **Time.** `time_ms` times every version with identity experts, so only
   communication is measured. All ranks line up at a barrier before each timed call,
   and the reported time is the slowest rank's average.

With more ranks than recordings, each recording's tokens are split across several
ranks, so the total workload stays the same as ranks are added.
