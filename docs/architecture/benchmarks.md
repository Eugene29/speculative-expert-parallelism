# Benchmarks

Back to the [architecture map](../architecture.md).

Every benchmark is a standalone script launched with `srun`, one process per GPU.
`scripts/run_benchmark.sh` sets up NCCL and the environment.

| Script | What it measures |
|---|---|
| `benchmark/all2all/combine_dispatch.py` | Every dispatch-and-combine version, end to end on recorded routing: plain, RBD and two-lane, with Tutel, DeepSpeed's MOELayer and DeepEP as baselines. Also the same versions with the hot experts replicated on every node. |
| `scripts/ds_moe.py` | The whole Qwen model, with any of the same modes as the dispatch of every MoE layer (`--modes`). One model load, a logits check, then rounds of one timed and one traced forward per mode. For every mode it prints comm, compute and overlapped time, for the forward and split into inside and outside the MoE layers, a table per step, and a Chrome trace per rank under `LOGS/events/`. |
| `src/phase_clock.py` | The events behind `ds_moe.py --profile events`: every step a mode marks, as comm or compute on the GPU's clock, with its link and its overlap. A library's own packing inside its transfer call counts as comm, and an all2all that carries same-node and other-node rows is one event with link `both`. |
| `benchmark/plan_cost.py` | The compute cost of RBD's plan on one GPU, at the shape of a layer of the target model. Every rank is emulated in one process, so no NCCL is involved. |
| `benchmark/guest_choice.py` | What each way of choosing guests leaves on the network: copies, pilots, how evenly the nodes receive them, and how many guests one rank sends. CPU only. |
| `benchmark/all2all/contention.py` | One all2all and one expert block, alone and overlapped on every rank: how much each slows the other. |
| `benchmark/replica_count.py` | The rows that cross between nodes under every replica rule, budget and routing, on recorded routing. CPU only. |
| `benchmark/hot_experts.py` | How stable the hot experts are from batch to batch, on recorded routing. CPU only. |
| `benchmark/activation_spread.py` | How widely tokens spread over the experts at different batch sizes, and the pilots that guests leave at each, on recorded routing. |
| `benchmark/rbd_benefit.py` | The rows RBD saves between nodes for a routing and a placement, and the fit of RBD's measured gain to them over several model steps. CPU only. |
| `scripts/login_dry_run.py` | Any of these scripts on the login node over gloo, to check that code runs and modes agree. Timings mean nothing there. |
| `benchmark/trace_report.py` | Where the GPU time of a `combine_dispatch --trace` run went, as a table and a timeline picture. CPU only. |
| `benchmark/all2all/even_vs_uneven.py` | One all2all with the recorded (uneven) split sizes against even ones. |
| `benchmark/internode/expert_all2all.py` | All2all of expert weights across nodes. |
| `benchmark/internode/expert_send_recv.py` | Send/recv of expert weights between two ranks, within a node or between nodes. |
| `benchmark/internode/expert_send_recv_parallel.py` | The same transfers in parallel, as a ring across nodes. |
| `benchmark/internode/expert_load.py` | Loading expert weights onto every GPU at once from host memory, from the same node or from another node, alone and during an all2all. |
| `benchmark/compute.py` | GEMM and Flash Attention compute, no communication. |

## How `combine_dispatch` runs

`--modes` picks the versions, all built by `benchmark/all2all/modes.py`: `plain`,
`plain_fused`, `rbd`, `two_lane`, `autoep`, `tutel`, `ds_moe`, `xmoe` and `deepep`. `plain` always runs. The Tutel, DeepSpeed and DeepEP adapters are
in `benchmark/all2all/baselines.py` and X-MoE's in `benchmark/all2all/xmoe.py`,
imported only when selected.

A mode takes settings as `name:key=value,key=value`, which become keyword arguments of
its function. One run can therefore time a mode under several settings, such as
`tutel tutel:capacity_factor=1 tutel:capacity_factor=1,degree=2`. The baselines'
settings are listed in `baselines.py`.

1. **Record.** A model run (`scripts/ds_moe.py --record`) saves which experts every token
   chose: one file per recorded rank, `experts_rank{r}.pt`, holding a
   `[num_layers, num_tokens, k]` tensor of expert ids.
2. **Launch.** `init_dist` starts NCCL from the SLURM environment, and
   `check_node_layout` checks that ranks fill the nodes in blocks. The RBD modes and
   `xmoe` need two nodes or more. `deepep` needs every rank on one node with the
   installed build, and runs across nodes with the build of `patches/` first on
   `PYTHONPATH` (the `deepep` steps of `scripts/jobs/baselines_replicas.sh`).
3. **Inputs.** `load_routing` gives this rank its share of one layer's recording, and
   `make_inputs` adds random tokens and weights. The routing is the only real data;
   token values do not affect communication cost. With `--replicas RULE`, rank 0 first
   chooses `--budget` replicas per node (`spec_ep.replicas.choose_replicas`), from the
   timed routing or from another recording (`--choose-from DIR`), and every rank
   reroutes its copies to them: to replicas on the token's own node only, or with
   `--cover` to the fewest nodes that hold all of a token's experts. The script
   prints what that rerouting costs per call; it is not part of any mode's time.
4. **Count.** `inter_node_rows` counts the rows plain and RBD send between nodes.
5. **Check.** Every mode runs once with an expert that scales each row by its
   expert id, and its output is compared with plain. A row sent to the wrong expert,
   merged into the wrong token or given the wrong weight changes the output. A padded
   mode with a capacity limit drops copies, so it is compared with plain run on the
   copies it keeps, and the script prints how many it drops.
6. **Time.** `time_ms` times every mode with identity experts, so only communication
   and each mode's own planning are measured. The script runs `--repeats` rounds and
   times every mode in turn in each, so drift over the run favors none. All ranks line
   up at a barrier before each timed call, and each timed call reports its slowest
   rank. The script reports each mode's median, min and max over all rounds.
   `--expert ffn` times with a real feed-forward expert instead.
7. **Trace.** With `--trace DIR`, every mode runs once more under `torch.profiler`
   and every rank writes `DIR/{mode}_rank{r}.json`. `benchmark/trace_report.py DIR`
   turns those into a table of GPU time and a picture.

The recordings are split across ranks, or joined, to fit the number of ranks, so the
total workload stays the same.
