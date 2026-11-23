# Architecture map

## What the code does

In a mixture-of-experts layer every token is processed by `k` experts, and the
experts are spread over GPUs. Each layer therefore has to send every token to the
ranks that hold its experts, run the experts, and bring the outputs back.

This repo implements two ways to do that round trip, plain all2all and
Redundancy-Bypassing Dispatch (RBD), plus two variants of RBD, two-lane and the
pipeline, and a way to choose replicas of experts that cut RBD's inter-node rows. It
benchmarks them against Tutel, DeepSpeed's MOELayer and AutoEP, X-MoE and DeepEP, on
routing recorded from a real model and inside the model itself. Features that were dropped and may come back are listed in
[todo.md](todo.md).

## Pages

| Page | Read it to learn |
|---|---|
| [concepts.md](architecture/concepts.md) | The vocabulary, one token's round trip step by step, and the assumptions. |
| [plain.md](architecture/plain.md) | The plain reference that RBD is checked and timed against. |
| [rbd.md](architecture/rbd.md) | How one RBD call works: planning, the `Plan`, the five stages, and the pilot rules. |
| [variants.md](architecture/variants.md) | What two-lane and the pipeline change in an RBD call. |
| [benchmarks.md](architecture/benchmarks.md) | What each benchmark measures and how `combine_dispatch` runs. |

## Layers

Imports point one way: each layer imports only from the layers above it in this list.

1. **Shared operations: `src/ops.py`.** `all2all`, `all2all_async` and `gather_sum`.
   None knows which algorithm calls it.
2. **The two algorithms: `src/plain.py` and `src/rbd.py`.** They are peers; neither
   imports the other. `src/replicas.py` sits beside them: it only rewrites a routing
   before either runs, and imports neither.
3. **The RBD variants: `src/two_lane.py` and `src/pipeline.py`.** Each is built on
   `rbd.py` and `ops.py`; neither imports the other.
4. **Benchmarks: `benchmark/`.** Standalone scripts. The few helpers they share are
   in `benchmark/utils.py`.

`src/` installs as the package `spec_ep`.

## File map

| File | Role | Public names |
|---|---|---|
| `src/ops.py` | Operations shared by the algorithms. | `all2all`, `all2all_async`, `gather_sum` |
| `src/plain.py` | The plain reference. | `plain_combine_dispatch` |
| `src/rbd.py` | RBD: planning and the five stages. | `Plan`, `make_plan`, `stage1_dispatch`, `stage2_dispatch`, `stage2_combine`, `stage1_combine`, `final_merge`, `rbd_combine_dispatch`, `PILOT_RULES` |
| `src/replicas.py` | Replicas of experts on other nodes: choosing them, routing to them, and rewriting a routing so the dispatch functions use them. | `RULES`, `choose_replicas`, `serving_node`, `replica_ids`, `relabel`, `inter_node_rows` |
| `benchmark/utils.py` | Launch, recorded routing, inputs, timing. | `init_dist`, `check_node_layout`, `load_routing`, `load_sample`, `expert_counts`, `make_inputs`, `time_ms` |
| `src/two_lane.py` | RBD with stage 1 in two lanes. | `two_lane_plan`, `two_lane_dispatch`, `two_lane_combine`, `two_lane_rbd_combine_dispatch` |
| `src/pipeline.py` | RBD and plain on slices of the tokens. | `pipelined_rbd_combine_dispatch`, `pipelined_plain_combine_dispatch` |
| `benchmark/all2all/modes.py` | Every dispatch-and-combine version behind one call, `mode(tokens, weights, expert, route)`. The layer benchmark and the model script both build their modes here. | `MODES`, `parse_mode`, `build_mode` |
| `benchmark/all2all/combine_dispatch.py` | Every mode against plain on recorded routing, optionally with replicas. | |
| `benchmark/all2all/baselines.py` | Tutel, DeepSpeed's MOELayer and AutoEP, and DeepEP behind the arguments of `plain_combine_dispatch`, with each library's settings as keyword arguments. | `tutel_combine_dispatch`, `ds_moe_combine_dispatch`, `autoep_combine_dispatch`, `capacity_kept`, `make_deepep_buffer`, `deepep_combine_dispatch` |
| `benchmark/all2all/xmoe.py` | X-MoE's own RBD code behind the same arguments. | `xmoe_combine_dispatch` |
| `benchmark/replica_count.py` | Rows that cross between nodes under every replica rule, budget and routing. CPU only. | |
| `benchmark/hot_experts.py` | How stable the hot experts are across batches of recorded routing. | |
| `benchmark/phase_clock.py` | Times every step a mode marks with `record_function("<mode>/<step>")` on the GPU's clock, with CUDA events and no profiler. The steps add up to the forward. | `PhaseClock`, `rank_mean`, `phase_line` |
| `benchmark/trace_report.py` | GPU time by kind and a timeline picture from `combine_dispatch --trace`. | |
| `benchmark/all2all/even_vs_uneven.py` | One all2all with the recorded uneven split sizes against even split sizes. | |
| `benchmark/internode/` | Send, receive and all2all of expert weights within and between nodes, and loading them from host memory (`expert_load.py`). | |
| `benchmark/compute.py` | GEMM and Flash Attention compute, no communication. | |
| `benchmark/summarize.py` | Markdown tables from the outputs of `combine_dispatch` and `ds_moe` runs. | |
| `patches/` | Patches and build scripts for third-party code the baselines need: DeepEP across nodes over NVSHMEM, and X-MoE's two RBD fixes. | |
| `tests/checks.py` | The check the tests share: layouts, routings and the locally computed answer, on CPU. | `LAYOUTS`, `Layout`, `run_check` |
| `tests/test_rbd.py`, `tests/test_two_lane.py`, `tests/test_pipeline.py` | RBD, two-lane and the pipeline through that check, for every pilot rule. | |
| `tests/test_phase_clock.py` | The phase clock on one GPU: its phases add up to the wall time. | |
| `tests/test_replicas.py` | Replica choice, routing and relabeling, on CPU. | |
| `scripts/`, `configs/` | Model-level runs through DeepSpeed: `scripts/ds_moe.py --modes` times the model with any mode of `modes.py` as its MoE layers' dispatch, and `--record` saves the routing. Its helpers are in `scripts/ds_utils.py`. `scripts/record_routing.py` records routing on many batches with Hugging Face, and `scripts/jobs/` holds the job scripts behind the experiment pages. | |

## Naming rules

- A name with a leading underscore is private to its file. Other files import only
  public names.
- A function stays in the file that uses it. It moves to a shared file only when a
  second, unrelated module needs it.
- A module built on top of another imports from it directly. Peers share through
  `ops.py`.

## Suggested reading order

1. `src/plain.py`: the whole round trip in its simplest form.
2. `src/rbd.py`: `rbd_combine_dispatch` and the five stages, treating `Plan` as given.
3. `src/rbd.py`: `make_plan`, one numbered step at a time.
4. `src/two_lane.py`, then `src/pipeline.py`.
5. `benchmark/all2all/combine_dispatch.py`.
