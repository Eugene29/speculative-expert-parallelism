# Todos

The library is `ops.py`, `plain.py` and `rbd.py`, plus `two_lane.py` built on `rbd.py`,
`replicas.py`, which chooses replicas and routes to them, and `guests.py`, which
chooses guests and pulls their weights from the ranks that own them. The benchmark is
Qwen3.5-397B-A17B (400B 01): all 60 layers on 4 nodes of 80 GB GPUs, or its first 30
layers on 2 nodes with `--layers 30`. Standalone benchmark scripts sit on top, and
`benchmark/all2all/modes.py` builds every dispatch mode for both the layer benchmark
and the model. See [architecture.md](architecture.md).

Older versions of some features exist on abandoned branches, such as commit `f60a7ec`
on `refactor`. Treat those as previous effort: design each feature on the current code
and do not port the old one.

Experiments are written up in Notion, one page per experiment; the index and the
job scripts behind them are in [experiments/README.md](experiments/README.md).

## Warnings

- **The DeepSpeed fork's `comm_backend: "rbd"` has not been run since it was renamed
  to spec_ep's current functions.** The model benchmark does not use it: it sets the
  `combine_dispatch` hook of `AutoEPMoELayer` (`scripts/ds_moe.py --modes`). The fork's
  changes are uncommitted in the submodule. Its `comm_rbd_pipeline_degree` path
  imports `pipelined_rbd_combine_dispatch`, which is gone since 400B 01.
- **The fork's `comm_backend: "deepep"` is written for DeepEP V2, which does not
  compile for A100** (DeepEP 01). DeepEP runs in the model through the hook and the
  v1 adapter instead.
- **NCCL's channels per peer change the ranking of the modes** (NCCL 01). Time a mode
  at the default and at `NCCL_NCHANNELS_PER_NET_PEER=8`, and say which was used.

## Next up

- [ ] Inter-node transfer is the largest part of the forward on the target model: 700
  of 1,874 ms on 4 nodes with two-lane and `early=0.5` (400B 01). RBD already sends
  one row per token and node. NCCL's channels per peer are untried on this model
  (NCCL 01).
- [ ] One row per (token, rank) inside a node. Direct rows and stage 2 rows are one
  per (token, expert). With 5 copies on a node of 4 ranks a token reaches 3.05 ranks on
  average, so 39% of those rows are redundant on 2 nodes and 18% on 4 (a count, not a
  measurement). NVLink transfer not under another is 196 ms per forward on 4 nodes.
  The expert rank would expand a row to its experts and sum before returning.
- [ ] The count exchange. 42 to 45 ms per forward on 4 nodes, 0.7 ms per layer for a
  few kilobytes, more than the plan's compute (24 ms). 5.5 ms on 2 nodes (400B 01).
- [ ] `early` at few tokens per rank. It gains 4.1% on 2 nodes and 4.9% on 4 at 4,096
  tokens and is an option, not the default: three expert calls cost more when the
  batch is small (400B 01).
- [ ] Fold the routing to replicas into `make_plan`, as part of its kernel. It is a
  separate step now, 0.22 ms per call for per-node routing and 0.72 ms for cover
  routing, which is half of what cover routing gains on 4 nodes (Redundancy 01).
- [ ] Replicas in the model: `scripts/ds_moe.py` has no `--replicas` yet, so cover
  routing is timed on one layer only.
- [ ] Plain's count exchange. It takes 1.9 ms of plain's 10 ms for a few bytes
  (Baselines 01). Tutel and DeepSpeed's MOELayer need no such exchange.
- [ ] `even_vs_uneven.py`: count the off-rank bytes of every rank. It now multiplies
  rank 0's by the number of ranks. Keep dividing by the slowest rank's time.

## Performance

- [ ] The rest of the plan's source half. The token kernel is 68 us of 0.58 ms; the
  rest is about 45 small torch operations and the host sync. Doing that arithmetic on
  the host was slower in the model, because its upload is a second synchronous copy.
  Untried: a pinned, non-blocking upload, and expert ids built by the kernel that
  already walks the runs (400B 01).
- [ ] The stage 1 pack. `with_meta` copies every token to append the metadata and the
  gather copies the rows again; one kernel could write the send rows. Two-lane's
  dispatch compute is 18 ms per forward on 4 nodes (400B 01).
- [ ] Guests cannot pay for a transfer per layer at 4,096 tokens per rank: 8 guests
  are 201 MB of weights against 34 MB of pilots per rank and direction, and they
  remove 1 to 5% of the pilots (400B 01). The byte counts are equal near 40,000 tokens
  per rank (an estimate). Open: guests held across forwards, which is what replicas
  are, and a cap on what one rank sends, since the owner of the hot experts sends up to
  21 guests in a layer while every rank receives 8.
- [ ] Guests remove copies, not pilots, with top 10 routing: a pilot goes only when all
  of a token's copies on that node are guests. 16 oracle guests per rank leave 78% of
  the pilots on 4 nodes and 55% of the copies (400B 01). They would help a plain
  all2all more than RBD.
- [ ] RBD at 512 tokens per rank and below. AutoEP and plain take 0.17 to 0.18 s and
  RBD 0.205 s on the 35B model: every all2all and every operation costs its fixed
  latency there, and RBD makes more of both (Plan 01).
- [ ] The event totals of the guest mode. Its comm and compute inside the MoE layers
  add up to more than the forward, because a guest transfer shows as a long in-flight
  event; only the per-step rows are usable (Guests 02, 400B 01).
- [ ] RBD in one all2all with replicas is slower than without (5.6 to 6.1 ms on 2
  nodes at 8 channels per peer). The lead: the communicator of all ranks has 8 p2p
  channels in total against the node communicator's 32 (NCCL 01).
- [ ] Re-time cover routing on 4 nodes with the even tie rule. The 2026-11-22
  timings used a rule that split equal choices unevenly; the pilot counts are the
  same, the per-node loads are not (Redundancy 01).
- [ ] A cover search that scales past a few nodes. `serving_node(cover=True)` scores
  all 2^nodes node sets.

## Baselines

- [ ] DeepEP's RDMA chunk on 4 nodes. Its default there gives 9.8 ms per layer and
  128-token chunks 6.8 ms; only two tuned settings were tried, and one in the model
  (DeepEP 01). Its limiter (SM count, then puts through NVSHMEM's proxy thread) is
  inferred from a sweep, not profiled.
- [ ] The patched DeepEP build on one node against the installed v1.2.1.
- [ ] Tutel cannot mix float16 and other dtypes in one process; it caches its kernels
  for the first dtype. `tutel:half=True` must run without other Tutel modes.
- [ ] Tutel aborts the process at exit after its pipeline has run (static CUDA events).
  `scripts/ds_moe.py` leaves with `os._exit`; the layer benchmark has not hit it.
- [ ] Why the model's profiled forward stalled a few hundred milliseconds in one of
  its first layers (job 61767192) is likely but not proven. A full Python garbage
  collection takes 0.23 s per rank and up to 1.6 s on rank 0 after it builds a
  profile's table, and `scripts/ds_moe.py` now keeps the collector off inside every
  measured forward: no stall in 105 measured forwards since (Profiles 02). One of the old
  stalls was on rank 4, which ran no full collection in the new runs.
- [ ] A 4-node timeline of the model. Most traces of the 2026-11-22 job lost their
  GPU timestamps (fixed in `scripts/ds_moe.py` since); Profiles 01 uses the per-op
  tables instead.
- [ ] Why Tutel's per-peer layout costs 0.27 s per forward in the 4-node model when
  it costs 0.015 s on 2 nodes (Tutel 01).

## Checks still owed

- [ ] The 2-node model at the default and at 8 NCCL channels per peer in one job, and
  the 4-node model at 8 per peer without the total of 32. Both comparisons on NCCL 01
  are across jobs or at one raised setting only.
- [ ] Why plain, AutoEP and Tutel get 5 to 15% slower with more NCCL channels per
  peer when most of their rows cross between nodes. Contention on the network is a
  guess (NCCL 01).
- [ ] The modes that keep every copy differ from AutoEP by 11 to 16% in the model's
  last logits in bf16. On a small model in float32 they agree to 1e-5. A float32 run
  of the 35B model would confirm it is rounding there too.
- [ ] Why offload and inter-node prefetch do not hide under a dispatch (Expert
  loading 01). The shared PCIe link is a guess.
- [ ] Where the 8.7e-3 error against plain, the same for every mode, comes from.
- [ ] On the 60-layer 400B model in bf16, plain, RBD and two-lane differ from AutoEP by
  29 to 34% in the last logits with the same top token on 98% of the last tokens.
  Plain differs as much as RBD, and a small model in float32 agrees to 8e-6, so it
  reads as rounding, but a float32 run of the 400B does not fit in memory (400B 01).
- [ ] Rank-level load. The all2all after the experts takes 1.4 ms where the one
  before takes 0.8 ms for the same rows on 2 nodes. A rank whose experts finish early
  waits for the others; that this is the cause is inferred, not measured (400B 01).
- [ ] Run the `flops` and `comm` profilers of `scripts/ds_moe.py` on a model.

## Small code items

- [ ] `scripts/ds_utils.py` reads the environment when imported.
- [ ] `benchmark/all2all/xmoe.py`, `baselines.py` and the model script each start
  `deepspeed.comm` in their own way; one helper would do.

## Features to bring back

Each comes back as one file that imports only public names: the plan and stage
functions of `rbd.py`, and `ops.py`.

- [ ] Hot-expert swaps. Replicas are back (`src/replicas.py`); swaps are not.
- [ ] Low priority: the slice pipeline (`src/pipeline.py`) and the pilot rules `most`
  and `even`, removed with the sort-based plan. Commit c730d7e has them.
- [ ] Low priority: Expert placement searches.
- [ ] Low priority: Preemptive partition: bisect the experts by co-activation.

## Done

Finished items move here with the date and where the result is written up.

- [x] 2026-11-23. The plan as kernels: `make_plan`, one kernel over the tokens with
  atomic adds in place of both sorts. 1.16 to 0.65 ms per layer on one A100, and 18.9
  to 10.4 ms of plan compute per forward on the 400B model's first 30 layers (400B 01).
- [x] 2026-11-23. Guests pulled GPU to GPU into slots all layers share, in place of
  the host store: `GuestPool`, `exchange_requests`, and `balance` for the choice by
  pilots. They lose at 4,096 tokens per rank on 2 and 4 nodes (400B 01).
- [x] 2026-11-23. The experts get rows grouped by expert, so DeepSpeed's sort and
  scatter go: two-lane 2.057 to 1.970 s on the whole 400B model on 4 nodes (400B 01).
- [x] 2026-11-23. The direct rows' experts under the network lane
  (`two_lane_rbd_combine_dispatch(early=0.5)`): 1.970 to 1.874 s on 4 nodes (400B 01).
- [x] 2026-11-23. The 400B benchmark: `scripts/jobs/gpu_prefetch.sh`, `--layers`,
  `benchmark/plan_cost.py` and `benchmark/guest_choice.py` (400B 01).

- [x] 2026-11-23. A cheaper plan, after DeepEP's: a cached layout, metadata in the
  stage 1 rows, one sort for all slices in lane order. 93 to 35 ms per forward in
  two-lane at 4096 tokens per rank, and two-lane 0.539 to 0.483 s in one job (Plan 01).
- [x] 2026-11-23. Guests in the layer's own weight tensors: one expert call, no host
  sync, recorded guests loaded once. With the plan, static guests beat AutoEP at 1024
  tokens per rank, 0.221 against 0.230 to 0.247 s, and not at 512 (Plan 01).

- [x] 2026-11-22. The pipeline over two-lane (`pipelined_rbd_combine_dispatch(lanes=2)`),
  with and without guests. It is slower than two-lane without slices on the 35B model
  and 1% faster on the 122B model (Guests 02).
- [x] 2026-11-22. Few tokens per rank, and how activation spreads with batch size:
  plain wins at 128 to 1024 tokens per rank (the fit puts the break-even near 1,500, not measured), and static prefetch beats speculative
  there (Guests 02).
- [x] 2026-11-22. A metric for RBD's benefit, the saved rows
  (`benchmark/rbd_benefit.py`), and guests chosen by the pilots they remove
  (`choose_guests_by_pilots`), which gained nothing end to end (Guests 02).
- [x] 2026-11-22. Guests on Qwen3.5-122B-A10B: the store is striped, pinned in shared
  chunks and releases its source pages, 85.4 GiB of host memory per rank (Guests 02).

- [x] 2026-11-22. Why pipelining gained nothing for Tutel and RBD, and one plan for
  all slices of the RBD pipeline, with its tail under the dispatch: 1.134x over RBD
  in the Qwen model (Pipeline 01).
- [x] 2026-11-22. Expert offload as guests: hot experts of other nodes loaded from
  pinned host memory one layer ahead, chosen by the next layer's router, on two-lane
  RBD: 1.30x over RBD and 1.10x over two-lane at 24 guests per rank (Guests 01).
- [x] 2026-11-22. Every library in one table, on one layer and in the Qwen model, on
  2 and 4 nodes: DeepEP, Tutel, AutoEP, DeepSpeed's MOELayer, RBD, two-lane and X-MoE
  through `benchmark/all2all/modes.py` (Baselines 02), with their profiles (Profiles 01).
- [x] 2026-11-22. spec_ep's RBD against X-MoE's own RBD code: `benchmark/all2all/xmoe.py`
  (Baselines 02).
- [x] 2026-11-22. DeepSpeed's AutoEP layer against its explicit MOELayer: the `autoep`
  and `ds_moe` modes (Baselines 02).
- [x] 2026-11-22. DeepEP across nodes on Perlmutter, on 2 and 4 nodes and in the
  model: DeepEP 7705f53 over NVSHMEM's libfabric transport, patches in `patches/`.
  DeepEP V2 does not compile for A100 (DeepEP 01).
- [x] 2026-11-22. Why Tutel's pipelining is slower than one all2all on 2 nodes: one
  NCCL message per expert, 32 per peer. `tutel:per_peer=True` sends one per peer. On
  4 nodes, with 16 per peer, Tutel's pipelining ties and the layout does not help
  (Tutel 01).
- [x] 2026-11-22. Why plain and RBD get slower with hot replicas while two-lane gets
  faster, and why two-lane beats one all2all: same-node peers of a multi-node NCCL
  communicator get 2 channels (NCCL 01).
- [x] 2026-11-22. Replicas chosen by the pilots they remove, per node
  (`choose_replicas(..., "pilots")`, Hot experts 02), and across all nodes with cover
  routing and the ring layout (Redundancy 01). `--hot` became `--replicas`.
- [x] 2026-11-22. The DeepSpeed fork runs any dispatch mode in the model through one
  hook, and its RBD backend uses spec_ep's current names.
- [x] 2026-11-22. Events instead of phase totals: `src/phase_clock.py` gives every
  step as comm or compute with its link and overlap, and `scripts/ds_moe.py --profile
  events` prints the totals inside and outside the MoE layers (Profiles 03).
- [x] 2026-11-22. Each step's time without a profiler: `benchmark/phase_clock.py`
  marks the steps with CUDA events and `scripts/ds_moe.py` prints every mode's forward
  as a sum of them, within 2.5% of the timed forward (Profiles 02).
- [x] 2026-11-21. Per-phase times: `benchmark/trace_report.py` prints each mode's GPU
  time by kind and its CPU-side phases from a `--trace` run (Baselines 01).
- [x] 2026-11-21. Settings for Tutel, DeepSpeed's MOELayer and DeepEP on one layer
  (Baselines 01).
- [x] 2026-11-21. Hot experts across batches, static against dynamic (Hot experts 01).
- [x] 2026-11-21. Loading experts from host memory against other GPUs, as a load time
  (Expert loading 01).
