# Todos

The library is `ops.py`, `plain.py` and `rbd.py`, which has all four pilot rules, plus
`two_lane.py` and `pipeline.py` built on `rbd.py`. Standalone benchmark scripts sit on
top. See [architecture.md](architecture.md).

Older versions of some features exist on abandoned branches, such as commit `f60a7ec`
on `refactor`. Treat those as previous effort: design each feature on the current code
and do not port the old one.

## Warnings

- **The DeepSpeed RBD backend is broken until the fork is updated.** The fork
  should be ignored until integration benchmark begins.

Experiments are written up in Notion, one page per experiment; the index and the
job scripts behind them are in [experiments/README.md](experiments/README.md).

## Next up

- [ ] One plan for all slices of the pipeline. Each slice now pays a whole plan, about
  2 ms, which is why pipelined RBD does not beat RBD.
- [ ] Plain's count exchange. It takes 1.9 ms of plain's 10 ms for a few bytes
  (Baselines 01). Tutel and DeepSpeed's MOELayer need no such exchange.
- [ ] `even_vs_uneven.py`: count the off-rank bytes of every rank. It now multiplies
  rank 0's by the number of ranks. Keep dividing by the slowest rank's time.

## Performance

- [ ] A cheaper `make_plan`. It is about a quarter of an RBD round trip, mostly
  host time.
- [ ] The pipeline over two-lane.

## Baselines

Settled in Baselines 01: the installed DeepEP fails on two nodes when it builds its
buffer; DeepEP's slowness was the adapter's `index_add_` reduce and 20 SMs; Tutel at
its default capacity runs in 12.5 ms and drops 28% of the copies on layer 0.

- [ ] DeepEP across nodes on Perlmutter. The installed v1.2.1 is one node only on
  A100. DeepEP v2 (`ElasticBuffer`) sends over NCCL and needs NCCL 2.30.4 or newer
  with GIN; the newest module here is 2.29.2. Open: whether v2 runs on A100, and
  whether GIN works over Slingshot.
- [ ] Why Tutel's pipelining is slower than one all2all: each chunked all2all takes
  as long as the whole one.
- [ ] Tutel cannot mix float16 and other dtypes in one process; it caches its kernels
  for the first dtype. `tutel:half=True` must run without other Tutel modes.

## Checks still owed

- [ ] Why RBD gets slower with hot replicas (Hot experts 01). Its transfers grow from
  5.9 to 7.2 ms although fewer rows cross nodes.
- [ ] Why the oracle hot set is slower than the static one on layer 20. The suspected
  cause, replicas piling load onto single ranks, is not verified.
- [ ] Why offload and inter-node prefetch do not hide under a dispatch (Expert
  loading 01). The shared PCIe link is a guess.
- [ ] Why two-lane beats one all2all: time same-node rows over the two-node group
  against the node's own group. A lead from Expert loading 01: a same-node send over
  the communicator of all ranks runs at 23 GB/s, and over the node's own at 79 GB/s.
  Plain, Tutel and DeepSpeed's MOELayer all send their same-node rows over the first.
- [ ] Where the 8.7e-3 error against plain, the same for every mode, comes from.
- [ ] Run the `flops` and `comm` profilers of `scripts/ds_moe.py` on a model.

## Small code items

- [ ] `scripts/ds_utils.py` reads the environment when imported.

## Features to bring back

Each comes back as one file that imports only public names: the plan and stage
functions of `rbd.py`, and `ops.py`.

- [ ] Expert offload: experts in host memory instead of on other ranks. Measured as a
  load time only (`benchmark/internode/expert_load.py`); keep them in pinned memory.
- [ ] Hot-expert swaps and replicas. `combine_dispatch --hot` emulates replicas by
  rerouting; nothing loads or holds a real replica yet. A replica per rank, not per
  node, is untried.
- [ ] Low priority: Expert placement searches.
- [ ] Low priority: Preemptive partition: bisect the experts by co-activation.

## Comparisons to bring back

- [ ] spec_ep's RBD against X-MoE's own RBD code.
- [ ] DeepSpeed's AutoEP layer against its explicit MoE layer.
- [ ] Per-phase times. `combine_dispatch.py --trace` shows the overlap but gives no table.
- [ ] Planned RBD: the plan is made ahead of time.
