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

## Next up

- [ ] One plan for all slices of the pipeline. Each slice now pays a whole plan, about
  2 ms, which is why pipelined RBD does not beat RBD.
- [ ] `even_vs_uneven.py`: count the off-rank bytes of every rank. It now multiplies
  rank 0's by the number of ranks. Keep dividing by the slowest rank's time.

## Performance

- [ ] A cheaper `make_plan`. It is about a quarter of an RBD round trip, mostly
  host time.
- [ ] The pipeline over two-lane.

## Baselines

- [ ] DeepEP across nodes on Perlmutter. The installed v1.2.1 is one node only on
  A100. DeepEP v2 (`ElasticBuffer`) sends over NCCL and needs NCCL 2.30.4 or newer
  with GIN; the newest module here is 2.29.2. Open: whether v2 runs on A100, and
  whether GIN works over Slingshot.
- [ ] Try the installed DeepEP on two nodes once, to confirm it fails.
- [ ] Why DeepEP is 1.7 times slower than plain with the fused merge on one node.
- [ ] Tutel with its default capacity, which drops tokens. So far it ran only with
  zero-drop capacity, which pads 7 to 30 times.

## Checks still owed

- [ ] Repeat the main `combine_dispatch` table on another day or pair of nodes.
- [ ] Why two-lane beats one all2all: time same-node rows over the two-node group
  against the node's own group.
- [ ] Where the 8.7e-3 error against plain, the same for every mode, comes from.
- [ ] Run the `flops` and `comm` profilers of `scripts/ds_moe.py` on a model.

## Small code items

- [ ] `scripts/ds_utils.py` reads the environment when imported.
- [ ] The comment in `time_ms` about warm-up building communicators may be out of date.

## Features to bring back

Each comes back as one file that imports only public names: the plan and stage
functions of `rbd.py`, and `ops.py`.

- [ ] Expert offload: experts in host memory instead of on other ranks.
- [ ] Hot-expert swaps and replicas.
- [ ] Low priority: Expert placement searches.
- [ ] Low priority: Preemptive partition: bisect the experts by co-activation.

## Comparisons to bring back

- [ ] spec_ep's RBD against X-MoE's own RBD code.
- [ ] DeepSpeed's AutoEP layer against its explicit MoE layer.
- [ ] Per-phase times. `combine_dispatch.py --trace` shows the overlap but gives no table.
- [ ] Planned RBD: the plan is made ahead of time.
