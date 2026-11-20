# Refactor todos

The library is `ops.py`, `plain.py` and `rbd.py` (with all four pilot rules), with
standalone benchmark scripts on top; see [architecture.md](architecture.md).
Everything listed below was dropped in the refactor and can be recovered from commit
`f60a7ec` on the `refactor` branch.

## Warnings

- **The dropped code exists in one local place.** The `refactor` branch is not on
  the remote. Push or tag it before deleting anything.
- **The DeepSpeed RBD backend is broken until the fork is updated.** The fork
  imports `RBD_combine_dispatch`, which is now `rbd_combine_dispatch` (with
  `num_experts=` and `pilot_rule=`), so the `qwen_ep*_rbd.json` configs fail at
  import. Accepted, since DeepSpeed is ignored for now.
- **Do not delete `scripts/ds_moe.py`.** It records the routing that the benchmarks
  replay.
- **Keep the node layout check.** If ranks do not fill nodes in blocks, RBD still
  gives the right output but the timings mean nothing.
- **Plain is not tested on its own.** It is only the reference inside
  `rbd_vs_plain`, which needs GPU nodes.

## Features to bring back

Each comes back as one file that imports only public names: the plan and stage
functions of `rbd.py`, and `ops.py`.

- [ ] EP pipeline: run expert compute while the next slice of tokens is on the network.
- [ ] Two-lane RBD: same-node rows on the node's own group, apart from the pilots.
- [ ] Expert offload: experts in host memory instead of on other ranks.
- [ ] Hot-expert swaps and replicas.
- [ ] Expert placement searches and their cost model.
- [ ] Preemptive partition: bisect the experts by co-activation.

## Comparisons to bring back

- [ ] spec_ep's RBD against X-MoE's own RBD code.
- [ ] DeepSpeed's AutoEP layer against its explicit MoE layer.
- [ ] Strong against weak scaling.
- [ ] Per-phase and compute/communication overlap profiles.
- [ ] RBD with the plan made ahead of time ("planned").
