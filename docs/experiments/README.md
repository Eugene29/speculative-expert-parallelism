# Experiments

Each experiment is written up as one Notion page under
[Experiments](https://app.notion.com/p/41fe2e64355e81119118f2eee57875a3), named
`<Topic> <NN>: <question>`. This file maps each page to the code that produced it.

| Page | Scripts | Job steps | Raw outputs on Perlmutter |
|---|---|---|---|
| [Baselines 01](https://app.notion.com/p/41fe2e64355e81359858e05464b6fdf3): settings for Tutel, DeepSpeed's MOELayer and DeepEP | `benchmark/all2all/combine_dispatch.py`, `benchmark/all2all/baselines.py`, `benchmark/trace_report.py` | `deepep`, `tutel`, `deepep2` | `logs/2026-11-21_exp/`, `logs/2026-11-21_exp2/` |
| [Hot experts 01](https://app.notion.com/p/41fe2e64355e81a5bc76ce61b9f7c9fb): hot experts across batches, static against dynamic | `scripts/record_routing.py`, `benchmark/hot_experts.py`, `combine_dispatch.py --hot` | `record`, `hot` | `logs/2026-11-21_exp/routing/`, `hot_*.out` in `logs/2026-11-21_exp4/` and `_exp5/` |
| [Expert loading 01](https://app.notion.com/p/41fe2e64355e8156b2a6c6f82ee2d6ed): host memory against other GPUs | `benchmark/internode/expert_load.py` | `load1`, `load` | `load_*.out` in `logs/2026-11-21_exp/`, `_exp2/`, `_exp3/` |

The job steps are those of `scripts/jobs/baselines_hot_load.sh`, picked with `STEPS`.
`logs/` is not in git.

The `hot_*.out` files in `logs/2026-11-21_exp/` and `_exp2/` are superseded. Those
runs picked the hot experts from all experts, including the ones the node already
held, so about half the replicas moved nothing.

`decisions.tsv` is the decision trail of the run that produced these pages: one row
per decision, with what was chosen, why, and where the evidence is.
