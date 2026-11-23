# Experiments

Each experiment is written up as one Notion page under
[Experiments](https://app.notion.com/p/41fe2e64355e81119118f2eee57875a3), named
`<Topic> <NN>: <question>`. This file maps each page to the code that produced it.

| Page | Scripts | Job steps | Raw outputs on Perlmutter |
|---|---|---|---|
| [Baselines 01](https://app.notion.com/p/41fe2e64355e81359858e05464b6fdf3): settings for Tutel, DeepSpeed's MOELayer and DeepEP | `benchmark/all2all/combine_dispatch.py`, `benchmark/all2all/baselines.py`, `benchmark/trace_report.py` | `deepep`, `tutel`, `deepep2` | `logs/2026-11-21_exp/`, `logs/2026-11-21_exp2/` |
| [Hot experts 01](https://app.notion.com/p/41fe2e64355e81a5bc76ce61b9f7c9fb): hot experts across batches, static against dynamic | `scripts/record_routing.py`, `benchmark/hot_experts.py`, `combine_dispatch.py --hot` | `record`, `hot` | `logs/2026-11-21_exp/routing/`, `hot_*.out` in `logs/2026-11-21_exp4/` and `_exp5/` |
| [Expert loading 01](https://app.notion.com/p/41fe2e64355e8156b2a6c6f82ee2d6ed): host memory against other GPUs | `benchmark/internode/expert_load.py` | `load1`, `load` | `load_*.out` in `logs/2026-11-21_exp/`, `_exp2/`, `_exp3/` |
| [Tutel 01](https://app.notion.com/p/420e2e64355e81c199c9fc98de3d0897): why Tutel's pipelining is slower than one all2all | `benchmark/all2all/baselines.py` (`per_peer`), `/pscratch/sd/e/eku/tutel_pipeline/a2a_chunks.py`, `scripts/ds_moe.py --modes` | `pipe`, `tutel`, `model` of `baselines_replicas.sh` | `logs/2026-11-22_exp/2n/` |
| [NCCL 01](https://app.notion.com/p/420e2e64355e81058ee0e7b7b340b1e8): why same-node rows were slow in the all2all of all ranks | `benchmark/all2all/combine_dispatch.py` under `NCCL_NCHANNELS_PER_NET_PEER` | `channels` of `baselines_replicas.sh` | `logs/2026-11-22_exp/2n_b/channels_*.out` |
| [Baselines 02](https://app.notion.com/p/420e2e64355e811297aedc8c3e809a9b): six libraries on 2 and 4 nodes, one layer and the Qwen model | `benchmark/all2all/modes.py`, `combine_dispatch.py`, `baselines.py`, `xmoe.py`, `scripts/ds_moe.py --modes` | `micro`, `xmoe`, `deepep`, `model`, `model_deepep`, `model_tuned` | `logs/2026-11-22_exp/2n/`, `2n_b/`, `2n_c/`, `2n_d/`, `4n/`, `4n_b/` |
| [Hot experts 02](https://app.notion.com/p/420e2e64355e81908ec8f17b64650cd4): replicas chosen to cut each node's inter-node rows | `src/replicas.py`, `benchmark/replica_count.py`, `combine_dispatch.py --replicas` | `replicas` | `logs/2026-11-22_exp/replica_count_*.out`, `2n_c/replicas_*.out`, `4n/replicas_*.out` |
| [Profiles 01](https://app.notion.com/p/420e2e64355e81119062fadf3fe9c99f): where each library's time goes, on one layer and in the Qwen model | `benchmark/trace_report.py`, `combine_dispatch.py --trace`, `scripts/ds_moe.py --profile torch` | `micro`, `xmoe`, `deepep`, `model`, `model_deepep`, `model_tuned` | `traces/`, `model*/traces/`, `trace_report_*_rank0.txt` under `logs/2026-11-22_exp/*/` |
| [Profiles 02](https://app.notion.com/p/420e2e64355e81e09a34f2483ac1758e): each component's latency in the Qwen model, timed with CUDA events | `src/phase_clock.py` (then `benchmark/phase_clock.py`), `scripts/ds_moe.py --profile events` (then `phases`), `benchmark/summarize.py --by phase` | `model`, `model_deepep` | `logs/2026-11-22_trace/2n/`, the stall probe in `logs/2026-11-22_trace/probe/` |
| [Profiles 03](https://app.notion.com/p/421e2e64355e813a9b5deb8391f77275): comm, compute and overlap of every event in the Qwen model, on 2 and 4 nodes | `src/phase_clock.py`, `scripts/ds_moe.py --profile events`, `/pscratch/sd/e/eku/events_trace_2026-11-22/report.py` and `check_events.py` | `model_events` | `logs/2026-11-22_events/2n/`, `4n/`, `report.md` |
| [Redundancy 01](https://app.notion.com/p/420e2e64355e81c0a5f8eb597ad11b4e): cover routing, raising token redundancy across all nodes | `src/replicas.py` (`cover`, `ring`, `serving_node(cover=True)`), `benchmark/replica_count.py` | `replicas`, `channels` on 4 nodes | `logs/2026-11-22_exp/replica_count_4nodes.out`, `4n/replicas_*.out` |
| [DeepEP 01](https://app.notion.com/p/420e2e64355e813c9eb9fa41bae40e34): DeepEP across nodes on Perlmutter | `patches/`, `benchmark/all2all/baselines.py` (`deepep`) | `deepep`, `model_deepep` | `/pscratch/sd/e/eku/deepep_multinode/logs/job_61768851/` |
| [Pipeline 01](https://app.notion.com/p/421e2e64355e81edb9b6c4356d34e791): why pipelining gained nothing for Tutel and RBD, and the fix | `src/rbd.py` (`start_plans`, `finish_plans`), `src/pipeline.py`, `benchmark/all2all/contention.py`, `benchmark/pipeline_report.py`, `scripts/ds_moe.py --profile events` | `contention`, `pipe`, `model` of `pipeline_prefetch.sh` | `logs/2026-11-22_pipeline/2n/`, `2n_b/` |
| [Guests 01](https://app.notion.com/p/421e2e64355e81578af1eb590518db0f): hot experts of other nodes loaded from host memory, chosen by prediction | `src/guests.py`, `scripts/ds_guests.py`, `scripts/ds_moe.py --modes rbd_guests:...` | `guests`, `slots`, `final` of `pipeline_prefetch.sh` | `logs/2026-11-22_pipeline/2n_guests/`, `2n_b/`, `2n_c/`, `2n_d/` |
| [Guests 02](https://app.notion.com/p/421e2e64355e8187aba7da6040049580): few tokens, profiles against the libraries, an RBD benefit metric, and Qwen3.5-122B | `benchmark/activation_spread.py`, `benchmark/rbd_benefit.py`, `src/pipeline.py` (`lanes=2`), `src/guests.py` (`choose_guests_by_pilots`), `scripts/ds_moe.py --sequences` | `ab`, `few`, `profile`, `others`, `big`, `big_guests`, `big_others` of `pipeline_prefetch.sh` | `logs/2026-11-23_merge/2n/`, `2n_b/`, `2n_d/` in the worktree `spec_ep_worktrees/merge-pipeline-prefetch` |
| [Plan 01](https://app.notion.com/p/421e2e64355e81b9a3bdce9b5a712dda): RBD's plan after DeepEP's, and guests against AutoEP at few tokens | `src/rbd.py` (`start_plans`, `finish_plan`), `scripts/ds_guests.py`, `scripts/ds_moe.py --modes` | `ab`, `few`, `others` of `pipeline_prefetch.sh` | `logs/2026-11-23_plan/tree_a/logs/2n_a/`, `tree_b/logs/2n_b/`, `tree_c/logs/4n/` in the worktree `spec_ep_worktrees/plan-inband` |

The job steps of the first three pages are those of `scripts/jobs/baselines_hot_load.sh`, and
the later ones those of `scripts/jobs/baselines_replicas.sh`, picked with `STEPS`; the
Pipeline and Guests pages use `scripts/jobs/pipeline_prefetch.sh`. A
`baselines_replicas.sh` job leaves the script it ran as `job.sh` in its output directory.
`benchmark/summarize.py` turns a job's `.out` files into the tables on the pages.
`logs/` is not in git.

The `hot_*.out` files in `logs/2026-11-21_exp/` and `_exp2/` are superseded. Those
runs picked the hot experts from all experts, including the ones the node already
held, so about half the replicas moved nothing.

`decisions.tsv` is the decision trail of the run that produced these pages: one row
per decision, with what was chosen, why, and where the evidence is.
