# Architecture map

## What the code does

In a mixture-of-experts layer every token is processed by `k` experts, and the
experts are spread over GPUs. Each layer therefore has to send every token to the
ranks that hold its experts, run the experts, and bring the outputs back.

This repo implements two ways to do that round trip, plain all2all and
Redundancy-Bypassing Dispatch (RBD), and benchmarks them on routing recorded from a
real model. Features that were dropped and may come back are listed in
[todo.md](todo.md).

## Pages

| Page | Read it to learn |
|---|---|
| [concepts.md](architecture/concepts.md) | The vocabulary, one token's round trip step by step, and the assumptions. |
| [plain.md](architecture/plain.md) | The plain reference that RBD is checked and timed against. |
| [rbd.md](architecture/rbd.md) | How one RBD call works: planning, the `Plan`, the five stages, and the pilot rules. |
| [benchmarks.md](architecture/benchmarks.md) | What each benchmark measures and how `rbd_vs_plain` runs. |

## Layers

Imports point one way: each layer imports only from the layers above it in this list.

1. **Shared operations: `src/ops.py`.** `all2all` and `gather_sum`. Neither knows
   which algorithm calls it.
2. **The two algorithms: `src/plain.py` and `src/rbd.py`.** They are peers; neither
   imports the other.
3. **Benchmarks: `benchmark/`.** Standalone scripts. The few helpers they share are
   in `benchmark/utils.py`.

`src/` installs as the package `spec_ep`.

## File map

| File | Role | Public names |
|---|---|---|
| `src/ops.py` | Operations shared by plain and RBD. | `all2all`, `gather_sum` |
| `src/plain.py` | The plain reference. | `plain_combine_dispatch` |
| `src/rbd.py` | RBD: planning and the five stages. | `Plan`, `make_plan`, `stage1_dispatch`, `stage2_dispatch`, `stage2_combine`, `stage1_combine`, `final_merge`, `rbd_combine_dispatch`, `PILOT_RULES` |
| `benchmark/utils.py` | Launch, recorded routing, inputs, timing. | `init_dist`, `check_node_layout`, `load_routing`, `make_inputs`, `time_ms` |
| `benchmark/all2all/rbd_vs_plain.py` | RBD against plain on recorded routing. | |
| `benchmark/all2all/even_vs_uneven.py` | One all2all with the recorded uneven split sizes against even split sizes. | |
| `benchmark/internode/` | Send, receive and all2all of expert weights within and between nodes. | |
| `benchmark/compute.py` | GEMM and Flash Attention compute, no communication. | |
| `tests/test_rbd.py` | RBD against a locally computed answer, on CPU, for every pilot rule. | |
| `scripts/`, `configs/`, `src/utils*.py` | Model-level runs through DeepSpeed, including the routing recorder. | |

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
4. `benchmark/all2all/rbd_vs_plain.py`.
