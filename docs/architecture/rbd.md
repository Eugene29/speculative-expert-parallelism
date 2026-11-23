# One RBD call

Back to the [architecture map](../architecture.md). Terms are defined in
[concepts.md](concepts.md).

Each rank holds `tokens [T, H]`, `weights [T, k]` and `route [T, k]`. All ranks call
`rbd_combine_dispatch` together, and each gets back `output [T, H]`.

A call has two parts. The plan decides where every row goes. The five stages then
move the rows.

## The plan: `make_plan` in `src/rbd.py`

The plan follows DeepEP's: a layout computed where the tokens are, and rows that carry
what their receiver must know. The only planning communication is the count exchange
that NCCL needs before it can receive. The plan is GPU only: it is three Triton
kernels, two in `make_plan` and one on the pilot rank.

`make_plan` is the source rank's half. No token data moves.

| Step | What happens | Communication |
|---|---|---|
| 1. `_plan_kernel` | One program per 128 tokens. For every copy: whether it is direct, and if not, whether it is its node's pilot and which rank receives that pilot. An atomic add counts the row in its bucket and returns the row's slot there. Two more count the copy for stage 2. The kernel also writes every token's metadata and combine weights. | None. |
| 2. Exchange counts | The ranks exchange the count table, one row per destination rank. | All2all 1, small. |
| 3. Place the rows | A bucket's first row is a prefix sum over the buckets in sending order. A stage 1 row's place is its bucket's start plus its slot. | None. |
| 4. Read sizes | Every split size is read from the GPU at once. | The only host sync. |
| 5. `_group_kernel` | One program per run of rows this rank's experts get. It orders the direct rows by expert, and the stage 2 rows by expert, and writes each row's place in that order and the expert ids. | None. |

The pilot rank's half is `_fan_out`, inside `stage2_dispatch`. A pilot's metadata names
all of its token's experts; the ones on this node are the copies of stage 2.
`_fanout_kernel` gives each its stage 2 row with one atomic add on a cursor per expert.
It does not wait for the rows: its work is queued behind them on the GPU.

Notes:

- **Buckets.** A stage 1 bucket is a (destination rank, expert) pair for direct rows
  and a destination rank for pilots. Rows of one bucket are interchangeable, since
  they share an expert or carry their own metadata, so the order the atomic adds give
  them is free. That is what replaces the sort. It also means the row order inside a
  bucket differs from run to run, which changes no output.
- **Counts.** A destination's table row has four groups of columns: direct rows per
  expert, one pilot count, the stage 2 rows the destination forwards per (rank of its
  node, expert), and the stage 2 rows it receives per (pilot rank, expert).
- **Expert ids are never sent.** Rows arrive grouped by bucket, so the per-expert
  counts from step 2 are enough to rebuild them.
- **Metadata.** The weights travel with the pilot because the pilot rank, not the
  source rank, applies them to remote copies. M is `8k` bytes in the tokens' dtype: 40
  columns for bf16 at k = 10, 1% of a 4096-wide row. Direct rows carry them unused.
- **Lanes.** `make_plan(lanes=2)` only changes the sending order of the buckets:
  every rank's pilots first, then every rank's direct rows.
- **Grouping.** A part's rows arrive by source rank and then expert, so the rows of
  one (rank, expert) are a run, and the count exchange already gave this rank the
  length of every run. Step 5 moves whole runs: a part's rows are by expert and then
  rank afterwards. The experts then need no sort, which in DeepSpeed's hook is a
  gather and a scatter of every row (39 ms of a 798 ms forward on the 400B model).
- **What bounds it.** On one A100 at the 400B layer shape the token kernel takes
  31 us to launch and 37 us to run. The rest of `make_plan`, about 0.5 ms, is some 45
  small torch operations and the host sync. Moving that arithmetic to the host was
  tried and was slower in the model.

## The `Plan`

| Used by | Field | Shape | Meaning |
|---|---|---|---|
| Stage 1 | `send1_token` | `[rows sent]` | Token of each stage 1 row. |
| Stage 1 | `send1`, `recv1` | one int per rank of the EP group | Rows to and from each rank. |
| Stage 1 | `direct` | `(start, end)` | Where this node's rows sit in the received buffer. The rest are pilots. |
| Stage 2 | `pilot_row` | `[num_pilots]` | Received stage 1 row of each pilot. |
| Stage 2 | `send2`, `recv2` | one int per rank of the node | Rows to and from each rank of this node. |
| Stage 2 | `send2_start`, `first_expert` | `[G * L]`, int | First stage 2 row sent for each expert of this node, and the node's first expert. |
| Experts | `direct_order`, `direct_place` | `[num_direct]` | Received stage 1 row of each direct row, by expert, and where each direct row is in that order. |
| Experts | `stage2_order`, `stage2_place` | `[sum(recv2)]` | The same for the stage 2 rows, as rows of the stage 2 buffer. |
| Experts | `expert_ids` | `[expert inputs]` | Experts of the direct rows, then of the stage 2 rows, each part ascending. |
| Final merge | `final_index`, `final_weight` | `[T, k]` | Returned stage 1 row of each copy (-1 if a pilot covers it), and its weight. |
| Stage 1 | `meta` | `[T, 2k]` int32 | Every token's routing, then the bits of its weights. |

`stage2_dispatch` also returns a `Merge` for `stage2_combine`: `index` and `weight`,
both `[num_pilots, k]`, the returned replica rows of each pilot and their weights.

## The five stages in `src/rbd.py`

| Stage | Runs over | Input | Output | Row order of the output |
|---|---|---|---|---|
| `stage1_dispatch` | EP group | `with_meta(tokens, meta)[send1_token]` | `received [sum(recv1), H + M]` | By source rank. This node's direct rows form one block, with pilots on either side. |
| `run_direct` | this rank | `received[direct_order, :H]` | the direct rows of `back [sum(recv1), H]` | As `received`: the experts run on rows grouped by expert, and `direct_place` puts their outputs back. |
| `stage2_dispatch` | local group | `received[send2_row, :H]` | `rows [sum(recv2), H]`, and the `Merge` | Grouped by expert. |
| `run_experts` | this rank | `rows`, `expert_ids[num_direct:]` | `outputs`, same shape | Same as `rows`. |
| `stage2_combine` | local group | `outputs[stage2_place]` | the pilot rows of `back` | One weighted sum in each pilot's row. |
| `stage1_combine` | EP group | `back` | `returned [sum(send1), H]` | The stage 1 send order. |
| `final_merge` | this rank | `returned` | `output [T, H]` | Token order. |

`rbd_combine_dispatch` calls these in order. The stage functions are public, so a
variant can reorder or replace them, as two-lane does.

The expert function is called twice, on the direct rows and on the stage 2 rows, each
time with rows grouped by expert and ids ascending. A caller whose experts want
grouped rows can therefore skip its own sort: `scripts/ds_moe.py` does, for the modes
in `benchmark/all2all/modes.GROUPED`.

Things worth knowing:

- **Every remote copy passes through stage 2.** This includes a copy whose expert is
  on the pilot rank itself; that row is a send from the rank to itself.
- **Where weights are applied.** Remote copies are scaled and summed on the pilot
  rank in `stage2_combine`, so pilot sums cross the network already weighted. Direct
  copies are scaled on the source rank in `final_merge`, where pilot sums get
  weight 1.
- **The weighted sums use `gather_sum`** from `src/ops.py`: a Triton kernel on GPU,
  plain PyTorch on CPU and for float64, which the tests use.
- **Cost of one call.** One small planning all2all, one host sync, three plan
  kernels, two expert calls, and four data all2alls: two over the EP group and two
  inside the node.

## Pilot rules

The pilot rule picks which rank on the other node receives a token's pilot.

| Rule | Pilot rank | Trade-off |
|---|---|---|
| `rail` (default) | The rank with the source rank's position within its node. | Spreads pilots evenly. Costs one more NVLink copy when that rank holds none of the token's experts. |
| `first` | The rank of the token's first copy on that node, in route order. | Ranks holding popular top-1 experts receive the most pilots. |

The rule is a branch of `_plan_kernel` in `src/rbd.py`. The rules `even` and `most`
were removed with the sort-based plan (commit c730d7e has them): no experiment used
them, and each needed work across tokens that the kernel does not do.
