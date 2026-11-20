# One RBD call

Back to the [architecture map](../architecture.md). Terms are defined in
[concepts.md](concepts.md).

Each rank holds `tokens [T, H]`, `weights [T, k]` and `route [T, k]`. All ranks call
`rbd_combine_dispatch` together, and each gets back `output [T, H]`.

A call has two parts. Planning decides where every row goes. The five stages then
move the rows.

## Planning: `make_plan` in `src/rbd.py`

No token data moves during planning.

| Step | What happens | Communication |
|---|---|---|
| 1. Classify copies | For every copy: its expert's rank and node, whether it is direct, and if not, whether it is its node's pilot and which rank receives that pilot. | None, except under the `even` rule (see below). |
| 2. Exchange counts | Each rank fills an integer table with one row per destination rank and exchanges it. | All2all 1, small. |
| 3. Read sizes | Every split size is read from the GPU at once. | The only host sync. |
| 4. Order stage 1 sends | Copies are sorted by destination rank, then by expert. Copies covered by a pilot are not sent. | None. |
| 5. Exchange pilot metadata | For each pilot: which experts on the destination node it stands for, and their weights. | All2all 2, small. |
| 6. Build the `Plan` | The index and weight tensors each stage needs. | None. |

Notes on individual steps:

- **Step 2.** A destination's table row has four groups of columns: direct rows per
  expert, one pilot count, stage 2 rows the destination will forward to each rank of
  its node, and stage 2 rows it will receive per (forwarding rank, expert).
- **Step 4.** Expert ids are never sent. Rows are sorted by expert inside each block,
  so the per-expert counts from step 2 are enough to rebuild them.
- **Step 5.** The weights travel with the metadata because the pilot rank, not the
  source rank, applies them to remote copies.

## The `Plan`

| Used by | Field | Shape | Meaning |
|---|---|---|---|
| Stage 1 | `send1_token` | `[rows sent]` | Token each stage 1 row carries. |
| Stage 1 | `send1`, `recv1` | one int per rank of the EP group | Rows to and from each rank. |
| Stage 1 | `direct` | `(start, end)` | Where this node's rows sit in the received buffer. The rest are pilots. |
| Stage 2 | `send2_row` | `[rows sent]` | Received stage 1 row that each stage 2 row copies. |
| Stage 2 | `send2`, `recv2` | one int per rank of the node | Rows to and from each rank of this node. |
| Experts | `expert_ids` | `[expert inputs]` | Expert of each input row. |
| Stage 2 combine | `merge_index`, `merge_weight` | `[num_pilots, k]` | Returned replica rows of each pilot, and their weights. |
| Stage 2 combine | `pilot_row` | `[num_pilots]` | Stage 1 row that each pilot's sum is written to. |
| Final merge | `final_index`, `final_weight` | `[T, k]` | Returned stage 1 row of each copy (-1 if a pilot covers it), and its weight. |

## The five stages in `src/rbd.py`

| Stage | Runs over | Input | Output | Row order of the output |
|---|---|---|---|---|
| `stage1_dispatch` | EP group | `tokens[send1_token]` | `received [sum(recv1), H]` | By source rank. This node's direct rows form one block, with pilots on either side. |
| `stage2_dispatch` | local group | `received[send2_row]` | `inputs [num_direct + sum(recv2), H]` | Direct rows, then stage 2 rows. |
| (experts) | this rank | `inputs`, `expert_ids` | `outputs`, same shape | Same as `inputs`. |
| `stage2_combine` | local group | `outputs[num_direct:]` | `back [sum(recv1), H]` | Same as `received`: direct outputs in their block, one weighted sum in each pilot's row. |
| `stage1_combine` | EP group | `back` | `returned [sum(send1), H]` | The stage 1 send order. |
| `final_merge` | this rank | `returned` | `output [T, H]` | Token order. |

`rbd_combine_dispatch` calls these in order, with the expert function between
`stage2_dispatch` and `stage2_combine`. The stage functions are public, so a later
variant can reorder or replace them.

Things worth knowing:

- **Every remote copy passes through stage 2.** This includes a copy whose expert is
  on the pilot rank itself; that row is a send from the rank to itself.
- **Where weights are applied.** Remote copies are scaled and summed on the pilot
  rank in `stage2_combine`, so pilot sums cross the network already weighted. Direct
  copies are scaled on the source rank in `final_merge`, where pilot sums get
  weight 1.
- **The weighted sums use `gather_sum`** from `src/ops.py`: a Triton kernel on GPU,
  plain PyTorch on CPU. The tests run the CPU path.
- **Cost of one call.** Two small planning all2alls, one host sync, and four data
  all2alls: two over the EP group and two inside the node.

## Pilot rules

The pilot rule picks which rank on the other node receives a token's pilot.

| Rule | Pilot rank | Trade-off |
|---|---|---|
| `rail` (default) | The rank with the source rank's position within its node. | Spreads pilots evenly. Costs one more NVLink copy when that rank holds none of the token's experts. |
| `even` | Round-robin over the node's ranks, so arrivals differ by at most one. | Needs one more collective before planning (an all-gather of per-node pilot counts). May add stage 2 traffic. |
| `first` | The rank of the token's first copy on that node, in route order. | Ranks holding popular top-1 experts receive the most pilots. |
| `most` | The rank holding the most of the token's copies on that node. | Fewest stage 2 rows, but ranks of popular experts receive the most pilots. |

The rule is implemented in `_select_pilot_rank` in `src/rbd.py`.
