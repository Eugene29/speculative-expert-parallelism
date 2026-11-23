# Concepts

Back to the [architecture map](../architecture.md).

## Vocabulary

| Term | Meaning |
|---|---|
| rank | One GPU process. |
| node | A machine holding `G` ranks (`gpus_per_node`). Ranks are numbered node by node. |
| EP group | The ranks that together hold one copy of every expert (`ep_group`). By default, all ranks. |
| local group | The ranks of one node (`local_group`). |
| `E`, `L` | Experts in total, and experts per rank (`L = E // ranks in the EP group`). |
| token | One row of the layer input, `[H]`. |
| route | `[T, k]` expert ids: the `k` experts of each of this rank's `T` tokens. |
| weights | `[T, k]` combine weights; `weights[t, j]` scales the output of expert `route[t, j]`. |
| copy | One (token, expert) pair. A rank has `T * k` copies. |
| direct copy | A copy whose expert is on the token's own node. |
| pilot | The single row sent to another node on behalf of all of a token's copies there. |
| pilot rank | The rank on that node which receives the pilot. Chosen by the pilot rule. |
| replica | A row the pilot rank forwards to one expert on its node. |
| expert function | `expert(rows, expert_ids) -> outputs`, same row order. Supplied by the caller. RBD and two-lane call it with rows grouped by expert, ids ascending. |

## One token's round trip

The setting is two nodes of four ranks each. Token `t` starts on rank 1, which is on
node 0. It is routed to four experts:

- expert `a` on rank 2, on the same node;
- experts `b` and `c` on rank 5, on node 1;
- expert `d` on rank 7, on node 1.

With RBD the round trip has six steps.

1. **Stage 1 dispatch.** Rank 1 sends two rows. One goes straight to rank 2 for
   expert `a`; this is a direct row. The other goes to rank 5 and stands for all
   three experts on node 1; this is the pilot. Under the `rail` rule, rank 5 is
   chosen because it has the same position within its node as rank 1 has in its own.
2. **Stage 2 dispatch.** Rank 5 copies the pilot once per expert on its node. Two
   copies stay on rank 5, for `b` and `c`. One goes to rank 7, for `d`. Nothing
   leaves node 1 in this step.
3. **Experts.** Ranks 2, 5 and 7 run their experts on the rows they hold.
4. **Stage 2 combine.** Rank 7 returns the output of `d` to rank 5. Rank 5 scales the
   three outputs by their weights and adds them into a single row.
5. **Stage 1 combine.** Rank 2 returns the output of `a` to rank 1, and rank 5
   returns its single summed row to rank 1.
6. **Final merge.** Rank 1 scales the output of `a` by its weight and adds the summed
   row from rank 5. That is the token's output.

One row crossed the network in each direction. A plain all2all sends one row per
copy, so it would have sent three rows across the network for this token and three
back. That saving is the point of RBD.

## Assumptions

- **Contiguous experts.** Expert `e` lives on rank `e // L` of the EP group.
- **Contiguous nodes.** Ranks are numbered node by node, `G` per node.
  `check_node_layout` in `benchmark/utils.py` checks this at launch.
- **Collective calls.** Every rank must make the same call at the same time,
  including ranks that have no tokens.
- **Forward only.** The all2alls carry no gradients.
