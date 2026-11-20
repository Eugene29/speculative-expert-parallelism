# The plain reference

Back to the [architecture map](../architecture.md). Terms are defined in
[concepts.md](concepts.md).

`plain_combine_dispatch` in `src/plain.py` is the baseline RBD is checked and timed
against. It sends one row per copy and has no notion of nodes.

1. **Plan.** Count the copies going to each rank and exchange the counts. Then send
   each copy's expert id. Two small all2alls.
2. **Dispatch.** Send one row per copy to its expert's rank. One data all2all.
3. **Experts.** Run the expert function on the received rows.
4. **Combine.** Send every output row back to its source rank. One data all2all.
5. **Merge.** On the source rank, scale each returned row by its weight and add the
   rows of each token.

With `fused_merge=True` the merge uses `gather_sum` from `src/ops.py` instead of
`index_add_`. Comparing the two separates the gain of a faster merge kernel from the
gain of sending fewer rows over the network.
