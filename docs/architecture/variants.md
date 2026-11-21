# RBD variants: two-lane and the pipeline

Back to the [architecture map](../architecture.md).

Both variants keep RBD's plan and change only how rows travel. Each is one file
built on the public names of `rbd.py` and `ops.py`, and each takes the arguments of
`rbd_combine_dispatch`.

## Two-lane: `src/two_lane.py`

RBD's stage 1 is one all2all over `ep_group` that carries two kinds of rows: direct
rows, whose expert is on the source's node, and pilots, which cross between nodes.
Two-lane sends them in two all2alls that run at the same time: direct rows over
`local_group`, pilots over `ep_group`. The same split applies on the way back.

1. `two_lane_plan` reorders an RBD `Plan` so that each lane's rows are one block:
   pilots first, then direct rows, on both the send and the receive side.
2. `two_lane_dispatch` and `two_lane_combine` replace `stage1_dispatch` and
   `stage1_combine`. Each starts the local lane with `all2all_async`, runs the
   network lane, then waits for the local lane.
3. `stage2_dispatch`, `stage2_combine` and `final_merge` run unchanged on the
   reordered plan.

## The pipeline: `src/pipeline.py`

In a blocking round trip the GPU idles while rows are on the network. The pipeline
cuts a rank's tokens into `degree` slices and gives each slice its own round trip, so
the experts of one slice run while the next slice is on the network.

1. Per slice: plan it, then start its dispatch without waiting.
2. Per slice: wait for its rows, run the experts, and start its combine without
   waiting.
3. Per slice: wait for its returned rows and merge them.

`pipelined_rbd_combine_dispatch` uses RBD's stages and `pipelined_plain_combine_dispatch`
the plain all2all. `expert` is called once per slice.

The pipeline can hide at most the expert time, and it pays one plan per slice. A plan
is host work that does not shrink with the slice, so the pipeline pays off only when
the experts cost more than the extra plans.
