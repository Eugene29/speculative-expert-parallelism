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

1. Plan every slice as far as its stage 1 rows (`start_plans`), start every slice's
   dispatch without waiting, then build the rest of the plans (`finish_plans`) while
   the rows are on the network.
2. Per slice: wait for its rows, run the experts, and start its combine without
   waiting.
3. Per slice: wait for its returned rows and merge them.

`pipelined_rbd_combine_dispatch` uses RBD's stages and `pipelined_plain_combine_dispatch`
the plain all2all. `expert` is called once per slice.

The RBD pipeline plans all slices in one exchange: `start_plans` gives the counts one
block of columns per slice and reads every slice's split sizes in a single host sync,
and one small all2all carries the pilot metadata of every slice. `finish_plans` is
the rest of the plan, about 30 small operations per slice that are host-bound and
need no other rank; it runs under the dispatches, so even at degree 1 the pipeline is
ahead of `rbd_combine_dispatch`, which finishes its plan first. The plain pipeline
still plans per slice.

What a slice can hide is its stage 2 and its experts, under the stage 1 rows of the
other slices. Two things bound it. With `degree` slices, the first dispatch and the
last combine have nothing to run under, so at most `1 - 1/degree` of stage 1 is
covered. And the transfers and the experts share the GPU: NCCL moves rows with its own
kernels, so each runs slower while the other does
(`benchmark/all2all/contention.py` measures by how much).

## Guests: `src/guests.py`

A guest is an expert of another node that a rank runs itself for one layer, from
weights it loads out of host memory. A copy routed to a guest stays on its node, so
guests of the experts a node asks other nodes for most remove pilots.

1. `remote_demand` counts how often a rank's tokens ask for each expert of another
   node, in any routing: the layer's own, a recording, or a prediction.
2. `choose_guests` makes the experts in most demand guests. Unique placement adds the
   demand of a node's ranks and deals the hottest experts over them, so no two ranks
   of a node load the same one. Redundant placement lets every rank take its own
   hottest experts, with no collective.
3. A guest is a new expert id after its rank's own experts, the numbering of
   `replicas.py`, so `rbd_combine_dispatch` and the pipeline run unchanged on
   `Guests.grown[route]`.
4. `GuestPool` holds the guests' weights on the GPU and refills them from pinned host
   memory on a side stream. Two pools take turns, one for the running layer and one
   for the next, so guests cost `2 * slots` experts of GPU memory whatever the depth.

In the model (`scripts/ds_guests.py`, mode `rbd_guests`), MoE layer i chooses and
starts loading the guests of layer i + 1, then runs with the guests chosen during
layer i - 1. It runs on two-lane RBD by default: a guest turns a pilot into same-node
rows, which the all2all over all ranks carries slowly and the node's own communicator
does not. The next layer's guests come from its router applied to this layer's
tokens (speculative), from a recording of another batch (static), or of this batch
(oracle).
