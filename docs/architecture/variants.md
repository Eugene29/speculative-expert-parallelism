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

1. `start_plans(lanes=2)` orders the stage 1 rows so that each lane's rows are one
   block: pilots first, then direct rows, on both the send and the receive side.
2. `two_lane_dispatch` and `two_lane_combine` replace `stage1_dispatch` and
   `stage1_combine`. Each starts the local lane with `all2all_async`, runs the
   network lane, then waits for the local lane.
3. `finish_plan`, `stage2_dispatch`, `stage2_combine` and `final_merge` run
   unchanged.

## The pipeline: `src/pipeline.py`

In a blocking round trip the GPU idles while rows are on the network. The pipeline
cuts a rank's tokens into `degree` slices and gives each slice its own round trip, so
the experts of one slice run while the next slice is on the network.

1. Plan the source half of every slice (`start_plans`) and start every slice's
   dispatch without waiting.
2. Per slice: wait for its rows, finish its plan from them (`finish_plan`), run the
   experts, and start its combine without waiting.
3. Per slice: wait for its returned rows and merge them.

`pipelined_rbd_combine_dispatch` uses RBD's stages and `pipelined_plain_combine_dispatch`
the plain all2all. `expert` is called once per slice.

The RBD pipeline plans all slices as one batch: `start_plans` gives the counts one
block of columns per slice, reads every slice's split sizes in a single host sync, and
orders every slice's rows in one sort. A slice's `finish_plan` is queued behind its
rows and does not wait for them. The plain pipeline still plans per slice.

What a slice can hide is its stage 2 and its experts, under the stage 1 rows of the
other slices. Two things bound it. With `degree` slices, the first dispatch and the
last combine have nothing to run under, so at most `1 - 1/degree` of stage 1 is
covered. And the transfers and the experts share the GPU: NCCL moves rows with its own
kernels, so each runs slower while the other does
(`benchmark/all2all/contention.py` measures by how much).

With `lanes=2` the RBD pipeline sends every slice's stage 1 in the two lanes of
`two_lane.py`: `two_lane_all2all_async` starts both lanes without waiting, on rows
that `start_plans(lanes=2)` put in lane order.

## Guests: `src/guests.py`

A guest is an expert of another node that a rank runs itself for one layer, from
weights it loads out of host memory. A copy routed to a guest stays on its node, so
guests of the experts a node asks other nodes for most remove pilots.

1. `remote_demand` counts how often a rank's tokens ask for each expert of another
   node, in any routing: the layer's own, a recording, or a prediction.
2. `choose_guests` makes the experts in most demand guests. Unique placement adds the
   demand of a node's ranks, and the node's rank q takes the hottest experts e with
   `e % G == q`. No two ranks of a node load the same one, and a rank only ever hosts
   one remote expert in G, so its host store is that much smaller. Redundant placement
   lets every rank take its own hottest experts, with no collective.
   `choose_guests_by_pilots` ranks experts by `pilot_credit`, the pilots a guest of
   them helps remove, in a few rounds. It leaves 3 to 7% fewer pilots than demand and
   costs more than that saves when it runs per layer (Guests 02).
3. A guest is a new expert id after its rank's own experts, the numbering of
   `replicas.py`, so `rbd_combine_dispatch` and the pipeline run unchanged on
   `Guests.grown[route]`.
4. `GuestPool` fills the guests' weights on the GPU from pinned host memory on a side
   stream. The tensors it fills are its caller's.

In the model (`scripts/ds_guests.py`, mode `rbd_guests`), every layer's weight tensors
get `slots` rows after the rank's own experts, and the pool fills those. One expert
call then runs a rank's experts and its guests, with no host sync, at
`layers * slots` experts of GPU memory. It runs on two-lane RBD by default: a guest
turns a pilot into same-node rows, which the all2all over all ranks carries slowly and
the node's own communicator does not. A layer's guests come from its router applied to
the previous layer's tokens (speculative), from a recording of another batch (static),
or of this batch (oracle). Speculative guests are chosen and loaded during the layer
before. Recorded guests are loaded once, when the mode is set.
