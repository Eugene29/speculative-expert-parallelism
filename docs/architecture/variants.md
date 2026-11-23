# RBD variants: two-lane and guests

Back to the [architecture map](../architecture.md).

Both keep RBD's plan. Two-lane changes how rows travel, and guests change which rows
there are. Each is one file built on public names, and each takes the arguments of
`rbd_combine_dispatch`.

## Two-lane: `src/two_lane.py`

RBD's stage 1 is one all2all over `ep_group` that carries two kinds of rows: direct
rows, whose expert is on the source's node, and pilots, which cross between nodes.
Two-lane sends them in two all2alls that run at the same time: direct rows over
`local_group`, pilots over `ep_group`. The same split applies on the way back.

1. `make_plan(lanes=2)` orders the stage 1 rows so that each lane's rows are one
   block: pilots first, then direct rows, on both the send and the receive side.
2. `two_lane_rbd_combine_dispatch` starts both lanes with `all2all_async` in place of
   `stage1_dispatch` and `stage1_combine`.
3. `stage2_dispatch`, `stage2_combine` and `final_merge` run unchanged.

With `early` the experts do not wait for the network lane. The direct rows arrive
over NVLink in under a millisecond while the pilots take two to three, and their
outputs are not needed before the pilot sums are back. So the experts run on a share
`early` of the direct rows while the pilots travel out, and on the rest while the
pilot sums travel back. No transfer is added and the expert function is called up to
three times. NCCL moves rows with GPU kernels of its own, which slow down under the
experts, and splitting the expert call has a cost of its own, so most of the overlap is paid back: on the 400B
model's first 30 layers on 2 nodes, 170 ms of overlap per forward bought 33 ms.

The slice pipeline (`src/pipeline.py`, commit c730d7e) was removed: it lost to two-lane
at every size but one (Plan 01).

## Guests: `src/guests.py`

A guest is an expert of another node that a rank runs itself for one layer, from
weights it pulls from the rank that owns them. A copy routed to a guest stays on its
node, so guests of the experts a node asks other nodes for most remove pilots.

1. `remote_demand` counts how often a rank's tokens ask for each expert of another
   node, in any routing: the layer's own, a recording, or a prediction.
2. `choose_guests` makes the experts in most demand guests, which removes the most
   copies. Unique placement adds the demand of a node's ranks, and the node's rank q
   takes the hottest experts e with `e % G == q`, so no two ranks of a node pull the
   same one. Redundant placement lets every rank take its own hottest experts, with no
   collective. `choose_guests_by_pilots` ranks experts by `pilot_credit`, the pilots a
   guest of them helps remove, in a few rounds. With `balance > 0` it weights that
   credit by how many pilots the expert's node receives, which takes guests from the
   busiest node first.
3. A guest is a new expert id after its rank's own experts, the numbering of
   `replicas.py`, so `rbd_combine_dispatch` runs unchanged on `Guests.grown[route]`.
4. `exchange_requests` tells every rank which guests the others want: one all-gather
   and one host sync, which gives the split sizes of the transfer.
5. `GuestPool` holds two sets of `slots` experts. `fetch` fills the idle set from the
   owners in one all2all per weight tensor, on a side stream and a communicator that
   nothing else uses, so the transfer runs under whatever the layer does meanwhile.
   `take` swaps the sets, and `wait` makes the stream wait for the set in use.

In the model (`scripts/ds_guests.py`, mode `rbd_guests`) all layers share one pool, so
guests take `2 * slots` experts of GPU memory whatever the depth. In MoE layer i every
rank starts pulling layer i + 1's guests. They come from layer i + 1's router applied
to layer i's tokens (speculative), from a recording of another batch (static), or of
this batch (oracle). The recorded sources know their transfers when the mode is built
and add no sync per layer; their layer 0 guests travel under the last layer of the
forward before. The expert call runs a rank's experts and its guests in one output by
launching DeepSpeed's grouped GEMM kernel once per weight tensor, with no host sync. It
runs on two-lane RBD by default: a guest turns a pilot into same-node rows, which the
all2all over all ranks carries slowly and the node's own communicator does not.

On the 400B model at 4,096 tokens per rank guests make the forward slower: 8 guests
are 201 MB of weights a layer against 34 MB of pilots per rank and direction, and with
top 10 routing they remove few pilots, since a pilot goes only when all of a token's
copies on that node are guests (400B 01).

The earlier design loaded guests from pinned host memory into `slots` rows of every
layer's weights (commit c730d7e): `layers * slots` experts of GPU memory and a copy of
the other nodes' experts in host memory.
