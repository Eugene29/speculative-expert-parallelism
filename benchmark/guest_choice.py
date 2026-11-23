"""What each way of choosing guests leaves on the network, on recorded routing. CPU only.

Spawns one CPU process per rank over gloo and replays a recording layer by layer. For
every placement (unique, redundant), choice (demand, pilots, pilots with balance) and
slot count, every rank chooses guests from the layer's own routing, the best any
prediction could do, and the ranks add up what RBD would then send:

    copies: the (token, expert) pairs whose expert is on another node, which is what a
        plain all2all sends between nodes.
    pilots: the (token, other node) pairs, which is what RBD sends between nodes.
    busiest node: the pilots the node that receives the most gets, as a multiple of
        the mean over the nodes. An all2all is as slow as its busiest link.
    most sent: the most guests one rank has to send in a layer, in experts. The rank
        that owns the hot experts sends them to every rank that asks.

Copies and pilots are given as a share of what the routing sends with no guests.
--nodes can exceed the recording's: a recording of 8 ranks replays as 16 ranks of half
the tokens each, since a token's routing does not depend on where the experts are.

Usage:
    python -m benchmark.guest_choice LOGS/model --nodes 2 4 --slots 4 8 16
"""
import argparse
import tempfile
from itertools import product
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spec_ep.guests import choose_guests, choose_guests_by_pilots, exchange_requests, remote_demand

G = 4
CHOICES = {"demand": None, "pilots": 0.0, "pilots, balance 2": 2.0}


def count(route: torch.Tensor, grown: torch.Tensor, per_rank: int, node: int, num_nodes: int) -> torch.Tensor:
    """Count what one rank's tokens send to other nodes under a relabeled routing.

    Returns:
        sent: [1 + num_nodes] the copies for other nodes, then the pilots to each node.
    """
    served = grown[route] // per_rank // G
    visits = torch.zeros(route.shape[0], num_nodes).scatter_(1, served, 1.0).sum(dim=0)
    visits[node] = 0
    return torch.cat([(served != node).sum().view(1), visits])


def replay(
    rank: int,
    init_file: str,
    run_dir: Path,
    world: int,
    E: int,
    slot_counts: list[int],
) -> None:
    """Replay the recording on one rank and print the table on rank 0."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    local_group, _ = dist.new_subgroups(group_size=G)
    recorded = len(list(run_dir.glob("experts_rank*.pt")))
    share = world // recorded
    routing = torch.load(run_dir / f"experts_rank{rank // share}.pt").long()
    routing = routing[:, rank % share :: share]
    L, node, num_nodes = E // world, rank // G, world // G
    home = torch.arange(E)

    if rank == 0:
        print(f"{world // G} nodes x {G} ranks, {E} experts, {routing.shape[0]} layers, "
              f"{routing.shape[1]} tokens per rank, top {routing.shape[2]}\n")
        print("| placement | choice | slots | copies left | pilots left | busiest node | most sent |")
        print("|---|---|---|---|---|---|---|")
    for placement, (choice, balance), slots in product(("unique", "redundant"), CHOICES.items(), slot_counts):
        node_group = local_group if placement == "unique" else None
        none, left, most_sent = torch.zeros(1 + num_nodes), torch.zeros(1 + num_nodes), torch.zeros(1)
        busiest = []
        for route in routing:
            if balance is None:
                guests = choose_guests(remote_demand(route, E, node, num_nodes), slots, G, node_group)
            else:
                guests = choose_guests_by_pilots(route, E, slots, G, node_group, balance=balance)
            layer = count(route, guests.grown, L + slots, node, num_nodes)
            dist.all_reduce(layer)
            busiest.append((layer[1:].max() / layer[1:].mean()).item())
            left += layer
            none += count(route, home, L, node, num_nodes)
            sent = torch.tensor([float(sum(exchange_requests(guests.slot_expert, L).send))])
            dist.all_reduce(sent, op=dist.ReduceOp.MAX)
            most_sent = torch.maximum(most_sent, sent)
        dist.all_reduce(none)
        if rank == 0:
            print(f"| {placement} | {choice} | {slots} | {left[0] / none[0]:.1%} | "
                  f"{left[1:].sum() / none[1:].sum():.1%} | {sum(busiest) / len(busiest):.3f}x | "
                  f"{most_sent.item():.0f} |")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="Directory with experts_rank{r}.pt")
    parser.add_argument("--nodes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--experts", type=int, default=512, help="E, total experts")
    parser.add_argument("--slots", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()
    for nodes in args.nodes:
        init_file = str(Path(tempfile.mkdtemp()) / "init")
        mp.spawn(
            replay, args=(init_file, args.run_dir, nodes * G, args.experts, args.slots),
            nprocs=nodes * G,
        )
        print()
