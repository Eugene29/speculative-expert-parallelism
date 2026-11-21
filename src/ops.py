"""Operations shared by the dispatch algorithms: all2all, all2all_async and gather_sum.

None knows which algorithm calls it. all2all moves rows between the ranks of a
process group, and all2all_async does the same without waiting for them to arrive.
gather_sum merges rows on one rank: a weighted sum of gathered rows, in a Triton
kernel on GPU and in PyTorch ops on CPU.

Usage:
    from spec_ep.ops import all2all, all2all_async, gather_sum
"""
from collections.abc import Callable

import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

__all__ = ["all2all", "all2all_async", "gather_sum"]


def all2all(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-size all2all of rows grouped by destination. Collective over group.

    Args:
        rows: [sum(send), ...] rows to send, grouped by destination rank.
        send: Rows sent to each rank of group.
        recv: Rows received from each rank of group.
        group: Process group. None means the global group.
        out: Buffer to receive into. Allocated if None.

    Returns:
        out: [sum(recv), ...] received rows, grouped by source rank.
    """
    if out is None:
        out = rows.new_empty((sum(recv), *rows.shape[1:]))
    dist.all_to_all_single(out, rows, recv, send, group=group)
    return out


def all2all_async(
    rows: torch.Tensor,
    send: list[int],
    recv: list[int],
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> Callable[[], torch.Tensor]:
    """Start an all2all and return before the rows have arrived. Collective over group.

    Takes the arguments of all2all. Calls on one group run in the order they were
    started, so work on another group, or on this rank, overlaps with the transfer.

    Returns:
        wait: Blocks until the rows have arrived, then returns out as all2all does.
    """
    if out is None:
        out = rows.new_empty((sum(recv), *rows.shape[1:]))
    work = dist.all_to_all_single(out, rows, recv, send, group=group, async_op=True)

    def wait() -> torch.Tensor:
        work.wait()
        return out

    return wait


if triton is not None:

    @triton.jit
    def _gather_sum_kernel(
        src,
        index,
        weight,
        out,
        out_rows,
        H,
        src_stride,
        out_stride,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_OUT_ROWS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        in_row = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(K):
            i = tl.load(index + row * K + j)
            w = tl.load(weight + row * K + j, mask=i >= 0, other=0.0)
            x = tl.load(src + i * src_stride + cols, mask=in_row & (i >= 0), other=0.0)
            acc += w * x.to(tl.float32)
        dst = row
        if HAS_OUT_ROWS:
            dst = tl.load(out_rows + row)
        tl.store(out + dst * out_stride + cols, acc.to(out.dtype.element_ty), mask=in_row)


def gather_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    out_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[out_rows[r]] = sum_j weight[r, j] * src[index[r, j]], skipping index -1.

    Sums in float32 (float64 for float64 src). Triton on CUDA, else PyTorch ops;
    float64 always takes the PyTorch path.

    Args:
        src: [num_src, H] rows to gather from.
        index: [num_out, K] row of src for each term, or -1 for no term.
        weight: [num_out, K] weight of each term.
        out: Buffer to write into; rows not written are left untouched. Allocated as
            [num_out, H] if None.
        out_rows: [num_out] row of out for each sum. Must be unique. Defaults to r.

    Returns:
        out: The buffer holding the sums.
    """
    num_out, K = index.shape
    H = src.shape[1]
    if out is None:
        out = src.new_empty((num_out, H))

    if triton is None or not src.is_cuda or src.dtype == torch.float64:
        padded = torch.cat([src, src.new_zeros((1, H))])
        safe = torch.where(index >= 0, index, src.shape[0])
        sum_dtype = torch.promote_types(src.dtype, torch.float32)
        valid_weight = torch.where(index >= 0, weight, 0).to(sum_dtype)
        merged = (padded[safe].to(sum_dtype) * valid_weight.unsqueeze(-1)).sum(dim=1)
        if out_rows is None:
            out[:num_out] = merged.to(out.dtype)
        else:
            out[out_rows] = merged.to(out.dtype)
        return out

    if num_out == 0 or H == 0:
        return out
    if src.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("gather_sum needs rows that are contiguous along H.")
    BLOCK = min(triton.next_power_of_2(H), 1024)
    grid = (num_out, triton.cdiv(H, BLOCK))
    _gather_sum_kernel[grid](
        src,
        index.contiguous(),
        weight.float().contiguous(),
        out,
        out_rows.contiguous() if out_rows is not None else index,
        H,
        src.stride(0),
        out.stride(0),
        K=K,
        BLOCK=BLOCK,
        HAS_OUT_ROWS=out_rows is not None,
        num_warps=4,
    )
    return out
