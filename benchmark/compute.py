"""Forward-only GEMM and Flash Attention benchmarks.

Takes in hyper-parameters and computes the memory and TFLOPs of GEMM 
and Flash Attention.

Usage:
    python benchmark/compute.py
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DTYPE = torch.bfloat16
WARMUP = 5
ITERATIONS = 5

H = 2048
H2 = 512

BATCH, HEADS, SEQ_LEN, HEAD_DIM = 8, 32, 4096, 64
CAUSAL = True


def benchmark(fn):
    """Return (average GPU ms per forward pass, peak allocated MiB).
    """
    torch.cuda.reset_peak_memory_stats()
    
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERATIONS):
        fn()
    end.record()
    end.synchronize()
    
    peak = torch.cuda.max_memory_allocated()

    return start.elapsed_time(end) / ITERATIONS, peak / 2**20


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    print(f"GPU: {torch.cuda.get_device_name()} | dtype: {DTYPE}")

    a = torch.randn(H, H2, device="cuda", dtype=DTYPE)
    b = torch.randn(H2, H, device="cuda", dtype=DTYPE)
    out = torch.empty(H, H, device="cuda", dtype=DTYPE)

    gemm_ms, gemm_mem_mib = benchmark(lambda: torch.mm(a, b, out=out))
    gemm_flops = 2 * H **2 * H2
    gemm_tflops = gemm_flops / (gemm_ms / 1000) / 1e12
    print(
        f"GEMM: {gemm_ms:.3f} ms | {gemm_tflops:.2f} TFLOP/s | "
        f"peak memory: {gemm_mem_mib:.1f} MiB"
    )
    del a, b, out

    shape = (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=DTYPE)
    k = torch.randn(shape, device="cuda", dtype=DTYPE)
    v = torch.randn(shape, device="cuda", dtype=DTYPE)

    def attention():
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=CAUSAL
        )

    # Force Flash Attention; fail if unsupported instead of falling back.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attention_ms, attention_mem_mib = benchmark(attention)
        
    pairs = SEQ_LEN * (SEQ_LEN + 1) // 2 if CAUSAL else SEQ_LEN**2

    attention_flops = 4 * BATCH * HEADS * pairs * HEAD_DIM
    attention_tflops = attention_flops / (attention_ms / 1000) / 1e12

    print(
        f"Flash Attention: {attention_ms:.3f} ms | "
        f"{attention_tflops:.2f} TFLOP/s | "
        f"peak memory: {attention_mem_mib:.1f} MiB"
    )

if __name__ == "__main__":
    main()
