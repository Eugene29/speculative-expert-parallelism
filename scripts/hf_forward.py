"""Time Hugging Face inference of the MoE model, the baseline with no expert parallelism.

Loads the text backbone with device_map="auto", which spreads the layers over the
GPUs this process can see, and times the same prefill forward scripts/ds_moe.py
times under AutoEP: sequences of --seq-len tokens cut from data/short_text.txt,
eager, use_cache=False. Sequence b holds the tokens rank b gets in ds_moe.py, so a
batch of B sequences is the work B ranks do there in one forward.

For every batch size it runs three timed forwards, then one under torch.profiler,
and leaves LOGS/profile/trace_hf_b{B}.json and key_averages_hf_b{B}.txt.

Usage:
    LOGS=logs/$(date +%Y-%m-%d_%H-%M-%S) srun -N 1 -n 1 --gpus 4 \
        python -u scripts/hf_forward.py --batch-sizes 1 4
"""
import argparse
import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM


def load_model(model_id: str) -> Qwen3_5MoeForCausalLM:
    """Load the text backbone over every visible GPU.

    Returns:
        model: The model in eval mode, its layers placed by device_map="auto".
    """
    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading_info["missing_keys"]:
        raise RuntimeError(f"Missing model weights: {loading_info['missing_keys']}")
    model.config.use_cache = False
    return model.eval()


def get_batch(
    model_id: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Cut batch_size sequences of seq_len tokens from the text, as ds_moe.py's ranks do.

    Returns:
        batch: input_ids and attention_mask, each [batch_size, seq_len].
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = " ".join(Path("data/short_text.txt").read_text(encoding="utf-8").split())
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    needed = batch_size * seq_len
    if tokens.input_ids.shape[1] < needed:
        raise ValueError(f"Provide more text: {needed} tokens are required, "
                         f"got {tokens.input_ids.shape[1]}.")
    return {
        "input_ids": tokens.input_ids[0, :needed].view(batch_size, seq_len).to(device),
        "attention_mask": tokens.attention_mask[0, :needed].view(batch_size, seq_len).to(device),
    }


def sync_all() -> None:
    """Wait for every visible GPU to finish its queued work."""
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def time_forward(model: Qwen3_5MoeForCausalLM, batch: dict[str, torch.Tensor]) -> float:
    """Time one forward.

    Returns:
        seconds: Wall time of the forward, GPU work included.
    """
    sync_all()
    start = time.time()
    model(**batch, use_cache=False)
    sync_all()
    return time.time() - start


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        args: model, seq_len and batch_sizes.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1],
                        help="Sequences per forward; each one is a rank's work in ds_moe.py")
    return parser.parse_args()


def main() -> None:
    """Load the model, then time and profile a forward for every batch size."""
    args = parse_args()
    out = Path(os.environ["LOGS"]) / "profile"
    out.mkdir(parents=True, exist_ok=True)
    gpus = torch.cuda.device_count()

    start = time.time()
    model = load_model(args.model)
    print(f"loaded {args.model} over {gpus} GPUs in {time.time() - start:.0f} s", flush=True)

    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            batch = get_batch(args.model, model.device, args.seq_len, batch_size)
            for i in range(3):
                seconds = time_forward(model, batch)
                tokens = batch_size * args.seq_len
                print(f"hf batch {batch_size} x {args.seq_len}: forward {i} took {seconds:.3f} s, "
                      f"{tokens / seconds:.0f} tokens/s, {tokens / seconds / gpus:.0f} per GPU",
                      flush=True)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function("forward"):
                    model(**batch, use_cache=False)
                sync_all()
            prof.export_chrome_trace(str(out / f"trace_hf_b{batch_size}.json"))
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
            (out / f"key_averages_hf_b{batch_size}.txt").write_text(table)

    for d in range(gpus):
        print(f"gpu {d}: peak allocated {torch.cuda.max_memory_allocated(d) / 2**30:.2f} GiB",
              flush=True)


if __name__ == "__main__":
    main()
