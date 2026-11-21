"""Record which experts every token of the MoE model chooses, on many batches of text.

Loads Qwen3.6-35B-A3B with Hugging Face over the GPUs this process can see and runs
one prefill forward per batch. Batch b is --seq-len tokens of --text starting at
token b * seq_len, so --batches picks where in the text each batch comes from. The
i-th batch given is saved as LOGS/experts_rank{i}.pt, a [num_layers, seq_len, k] int16
tensor of expert ids: the file scripts/ds_moe.py --record writes for rank i, so the
benchmarks read either. LOGS/batches.txt lists the batches in that order.

Batch r of data/short_text.txt is what rank r gets in ds_moe.py, so recording a batch
that ds_moe.py also recorded checks this script against it.

Usage:
    LOGS=logs/$(date +%Y-%m-%d_%H-%M-%S) srun -N 1 -n 1 --gpus 4 \
        python -u scripts/record_routing.py --text data/long_text.txt --batches 0 1 2 40 80
"""
import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hf_forward import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B")
    parser.add_argument("--text", default="data/long_text.txt")
    parser.add_argument("--seq-len", type=int, default=4096, help="Tokens per batch")
    parser.add_argument("--batches", type=int, nargs="+", default=[0],
                        help="Batches to record, as offsets into the text in units of --seq-len")
    args = parser.parse_args()
    out = Path(os.environ["LOGS"])
    out.mkdir(parents=True, exist_ok=True)

    text = " ".join(Path(args.text).read_text(encoding="utf-8").split())
    input_ids = AutoTokenizer.from_pretrained(args.model)(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    if (max(args.batches) + 1) * args.seq_len > input_ids.numel():
        raise ValueError(f"{args.text} has {input_ids.numel()} tokens, fewer than batch "
                         f"{max(args.batches)} of {args.seq_len} tokens needs.")

    start = time.time()
    model = load_model(args.model)
    print(f"loaded {args.model} in {time.time() - start:.0f} s", flush=True)

    chosen = []
    for block in model.modules():
        if type(block).__name__ == "Qwen3_5MoeSparseMoeBlock":
            block.gate.register_forward_hook(
                lambda _module, _inputs, output: chosen.append(output[2].to(torch.int16).cpu())
            )

    (out / "batches.txt").write_text(" ".join(map(str, args.batches)) + "\n")
    with torch.inference_mode():
        for i, batch in enumerate(args.batches):
            chosen.clear()
            ids = input_ids[batch * args.seq_len : (batch + 1) * args.seq_len]
            model(input_ids=ids.unsqueeze(0).to(model.device), use_cache=False)
            torch.save(torch.stack(chosen), out / f"experts_rank{i}.pt")
            print(f"batch {batch} -> experts_rank{i}.pt {tuple(torch.stack(chosen).shape)}",
                  flush=True)
