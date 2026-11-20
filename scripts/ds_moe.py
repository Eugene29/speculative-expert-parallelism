"""Run MoE model with deepspeed's AutoEP

Note:
    Using Qwen3.6 35B-A3B
"""

import argparse
import json
import os
import time
import torch
import deepspeed
import deepspeed.comm as dist

from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from pathlib import Path

from utils import print0, shutdown_comm, get_rank_and_world_size
from utils_profile import AutoEPMoELayer, count_params, end_flops_profile, add_expert_statistics_hook, record_function, report_peak_mem, save_torch_profile, start_flops_profile, torch_profile_ctx


rank, world_size = get_rank_and_world_size()
logs = Path(os.environ["LOGS"])


def load_model(model_id):
    print0("Loading model and tokenizer...")
    model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=None,             # Load on CPU; DeepSpeed handles GPU placement
        attn_implementation="sdpa",
        output_loading_info=True,
    )

    total, expert = count_params(model)
    print0(f"Full model: {total/1e9:.2f}B params ({expert/1e9:.2f}B in experts, "
           f"{(total-expert)/1e9:.2f}B dense)")

    assert model.config.model_type == "qwen3_5_moe_text"
    model.config.use_cache = False  # Q. What is this? 

    # Do not benchmark with accidentally uninitialized text weights.
    if loading_info["missing_keys"]:
        raise RuntimeError(
            f"Missing model weights: {loading_info['missing_keys']}"
        )
    return model
        
def init_deepspeed(model, ds_config_pth):
    with open(ds_config_pth, 'r') as file:
        ds_config = json.load(file)

    # Then apply the AutoEP config from above.
    print0("Applying autoEP...")
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    engine.eval()
    return engine
    
    
def switch_comm(engine, ds_config_pth):
    """Switch every AutoEP layer to another config's communication settings.

    Lets one job time several comm backends on the same loaded model and nodes.
    Only the expert_parallel comm_* keys are applied; nothing is re-validated, so
    pass the most demanding config to deepspeed.initialize first.
    """
    with open(ds_config_pth, 'r') as file:
        ep = json.load(file)["expert_parallel"]
    for layer in engine.module.modules():
        if isinstance(layer, AutoEPMoELayer):
            layer.comm_backend = ep.get("comm_backend", "comm")
            layer.comm_rbd_pilot = ep.get("comm_rbd_pilot", "rail")
            layer.comm_rbd_pipeline_degree = ep.get("comm_rbd_pipeline_degree", 1)
            layer._bind_rbd_node_group()


def get_batch(model_id, dev, seq_len):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    text = Path("data/short_text.txt").read_text(encoding="utf-8")
    text = " ".join(text.split())
    
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    strt, end = rank*seq_len, (rank+1)*seq_len
    
    if tokens.input_ids[:, strt:end].shape[1] < seq_len:
        raise ValueError(f"Provide more text: at least {seq_len} tokens are required. "
                         f"[{rank}] has {tokens.input_ids[:, strt:end].shape[1]} seq.")

    batch = {
        "input_ids": tokens.input_ids[:, strt:end].to(dev),
        "attention_mask": tokens.attention_mask[:, strt:end].to(dev),
    }
    
    return batch

def record_expert_routing(engine, batch):
    """Save expert routing statistics (tokens per expert)
    
    Q. What does it store? 
    """
    sent = {}
    add_expert_statistics_hook(engine.module, sent)
    engine(**batch, use_cache=False)
    torch.save(torch.stack(list(sent.values())), logs / f"experts_rank{rank}.pt")

def record_latency(engine, batch):
    """Purely times the latency"""
    # dist.barrier() # TODO: 
    torch.cuda.synchronize(engine.device)
    dist.barrier()
    strt = time.time()
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    dist.barrier()
    dur = time.time() - strt
    print(f"[{rank}] dur: {dur}", flush=True)
    # TFlops_per_gpu = total_flop_count / dur / 1e12 / world_size
    # print(f"[{rank}] real TFlops: {TFlops_per_gpu}", flush=True)
    
def run_pytorch_profile(engine, batch, outdir=logs):
    """Run pytorch profiler"""
    with torch_profile_ctx() as tprof:
        with record_function("forward"):
            outputs = engine(**batch, use_cache=False)
        torch.cuda.synchronize(engine.device)
    save_torch_profile(tprof, outdir=outdir, verbose=True)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-pth", type=str, nargs="+", default=["configs/qwen_ep8.json"],
                        help="Path to the DeepSpeed config-pth (e.g. configs/qwen_ep4.json). With several, "
                             "the model is loaded once with the first and the others only switch the "
                             "AutoEP comm settings; each gets its own LOGS/<config name>/profile")
    parser.add_argument("--model", type=str, default="/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B",
                        help="Model directory; must be a qwen3_5_moe checkpoint")
    parser.add_argument("--seq-len", type=int, default=4096, help="Tokens per rank")
    # deepspeed launcher may inject --local_rank; ignore unknown args
    args, _ = parser.parse_known_args()
    return args

def run_ds_flops_profile(engine, batch):
    """DeepSpeed Flops profiler"""
    prof = start_flops_profile(engine)
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    end_flops_profile(prof, outdir=logs)

def run_ds_comm_profile(engine, batch):
    """DeepSpeed comms logger: per-op count, msg size, latency, algbw/busbw."""
    # TODO: hangs at the end for some reason.
    if hasattr(dist.comms_logger, "reset_data"):
        dist.comms_logger.reset_data()  # drop warmup/init comms
    outputs = engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    dist.log_summary(show_straggler=True)   # collective; call on ALL ranks

def main():
    args = parse_args()
    model_id, seq_len = args.model, args.seq_len

    model = load_model(model_id)
    engine = init_deepspeed(model, args.config_pth[0])
    batch = get_batch(model_id, engine.device, seq_len)

    with torch.inference_mode():
        for config_pth in args.config_pth:
            outdir = logs
            if len(args.config_pth) > 1:
                switch_comm(engine, config_pth)
                outdir = logs / Path(config_pth).stem
                (outdir / "profile").mkdir(parents=True, exist_ok=True)
                print0(f"=== {config_pth}")
            # compile + warmup
            # print0("first")
            # record_latency(engine, batch)
            # engine.compile()
            record_latency(engine, batch)
            record_latency(engine, batch)
            record_latency(engine, batch)

            # run_ds_flops_profile(engine, batch)

            # engine(**batch, use_cache=False)
            # record_expert_routing(engine, batch)
            # record_latency(engine, batch)
            # run_ds_comm_profile(engine, batch)
            run_pytorch_profile(engine, batch, outdir)
            # run_ds_flops_profile(engine, batch)

    # bookeeping
    report_peak_mem(engine.device)
    shutdown_comm()

if __name__ == "__main__":
    main()