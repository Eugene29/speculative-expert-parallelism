"""Time DeepSpeed AutoEP inference of the MoE model, and optionally record its routing.

Loads Qwen3.6-35B-A3B on CPU, lets DeepSpeed's AutoEP spread the experts over the
ranks, and runs a prefill forward on --seq-len tokens per rank cut from
data/short_text.txt. For every config it runs three timed forwards, then one more
forward under each profiler named by --profile. With several configs the model is
loaded once, only the AutoEP comm settings switch, and each config writes under
LOGS/<config name>/ instead of LOGS/.

The profilers:
    torch (default): torch.profiler. Leaves profile/trace_rank{r}.json for every rank
        and profile/key_averages.txt, the 30 ops with the most CUDA time on rank 0.
    flops: DeepSpeed's flops profiler. Leaves flops_profile_{r}.txt, with flops,
        MACs, parameters and latency per module.
    comm: DeepSpeed's comms logger. Prints the count, message size, latency and
        bandwidth of every communication op. The config must turn it on with
        "comms_logger": {"enabled": true}.

With --record it first saves the experts every token chose to
LOGS/experts_rank{r}.pt, which the benchmarks replay.

Usage:
    # In a SLURM allocation. run_ds.sh sets up NCCL, LOGS and the python env, then runs
    # srun --ntasks-per-node=4 python -u scripts/ds_moe.py --config-pth configs/qwen_ep4.json
    bash scripts/run_ds.sh

    # Another config, and record the routing.
    CONFIG=configs/qwen_ep8.json RECORD=1 bash scripts/run_ds.sh

    # All three profilers.
    PROFILE="torch flops comm" bash scripts/run_ds.sh
"""
import argparse
import json
import os
import time
from pathlib import Path

import deepspeed
import deepspeed.comm as dist
import torch
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from torch.profiler import record_function
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

from ds_utils import (
    add_expert_statistics_hook,
    count_params,
    end_flops_profile,
    print0,
    rank,
    report_peak_mem,
    save_torch_profile,
    shutdown_comm,
    start_flops_profile,
    torch_profile_ctx,
)

logs = Path(os.environ["LOGS"])


def load_model(model_id: str) -> Qwen3_5MoeForCausalLM:
    """Load the text backbone on CPU and print its parameter counts.

    Returns:
        model: The model, not yet on any GPU.
    """
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
    model.config.use_cache = False

    if loading_info["missing_keys"]:
        raise RuntimeError(
            f"Missing model weights: {loading_info['missing_keys']}"
        )
    return model


def init_deepspeed(
    model: Qwen3_5MoeForCausalLM,
    ds_config_pth: str,
) -> deepspeed.DeepSpeedEngine:
    """Wrap the model in a DeepSpeed engine, which applies AutoEP from the config.

    Returns:
        engine: The engine in eval mode.
    """
    with open(ds_config_pth, 'r') as file:
        ds_config = json.load(file)

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


def get_batch(
    model_id: str,
    dev: torch.device,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Cut this rank's seq_len tokens from data/short_text.txt.

    Rank r gets tokens r * seq_len to (r + 1) * seq_len, so no two ranks share any.

    Returns:
        batch: input_ids and attention_mask, each [1, seq_len].
    """
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


def record_expert_routing(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
) -> None:
    """Run one forward and save the experts every token on this rank chose.

    Writes LOGS/experts_rank{r}.pt: one [num_layers, num_tokens, k] int16 tensor of
    expert ids. The hooks come off afterwards, so later forwards are timed without them.
    """
    sent = {}
    handles = add_expert_statistics_hook(engine.module, sent)
    engine(**batch, use_cache=False)
    for handle in handles:
        handle.remove()
    torch.save(torch.stack(list(sent.values())), logs / f"experts_rank{rank}.pt")


def record_latency(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
) -> None:
    """Time one forward between barriers and print this rank's seconds. Collective."""
    torch.cuda.synchronize(engine.device)
    dist.barrier()
    strt = time.time()
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    dist.barrier()
    dur = time.time() - strt
    print(f"[{rank}] dur: {dur}", flush=True)


def run_pytorch_profile(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
    outdir: Path,
) -> None:
    """Run one forward under torch.profiler and save the trace under outdir/profile."""
    with torch_profile_ctx() as tprof:
        with record_function("forward"):
            engine(**batch, use_cache=False)
        torch.cuda.synchronize(engine.device)
    save_torch_profile(tprof, outdir=outdir, verbose=True)


def run_ds_flops_profile(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
    outdir: Path,
) -> None:
    """Run one forward under DeepSpeed's flops profiler and save the table under outdir."""
    prof = start_flops_profile(engine.module)
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    end_flops_profile(prof, outdir)


def run_ds_comm_profile(
    engine: deepspeed.DeepSpeedEngine,
    batch: dict[str, torch.Tensor],
) -> None:
    """Run one forward and print DeepSpeed's comms logger summary of it. Collective.

    Prints nothing unless the config sets "comms_logger": {"enabled": true}.
    """
    dist.comms_logger.reset_data()
    engine(**batch, use_cache=False)
    torch.cuda.synchronize(engine.device)
    dist.log_summary(show_straggler=True)


def parse_args() -> argparse.Namespace:
    """Parse the command line, ignoring flags the DeepSpeed launcher adds.

    Returns:
        args: config_pth, model, seq_len, record and profile.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config-pth", type=str, nargs="+", default=["configs/qwen_ep8.json"],
                        help="Path to the DeepSpeed config-pth (e.g. configs/qwen_ep4.json). With several, "
                             "the model is loaded once with the first and the others only switch the "
                             "AutoEP comm settings; each gets its own LOGS/<config name>/profile")
    parser.add_argument("--model", type=str, default="/pscratch/sd/e/eku/spec_ep/models/Qwen3.6-35B-A3B",
                        help="Model directory; must be a qwen3_5_moe checkpoint")
    parser.add_argument("--seq-len", type=int, default=4096, help="Tokens per rank")
    parser.add_argument("--record", action="store_true",
                        help="Save the expert routing to LOGS/experts_rank{r}.pt before timing")
    parser.add_argument("--profile", nargs="*", choices=["torch", "flops", "comm"],
                        default=["torch"],
                        help="Profilers to run after the timed forwards, one forward each")
    # deepspeed launcher may inject --local_rank; ignore unknown args
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """Load the model once, then time and profile a forward under every config."""
    args = parse_args()
    model_id, seq_len = args.model, args.seq_len

    model = load_model(model_id)
    engine = init_deepspeed(model, args.config_pth[0])
    batch = get_batch(model_id, engine.device, seq_len)

    with torch.inference_mode():
        if args.record:
            record_expert_routing(engine, batch)

        for config_pth in args.config_pth:
            outdir = logs
            if len(args.config_pth) > 1:
                switch_comm(engine, config_pth)
                outdir = logs / Path(config_pth).stem
                (outdir / "profile").mkdir(parents=True, exist_ok=True)
                print0(f"=== {config_pth}")
            record_latency(engine, batch)
            record_latency(engine, batch)
            record_latency(engine, batch)

            if "torch" in args.profile:
                run_pytorch_profile(engine, batch, outdir)
            if "flops" in args.profile:
                run_ds_flops_profile(engine, batch, outdir)
            if "comm" in args.profile:
                run_ds_comm_profile(engine, batch)

    report_peak_mem(engine.device)
    shutdown_comm()


if __name__ == "__main__":
    main()
