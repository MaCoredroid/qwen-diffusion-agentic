#!/usr/bin/env python
"""SECTION Y trainer — full-trajectory AR-self-distillation @12288 with a two-stream
denoise-preservation co-step (Y.1 / Y.2).

DISTILL TERM (the load-bearing lever, X.2 finisher impl.#2): the SAME proven single-stream
QLoRA-4bit + chunked-CE causal trainer as swe_sft_arm1_qlora_train.py, run at block 12288 on
the windowed pool whose LABELS are the same-weights AR-greedy targets (y_ar_distill_data.py).
Because the objective is CE(clean-stream logits, AR-greedy labels), the distill term needs NO
trainer-math change — the label set IS the self-distillation. UNIFORM weight 1.0 (the labels
already encode value-always / reasoning-sampled coverage; no per-class multiplier, X.1 lesson).

DENOISE-PRESERVATION CO-STEP (Y.5 risk-1 guard: "co-train the two-stream L_diff on a <=4096
slice every step; never a pure single-stream run"): every step, additionally run the model's
OWN two-stream FLARE forward (modeling.py `_flare_two_stream_training_forward`, the exact
`noisy_to_noisy_mask` L_diff geometry W-2 draft-verify rides on) on a <=4096 slice of the same
window, with GROUND-TRUTH keeper labels. This reuses ALL proven two-stream code; Y adds only
the alternation + the eager/SDPA attention toggle (the two-stream custom bool masks are
IGNORED by the single-stream SDPA-causal patch, so eager attention MUST be restored around the
denoise forward). Uniform mask regime 0.3-0.8 (conversion default); NO argument/read-window
upweight env (kept uniform per Y). Toggle with --denoise-slice 0 to disable (core-only).

Segmented bit-faithful resume (fixed cosine horizon + seeded data schedule + full RNG restore)
so the runner can release the GPU between 100-step segments for the KL probe + KILL-T1 canary.
"""
import argparse
import functools
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
REPO = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(REPO / "scripts"))
from swe_sft_arm1_qlora_train import (  # reuse the PROVEN helpers verbatim
    install_sdpa_attention, load_dataset, build_index_schedule,
    save_checkpoint, latest_checkpoint, prune_checkpoints,
)

# FLARE two-stream env the denoise co-step sets (conversion default regime; uniform).
DENOISE_ENV = {
    "FASTDLLM_FLARE_TWO_STREAM": "1",
    "FLARE_TWO_STREAM": "1",
    "FASTDLLM_FLARE_GDN_ROUTE": "route_i",
    "FASTDLLM_FLARE_MASK_RATE_MIN": "0.3",
    "FASTDLLM_FLARE_MASK_RATE_MAX": "0.8",
    "FASTDLLM_BATCH_FLARE_NOISY_GDN": "1",
    "FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN": "1",
    "FASTDLLM_GDN_KERNEL": "fla",
}


def denoise_slice_from_row(ids, labels, slice_len, bd_size):
    """<=slice_len contiguous slice ending at the last covered label position (the near-cap
    region), length trimmed to a multiple of bd_size; denoise labels = ground-truth token at
    the covered positions (predict-the-masked-real-token = the preserved denoiser)."""
    covered = [p for p, l in enumerate(labels) if l != -100]
    if not covered:
        return None
    pmax = covered[-1]
    end = pmax + 1
    start = max(0, end - slice_len)
    length = end - start
    length -= length % bd_size          # exact multiple of bd_size (two-stream requirement)
    if length < bd_size:
        return None
    start = end - length
    sl_ids = ids[start:end]
    sl_labels = [-100] * length
    any_lab = False
    for j in range(length):
        p = start + j
        if labels[p] != -100:
            sl_labels[j] = ids[p]       # ground-truth token (denoise target)
            any_lab = True
    if not any_lab:
        return None
    return sl_ids, sl_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--stop-at-step", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=8)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets",
                    default="q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_up_proj,down_proj")
    ap.add_argument("--seed", type=int, default=71201)
    ap.add_argument("--logits-chunk", type=int, default=2048)
    ap.add_argument("--denoise-slice", type=int, default=4096, help="<=4096 two-stream denoise-preservation slice; 0 disables")
    ap.add_argument("--denoise-weight", type=float, default=1.0)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--metrics", default="")
    args = ap.parse_args()

    stop_at = args.stop_at_step or args.horizon
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    rows = load_dataset(args.dataset)
    row_lengths = [len(r[0]) for r in rows]
    n_rows = len(rows)
    schedule = build_index_schedule(n_rows, args.horizon, args.seed)
    import hashlib
    schedule_sha = hashlib.sha256(",".join(map(str, schedule)).encode()).hexdigest()[:16]
    bd_size = 32
    print(f"[y-train] rows={n_rows} max_len={max(row_lengths)} horizon={args.horizon} stop_at={stop_at} "
          f"block={args.block_size} denoise_slice={args.denoise_slice} sched_sha={schedule_sha}", flush=True)

    # ---- model: 4-bit NF4 QLoRA, SDPA-causal (single-stream distill) ----
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                             llm_int8_skip_modules=["lm_head"])
    print("[y-train] loading base (4bit-nf4) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 device_map={"": 0})
    model.config.use_cache = False
    attn_cls = install_sdpa_attention(model)
    sdpa_fwd = attn_cls.forward                  # the installed SDPA-causal forward
    eager_fwd = attn_cls._orig_eager_forward     # the original eager forward (two-stream custom masks)
    print("[y-train] SDPA causal installed; eager forward captured for the two-stream denoise toggle", flush=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=args.lora_targets.split(","), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()   # embed output requires grad -> reentrant denoise checkpoint builds its graph
    model.print_trainable_parameters()
    base = model.get_base_model()
    inner = base.model
    lm_head = base.lm_head
    inner.gradient_checkpointing = True
    # the two-stream denoise path calls inner._gradient_checkpointing_func (the single-stream
    # SFT path never does); ensure it exists so the checkpointed two-stream branch is valid.
    if not hasattr(inner, "_gradient_checkpointing_func"):
        inner._gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    model.train()
    device = next(model.parameters()).device

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    warmup_steps = round(args.warmup_ratio * args.horizon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.horizon)

    # ---- resume (bit-faithful) ----
    global_step = 0
    resume_dir = latest_checkpoint(args.output_dir) if args.resume == "auto" else (
        None if args.resume in ("none", "") else Path(args.resume))
    if resume_dir is not None and Path(resume_dir).exists():
        rd = Path(resume_dir)
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        adapter_sd = load_file(str(rd / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, adapter_sd)
        assert len(adapter_sd) > 0, "resume: adapter empty"
        optimizer.load_state_dict(torch.load(rd / "optimizer.pt", map_location=device, weights_only=False))
        scheduler.load_state_dict(torch.load(rd / "scheduler.pt", map_location="cpu", weights_only=False))
        rng = torch.load(rd / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        st = json.loads((rd / "trainer_state.json").read_text())
        global_step = int(st["global_step"])
        assert st["data_schedule_sha"] == schedule_sha, f"resume schedule mismatch {st['data_schedule_sha']} != {schedule_sha}"
        print(f"[y-train] RESUME from {rd} at step {global_step}", flush=True)

    metrics_fh = open(args.metrics, "a") if args.metrics else None
    stop_flag = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.__setitem__("stop", True))

    def ce_chunk(h_c, t_c):
        return F.cross_entropy(lm_head(h_c).float(), t_c, ignore_index=-100, reduction="sum")

    def distill_loss(input_ids, labels):
        hs = inner(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        pred_h = hs[:-1]; pred_t = labels[0][1:]
        n_valid = (pred_t != -100).sum()
        total = hs.new_zeros((), dtype=torch.float32)
        C = args.logits_chunk
        for s in range(0, pred_h.size(0), C):
            total = total + torch.utils.checkpoint.checkpoint(ce_chunk, pred_h[s:s + C], pred_t[s:s + C], use_reentrant=False)
        return total / n_valid.clamp(min=1).to(total.dtype), int(n_valid)

    def denoise_loss(sl_ids, sl_labels):
        """model's OWN two-stream FLARE forward (L_diff + clean L_AR) on a <=4096 slice; eager
        attention restored (two-stream bool masks are ignored by the SDPA-causal patch)."""
        torch.cuda.empty_cache()  # release the 12288 distill step's reserved fragments first
        ids = torch.tensor([sl_ids], dtype=torch.long, device=device)
        lab = torch.tensor([sl_labels], dtype=torch.long, device=device)
        am = torch.ones_like(ids)
        for k, v in DENOISE_ENV.items():
            os.environ[k] = v
        attn_cls.forward = eager_fwd            # two-stream custom bool masks (SDPA patch ignores them)
        # checkpoint the two-stream LAYERS (low retained memory) but with use_reentrant=True so
        # the FLARE GDN recompute is NOT metadata-checked (use_reentrant=False trips on it); embed
        # outputs require grad (prepare_model_for_kbit_training -> enable_input_require_grads), so
        # reentrant checkpointing is valid. Restore the distill func (reentrant=False) after.
        prev_ckpt_func = getattr(inner, "_gradient_checkpointing_func", None)
        inner._gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=True)
        inner.gradient_checkpointing = True
        try:
            out = model(input_ids=ids, labels=lab, attention_mask=am)
            loss = out.loss
        finally:
            inner._gradient_checkpointing_func = prev_ckpt_func
            inner.gradient_checkpointing = True  # kept on for the 12288 single-stream distill forward
            attn_cls.forward = sdpa_fwd
            for k in DENOISE_ENV:
                os.environ.pop(k, None)
        parts = getattr(base, "_last_flare_loss_parts", {}) or {}
        return loss, parts

    extra = {"n_rows": n_rows, "schedule_sha": schedule_sha}
    post_load_gib = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"[y-train] post-load resident={post_load_gib:.2f} GiB", flush=True)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    print(f"[y-train] TRAIN start step={global_step} -> stop_at={stop_at} warmup={warmup_steps}", flush=True)

    while global_step < stop_at:
        if args.denoise_slice > 0:
            torch.cuda.empty_cache()  # start the 12288 distill clean (denoise co-step leaves fragments)
        idx = schedule[global_step]
        ids, labels = rows[idx]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        label_ids = torch.tensor([labels], dtype=torch.long, device=device)
        d_loss, n_valid = distill_loss(input_ids, label_ids)
        (d_loss / args.grad_accum).backward()

        den_val = float("nan"); den_diff = float("nan")
        if args.denoise_slice > 0:
            sl = denoise_slice_from_row(ids, labels, args.denoise_slice, bd_size)
            if sl is not None:
                dn_loss, parts = denoise_loss(sl[0], sl[1])
                ((dn_loss * args.denoise_weight) / args.grad_accum).backward()
                den_val = float(dn_loss.detach().float().cpu())
                if parts.get("diff") is not None:
                    den_diff = float(parts["diff"].float().cpu())

        do_step = ((global_step + 1) % args.grad_accum == 0)
        gnorm = None
        if do_step:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.logging_steps == 0 or global_step == stop_at:
            lr_now = scheduler.get_last_lr()[0]
            mem = torch.cuda.max_memory_allocated() / 1e9
            gn = float(gnorm) if gnorm is not None else float("nan")
            print(f"{{'distill_loss': {float(d_loss.detach().float().cpu()):.6f}, 'denoise_loss': {den_val:.6f}, "
                  f"'denoise_diff': {den_diff:.6f}, 'grad_norm': {gn:.4f}, 'learning_rate': {lr_now:.3e}, "
                  f"'step': {global_step}, 'seq_len': {len(ids)}, 'n_valid': {n_valid}, 'peak_gib': {mem:.2f}, "
                  f"'epoch': {global_step / n_rows:.4f}}}", flush=True)
            if metrics_fh:
                metrics_fh.write(json.dumps({"t": time.time(), "step": global_step,
                                             "distill_loss": float(d_loss.detach().float().cpu()),
                                             "denoise_loss": den_val, "denoise_diff": den_diff,
                                             "lr": lr_now, "grad_norm": gn, "seq_len": len(ids),
                                             "n_valid": n_valid, "peak_gib": round(mem, 2),
                                             "elapsed_s": round(time.time() - t0, 1)}) + "\n")
                metrics_fh.flush()

        if global_step % args.save_steps == 0 and global_step < stop_at:
            save_checkpoint(Path(args.output_dir) / f"checkpoint-{global_step}", model, optimizer, scheduler, global_step, args, extra)
            prune_checkpoints(args.output_dir, args.save_total_limit)
            print(f"[y-train] checkpoint saved: checkpoint-{global_step}", flush=True)

        if stop_flag["stop"]:
            save_checkpoint(Path(args.output_dir) / f"checkpoint-{global_step}", model, optimizer, scheduler, global_step, args, extra)
            print(f"[y-train] SIGTERM -> checkpoint at step {global_step}; exit for resume", flush=True)
            if metrics_fh: metrics_fh.close()
            return

    save_checkpoint(Path(args.output_dir) / f"checkpoint-{global_step}", model, optimizer, scheduler, global_step, args, extra)
    prune_checkpoints(args.output_dir, args.save_total_limit)
    if global_step >= args.horizon:
        model.save_pretrained(str(Path(args.output_dir) / "adapter_final"))
    dt = time.time() - t0
    print(f"[y-train] DONE step={global_step} wall={dt:.1f}s ({dt / max(1, global_step):.2f}s/step)", flush=True)
    if metrics_fh: metrics_fh.close()


if __name__ == "__main__":
    main()
