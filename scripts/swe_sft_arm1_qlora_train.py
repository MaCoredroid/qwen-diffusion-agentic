#!/usr/bin/env python
"""SWE-SFT arm-1 (M_swe_S) -- AR-STYLE SINGLE-STREAM CAUSAL QLoRA trainer.

AMENDMENT (monitor GPU->training-handover decision, resolution 2): this REPLACES
the two-stream FLARE path (scripts/swe_sft_arm1_segment.sh -> train_s2_finetune.py
+ FASTDLLM_S2_PRETOK). The two-stream trainer concatenates clean+noisy to length
2L and materialises [2L, vocab=248320] logits (~16 GB) -> measured OOM above
block 8192 (runs/swe_sft_arm1/ARM1_LAUNCH_STATUS.md). This trainer trains a plain
autoregressive next-token objective on the SERVE-EXACT keeper trajectories --
exactly the "AR-side SFT" the design's arm S calls for -- and is memory-lean
enough to reach the design's block_size.

EVIDENCE for the amendment (why plain-AR-SFT-then-reconvert is sound): the
convert-after-RL preservation audit (#29, convert_after_rl_result.md, commit
b019b86) certified that plain training + fresh re-conversion PRESERVES fresh gains
(McNemar zero net-loss, two seeds). train==serve parity is enforced at the
CONVERSION stage (k_raise_campaign_design.md), NOT required bit-for-bit during the
SFT stage. So an AR single-stream SFT is a legitimate arm-S base producer.

WHY THIS FITS LONG SEQUENCES (three levers, all measured by the probe ladder):
  1. SINGLE stream (not 2L)             -> halves the sequence the transformer sees.
  2. SDPA causal attention (monkeypatch, THIS PROCESS ONLY; served modeling.py is
     untouched) for the 8 full_attention layers. The shipped forward is EAGER and
     materialises attn_weights [1,16,L,L] (~34 GB/layer at 32k) -> the true wall.
     SDPA(is_causal=True) is the flash/mem-efficient O(L) causal kernel and is
     mathematically the same causal softmax attention. The 24 GDN linear_attention
     layers are already O(L). (This numerical-not-bitwise attention swap is fine:
     serve-parity is a conversion-stage gate, #29, not an SFT-stage requirement.)
  3. CHUNKED cross-entropy (Liger-style, hand-rolled; liger_kernel absent in venv):
     lm_head+CE is gradient-checkpointed over sequence chunks so the full
     [L, vocab] logits (16 GB bf16 at 32k) is never materialised -- peak is one
     chunk [chunk, vocab].
  + 4-bit NF4 QLoRA base (weights ~5.5 GB) + per-layer gradient checkpointing.

OBJECTIVE (matches the FLARE clean-stream AR loss, modeling.py:2149-2164):
  shift by one -- hidden[i] predicts token[i+1]; loss over assistant-label
  positions only (labels[i+1] != -100); mean over valid target tokens.

FAITHFUL CHUNKED-RESUME (bit-faithful manifests): fixed cosine HORIZON so the LR
schedule is identical to one continuous run; a deterministic seeded data-index
schedule so step k always consumes the same episode; checkpoint saves adapter +
optimizer + scheduler + RNG (torch/cuda/python/numpy) + step; resume restores all
and continues. STOP_AT_STEP stops a segment early (chunked ~100-step segments).
"""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

REPO = Path("/home/mark/qwen_diffusion")


# --------------------------------------------------------------------------- #
# SDPA causal attention monkeypatch (THIS PROCESS ONLY; served file untouched) #
# --------------------------------------------------------------------------- #
def install_sdpa_attention(model):
    """Replace the eager O(L^2) full_attention forward with an O(L) SDPA causal
    forward. Only touches Fast_dLLM_Qwen3_5Attention instances of THIS in-memory
    model class object (loaded via trust_remote_code); the on-disk modeling.py the
    server loads is never modified. Valid only for the plain causal (mdm split_size
    is None) forward this trainer uses; delegates to the original otherwise."""
    attn_cls = None
    for m in model.modules():
        if m.__class__.__name__ == "Fast_dLLM_Qwen3_5Attention":
            attn_cls = m.__class__
            break
    if attn_cls is None:
        raise RuntimeError("no Fast_dLLM_Qwen3_5Attention module found to patch")
    if getattr(attn_cls, "_sdpa_patched", False):
        return attn_cls
    orig_forward = attn_cls.forward
    import importlib
    _repeat_kv = importlib.import_module(attn_cls.__module__).repeat_kv

    def sdpa_forward(self, hidden_states, position_embeddings, attention_mask=None, split_size=None):
        # mdm / block-diffusion path is never used by this AR trainer -> delegate.
        if split_size is not None:
            return orig_forward(self, hidden_states, position_embeddings, attention_mask, split_size)
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # additive (non-causal) mask -> not our case; be safe, delegate.
            return orig_forward(self, hidden_states, position_embeddings, attention_mask, split_size)
        input_shape = hidden_states.shape[:-1]
        q, k, v, gate = self._project(hidden_states, position_embeddings, split_size)

        # expand kv heads exactly as the eager path did (repeat_kv from the
        # model's own dynamic module, captured at patch-install time).
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

    attn_cls.forward = sdpa_forward
    attn_cls._sdpa_patched = True
    attn_cls._orig_eager_forward = orig_forward
    return attn_cls


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #
def load_dataset(path):
    d = json.loads(Path(path).read_text())
    assert d.get("type") == "text_only", f"expected LMFlow text_only, got {d.get('type')}"
    rows = []
    for inst in d["instances"]:
        ids = inst["input_ids"]
        labels = inst["labels"]
        assert len(ids) == len(labels)
        rows.append((ids, labels))
    return rows


def build_index_schedule(n_rows, horizon, seed, longest_first=False, row_lengths=None):
    """Deterministic per-step sample index. longest_first (probe) => worst-case
    VRAM first. Otherwise seeded epoch permutations so resume is data-faithful."""
    if longest_first:
        order = sorted(range(n_rows), key=lambda i: row_lengths[i], reverse=True)
        sched = [order[s % n_rows] for s in range(horizon)]
        return sched
    g = torch.Generator()
    g.manual_seed(seed)
    sched = []
    while len(sched) < horizon:
        perm = torch.randperm(n_rows, generator=g).tolist()
        sched.extend(perm)
    return sched[:horizon]


# --------------------------------------------------------------------------- #
# checkpoint io                                                               #
# --------------------------------------------------------------------------- #
def save_checkpoint(ckpt_dir, model, optimizer, scheduler, global_step, args, extra):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))  # adapter_model.safetensors + adapter_config.json
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch.save(rng, ckpt_dir / "rng.pt")
    state = {
        "global_step": global_step,
        "horizon": args.horizon,
        "seed": args.seed,
        "block_size": args.block_size,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "targets": args.lora_targets},
        "grad_accum": args.grad_accum,
        "logits_chunk": args.logits_chunk,
        "base_model": args.model,
        "dataset": args.dataset,
        "n_rows": extra["n_rows"],
        "data_schedule_sha": extra["schedule_sha"],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ckpt_dir / "trainer_state.json").write_text(json.dumps(state, indent=2))
    return ckpt_dir


def latest_checkpoint(output_dir):
    d = Path(output_dir)
    if not d.exists():
        return None
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return cks[-1] if cks else None


def prune_checkpoints(output_dir, keep):
    cks = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for p in cks[:-keep] if keep > 0 else []:
        import shutil
        shutil.rmtree(p, ignore_errors=True)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "models/qwen3.5-9b-fastdllm-mtplus1-merged"))
    ap.add_argument("--dataset", required=True, help="LMFlow text_only json (input_ids+labels)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--stop-at-step", type=int, default=0, help="0 => run to horizon")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=6)
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
    ap.add_argument("--seed", type=int, default=71101)
    ap.add_argument("--logits-chunk", type=int, default=2048)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--longest-first", action="store_true", help="probe: worst-case-VRAM order")
    ap.add_argument("--resume", default="auto", help="auto|none|<checkpoint dir>")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--bf16-base", action="store_true", help="load base in bf16 (no 4-bit); default is 4-bit QLoRA")
    args = ap.parse_args()

    stop_at = args.stop_at_step or args.horizon
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- data ----
    rows = load_dataset(args.dataset)
    row_lengths = [len(r[0]) for r in rows]
    if args.max_train_samples and args.max_train_samples < len(rows):
        if args.longest_first:
            keep = sorted(range(len(rows)), key=lambda i: row_lengths[i], reverse=True)[:args.max_train_samples]
        else:
            keep = list(range(args.max_train_samples))
        rows = [rows[i] for i in keep]
        row_lengths = [len(r[0]) for r in rows]
    n_rows = len(rows)
    schedule = build_index_schedule(n_rows, args.horizon, args.seed,
                                    longest_first=args.longest_first, row_lengths=row_lengths)
    import hashlib
    schedule_sha = hashlib.sha256(",".join(map(str, schedule)).encode()).hexdigest()[:16]
    print(f"[qlora] rows={n_rows} max_len={max(row_lengths)} horizon={args.horizon} "
          f"stop_at={stop_at} block={args.block_size} sched_sha={schedule_sha} longest_first={args.longest_first}",
          flush=True)

    # ---- model (4-bit QLoRA by default) ----
    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    if not args.bf16_base:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        )
    print(f"[qlora] loading base ({'bf16' if args.bf16_base else '4bit-nf4'}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False

    install_sdpa_attention(model)
    print("[qlora] SDPA causal attention installed on full_attention layers (this process only)", flush=True)

    if not args.bf16_base:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    base = model.get_base_model()
    inner = base.model
    lm_head = base.lm_head
    # belt-and-suspenders: ensure per-layer gradient checkpointing is live on inner
    inner.gradient_checkpointing = True
    print(f"[qlora] inner.gradient_checkpointing={inner.gradient_checkpointing} "
          f"has_ckpt_func={hasattr(inner, '_gradient_checkpointing_func')}", flush=True)

    model.train()
    device = next(model.parameters()).device

    # ---- optimizer + fixed-horizon cosine schedule ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    warmup_steps = round(args.warmup_ratio * args.horizon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.horizon)

    # ---- resume ----
    global_step = 0
    resume_dir = None
    if args.resume == "auto":
        resume_dir = latest_checkpoint(args.output_dir)
    elif args.resume not in ("none", ""):
        resume_dir = Path(args.resume)
    if resume_dir is not None and Path(resume_dir).exists():
        rd = Path(resume_dir)
        print(f"[qlora] RESUME from {rd}", flush=True)
        # load adapter weights into the peft model via the canonical peft loader
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        adapter_sd = load_file(str(rd / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, adapter_sd)
        n_unexp = len(getattr(res, "unexpected_keys", []) or [])
        print(f"[qlora] resume adapter: {len(adapter_sd)} tensors loaded (unexpected={n_unexp})", flush=True)
        assert len(adapter_sd) > 0, "resume: adapter_model.safetensors empty"
        optimizer.load_state_dict(torch.load(rd / "optimizer.pt", map_location=device, weights_only=False))
        scheduler.load_state_dict(torch.load(rd / "scheduler.pt", map_location="cpu", weights_only=False))
        rng = torch.load(rd / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        st = json.loads((rd / "trainer_state.json").read_text())
        global_step = int(st["global_step"])
        assert st["data_schedule_sha"] == schedule_sha, \
            f"resume schedule mismatch {st['data_schedule_sha']} != {schedule_sha}"
        # resume manifest (bit-faithful continuity record)
        man = {
            "resumed_from": str(rd), "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "global_step": global_step, "horizon": args.horizon, "stop_at_step": stop_at,
            "seed": args.seed, "data_schedule_sha": schedule_sha,
            "schedule_continues_same_order": True, "cosine_horizon_fixed": True,
            "restored": ["adapter", "optimizer", "scheduler", "rng(torch/cuda/py/np)", "global_step"],
        }
        (Path(args.output_dir) / f"resume_manifest_step{global_step}.json").write_text(json.dumps(man, indent=2))
        print(f"[qlora] resumed at global_step={global_step}", flush=True)

    metrics_fh = open(args.metrics, "a") if args.metrics else None

    # ---- SIGTERM -> checkpoint-then-exit at next step boundary ----
    stop_flag = {"stop": False}

    def _sigterm(_signo, _frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGTERM, _sigterm)

    def ce_chunk(h_c, t_c):
        logits = lm_head(h_c)  # [c, vocab] bf16
        return F.cross_entropy(logits.float(), t_c, ignore_index=-100, reduction="sum")

    def compute_loss(input_ids, labels):
        out = inner(input_ids=input_ids, use_cache=False)
        hs = out.last_hidden_state[0]          # [L, H]
        tgt = labels[0]                        # [L]
        pred_h = hs[:-1]                       # predicts positions 1..L-1
        pred_t = tgt[1:]
        n_valid = (pred_t != -100).sum()
        total = hs.new_zeros((), dtype=torch.float32)
        C = args.logits_chunk
        for stt in range(0, pred_h.size(0), C):
            h_c = pred_h[stt:stt + C]
            t_c = pred_t[stt:stt + C]
            total = total + torch.utils.checkpoint.checkpoint(ce_chunk, h_c, t_c, use_reentrant=False)
        loss = total / n_valid.clamp(min=1).to(total.dtype)
        return loss, int(n_valid)

    extra = {"n_rows": n_rows, "schedule_sha": schedule_sha}
    # reset peak-memory stats AFTER load so metrics reflect the true training peak
    # (not the transient 4-bit quantization/load spike) -- this is the probe reading.
    post_load_gib = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"[qlora] post-load resident={post_load_gib:.2f} GiB (peak stats reset for training)", flush=True)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    print(f"[qlora] TRAIN start step={global_step} -> stop_at={stop_at} warmup={warmup_steps}", flush=True)

    while global_step < stop_at:
        idx = schedule[global_step]
        ids, labels = rows[idx]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        label_ids = torch.tensor([labels], dtype=torch.long, device=device)
        loss, n_valid = compute_loss(input_ids, label_ids)
        (loss / args.grad_accum).backward()

        do_step = ((global_step + 1) % args.grad_accum == 0)
        gnorm = None
        if do_step:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if global_step % args.logging_steps == 0 or global_step == stop_at:
            lr_now = scheduler.get_last_lr()[0]
            mem = torch.cuda.max_memory_allocated() / 1e9
            lossf = float(loss.detach().float().cpu())
            gn = float(gnorm) if gnorm is not None else float("nan")
            print(f"{{'loss': {lossf:.6f}, 'grad_norm': {gn:.4f}, 'learning_rate': {lr_now:.3e}, "
                  f"'step': {global_step}, 'seq_len': {len(ids)}, 'n_valid': {n_valid}, "
                  f"'peak_gib': {mem:.2f}, 'epoch': {global_step / n_rows:.4f}}}", flush=True)
            if metrics_fh:
                metrics_fh.write(json.dumps({
                    "t": time.time(), "step": global_step, "loss": lossf, "lr": lr_now,
                    "grad_norm": gn, "seq_len": len(ids), "n_valid": n_valid,
                    "peak_gib": round(mem, 2), "elapsed_s": round(time.time() - t0, 1)}) + "\n")
                metrics_fh.flush()

        if global_step % args.save_steps == 0 and global_step < stop_at:
            ck = save_checkpoint(Path(args.output_dir) / f"checkpoint-{global_step}",
                                 model, optimizer, scheduler, global_step, args, extra)
            prune_checkpoints(args.output_dir, args.save_total_limit)
            print(f"[qlora] checkpoint saved: {ck}", flush=True)

        if stop_flag["stop"]:
            ck = save_checkpoint(Path(args.output_dir) / f"checkpoint-{global_step}",
                                 model, optimizer, scheduler, global_step, args, extra)
            print(f"[qlora] SIGTERM -> checkpoint {ck} at step {global_step}; exiting for resume", flush=True)
            if metrics_fh:
                metrics_fh.close()
            return

    # final checkpoint at the horizon/stop
    ck = save_checkpoint(Path(args.output_dir) / f"checkpoint-{global_step}",
                         model, optimizer, scheduler, global_step, args, extra)
    prune_checkpoints(args.output_dir, args.save_total_limit)
    if global_step >= args.horizon:
        model.save_pretrained(str(Path(args.output_dir) / "adapter_final"))
    dt = time.time() - t0
    print(f"[qlora] DONE step={global_step} wall={dt:.1f}s ({dt / max(1, global_step):.2f}s/step) ckpt={ck}", flush=True)
    if metrics_fh:
        metrics_fh.close()


if __name__ == "__main__":
    main()
