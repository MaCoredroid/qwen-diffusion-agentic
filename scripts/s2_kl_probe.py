#!/usr/bin/env python
"""S2 pilot — in-training KL-to-base retention monitor (spec sec.5 / KILL-retention).

Computes mean token-level KL(base || student) over ANSWER positions on a fixed,
leakage-safe held-out GSM8K-train probe (disjoint from training corpus + from the
30-gate / 20-retention eval sets). One 4-bit base is loaded; the LoRA adapter is
toggled ON (student) / OFF (base) via peft disable_adapter() -> no second model.
Clean teacher-forced causal forward (eval mode, two-stream off), fp32 softmax.

KILL-retention: rolling KL-to-base > 0.05 (campaign cap) => halt, do not continue.

Usage:
  s2_kl_probe.py --adapter runs/s2_pilot/Apilot_step400_seed90101/checkpoint-100 \
                 --step 100 --out runs/s2_pilot/Apilot_step400_seed90101/kl_to_base.jsonl
  (omit --adapter for the step-0 base==student sanity, KL must be ~0)
"""
import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/qwen3.5-9b-fastdllm-mtplus1-merged")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--probe", default="runs/s2_pilot/s2_kl_probe.json")
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--out", default="")
    ap.add_argument("--kl-cap", type=float, default=0.05)
    args = ap.parse_args()

    # clean causal forward: make sure no two-stream env leaks in
    for k in ("FASTDLLM_FLARE_TWO_STREAM", "FLARE_TWO_STREAM"):
        os.environ.pop(k, None)

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()

    have_adapter = bool(args.adapter) and os.path.isdir(args.adapter)
    if have_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    probe = json.load(open(args.probe))["probe"]
    dev = next(model.parameters()).device

    @torch.no_grad()
    def logits_for(input_ids, use_adapter):
        ids = torch.tensor([input_ids], dtype=torch.long, device=dev)
        if have_adapter and not use_adapter:
            with model.disable_adapter():
                out = model(input_ids=ids)
        else:
            out = model(input_ids=ids)
        return out.logits[0].float()  # [T, V] fp32

    per_seq = []
    for pr in probe:
        input_ids = list(pr["prompt_ids"]) + list(pr["answer_ids"])
        pl = pr["prompt_len"]
        T = len(input_ids)
        # positions predicting an answer token: logits[i] -> token i+1, for i+1 in [pl, T)
        pos = list(range(pl - 1, T - 1))
        ls = logits_for(input_ids, use_adapter=True)
        lb = logits_for(input_ids, use_adapter=False)
        idx = torch.tensor(pos, device=dev)
        ps = torch.log_softmax(ls[idx], dim=-1)  # student logprobs
        pb = torch.log_softmax(lb[idx], dim=-1)  # base logprobs
        # KL(base || student) = sum_x exp(pb) * (pb - ps)
        kl = (pb.exp() * (pb - ps)).sum(dim=-1)  # [n_answer]
        per_seq.append(float(kl.mean().item()))

    mean_kl = sum(per_seq) / len(per_seq)
    tripped = mean_kl > args.kl_cap
    rec = {
        "step": args.step,
        "adapter": args.adapter or None,
        "mean_kl_to_base": mean_kl,
        "per_seq_kl": per_seq,
        "n_probe": len(per_seq),
        "kl_cap": args.kl_cap,
        "kill_retention_tripped": bool(tripped),
    }
    print(json.dumps(rec))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    # non-zero exit signals a tripped kill to the caller (for halt-on-kill)
    sys.exit(3 if tripped else 0)


if __name__ == "__main__":
    main()
