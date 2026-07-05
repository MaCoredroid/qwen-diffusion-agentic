#!/usr/bin/env python3
"""S2 pilot DATA step -- build the GSM8K TRAIN-split self-generation prompt pool.

Reproduces, TOKEN-EXACT, the anchor free-text prompt format behind the 26/30 K-gate
(runs/l1_census/gsm8k_prompts_clean.json = fixed 5-shot(train[0:5]) + test[i]).
Here we keep the SAME fixed 5-shot prefix (GSM8K train[0:5]) and swap the target
question to TRAIN examples with index >= 5 (so no target is one of the 5 fewshot
exemplars). Verified byte-identical to clean[0..4] before writing (see --verify).

Output pool records mirror gsm8k_prompts_clean.json schema so the census/hardened
engine runner consumes them unchanged, plus a `train_idx` and normalized-question
hash for the leakage dedupe in build_s2_traj_corpus.py.

Prompt format per shot:
  <|im_start|>user\nQuestion: {q}\nAnswer:<|im_end|>\n<|im_start|>assistant\n{gold}<|im_end|>\n
Target (generation prompt) appends:
  <|im_start|>user\nQuestion: {q}\nAnswer:<|im_end|>\n<|im_start|>assistant\n

Pins: block_size=32, mask_id=248077, grammar_topk=256,
      stop_token_ids=[248044,248045,248046,248059] (native <|endoftext|>,<|im_start|>,<|im_end|>,</tool_call>).
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
CLEAN = ROOT / "runs/l1_census/gsm8k_prompts_clean.json"
OUT = Path(os.environ.get("S2_POOL_OUT", str(ROOT / "runs/s2_pilot/gsm8k_train_prompts.json")))

BLOCK_SIZE = 32
MASK_ID = 248077
GRAMMAR_TOPK = 256
STOP_TOKEN_IDS = [248044, 248045, 248046, 248059]
N_FEWSHOT = 5
POOL_SIZE = int(os.environ.get("S2_POOL_SIZE", "3500"))  # target-generation stops early


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def qhash(q: str) -> str:
    return hashlib.sha256(norm_q(q).encode("utf-8")).hexdigest()


def shot(q: str, a: str) -> str:
    return (f"<|im_start|>user\nQuestion: {q}\nAnswer:<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>\n")


def target(q: str) -> str:
    return f"<|im_start|>user\nQuestion: {q}\nAnswer:<|im_end|>\n<|im_start|>assistant\n"


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    prefix = "".join(shot(train[i]["question"], train[i]["answer"]) for i in range(N_FEWSHOT))

    def build(q: str):
        return tok(prefix + target(q), add_special_tokens=False)["input_ids"]

    # --- correctness gate: byte-identical reconstruction of the clean gate set ---
    clean = json.loads(CLEAN.read_text())
    for k in range(5):
        rebuilt = build(test[k]["question"])
        if rebuilt != clean[k]["prompt_ids"]:
            print(f"[FATAL] prompt reconstruction MISMATCH at clean[{k}]", flush=True)
            return 2
    print("[ok] byte-exact reconstruction of clean[0..4] (anchor prompt format verified)", flush=True)

    recs = []
    for train_idx in range(N_FEWSHOT, N_FEWSHOT + POOL_SIZE):
        q = train[train_idx]["question"]
        a = train[train_idx]["answer"]
        ids = build(q)
        recs.append({
            "idx": len(recs),               # 0-based pool position (engine runner key)
            "train_idx": train_idx,          # GSM8K train split index
            "question": q,
            "gold_answer": a,
            "q_norm_sha256": qhash(q),
            "prompt_ids": ids,
            "prompt_len": len(ids),
            "block_size": BLOCK_SIZE,
            "mask_id": MASK_ID,
            "grammar_topk": GRAMMAR_TOPK,
            "stop_token_ids": STOP_TOKEN_IDS,
        })
    OUT.write_text(json.dumps(recs))
    plens = [r["prompt_len"] for r in recs]
    print(f"[ok] wrote {len(recs)} train-split prompts -> {OUT}", flush=True)
    print(f"[ok] pool train_idx range [{N_FEWSHOT}, {N_FEWSHOT + POOL_SIZE - 1}], "
          f"prompt_len min/max {min(plens)}/{max(plens)}", flush=True)
    # sanity: fewshot exemplars (train[0:5]) must NOT appear as targets
    fewshot_hashes = {qhash(train[i]["question"]) for i in range(N_FEWSHOT)}
    overlap = sum(1 for r in recs if r["q_norm_sha256"] in fewshot_hashes)
    print(f"[ok] targets overlapping fewshot exemplars: {overlap} (expect 0)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
