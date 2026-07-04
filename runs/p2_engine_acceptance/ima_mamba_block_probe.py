#!/usr/bin/env python3
"""Decisive IMA root-cause probe: does raising mamba_block_size above the prompt
length (so the GDN state stays in ONE mamba block) remove the illegal-memory-access?

If the same episode-0 prompt (1041 computed tokens) crashes with mamba_block_size=1024
(2 mamba blocks) but SUCCEEDS with mamba_block_size=4096 (1 mamba block), the IMA is
the mamba-block-boundary crossing during a FLARE canvas denoise read.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))

from parity_audit_flare_engine import VllmFlareEngineAdapter, TurnContext  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_flare_northstar_matched import load_chat_template, render_matched_prompt  # noqa: E402
from eval_flare_multiturn_percall_waves import build_episodes  # noqa: E402
from eval_toolcall_jsonl import tool_schema_by_name  # noqa: E402

MODEL = str(ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16")
INPUT = ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl"
CHAT = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
MAMBA_BLOCK = int(os.environ.get("PROBE_MAMBA_BLOCK", "4096"))
DECODE = os.environ.get("PROBE_DECODE", "canvas")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
chat = load_chat_template(CHAT)


class _Args:
    input_jsonl = INPUT
    episode_limit = 1
    min_turns = 3
    max_turns = 6


eps = build_episodes(_Args())
ep = eps[0]
prompt = render_matched_prompt(tok, [dict(m) for m in ep["prompt_messages"]], ep["tools"], chat)
pids = tok([prompt], return_tensors="pt", add_special_tokens=False).input_ids
plen = int(pids.shape[1])
print(f"PROBE mamba_block={MAMBA_BLOCK} decode={DECODE} prompt_len={plen}", flush=True)

stop = set()
if tok.eos_token_id is not None:
    stop.add(int(tok.eos_token_id))
for t in ("<|im_end|>", "</tool_call>"):
    stop.update(int(x) for x in tok(t, add_special_tokens=False).input_ids)

adapter = VllmFlareEngineAdapter(
    Path("/home/mark/shared/vllm_p2_pr42406"),
    model_path=MODEL,
    canvas_length=32,
    max_denoising_steps=8,
    decode_mode=DECODE,
    gpu_memory_utilization=0.82,
    max_model_len=4096,
    seed=20260701,
    engine_kwargs={"mamba_block_size": MAMBA_BLOCK},
)
ctx = TurnContext(
    model=None,
    tokenizer=tok,
    prompt_input_ids=pids,
    block_size=32,
    max_new_tokens=32,
    mask_id=248077,
    stop_token_ids=stop,
    top_p=0.95,
    temperature=0.0,
    schemas=tool_schema_by_name(ep["tools"]),
    grammar_topk=256,
)
t0 = time.time()
try:
    res = adapter.run_turn(ctx)
    gen = res.output_ids[plen:]
    out = {
        "status": "OK_NO_CRASH",
        "mamba_block": MAMBA_BLOCK,
        "decode": DECODE,
        "prompt_len": plen,
        "generated_count": len(gen),
        "generated_ids_head": [int(x) for x in gen[:16]],
        "wall_s": time.time() - t0,
    }
except Exception as exc:  # noqa: BLE001
    out = {
        "status": "CRASH",
        "mamba_block": MAMBA_BLOCK,
        "decode": DECODE,
        "prompt_len": plen,
        "error": repr(exc)[:300],
        "wall_s": time.time() - t0,
    }
print("PROBE_RESULT " + json.dumps(out), flush=True)
Path(os.environ["PROBE_OUT"]).write_text(json.dumps(out, indent=2) + "\n")
