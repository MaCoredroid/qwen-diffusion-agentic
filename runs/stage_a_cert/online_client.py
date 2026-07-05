#!/usr/bin/env python3
"""A6/A7 ONLINE client (Stage-A cert).

Drives the launcher-booted FLARE hybrid_clean server (:9952) via /v1/COMPLETIONS
with RAW prompt token ids (prompt=<prompt_ids>) so the served engine receives a
byte-identical prompt -- eliminating chat-template rendering as a variable. The
tool-call grammar is activated through the completions FLARE bridge (A6): the
tool schemas are JSON-encoded under vllm_xargs["flare_tools"]. The FLARE sampler
emits no logprobs, so token ids are recovered by re-encoding the server's raw
text (skip_special_tokens=False) with the same HF tokenizer.

A6 = 10 matched-20 turns, single-turn, POST /reset_prefix_cache before each (fresh
context). A7 = 3 full episodes, warm APC (reset once, then episode order, NO reset).

Bounded foreground; the server is a separate cage'd process. Writes
online_a6.jsonl / online_a7.jsonl.
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
from transformers import AutoTokenizer  # noqa: E402

BASE = "http://127.0.0.1:9952"
MODEL = "qwen3.5-9b-flare-hybrid-clean"
SERVED = str(ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16")
OUTDIR = ROOT / "runs/stage_a_cert"
REF = json.loads((ROOT / "runs/p2_engine_bench/matched20_ref.json").read_text())
REC_BY_GT = {r["global_turn"]: r for r in REF}
TS = json.loads((OUTDIR / "turnsets.json").read_text())
SEED, MARGIN = TS["seed"], TS["margin"]

tok = AutoTokenizer.from_pretrained(SERVED, trust_remote_code=True)


def reset_cache():
    requests.post(f"{BASE}/reset_prefix_cache", timeout=60)


def one_turn(rec):
    maxtok = int(rec["n_ref"]) + MARGIN
    body = {
        "model": MODEL,
        "prompt": [int(x) for x in rec["prompt_ids"]],
        "max_tokens": maxtok,
        "temperature": 0,
        "seed": SEED,
        "stop_token_ids": [int(x) for x in rec["stop_token_ids"]],
        "skip_special_tokens": False,           # keep specials so re-encode -> exact ids
        "spaces_between_special_tokens": True,
        "add_special_tokens": False,            # prompt is raw ids; do not augment
        # A6 serving-drift fix: the offline in-process LLM RequestOutput INCLUDES
        # the grammar-forced </tool_call> close token (id 248059, which is also a
        # stop token); the completions endpoint trims stop tokens by default
        # (include_stop_str_in_output=False). Align the cert harness so both sides
        # carry the identical terminal token -> a true engine-vs-engine byte cmp.
        "include_stop_str_in_output": True,
        "vllm_xargs": {
            "flare_tools": json.dumps(rec["tools"]),
            "grammar_topk": int(rec["grammar_topk"]),
        },
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/completions", json=body, timeout=600)
    wall = round(time.time() - t0, 3)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    text_full = ch["text"]
    # recover token ids by re-encoding the server text through the SAME tokenizer
    online_ids = [int(x) for x in tok.encode(text_full, add_special_tokens=False)]
    text_skip = tok.decode(online_ids, skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    return {
        "global_turn": rec["global_turn"], "episode": rec["episode"], "turn": rec["turn"],
        "episode_id": rec["episode_id"], "n_ref": int(rec["n_ref"]), "maxtok": maxtok,
        "finish_reason": ch.get("finish_reason"),
        "usage_completion_tokens": (j.get("usage") or {}).get("completion_tokens"),
        "online_ids": online_ids, "gen_text": text_skip, "gen_text_full": text_full,
        "wall_s": wall,
    }


def run_a6():
    out = (OUTDIR / "online_a6.jsonl").open("w")
    for gt in TS["a6_global_turns"]:
        rec = REC_BY_GT[gt]
        reset_cache()
        r = one_turn(rec); r["apc"] = "reset_fresh"
        out.write(json.dumps(r) + "\n"); out.flush()
        print(f"[online-A6] gt{gt:2d} n_ids={len(r['online_ids'])}/{r['n_ref']} "
              f"fin={r['finish_reason']} wall={r['wall_s']}", flush=True)
    out.close()


def run_a7():
    reset_cache()  # clean slate, then warm build-up within episodes (NO reset)
    out = (OUTDIR / "online_a7.jsonl").open("w")
    for ep in TS["a7_episodes"]:
        gts = sorted(r["global_turn"] for r in REF if r["episode"] == ep)
        for gt in gts:
            rec = REC_BY_GT[gt]
            r = one_turn(rec); r["apc"] = "warm"
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"[online-A7] ep{ep} gt{gt:2d} t{rec['turn']} n_ids={len(r['online_ids'])}/{r['n_ref']} "
                  f"fin={r['finish_reason']} wall={r['wall_s']}", flush=True)
    out.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a6", "all"):
        run_a6()
    if which in ("a7", "all"):
        run_a7()
    print("[online] DONE", flush=True)
