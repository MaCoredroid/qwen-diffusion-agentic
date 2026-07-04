#!/usr/bin/env python3
"""CPU precompute: reconstruct the FIRST few never-train BFCL/API-Bank episode
turn prompts (teacher-forced from the HF hybrid-clean never-train reference) and
VERIFY each reconstruction is byte-identical to the HF eval prompt
(sha256_text(prompt) == hf_row.prompt_sha256). Mirrors build_matched20_ref.py.

Writes runs/p2_engine_battery_v2/nevertrain3_ref.json (engine-driver schema).
No GPU. Never-train build params: episode-limit 60, min-turns 1, max-turns 8.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoTokenizer  # noqa: E402
from eval_flare_multiturn_percall_waves import build_episodes  # noqa: E402
from eval_toolcall_jsonl import tool_schema_by_name  # noqa: E402
from eval_flare_northstar_matched import (  # noqa: E402
    render_matched_prompt, tool_response_suffix, next_turn_user_message,
    decode_text, sha256_text, load_chat_template,
)

EXPORT = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"  # the engine's tokenizer
CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
HF_TURNS = ROOT / "runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl"
INPUT = ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl"
OUT = ROOT / "runs/p2_engine_battery_v2/nevertrain_ref.json"

MASK_ID = 248077
BLOCK_SIZE = 32
GRAMMAR_TOPK = 256
STOP_TOKEN_IDS = [248044, 248045, 248046, 248059]
N_EPISODES = 60  # build all; we run a 3-turn subset spanning families


def main():
    tok = AutoTokenizer.from_pretrained(str(EXPORT), trust_remote_code=True)
    chat_template = load_chat_template(CHAT_TEMPLATE)
    args = SimpleNamespace(input_jsonl=INPUT, min_turns=1, max_turns=8, episode_limit=N_EPISODES)
    episodes = build_episodes(args)

    hf_rows = [json.loads(l) for l in open(HF_TURNS)]
    hf = {(r["episode_idx"], r["turn_idx"]): r for r in hf_rows}

    records = []
    problems = []
    global_turn = 0
    for ep in episodes:
        ei = ep["episode_idx"]
        messages = [dict(m) for m in ep["prompt_messages"]]
        prompt = render_matched_prompt(tok, messages, ep["tools"], chat_template)
        schemas = tool_schema_by_name(ep["tools"])
        n_turns = len(ep["gold_blocks"])
        for turn_idx in range(n_turns):
            hr = hf.get((ei, turn_idx))
            if hr is None:
                continue
            prompt_ids = tok([prompt], add_special_tokens=False).input_ids[0]
            sha_ok = sha256_text(prompt) == hr["prompt_sha256"]
            plen_ok = len(prompt_ids) == hr["prompt_tokens"]
            if not sha_ok:
                problems.append(f"ep{ei}/t{turn_idx} prompt_sha256 MISMATCH")
            if not plen_ok:
                problems.append(f"ep{ei}/t{turn_idx} prompt_tokens MISMATCH recon={len(prompt_ids)} hf={hr['prompt_tokens']}")
            ref_new_ids = [int(x) for x in (hr.get("generated_token_ids") or [])]
            records.append({
                "global_turn": global_turn, "episode": ei, "turn": turn_idx,
                "episode_id": ep["id"], "source_family": hr.get("source_family"),
                "turns_in_episode": n_turns, "prompt_len": len(prompt_ids),
                "prompt_ids": [int(x) for x in prompt_ids], "ref_new_ids": ref_new_ids,
                "n_ref": len(ref_new_ids), "mask_id": MASK_ID, "block_size": BLOCK_SIZE,
                "grammar_topk": GRAMMAR_TOPK, "stop_token_ids": STOP_TOKEN_IDS,
                "schemas": schemas, "tools": ep["tools"], "gold_block": ep["gold_blocks"][turn_idx],
                "hf_exact_arguments": bool(hr.get("exact_arguments")),
                "hf_valid_tool_call": bool(hr.get("valid_tool_call")),
                "hf_generated_token_count": int(hr.get("generated_token_count") or 0),
                "prompt_sha256": hr["prompt_sha256"],
                "verify": {"sha_ok": sha_ok, "plen_ok": plen_ok},
            })
            global_turn += 1
            history_text = decode_text(tok, ref_new_ids)
            next_user = next_turn_user_message(ep, turn_idx + 1)
            prompt = prompt + history_text + tool_response_suffix(hr["tool_response_payload"], next_user)

    OUT.write_text(json.dumps(records, indent=2) + "\n")
    sha_all = all(r["verify"]["sha_ok"] for r in records)
    plen_all = all(r["verify"]["plen_ok"] for r in records)
    print(json.dumps({
        "n_records": len(records), "n_episodes": len(episodes),
        "ALL_prompt_sha256_match": sha_all, "ALL_prompt_tokens_match": plen_all,
        "families": sorted({r["source_family"] for r in records}),
        "sample": [{"gt": r["global_turn"], "ep": r["episode"], "t": r["turn"],
                    "fam": r["source_family"], "n_ref": r["n_ref"],
                    "hf_exact": r["hf_exact_arguments"], "sha_ok": r["verify"]["sha_ok"]}
                   for r in records[:10]],
        "n_problems": len(problems), "problems": problems[:10], "out": str(OUT),
    }, indent=2))
    if problems:
        print("PROBLEMS — reconstruction not byte-faithful.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
