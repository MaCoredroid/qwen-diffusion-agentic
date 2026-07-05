#!/usr/bin/env python3
"""CPU precompute: reconstruct all 184 never-train (BFCL/API-Bank) turn prompts
(prompt_ids) from the HF hybrid-clean reference row's teacher-forced generated
history, and VERIFY each reconstruction is byte-identical to what the HF
never-train eval fed the model (assert sha256_text(prompt) == hf_row.prompt_sha256
and len(prompt_ids) == hf_row.prompt_tokens).

Mirror of runs/p2_engine_bench/build_matched20_ref.py, retargeted at the
never-train slice:
  input   data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl (60 eps)
  HF ref  runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/
          diffusion_hybrid_forced_grammar_seq_values/turns.jsonl (184 turns, 83 exact)

The never-train HF row already stores per-turn generated_token_ids, so we do NOT
need to regenerate the HF reference via the HF bridge -- ref_new_ids come straight
from the stored HF token ids, and the reconstructed prompt is verified against the
stored prompt_sha256 / prompt_tokens the HF eval recorded.

The HF driver (scripts/eval_flare_northstar_hybrid_clean.py::run_hybrid) advances
each turn's prompt as:
    prompt = prompt + decode_text(tok, new_ids)
                    + tool_response_suffix(tool_response_payload, next_user)
This script reproduces that loop exactly (teacher-forced on the HF row's own
generated_token_ids + stored tool_response_payload), so the engine can then decode
each turn on the byte-identical prompt the HF backend saw.

No GPU / model needed. Writes runs/p2_engine_nevertrain/nevertrain_ref.json
(schema compatible with matched20_ref.json used by run_battery_v3b.py).
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
    render_matched_prompt,
    tool_response_suffix,
    next_turn_user_message,
    decode_text,
    sha256_text,
    load_chat_template,
)

EXPORT = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")
INPUT_JSONL = ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl"
HF_TURNS = ROOT / "runs/hybrid_broaden_nevertrain_v2/nevertrain_bfcl_apibank60/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl"
OUT = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"

# global constants for this export/eval (identical to matched20_ref build; the
# hybrid-clean HF eval is the same code path -- eval_flare_northstar_hybrid_clean).
MASK_ID = 248077
BLOCK_SIZE = 32
GRAMMAR_TOPK = 256
STOP_TOKEN_IDS = [248044, 248045, 248046, 248059]

# never-train episode-build parameters: keep ALL 60 episodes / 184 turns
# (source gold blocks range 1..8 turns; the dataset builder used --max-turns 8).
MIN_TURNS = 1
MAX_TURNS = 8
EPISODE_LIMIT = 60


def main():
    tok = AutoTokenizer.from_pretrained(str(EXPORT), trust_remote_code=True)
    chat_template = load_chat_template(CHAT_TEMPLATE)

    args = SimpleNamespace(
        input_jsonl=INPUT_JSONL,
        min_turns=MIN_TURNS, max_turns=MAX_TURNS, episode_limit=EPISODE_LIMIT,
    )
    episodes = build_episodes(args)

    hf_rows = [json.loads(l) for l in open(HF_TURNS)]
    hf = {(r["episode_idx"], r["turn_idx"]): r for r in hf_rows}
    n_hf_turns = len(hf_rows)

    records = []
    problems = []
    global_turn = 0
    for ep in episodes:
        ei = ep["episode_idx"]
        hf0 = hf.get((ei, 0))
        if hf0 is None:
            problems.append(f"ep{ei} missing HF row (episode_idx not in HF turns)")
            continue
        if hf0["episode_id"] != ep["id"]:
            problems.append(f"ep{ei} id mismatch: episode={ep['id']} hf={hf0['episode_id']}")
        messages = [dict(m) for m in ep["prompt_messages"]]
        prompt = render_matched_prompt(tok, messages, ep["tools"], chat_template)
        schemas = tool_schema_by_name(ep["tools"])
        n_turns = len(ep["gold_blocks"])
        for turn_idx in range(n_turns):
            hr = hf.get((ei, turn_idx))
            if hr is None:
                problems.append(f"ep{ei}/t{turn_idx} missing HF row")
                break
            prompt_ids = tok([prompt], add_special_tokens=False).input_ids[0]
            # ---- VERIFY reconstruction is byte-identical to the HF eval's prompt ----
            sha_ok = sha256_text(prompt) == hr["prompt_sha256"]
            plen_ok = len(prompt_ids) == hr["prompt_tokens"]
            if not sha_ok:
                problems.append(f"ep{ei}/t{turn_idx} prompt_sha256 MISMATCH")
            if not plen_ok:
                problems.append(f"ep{ei}/t{turn_idx} prompt_tokens MISMATCH: recon={len(prompt_ids)} hf={hr['prompt_tokens']}")
            ref_new_ids = [int(x) for x in (hr.get("generated_token_ids") or [])]
            rec = {
                "global_turn": global_turn,
                "episode": ei, "turn": turn_idx, "episode_id": ep["id"],
                "turns_in_episode": n_turns,
                "prompt_len": len(prompt_ids),
                "prompt_ids": [int(x) for x in prompt_ids],
                "ref_new_ids": ref_new_ids,
                "n_ref": len(ref_new_ids),
                "mask_id": MASK_ID, "block_size": BLOCK_SIZE,
                "grammar_topk": GRAMMAR_TOPK, "stop_token_ids": STOP_TOKEN_IDS,
                "schemas": schemas,
                "tools": ep["tools"],
                "gold_block": ep["gold_blocks"][turn_idx],
                "source_family": ep.get("source_family"),
                # HF reference metrics for this turn (parity-implied targets)
                "hf_exact_arguments": bool(hr.get("exact_arguments")),
                "hf_valid_tool_call": bool(hr.get("valid_tool_call")),
                "hf_generated_token_count": int(hr.get("generated_token_count") or 0),
                "hf_denoise_forwards_total": int(
                    ((hr.get("backend_meta") or {}).get("sampler_schedule_events") or {}).get("denoise_forwards_total") or 0),
                "hf_turn_wall_seconds": float(hr.get("turn_wall_seconds") or 0.0),
                "prompt_sha256": hr["prompt_sha256"],
                "verify": {"sha_ok": sha_ok, "plen_ok": plen_ok},
            }
            records.append(rec)
            global_turn += 1
            # advance the teacher-forced prompt using the HF row's own generated
            # history + stored synthetic tool response (exactly the hybrid loop)
            history_text = decode_text(tok, ref_new_ids)
            next_user = next_turn_user_message(ep, turn_idx + 1)
            prompt = prompt + history_text + tool_response_suffix(hr["tool_response_payload"], next_user)

    OUT.write_text(json.dumps(records, indent=2) + "\n")

    hf_exact = sum(1 for r in records if r["hf_exact_arguments"])
    hf_valid = sum(1 for r in records if r["hf_valid_tool_call"])
    sha_all = all(r["verify"]["sha_ok"] for r in records)
    plen_all = all(r["verify"]["plen_ok"] for r in records)
    print(json.dumps({
        "n_records": len(records),
        "n_episodes": len(episodes),
        "n_hf_turns": n_hf_turns,
        "hf_exact_args": f"{hf_exact}/{len(records)}",
        "hf_valid": f"{hf_valid}/{len(records)}",
        "ALL_prompt_sha256_match": sha_all,
        "ALL_prompt_tokens_match": plen_all,
        "total_ref_tokens": sum(r["n_ref"] for r in records),
        "problems": problems[:20],
        "n_problems": len(problems),
        "out": str(OUT),
    }, indent=2))
    if problems or not sha_all or not plen_all or len(records) != n_hf_turns:
        print("PROBLEMS DETECTED -- reconstruction not byte-faithful; STOP.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
