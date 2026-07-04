#!/usr/bin/env python3
"""CPU precompute: reconstruct all 63 matched-20 turn prompts (prompt_ids) from
the HF hybrid-clean reference row's teacher-forced generated history, and VERIFY
each reconstruction is byte-identical to what the HF matched-20 eval fed the model
(assert sha256_text(prompt) == hf_row.prompt_sha256 and len(prompt_ids)==prompt_tokens).

The matched-20 eval is a *generated-history* loop (each backend appends its own
sampled assistant text + the synthetic tool response). To drive the engine over
the identical 63 turns without cascade-drift confounds, we rebuild each turn's
prompt from the HF row's stored generated_token_ids + tool_response_payload
(teacher-forced), then let the engine generate on that exact prompt and compare
token-for-token vs the HF row's generated_token_ids.

No GPU / model needed. Writes runs/p2_engine_bench/matched20_ref.json (schema
compatible with gap5a_ref.json used by the proven engine driver).
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
HF_TURNS = ROOT / "runs/hybrid_forced_grammar_seq_values_v2/matched20/diffusion_hybrid_forced_grammar_seq_values/turns.jsonl"
GAP5A = ROOT / "runs/p2_engine_acceptance/gap5a_ref.json"
OUT = ROOT / "runs/p2_engine_bench/matched20_ref.json"

# global constants for this export/eval (from gap5a_ref.json / fairness_manifest)
MASK_ID = 248077
BLOCK_SIZE = 32
GRAMMAR_TOPK = 256
STOP_TOKEN_IDS = [248044, 248045, 248046, 248059]


def main():
    tok = AutoTokenizer.from_pretrained(str(EXPORT), trust_remote_code=True)
    chat_template = load_chat_template(CHAT_TEMPLATE)

    args = SimpleNamespace(
        input_jsonl=ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl",
        min_turns=3, max_turns=6, episode_limit=20,
    )
    episodes = build_episodes(args)
    assert len(episodes) == 20, f"expected 20 episodes, got {len(episodes)}"

    hf_rows = [json.loads(l) for l in open(HF_TURNS)]
    assert len(hf_rows) == 63, f"expected 63 HF turns, got {len(hf_rows)}"
    # index HF rows by (episode_idx, turn_idx)
    hf = {(r["episode_idx"], r["turn_idx"]): r for r in hf_rows}

    gap5a = {(r["episode"], r["turn"]): r for r in json.loads(GAP5A.read_text())}

    records = []
    problems = []
    global_turn = 0
    for ep in episodes:
        ei = ep["episode_idx"]
        # cross-check episode id vs HF row
        hf0 = hf[(ei, 0)]
        if hf0["episode_id"] != ep["id"]:
            problems.append(f"ep{ei} id mismatch: episode={ep['id']} hf={hf0['episode_id']}")
        messages = [dict(m) for m in ep["prompt_messages"]]
        prompt = render_matched_prompt(tok, messages, ep["tools"], chat_template)
        schemas = tool_schema_by_name(ep["tools"])
        n_turns = len(ep["gold_blocks"])
        for turn_idx in range(n_turns):
            hr = hf[(ei, turn_idx)]
            prompt_ids = tok([prompt], add_special_tokens=False).input_ids[0]
            # ---- VERIFY reconstruction is byte-identical to the HF eval's prompt ----
            sha_ok = sha256_text(prompt) == hr["prompt_sha256"]
            plen_ok = len(prompt_ids) == hr["prompt_tokens"]
            if not sha_ok:
                problems.append(f"ep{ei}/t{turn_idx} prompt_sha256 MISMATCH")
            if not plen_ok:
                problems.append(f"ep{ei}/t{turn_idx} prompt_tokens MISMATCH: recon={len(prompt_ids)} hf={hr['prompt_tokens']}")
            ref_new_ids = [int(x) for x in (hr.get("generated_token_ids") or [])]
            # cross-check against gap5a_ref pre-tokenized prompt_ids where available
            g = gap5a.get((ei, turn_idx))
            gap_ids_match = None
            gap_ref_match = None
            if g is not None:
                gap_ids_match = list(map(int, g["prompt_ids"])) == list(map(int, prompt_ids))
                gap_ref_match = list(map(int, g["ref_new_ids"])) == ref_new_ids
                if not gap_ids_match:
                    problems.append(f"ep{ei}/t{turn_idx} prompt_ids != gap5a_ref prompt_ids")
                if not gap_ref_match:
                    problems.append(f"ep{ei}/t{turn_idx} ref_new_ids != gap5a_ref ref_new_ids")
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
                # HF reference metrics for this turn (parity-implied targets)
                "hf_exact_arguments": bool(hr.get("exact_arguments")),
                "hf_valid_tool_call": bool(hr.get("valid_tool_call")),
                "hf_generated_token_count": int(hr.get("generated_token_count") or 0),
                "hf_denoise_forwards_total": int(
                    ((hr.get("backend_meta") or {}).get("sampler_schedule_events") or {}).get("denoise_forwards_total") or 0),
                "hf_turn_wall_seconds": float(hr.get("turn_wall_seconds") or 0.0),
                "prompt_sha256": hr["prompt_sha256"],
                "verify": {"sha_ok": sha_ok, "plen_ok": plen_ok,
                            "gap5a_prompt_ids_match": gap_ids_match,
                            "gap5a_ref_ids_match": gap_ref_match},
            }
            records.append(rec)
            global_turn += 1
            # advance the teacher-forced prompt using the HF row's own generated
            # history + stored synthetic tool response (exactly the matched-20 loop)
            history_text = decode_text(tok, ref_new_ids)
            next_user = next_turn_user_message(ep, turn_idx + 1)
            prompt = prompt + history_text + tool_response_suffix(hr["tool_response_payload"], next_user)

    OUT.write_text(json.dumps(records, indent=2) + "\n")

    hf_exact = sum(1 for r in records if r["hf_exact_arguments"])
    hf_valid = sum(1 for r in records if r["hf_valid_tool_call"])
    sha_all = all(r["verify"]["sha_ok"] for r in records)
    plen_all = all(r["verify"]["plen_ok"] for r in records)
    gap_checked = [r for r in records if r["verify"]["gap5a_prompt_ids_match"] is not None]
    gap_ids_all = all(r["verify"]["gap5a_prompt_ids_match"] for r in gap_checked)
    gap_ref_all = all(r["verify"]["gap5a_ref_ids_match"] for r in gap_checked)
    print(json.dumps({
        "n_records": len(records),
        "n_episodes": len(episodes),
        "hf_exact_args": f"{hf_exact}/63",
        "hf_valid": f"{hf_valid}/63",
        "ALL_prompt_sha256_match": sha_all,
        "ALL_prompt_tokens_match": plen_all,
        "gap5a_crosscheck_turns": len(gap_checked),
        "gap5a_prompt_ids_all_match": gap_ids_all,
        "gap5a_ref_ids_all_match": gap_ref_all,
        "total_ref_tokens": sum(r["n_ref"] for r in records),
        "problems": problems[:20],
        "n_problems": len(problems),
        "out": str(OUT),
    }, indent=2))
    if problems:
        print("PROBLEMS DETECTED — reconstruction not byte-faithful; STOP.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
