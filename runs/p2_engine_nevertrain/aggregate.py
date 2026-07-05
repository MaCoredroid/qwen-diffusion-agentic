#!/usr/bin/env python3
"""Aggregate + diagnose the never-train engine battery (184 turns).

Computes: byte-parity vs HF hybrid-clean reference, engine exact_args vs HF 83/184
(per-turn deviation diagnosis: fp-residue class like gt44 vs structural), valid
count, TRUE denoise forwards/turn, s/turn mean/p50/p90/worst, and audit counters
(value_projection_events==0, zero_forward_rows==0, verify_invariants all turns).
For every parity break, decodes the divergence token (engine vs ref) and classifies.
"""
import json
import statistics
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
TURNS = ROOT / "runs/p2_engine_nevertrain/nevertrain_turns.jsonl"
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
OUT = ROOT / "runs/p2_engine_nevertrain/aggregate.json"

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"), trust_remote_code=True)

rows = [json.loads(l) for l in open(TURNS) if l.strip()]
rows.sort(key=lambda r: r["global_turn"])
ref = {r["global_turn"]: r for r in json.loads(REF.read_text())}
assert len(rows) == 184, f"expected 184 turns, got {len(rows)}"
assert len({r["global_turn"] for r in rows}) == 184, "duplicate/missing global_turns"

def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

byte_parity = sum(1 for r in rows if r["byte_parity_full"])
parity_over_min = sum(1 for r in rows if r["byte_parity_over_min"])
eng_exact = sum(1 for r in rows if r["eng_exact_arguments"])
hf_exact = sum(1 for r in rows if r["hf_exact_arguments"])
eng_valid = sum(1 for r in rows if r["eng_valid_tool_call"])
hf_valid = sum(1 for r in rows if r["hf_valid_tool_call"])
exact_matches_hf = sum(1 for r in rows if r["exact_matches_hf"])
eng_wins = [r["global_turn"] for r in rows if r["eng_exact_arguments"] and not r["hf_exact_arguments"]]
eng_losses = [r["global_turn"] for r in rows if not r["eng_exact_arguments"] and r["hf_exact_arguments"]]

walls = [r["wall_s"] for r in rows]
fwds = [r["denoise_forwards"] for r in rows]
per_fwd = [r["per_forward_ms"] for r in rows if r["per_forward_ms"]]

# audit counters
proj_events = [(r["global_turn"], r["counters"]["value_projection_events"]) for r in rows if r["counters"]]
proj_nonzero = [g for g, v in proj_events if v != 0]
zero_forward_rows = [r["global_turn"] for r in rows if r["denoise_forwards"] == 0]
verify_bad = [r["global_turn"] for r in rows if not (r["verify"] or {}).get("ok")]
finish_not_stop = [(r["global_turn"], r["finish_reason"]) for r in rows if r["finish_reason"] != "stop"]

# parity breaks -- per-turn diagnosis
breaks = []
for r in rows:
    if r["byte_parity_full"]:
        continue
    g = r["global_turn"]
    fd = r["first_divergence"]
    e_tok = r["engine_tok_at_fd"]
    r_tok = r["ref_tok_at_fd"]
    # context: where in the ref sequence is the divergence relative to a tool-call
    # value region? decode a small window around fd from the reference.
    refids = ref[g]["ref_new_ids"]
    lo = max(0, (fd or 0) - 6)
    ctx_before = tok.decode(refids[lo:fd]) if fd is not None else None
    breaks.append({
        "global_turn": g, "episode": r["episode"], "turn": r["turn"],
        "source_family": r["source_family"],
        "first_divergence": fd,
        "n_gen": r["n_gen"], "n_ref": r["n_ref"],
        "len_match": r["n_gen"] == r["n_ref"],
        "eng_tok_at_fd": e_tok, "ref_tok_at_fd": r_tok,
        "eng_tok_str": tok.decode([e_tok]) if e_tok is not None else None,
        "ref_tok_str": tok.decode([r_tok]) if r_tok is not None else None,
        "ctx_before_fd": ctx_before,
        "eng_exact": r["eng_exact_arguments"], "hf_exact": r["hf_exact_arguments"],
        "eng_valid": r["eng_valid_tool_call"], "hf_valid": r["hf_valid_tool_call"],
        "exact_matches_hf": r["exact_matches_hf"],
        "proj": r["counters"]["value_projection_events"] if r["counters"] else None,
        "verify_ok": (r["verify"] or {}).get("ok"),
    })

summary = {
    "n_turns": len(rows),
    "byte_parity_full": f"{byte_parity}/184",
    "byte_parity_over_min": f"{parity_over_min}/184",
    "eng_exact_args": f"{eng_exact}/184",
    "hf_exact_args": f"{hf_exact}/184",
    "eng_valid": f"{eng_valid}/184",
    "hf_valid": f"{hf_valid}/184",
    "eng_exact_matches_hf_per_turn": f"{exact_matches_hf}/184",
    "eng_wins_over_hf": eng_wins,
    "eng_losses_vs_hf": eng_losses,
    "s_per_turn_mean": round(statistics.mean(walls), 4),
    "s_per_turn_p50": round(pct(walls, 0.50), 4),
    "s_per_turn_p90": round(pct(walls, 0.90), 4),
    "s_per_turn_worst": round(max(walls), 4),
    "s_per_turn_worst_gt": max(rows, key=lambda r: r["wall_s"])["global_turn"],
    "true_denoise_forwards_per_turn": round(statistics.mean(fwds), 3),
    "hf_denoise_forwards_per_turn": round(statistics.mean([ref[r["global_turn"]]["hf_denoise_forwards_total"] for r in rows]), 3),
    "per_forward_ms_mean": round(statistics.mean(per_fwd), 2),
    "total_gen_tokens": sum(r["n_gen"] for r in rows),
    "total_ref_tokens": sum(r["n_ref"] for r in rows),
    "audit": {
        "value_projection_events_all_zero": len(proj_nonzero) == 0,
        "proj_nonzero_turns": proj_nonzero,
        "zero_forward_rows": zero_forward_rows,
        "zero_forward_rows_count": len(zero_forward_rows),
        "verify_invariants_all_ok": len(verify_bad) == 0,
        "verify_bad_turns": verify_bad,
        "finish_not_stop": finish_not_stop,
    },
    "n_parity_breaks": len(breaks),
    "parity_break_turns": [b["global_turn"] for b in breaks],
    "breaks": breaks,
}
OUT.write_text(json.dumps(summary, indent=2) + "\n")
# print without the verbose per-break ctx for console
console = {k: v for k, v in summary.items() if k != "breaks"}
print(json.dumps(console, indent=2))
print("\n=== PARITY BREAK DETAIL ===")
for b in breaks:
    print(f"gt{b['global_turn']} ep{b['episode']}/t{b['turn']} [{b['source_family']}] "
          f"fd={b['first_divergence']} n={b['n_gen']}/{b['n_ref']} lenmatch={b['len_match']} "
          f"eng={b['eng_tok_at_fd']}({b['eng_tok_str']!r}) ref={b['ref_tok_at_fd']}({b['ref_tok_str']!r}) "
          f"eng_exact={b['eng_exact']}(hf={b['hf_exact']}) matches_hf={b['exact_matches_hf']} "
          f"proj={b['proj']} verify={b['verify_ok']}")
    print(f"     ctx_before_fd={b['ctx_before_fd']!r}")
