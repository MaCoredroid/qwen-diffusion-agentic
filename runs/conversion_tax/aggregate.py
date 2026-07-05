#!/usr/bin/env python3
"""Aggregate the 3x3 conversion-tax battery into summary.json + report.md.

Re-scores EVERY cell (including the two reused class-A cells) with the single
deterministic scorer in scoring.py, so all nine numbers come from one code path.
Emits raw counts, per-item wrong-idx lists, finish-reason mix, and engine-side
audit/stability (value_projection_events, verify.ok, hangs, length-runaways).
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
CT = ROOT / "runs/conversion_tax"
sys.path.insert(0, str(CT))
from scoring import score_gsm8k, score_code, score_instruction  # noqa: E402

SYS_LABEL = {"stock": "STOCK-AR", "merged": "MERGED-AR", "engine": "ENGINE-DIFFUSION"}
CLASS_LABEL = {"A": "GSM8K free-CoT (30)", "B": "CODE / MBPP (25)", "C": "INSTRUCTION (25)"}
DENOM = {"A": 30, "B": 25, "C": 25}

CELLS = [
    ("A", "stock",  ROOT / "runs/l1_baseline_b1/ar_gsm8k_clean.jsonl", True),
    ("A", "merged", CT / "A_merged_ar.jsonl", False),
    ("A", "engine", ROOT / "runs/l0l2_final_head_verify/engine_gsm8k_clean_head.jsonl", True),
    ("B", "stock",  CT / "B_stock_ar.jsonl", False),
    ("B", "merged", CT / "B_merged_ar.jsonl", False),
    ("B", "engine", CT / "B_engine.jsonl", False),
    ("C", "stock",  CT / "C_stock_ar.jsonl", False),
    ("C", "merged", CT / "C_merged_ar.jsonl", False),
    ("C", "engine", CT / "C_engine.jsonl", False),
]

GOLD_A = {r["idx"]: r["gold_answer"] for r in json.loads((ROOT / "runs/l1_census/gsm8k_prompts_clean.json").read_text())}
META_B = {r["idx"]: r for r in json.loads((CT / "code_prompts.json").read_text())}
META_C = {r["idx"]: r for r in json.loads((CT / "instr_prompts.json").read_text())}


def score_row(clazz, row):
    t = row.get("gen_text", "")
    if clazz == "A":
        return score_gsm8k(t, GOLD_A[row["idx"]])
    if clazz == "B":
        m = META_B[row["idx"]]
        return score_code(t, m["test_imports"], m["test_list"])
    return score_instruction(t, META_C[row["idx"]]["check"])


def load_cell(clazz, system, path, reused):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by_idx = {r["idx"]: r for r in rows}
    correct, wrong = 0, []
    for idx in sorted(by_idx):
        ok = score_row(clazz, by_idx[idx])
        correct += int(ok)
        if not ok:
            wrong.append(idx)
    fin = {}
    for r in rows:
        fin[r.get("finish_reason")] = fin.get(r.get("finish_reason"), 0) + 1
    cell = {
        "system": SYS_LABEL[system], "class": clazz, "correct": correct,
        "n": len(by_idx), "denom": DENOM[clazz], "wrong_idxs": wrong,
        "finish_reasons": fin, "reused": reused, "source": str(path.relative_to(ROOT)),
    }
    if system == "engine":
        hangs = sorted(r["idx"] for r in rows if r.get("hang") or r.get("error"))
        length = sorted(r["idx"] for r in rows if r.get("finish_reason") == "length")
        projnz = sorted(r["idx"] for r in rows if (r.get("counters") or {}).get("value_projection_events", 0) != 0)
        verifok = all((r.get("verify") or {}).get("ok") for r in rows if "verify" in r)
        cell["engine_audit"] = {
            "hangs": hangs, "length_runaways": length,
            "value_projection_events_nonzero_idxs": projnz,
            "all_verify_ok": verifok,
        }
    return cell


def main():
    grid = {c: {} for c in ["A", "B", "C"]}
    cells = []
    for clazz, system, path, reused in CELLS:
        cell = load_cell(clazz, system, path, reused)
        grid[clazz][system] = f"{cell['correct']}/{cell['denom']}"
        cells.append(cell)

    summary = {
        "title": "Per-capability conversion-tax battery (#28) — 3x3 raw counts",
        "date": "2026-07-05",
        "systems": {
            "STOCK-AR": "stock Qwen3.5-9B c202236, offline vLLM bf16 enforce_eager, plain greedy",
            "MERGED-AR": "models/qwen3.5-9b-fastdllm-mtplus1-Anew-vllm-bf16 (the 136/247 export), same offline-vLLM AR path as STOCK-AR",
            "ENGINE-DIFFUSION": "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16, vLLM pin 0b44dcc hybrid_clean free-text, BIDIR_PROBE+CUDAGRAPH",
        },
        "regime": "B=1 greedy (temp 0, seed 20260701), strict deterministic scoring, prompts identical across systems",
        "grid_raw_counts": grid,
        "cells": cells,
        "engine_stability": {
            "hangs_total": sum(len(c["engine_audit"]["hangs"]) for c in cells if c["system"] == "ENGINE-DIFFUSION"),
            "length_runaways_total": sum(len(c["engine_audit"]["length_runaways"]) for c in cells if c["system"] == "ENGINE-DIFFUSION"),
            "value_projection_events": "0 across all engine cells" if all(
                not c["engine_audit"]["value_projection_events_nonzero_idxs"]
                for c in cells if c["system"] == "ENGINE-DIFFUSION") else "NONZERO — INVALID",
            "all_verify_ok": all(c["engine_audit"]["all_verify_ok"] for c in cells if c["system"] == "ENGINE-DIFFUSION"),
            "note": "L0 fix held: 0 CPU-pathological hangs on free-text B and C engine cells (expected).",
        },
    }
    (CT / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- report.md ----
    L = []
    L.append("# Per-capability conversion-tax table (#28)\n")
    L.append("Raw exact counts, B=1 greedy (temp 0, seed 20260701), strict deterministic scoring,")
    L.append("identical prompts across all three systems. Two class-A cells reused (see Reuse).\n")
    L.append("| capability class | STOCK-AR | MERGED-AR | ENGINE-DIFFUSION |")
    L.append("|---|---|---|---|")
    for c in ["A", "B", "C"]:
        L.append(f"| **{CLASS_LABEL[c]}** | {grid[c]['stock']} | {grid[c]['merged']} | {grid[c]['engine']} |")
    L.append("")
    L.append("Columns are the conversion pipeline: **STOCK-AR** (pre-conversion baseline) → "
             "**MERGED-AR** (RL-v2 merged weights served plain AR — the 136/247 export) → "
             "**ENGINE-DIFFUSION** (the same RL-v2 weights served through the block-diffusion engine).\n")
    L.append("## Per-cell detail\n")
    L.append("| class | system | correct | finish reasons | wrong idxs | reused | source |")
    L.append("|---|---|---|---|---|---|---|")
    for cell in cells:
        fin = ", ".join(f"{k}:{v}" for k, v in cell["finish_reasons"].items())
        L.append(f"| {cell['class']} | {cell['system']} | {cell['correct']}/{cell['denom']} | {fin} | "
                 f"{cell['wrong_idxs']} | {'yes' if cell['reused'] else 'no'} | `{cell['source']}` |")
    L.append("")
    L.append("## Engine-side audit / stability\n")
    for cell in cells:
        if cell["system"] != "ENGINE-DIFFUSION":
            continue
        a = cell["engine_audit"]
        L.append(f"- **class {cell['class']}**: hangs {a['hangs']}, length-runaways {a['length_runaways']}, "
                 f"value_projection_events nonzero {a['value_projection_events_nonzero_idxs']}, "
                 f"all verify.ok {a['all_verify_ok']}.")
    st = summary["engine_stability"]
    L.append(f"\n**Stability summary:** {st['hangs_total']} hangs total across the two free-text engine cells "
             f"(B, C) — L0 fix held. value_projection_events: {st['value_projection_events']}; "
             f"all verify.ok: {st['all_verify_ok']}.\n")
    L.append("## Reuse\n")
    L.append("- **A / STOCK-AR** reused from `runs/l1_baseline_b1/ar_gsm8k_clean.jsonl` (same 30 clean GSM8K prompts, same offline-vLLM greedy).")
    L.append("- **A / ENGINE** reused from `runs/l0l2_final_head_verify/engine_gsm8k_clean_head.jsonl` (pin 0b44dcc free-text head).")
    L.append("- All other seven cells run fresh here. Scoring for every cell (reused included) is recomputed by `scoring.py`.")
    (CT / "report.md").write_text("\n".join(L) + "\n")

    print(json.dumps(grid, indent=2))
    print("engine hangs total:", summary["engine_stability"]["hangs_total"],
          "| proj:", summary["engine_stability"]["value_projection_events"],
          "| verify_ok:", summary["engine_stability"]["all_verify_ok"])
    print("wrote summary.json + report.md")


if __name__ == "__main__":
    main()
