#!/usr/bin/env python3
"""S2 pilot DATA step -- build the trajectory-consistency training corpus + manifest.

Consumes the raw self-generation jsonl (runs/s2_pilot/train_gen.jsonl produced by
run_s2_gen.py on the GSM8K TRAIN split via the promoted free-text path) and the
prompt pool (runs/s2_pilot/gsm8k_train_prompts.json), then:

  1. AUDIT-FILTER (design s2_pilot_design.md sec.4), keep a trajectory only if ALL:
       (i)   strictly correct        (row.correct, pred==gold)
       (ii)  verify.ok == True       (proj==0 & forwards==model_chosen & gen==fsm+model)
       (iii) value_projection_events == 0
       (iv)  clean-stop              (finish_reason == 'stop', not length/hang/degenerate)
     plus answer_ids present & non-empty.
  2. LEAKAGE DEDUPE: hash every kept train prompt's normalized question; drop any whose
     hash collides with the 30-prompt GATE set (GSM8K test first-30) or the 20-prompt
     RETENTION set (GSM8K test first-20). Report collision counts (expected 0). Also drop
     intra-corpus duplicate questions (keep first in pool order).
  3. TARGET SIZE: ~1000 audit-clean-correct trajectories; yield-floor rule -- if
     audit-clean-correct yield < 0.30, cut target to 700 (feasibility probe).
  4. WRITE training file (jsonl; prompt_ids + answer_ids=y + provenance) + manifest.json
     (counts, filter yields, dedupe collisions, sha256 hashes, teacher/pins/decode config).

Teacher checkpoint: models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16 (M_{t+1} served, pin 0b44dcc);
decode: hybrid_clean, grammar inert, K=1 greedy, seed 90101, block_size 32, mask 248077.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
POOL = ROOT / "runs/s2_pilot/gsm8k_train_prompts.json"
GEN = Path(os.environ.get("S2_GEN", str(ROOT / "runs/s2_pilot/train_gen.jsonl")))
GATE = ROOT / "runs/l1_census/gsm8k_prompts_clean.json"     # 30-prompt gate set (test first-30)
TRAIN_OUT = ROOT / "runs/s2_pilot/s2_traj_corpus.jsonl"
MANIFEST = ROOT / "runs/s2_pilot/s2_traj_corpus.manifest.json"

TARGET = int(os.environ.get("S2_TARGET_CLEAN", "1000"))
TARGET_FLOOR = 700
YIELD_FLOOR = 0.30

TEACHER = "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
TEACHER_PIN = "0b44dcc"           # promoted free-text serving path (vLLM P2 code pin)
VLLM_WORKSPACE = "/home/mark/shared/vllm_p2_pr42406"
DECODE = {"policy": "hybrid_clean", "grammar": "inert (tools=[])", "K": 1,
          "temperature": 0.0, "greedy": True, "seed": 90101, "block_size": 32,
          "mask_id": 248077, "grammar_topk": 256, "maxtok": 384,
          "stop_token_ids": [248044, 248045, 248046, 248059]}


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def qhash(q: str) -> str:
    return hashlib.sha256(norm_q(q).encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def load_retention_hashes():
    """RETENTION set = GSM8K test first-20 questions (design sec.4)."""
    f = ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl"
    qs = []
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                q = r.get("question") or r.get("prompt") or r.get("input")
                if q:
                    qs.append(q)
    if len(qs) < 20:
        # fall back to the datasets test split first-20
        from datasets import load_dataset
        te = load_dataset("openai/gsm8k", "main", split="test")
        qs = [te[i]["question"] for i in range(20)]
    return {qhash(q) for q in qs[:20]}, qs[:20]


def main():
    pool = {r["idx"]: r for r in json.loads(POOL.read_text())}
    gen_rows = []
    for line in GEN.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                gen_rows.append(json.loads(line))
            except Exception:
                pass

    # --- census of the raw generation set ---
    total = len(gen_rows)
    hangs = sum(1 for r in gen_rows if r.get("hang") or r.get("error"))
    completed = [r for r in gen_rows if not (r.get("hang") or r.get("error"))]
    n_stop = sum(1 for r in completed if r.get("finish_reason") == "stop")
    n_length = sum(1 for r in completed if r.get("finish_reason") == "length")
    n_correct = sum(1 for r in completed if r.get("correct"))
    n_verify = sum(1 for r in completed if (r.get("verify") or {}).get("ok"))
    n_proj0 = sum(1 for r in completed
                  if (r.get("counters") or {}).get("value_projection_events") == 0)

    def is_clean(r):
        c = r.get("counters") or {}
        return (r.get("correct") and (r.get("verify") or {}).get("ok")
                and c.get("value_projection_events") == 0
                and r.get("finish_reason") == "stop"
                and r.get("answer_ids") and len(r["answer_ids"]) > 0)

    clean = [r for r in completed if is_clean(r)]
    clean.sort(key=lambda r: r["idx"])  # deterministic pool order
    audit_yield = (len(clean) / total) if total else 0.0

    # --- leakage dedupe vs GATE (test first-30) + RETENTION (test first-20) ---
    gate = json.loads(GATE.read_text())
    gate_hashes = {qhash(g["question"]) for g in gate}
    ret_hashes, _ret_qs = load_retention_hashes()

    gate_collisions, ret_collisions, intra_dupes = 0, 0, 0
    seen = set()
    kept = []
    for r in clean:
        h = r.get("q_norm_sha256") or qhash(r["question"])
        gc = h in gate_hashes
        rc = h in ret_hashes
        if gc:
            gate_collisions += 1
        if rc:
            ret_collisions += 1
        if gc or rc:
            continue
        if h in seen:
            intra_dupes += 1
            continue
        seen.add(h)
        kept.append(r)

    # --- target-size + yield-floor rule ---
    target = TARGET
    floor_triggered = False
    if audit_yield < YIELD_FLOOR:
        target = TARGET_FLOOR
        floor_triggered = True
    final = kept[:target]

    # --- write training corpus ---
    with TRAIN_OUT.open("w") as fh:
        for i, r in enumerate(final):
            prec = pool[r["idx"]]
            assert prec["train_idx"] == r["train_idx"]
            rec = {
                "id": i,
                "pool_idx": r["idx"],
                "train_idx": r["train_idx"],
                "q_norm_sha256": r.get("q_norm_sha256") or qhash(r["question"]),
                "prompt_ids": prec["prompt_ids"],       # loss-masked clean prompt
                "prompt_len": prec["prompt_len"],
                "answer_ids": r["answer_ids"],           # teacher target y (committed tokens)
                "n_answer": len(r["answer_ids"]),
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "pred": r["pred"], "gold": r["gold"],
                "provenance": {
                    "denoise_forwards": r.get("denoise_forwards"),
                    "per_forward_ms": r.get("per_forward_ms"),
                    "counters": r.get("counters"),
                    "finish_reason": r.get("finish_reason"),
                },
            }
            fh.write(json.dumps(rec) + "\n")

    ans_lens = [len(r["answer_ids"]) for r in final]
    manifest = {
        "artifact": "s2_traj_corpus",
        "design": "s2_pilot_design.md (9ce9445) sec.4",
        "teacher_checkpoint": TEACHER,
        "teacher_pin": TEACHER_PIN,
        "vllm_engine_workspace": VLLM_WORKSPACE,
        "vllm_engine_pin": "0b44dcc",
        "decode": DECODE,
        "split": "GSM8K train (openai/gsm8k main)",
        "prompt_format": "fixed 5-shot(train[0:5]) + <|im_start|>user Question:/Answer: turn; "
                         "byte-exact reconstruction of gsm8k_prompts_clean[0..4] verified in builder",
        "counts": {
            "raw_generations": total,
            "hangs_errors": hangs,
            "completed": len(completed),
            "finish_stop": n_stop,
            "finish_length": n_length,
            "correct": n_correct,
            "verify_ok": n_verify,
            "value_projection_events_zero": n_proj0,
            "audit_clean_correct": len(clean),
            "kept_after_dedupe": len(kept),
            "written_to_corpus": len(final),
        },
        "filter_yields": {
            "audit_clean_correct_over_raw": round(audit_yield, 4),
            "correct_over_completed": round(n_correct / len(completed), 4) if completed else 0.0,
            "clean_over_correct": round(len(clean) / n_correct, 4) if n_correct else 0.0,
            "target": target,
            "target_floor_triggered": floor_triggered,
        },
        "leakage_dedupe": {
            "gate_set": "runs/l1_census/gsm8k_prompts_clean.json (GSM8K test first-30)",
            "retention_set": "data/phaseA_retention/gsm8k_main_test_first20.jsonl (GSM8K test first-20)",
            "normalization": "lower + collapse-whitespace + strip, sha256",
            "gate_collisions": gate_collisions,
            "retention_collisions": ret_collisions,
            "intra_corpus_duplicate_questions_dropped": intra_dupes,
            "gate_hash_count": len(gate_hashes),
            "retention_hash_count": len(ret_hashes),
        },
        "answer_len_stats": {
            "min": min(ans_lens) if ans_lens else 0,
            "max": max(ans_lens) if ans_lens else 0,
            "mean": round(sum(ans_lens) / len(ans_lens), 1) if ans_lens else 0,
            "total_target_tokens": sum(ans_lens),
        },
        "hashes": {
            "prompt_pool_json_sha256": sha256_file(POOL),
            "gate_set_json_sha256": sha256_file(GATE),
            "raw_gen_jsonl_sha256": sha256_file(GEN),
            "training_corpus_jsonl_sha256": sha256_file(TRAIN_OUT),
        },
        "files": {
            "training_corpus": str(TRAIN_OUT.relative_to(ROOT)),
            "prompt_pool": str(POOL.relative_to(ROOT)),
            "raw_generations": str(GEN.relative_to(ROOT)),
            "builder": "scripts/build_s2_traj_corpus.py",
            "prompt_builder": "scripts/build_s2_train_prompts.py",
            "generator": "runs/s2_pilot/run_s2_gen.py",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print("=" * 64)
    print(f"raw_generations={total}  hangs/errs={hangs}  completed={len(completed)}")
    print(f"finish: stop={n_stop} length={n_length}")
    print(f"correct={n_correct}  verify_ok={n_verify}  proj0={n_proj0}")
    print(f"AUDIT-CLEAN-CORRECT={len(clean)}  (yield over raw = {audit_yield:.3f})")
    print(f"leakage: gate_collisions={gate_collisions} retention_collisions={ret_collisions} "
          f"intra_dupes={intra_dupes}")
    print(f"kept_after_dedupe={len(kept)}  target={target}"
          f"{' (FLOOR TRIGGERED)' if floor_triggered else ''}  WRITTEN={len(final)}")
    if ans_lens:
        print(f"answer_len: min={min(ans_lens)} max={max(ans_lens)} "
              f"mean={sum(ans_lens)/len(ans_lens):.1f} total_tok={sum(ans_lens)}")
    print(f"corpus  -> {TRAIN_OUT}")
    print(f"manifest-> {MANIFEST}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
