# RUNG W-1c — OWED LiveCert battery (gate-ON twin) — VERDICT: STOP

Object = twin@plain (`qwen3.5-9b-fastdllm-mswe2-S-twinK1`), FLARE hybrid_clean, mask 248077,
max_model_len 32768, gmu 0.74, max_num_seqs 4, `VLLM_FLARE_BIDIR_PROBE=1` (the certified
full-reveal envelope). ONLY delta gate-ON vs gate-OFF = `VLLM_FASTDLLM_W1_DRAFT_VERIFY`.
Three bounded boots (gate-OFF ref, gate-ON, gate-OFF determinism control); server DOWN + GPU
idle (385 MiB, 0%) at exit; ~0.5 GPU-h of the ~4 budget. Harnesses + raw jsonl + logs in
`runs/w1c_livecert/`; consolidated `results.json`.

## The four owed items

### (a) LIVE PERTURBED FA BATTERY — registered bar MET (but see the DECISIVE control)
12 real serving requests (3 canonical spans × {off-by-one, single-sub, near-dup, whitespace}),
each seeding BOTH the canonical span AND a class-perturbed distractor; write_file `content` arg
must reproduce the canonical verbatim; temp-0; gate-ON vs gate-OFF byte-diff.

| bar | required | measured |
|---|---|---|
| deploy-class (single-sub + near-dup) full-span false-accepts | **= 0** | **0** ✅ |
| gate-ON byte-identical to gate-OFF K=1 | — | **12/12** |
| gate-ON emits canonical exactly | — | **12/12** |

The registered battery PASSES. **Caveat that turned decisive:** these cases seed a distractor
alongside the canonical, so the guard's common-prefix rule is exercised — they do NOT exercise
the **single-source copy-value** case, which (b) found DOES corrupt live.

### (b) W-0 SPAN-CORPUS LIVE THROUGHPUT — speedup real but modest; **byte-safety VIOLATED**
10-turn copy corpus (6 copy-heavy write_file, 4 read_file path), temp-0, per-request engine
counters (`w1[spans/toks/vfwd/rej]`, model_forwards, generated_tokens).

| metric | gate-OFF K=1 | gate-ON W1 | vs CPU-cert |
|---|---|---|---|
| copy-heavy tok/fwd | 1.19 | **2.80 (2.35×)** | cert projected **14.41** — live is ~5× below |
| blended tok/fwd | 1.24 | **2.43 (1.95×)** | cert blended 1.863× (tok/fwd basis holds) |
| copy-heavy ms/committed-tok | 24.35 | **15.42 (0.63×)** | — |
| path (non-copy) ms/committed-tok | 20.68 | **26.66 (1.29× SLOWER)** | reject-tax |
| reject-tax share (copy / path) | — | **0.859 / 1.0** | — |

The strict whole-span full-reveal verify **rejects ~86% of copy drafts on-policy** → the CPU-cert
14.41 copy-tok/fwd (teacher-forced ā=0.9913) does NOT reproduce live; live copy speedup is ~2.35×.
Non-copy turns net slower (wasted verify forwards). W-1b reject-tax lever (per-span propose gating
/ accept-matching-prefix) is a decode-behavior change, **not a one-knob env change — NOT enacted.**

### (c) FORMAL A6 — tool-call args exact on the sample; free-text drifts
5 real banked agentic turns, temp-0, gate-ON vs gate-OFF byte-diff. **All 5 FIRED** (real agentic
turns always carry copy mass). Tool-call turns **2/2 byte-identical + exact args**; free-text
reasoning turns diverge (1/3 content-diverged, reasoning-token drift on 2/3). "unfired byte-identical"
demonstrated on the **4/4 path turns** in (b) (0 firing → byte-identical). But (b) proves fired
copy-value turns CAN corrupt → "fired turns exact-args valid" is NOT a live guarantee.

### (d) 6-EPISODE C46 A/B — NOT RUN (disciplined stop)
The byte-lossless precondition (structural FA=0 / K=1-equal on copy mass) FAILED at (b). A
throughput A/B of a value-corrupting fast path is not a meaningful win and would inject corrupted
edits into episodes. Plan built (`d_plan.json`, 6 django ctx_overflow ids, seed-matched to banked
gate-OFF) and runner ready (`run_d.sh`); re-owed once the seam is fixed.

## THE DECISIVE CONTROL — determinism, 5× per snippet, gate-OFF vs gate-ON

| # | snippet | gate-OFF K=1 | gate-ON W1 |
|---|---|---|---|
| 0 | utils/retry.py | det, exact 5/5 | det, exact 5/5 |
| 1 | models/user.py | det, exact 5/5 | det, exact 5/5 |
| 2 | handlers/webhook.py | det, exact 5/5 | det, exact 5/5 |
| 3 | migrations/0042.sql | det, exact 5/5 | **NON-det, 4/5** |
| 4 | k8s/deployment.yaml | det, exact 5/5 | **NON-det, 4/5** |
| 5 | lib/parse.py | det, exact 5/5 | **det, 0/5 (100% corrupt)** |
| | **total exact** | **30/30** | **23/30** |

gate-OFF K=1 is **bit-reproducible and exact on all 6**. gate-ON W-1 firing **non-deterministically
corrupts copied tool-call VALUES on 3/6 snippets** (~23% of copy-heavy generations). idx5:
`value = value[:-1] + '+00:00'` → `'+00:` (a 31-token span was ACCEPTED and committed;
`projected_value_tokens_exact` tripwire = 0, MISSED it).

**Conclusion:** the CPU-certified structural-FA=0 does NOT transfer to the live engine seam
(`_hc_stage_verify`/`_hc_verify_read` in `qwen3_5_flare.py`). The wide read-only-denoise verify
canvas yields a temp-0 argmax that (a) is non-deterministic run-to-run and (b) differs from the
serial K=1 argmax (the W-0 (b) full-reveal leak, on-policy), so accepted spans diverge from K=1
and corrupt values. The value-projection tripwire does not catch it. This is exactly the byte-safety
the guard is supposed to guarantee — and it is violated live.

## VERDICT = STOP
Not dispatch-ready for the full C46-under-new-envelope run. The registered FA battery (a) passed,
but the W-0 span-corpus (b) + determinism control surface a reproducible live value corruption that
gate-OFF K=1 never exhibits. Fix the seam faithfulness (make the live verify commit only the serial-
K=1-equal prefix, deterministically) and re-run the LiveCert before any gate-ON dispatch.
