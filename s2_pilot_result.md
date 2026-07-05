# S2 pilot — EVAL adjudication (spec `s2_pilot_design.md` @ 9ce9445)

**Date:** 2026-07-05 · **Repo HEAD at gate:** `75ba599` (battery) → `05d5297` (GATE commit) ·
**Author:** S2 GATE adjudication pass. **This document SUPERSEDES the earlier BLOCKED verdict** (the
pilot had not run then; it has now run end-to-end — corpus built, `A_S2` trained, full battery
executed, all rows raw + audited).

---

## VERDICT: **PILOT KILL — the K-factor is a wall (reasoning-span K stays ≈1.0 at held exactness).**

Adjudicated on the design's **§0 verdict axis** — *"KILL ⇒ reasoning-span K stays ≤1 at held
exactness ⇒ the 5× claim is retired."* The measured outcome is exactly that: peak committed
**tok/fwd = 1.053** (A_S2 K=2, γ=0.90), **decisively below the 2.0 PASS bar** and below even the 1.5
INCONCLUSIVE floor, while accuracy is essentially held (net-loss ≤ 1, McNemar p = 1.000). The
parallel reasoning lane **does not open**: even the most-trained checkpoint at the most-permissive γ
commits ≥2 contiguous high-confidence tokens on only **5.3 %** of forwards.

**Honest classification of the failure mode (adjudicate EXACTLY):**
- The specific **§9 KILL-a *trigger* (net-loss > 2, accuracy-collapse) did NOT fire** — net-loss = 1,
  p = 1.0. This is **not** the feared SDTT-null "parallelism breaks exactness" mode.
- It is the **other** failure mode: **K never engages.** tok/fwd stays ≈1 *because* the model
  refuses to jointly commit adjacent reasoning tokens — so accuracy trivially holds (you are still
  essentially K=1 everywhere). Per §0/§8 the PILOT PASS *requires* the K-gate PASS (`tok/fwd ≥ 2.0`),
  which is not met, so the pilot **fails its primary bet** and the 5× claim is retired on evidence.
- The safety gates all **held** (retention 13/20 PASS, tool-call 0-lost PASS, audits clean), so the
  pilot did **not** damage a certified/retained capability — the bet dies purely on the K-factor,
  which is the cleanest possible KILL: the wall is the architecture's reasoning-token joint
  commitment, not our training hurting the model.

**Consequence (§8 PILOT KILL / §0):** retire the 5×-vs-AR claim; the campaign reverts to the honest
**"0.36× vs AR-cudagraph today, L2-bound ~2× ceiling, no path to 5×"** on this GDN-hybrid stack. L3
is **not funded.**

---

## The instrument (design §3) — authored + validated before any K=2 row

`scripts/eval_flare_freetext_cad.py` — sha256 `e12364e76e62059fe0ca4aeb1fcc8f6c4ebe34197dda25edbf58fce3e6104b87`
(pinned this run). Entropy-gated adaptive-K CAD free-text sampler; a monkeypatch of the promoted
engine `_hybrid_clean_step`. At `k_max=1 / γ=1.0` it is a **pure pass-through** to the single-[MASK]
probe; at `k_max=2` it stages k trailing masks, reads the k `+1`-shifted probe logits, and commits
the **leading contiguous run** with `c_i ≥ γ` (clip [1, k_max]), **numeric-blocked** (a sub-γ position
blocks the run → forces K=1), **native EOS-stop mid-run**, **never remask** (GDN state discipline / FR13
cache preserved).

**R1 byte-exact certificate — PASS (mandatory §3 gate, run before any K=2 row).** `k=1 / γ=1.0` on
base `rlv2-vllm-bf16`, seed 20260701, reproduced the anchor **26/30 strict · 0.8618 tok/fwd ·
gen_text 30/30 identical · (n_gen, denoise_fwd) = (3267, 3791) identical** vs
`runs/l0l2_final_head_verify/…engine_gsm8k_clean_head.jsonl`. The instrument is validated as a pure
extension of the promoted path.

---

## Object under test + provenance (design §1, §12)

- **A_S2** = `runs/s2_pilot/Apilot_step400_seed90101` — trajectory-consistency LoRA (`r=16/α=32`,
  attn+GDN targets), trained on `M_{t+1}-merged`. **Root adapter == checkpoint-120**, the
  **KILL-retention halt state** (the in-training KL guard early-stopped training there, §5 below) —
  i.e. the **most-trained / most-favorable state for the bet.** vLLM export
  `models/qwen3.5-9b-fastdllm-mtplus1-Apilot-vllm-bf16` (lora_merge_count=152, scale 2.0).
- **Base engine (control / anchor path):** `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16`.
- **Gate set:** `runs/l1_census/gsm8k_prompts_clean.json` (30, the 26/30 anchor set). **Seed 20260701.**
- **Corpus:** sha256 `8da67214…8e39ff`; **leakage-dedupe = 0 gate / 0 retention collisions** (§4).
- **KILL-0 base sanity:** `mtplus1-merged` `mask_token_id=248077`, `bd_size=32` ✓; R1 free-text
  sanity 26/30 ✓ → base is correct, training was licensed.

---

## Full battery — every row RAW (30-prompt clean set, seed 20260701)

`gate_summary.json` (`runs/s2_pilot/gate/`) is the machine record; every tok/fwd is EOS-trimmed
emitted, `denoise_forwards` counted.

| row | base / adapter | decode | strict | tok/fwd | K=2-commit % | proj |
|---|---|---|--:|--:|--:|--:|
| anchor K=1 (engine) | rlv2 export | K=1 | 26/30 | 0.862 | — | 0 |
| **R1** CAD k=1 γ1.0 (base, SANITY) | rlv2 export | CAD pass-through | 26/30 | 0.862 | 0.0 % | 0 |
| **(CTRL-decode)** K=2 γ0.90 | merged / **none** | CAD k_max=2 | 26/30 | **1.015** | 1.5 % | 0 |
| CTRL-decode K=2 γ0.95 | merged / none | CAD k_max=2 | 26/30 | 1.003 | 0.3 % | 0 |
| CTRL-decode K=2 γ0.99 | merged / none | CAD k_max=2 | **27/30** | 1.000 | 0.0 % | 0 |
| **(CTRL-K1)** A_S2 K=1 γ1.0 | merged / A_S2 | K=1 | 26/30 | 0.863 | 0.0 % | 0 |
| **(PRIMARY)** A_S2 K=2 γ0.90 | merged / A_S2 | CAD k_max=2 | 25/30 | **1.053** | **5.3 %** | 0 |
| A_S2 K=2 γ0.95 | merged / A_S2 | CAD k_max=2 | 26/30 | 1.014 | 1.4 % | 0 |
| A_S2 K=2 γ0.99 | merged / A_S2 | CAD k_max=2 | 26/30 | 1.000 | 0.0 % | 0 |

**McNemar (A_S2 K=2 vs A_S2 K=1, paired 30):** γ0.90 → b=1, c=0, **net-loss = 1, p = 1.000**;
γ0.95 → 0/0, net-loss 0, p = 1.0; γ0.99 → 0/0, p = 1.0. **KILL-a (net-loss > 2) NOT tripped at any γ.**

**The measured K-curve (the whole story in one line):** as γ relaxes 0.99 → 0.95 → 0.90, K=2-commit
climbs 0.0 % → 1.4 % → 5.3 % and tok/fwd 1.000 → 1.014 → 1.053 — a **real but tiny** propensity that
saturates ≈1.05 and never approaches 2.0. There is no γ at which the lane opens.

---

## The other three gates (§6 b/c/d) — all PASS, but they cannot rescue a failed primary

**(b) Retention N=20 — PASS (13/20 = anchor).** `full_context_sample_one` pinned, base
`mtplus1-merged`, adapter A_S2: half0 **8/10** + half1 **5/10** = **13/20 strict** (anchor 13/20,
PASS ≥ 13/20). GPU util 98.6 % on this row.
- **Honest tension flagged (design §5 / §9):** the **in-training rolling KL-to-base proxy tripped —
  0.0699 > 0.05 at step 120** (`kl_to_base.jsonl`, `kill_retention_tripped: true`), early-stopping
  training at global_step 120 of 400 planned (epoch 0.24). This is **why A_S2 == checkpoint-120.** The
  §5 KL guard is an in-training *early-stop* mechanism; the authoritative §8b retention adjudicator is
  the **behavioral N=20 gate, which HELD at the 13/20 anchor** on the halt-state checkpoint. So the
  proxy fired (its protective job) and the quantity it proxies for is intact — I record this as a
  flagged tension, **not** a tripped KILL-retention (the N=20 disjunct is 13/20, not ≤ 11/20).

**(c) Tool-call spot-10 — PASS (0 lost vs C0).** `eval_flare_northstar_hybrid_clean`, matched-20
first 3 episodes = 10 turns, hybrid-clean (K=1 FSM values, adaptive-K OFF on this path): A_S2
**exact_args 9/10**, C0 (base, no adapter) **exact_args 9/10** → **0 lost vs C0.** Per-turn forwards
byte-identical (519 fwd / 710 tok on both; the reasoning LoRA left the FSM tool-call path untouched,
as designed). The one non-exact arg is shared with C0 → not an A_S2 regression.

**(d) Audits — CLEAN (KILL-3 not tripped).** On **every** CAD row and both tool-call rows:
`value_projection_events == 0`, `zero_forward_rows == 0`, `parallel_commit_forced_tokens_counter == 0`,
`wave{1,2}_* == 0`, `verification_mode == no_projection_events`. **Every tok/fwd number above is
valid** (uncontaminated by phantom projection).

---

## Training delta vs CTRL-decode (§7 promotion discipline)

Training was **directionally positive but immaterial.** At held accuracy (γ0.95, both 26/30) A_S2
(1.014 tpf) > CTRL-decode (1.003 tpf); at γ0.90 A_S2 K=2-commit 5.3 % vs CTRL-decode 1.5 % (**~3.5×**
more joint commits) for +0.038 tpf. So the LoRA **did** move joint-commit propensity in the intended
direction — but **the absolute ceiling is ~1.05, an order of magnitude short of 2.0**, and
**CTRL-decode alone peaks at only 1.015**, so *neither* lever (decode policy nor training) approaches
the bar. This is not a "decode-only win" (CTRL-decode also fails) and not a "training win" (delta is
noise-scale against the 2.0 target). **Both confirm the same wall.**

---

## Adjudication against the pre-registered gates (§8, verbatim)

| # | measurement | anchor | PASS bar | measured | verdict |
|---|---|---|---|---|---|
| a | K-gate: A_S2 K=2 tok/fwd **and** accuracy | 0.862 · 26/30 | `tok/fwd ≥ 2.0` ∧ net-loss ≤ 2 ∧ p ≥ 0.05 | **1.053** tpf · net-loss 1 · p 1.0 | **FAIL** (tpf ≪ 2.0; KILL-a trigger not fired) |
| b | GSM8K retention N=20 | 13/20 | ≥ 13/20 | **13/20** (KL proxy tripped, flagged) | **PASS** |
| c | tool-call spot-10 vs C0 | C0-10 | 0 lost | **0 lost** (9/10 == 9/10) | **PASS** |
| d | value-projection audits | 0 | all counters 0 | all 0 | **CLEAN** |
| — | training delta vs CTRL-decode | — | A_S2 K=2 > CTRL at held acc | +0.011 tpf @ 26/30 (tiny) | positive, immaterial |

**PILOT PASS requires a∧b∧c PASS ∧ d clean ∧ delta positive.** Gate **(a) FAILS** ⇒ **NOT a PILOT
PASS.** Per §0's verdict axis (K ≤ 1 at held exactness) the primary bet is a **KILL of the 5×
claim** — reasoning-span K is a wall on this architecture. The §9 KILL-a *net-loss* trigger did not
fire (the failure is K-non-engagement, not exactness loss), and no safety gate tripped, so this is the
**cleanest form of the KILL: the lane simply never opens.**

---

## Why (design §11, corroborated)

The measured wall matches the honest adverse prior, not the hope. `l1` measured **top-1 conditional
≈ 0.238** on model-chosen reasoning tokens — high-entropy, the opposite of the `C≈0` copy spans
ParallelBench says are trainable-parallel-safe. The pilot's one structural reason for optimism was
that reasoning CoT carries a low-entropy *connective* fraction ("= ", "So the answer is ", operators)
that trajectory-CE could push to joint-commit. Training **did** move that fraction (5.3 % K=2 commits
at γ0.90, ~3.5× over decode-only) — the mechanism is **real but thin**: the numeric/derived positions
dominate the span, so the *average* never leaves ≈1. The GDN bidirectional-within-block copy circuit
does not refuse outright (K=2 is nonzero, exactness held), it just **has too little low-entropy mass
to average to 2.0.** ~1 GPU-day bought the answer: the K-factor is the 5× wall, and it is an
entropy/architecture wall, not a training-dose wall — consistent with retrain-freely at {300,500}
being unable to rescue a factor-of-two miss (do **not** extend past the 600 erosion cap to manufacture
a pass, §5/§9).

---

## Provenance (§12)

Battery: `runs/s2_pilot/gate/gate_summary.json` (HEAD `75ba599`). CAD sampler sha
`e12364e7…6104b87`; R1 byte-exact certificate PASS. Retention:
`runs/s2_pilot/gate/retention_half{0,1}/summary.json`. Tool-call:
`runs/s2_pilot/gate/toolcall_{a_s2,c0}/summary.json`; audits
`runs/s2_pilot/gate/audit_toolcall_{a_s2,c0}.json`. Training KL trace:
`runs/s2_pilot/Apilot_step400_seed90101/kl_to_base.jsonl` (step-120 trip); trainer halt
`…/trainer_state.json` (global_step 120 / max 400). Corpus manifest sha `8da67214…8e39ff`,
leakage-dedupe 0/0. GPU idle after gate (2.2 GB baseline); no stray processes.
</content>
