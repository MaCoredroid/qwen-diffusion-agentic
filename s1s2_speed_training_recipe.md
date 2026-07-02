# S1+S2 Speed-via-Training Recipe (workflow wa5wvsnvi, 2026-07-02)

13 agents, 0 errors; recipes verified against papers/code (CD4LM/DSCD 2601.02236 code-verified; SDTT fallback; FLARE/WSD budgets).

# S1+S2 Speed-via-Training Plan (S3/S4 stubs) — EXECUTABLE SPEC

Scope: covers S1 (budget retrain) + S2 (consistency distillation) in full; S3/S4 as triggered stubs. Sequenced AFTER the live quality-RL pilot on the single 5090. Commit + push each step to origin/main with narrated reasoning (standing workflow rule). All capability evals use the audited scorer: structural-projection-only, ZERO value projection, generated-token audit, forwards/turn >= 1.

---

## PHASE S1 — BUDGET RETRAIN (fix the checkpoint ceiling)

**Rationale.** Every prior run was 200–1000 steps at r=8–16 = 2–11% of FLARE's 9000-step full-FT budget; the block-mode anchor failure (cannot hit GSM8K 0.65 at K=B even with the legacy sampler) converges with "undertrained for block-mode generation." S1 spends a real conversion budget before any distillation. Per the retrain-freely rule, the Run-1 checkpoint is NOT reused as init; retrain from `models/qwen3.5-9b-fastdllm-init`.

### S1.0 Pre-flight VRAM smoke (mandatory, ~10 min)
No peak-VRAM record exists for block-512 two-stream (the 29,149 MiB anchor is block 1024). Run a 3-step smoke of the exact S1 config below, record `torch.cuda.max_memory_allocated()`:
- **Pass:** peak <= 30.5 GiB → launch S1.
- **Optional info run:** same smoke at r=128 (projected +2.0 GiB over r=16's trainable side). If <= 30.5 GiB, bank r=128 as the escalation config.
- **Fail at r=64 (not expected):** fall back r=32, alpha=64, same everything else.

### S1.1 Exact training config
Entry: copy `scripts/run_flare_redesign_run1.sh` → `scripts/run_s1_budget_retrain.sh` (same wrapper chain → `run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh` → `fast-dllm/v2/train_scripts/finetune.py`) with these overrides; everything not listed stays byte-identical to Run 1.

| Knob | Value | Note |
|---|---|---|
| LORA_R / LORA_ALPHA / DROPOUT | **64 / 128 / 0.05** | keeps alpha/r=2 scale of the proven config; r=64 is the banked S1 floor |
| LORA_TARGET_MODULES | q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj | unchanged (attn + GDN) |
| BLOCK_SIZE / TRAIN_BD_SIZE | **512 / 32** | 512 is the comfort zone; 1024+r=64 is edge-plausible but not worth the OOM risk |
| FASTDLLM_GDN_KERNEL | **torch** (explicit override) | Run 1's script defaults to `fla` (line 40) — a defect at higher rank: FLA is +1.3–2.3 GiB and 21–52% slower. Torch is the documented winner. |
| Batch / GRAD_ACCUM | 1 / **2** | batch>1 OOM-blocked; accum is free. Effective batch 2 for gradient noise at higher rank. |
| MAX_STEPS | **2000 optimizer steps** (= 4000 micro-forwards = **10x Run-1 sample-views**, ~0.8 epoch of 5055) | top of the banked 5–10x band |
| LR schedule | **WSD, peak 1e-5**: warmup 0→1e-5 over steps 0–100; stable 1e-5 steps 100–1700; linear decay 1700–2000 to 1e-6 | HF `lr_scheduler_type=warmup_stable_decay`, `lr_scheduler_kwargs={"num_stable_steps":1600,"num_decay_steps":300,"min_lr_ratio":0.1}`. Fallback if transformers version lacks it: `cosine_with_min_lr` (min_lr_ratio 0.1, warmup 100). |
| Data | **reuse `data/flare_redesign_run1_copy_retention_mix`** as-is (builder: `build_flare_redesign_run1_copy_mix.py`), MAX_TRAIN_SAMPLES=5055 | it passed retention AND improved copy-args; no rebuild |
| Loss flags | unchanged: FASTDLLM_FLARE_TWO_STREAM=1, GDN route_i, mask rate U[0.3,0.8], FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE=1 (0.02–0.12), VALUE_SPAN_LOSS_WEIGHT=2.0, VALUE_SPAN_MASK_PROB=1.0, fast_dllm_v2_native, TRUNCATION_SIDE=left | native-format rule |
| Grad ckpt / alloc | ON (use_reentrant:false) / expandable_segments:True | ckpt-off OOMs; do not touch |
| SEED / DATA_SEED | 71101 / 71101 | comparability with Run 1 |
| Checkpoints | save every 250 optimizer steps | intermediate gate probes |
| OUTPUT_DIR | `runs/s1_budget_retrain_r64_qwen35_9b` | |

**Wall-clock:** 4000 micro-steps × ~5.2 s ≈ **5.8 h train** (overnight). Profile at step ~50 per the GPU-util standard: torch path baseline is ~65%; if util drops below that, stop and fix the host-bound defect before continuing.

### S1 GATE (all three required; evals ~3–4 h)
1. **Block-mode anchor (the S1 raison d'être): GSM8K >= 0.65 at K=B via the VALIDATED legacy sampler** (the mutable-remask sampler is disqualified — remasking breaks GDN state discipline). This is the anchor the current checkpoint fails.
2. **GSM8K careful decode >= 0.70** (Run-1 config scored 0.75; base 0.65).
3. **Frozen battery no-regress:** copy-arg heldout >= 41/52 and public >= 55/60 at careful decode, audited scorer.
Also record (not gated): the block-quality curve — GSM8K at K=B, B/2, B/4, B/8 — as the S2 baseline measurement.

**Escalation (one shot):** if the anchor fails at step 2000 AND at ckpts 1000/1500 shows no upward trend, retrain once at r=128/alpha=256 (only if smoke passed) with MAX_STEPS=3000 (~8.7 h).
**Kill S1:** anchor still fails after escalation → the "checkpoint ceiling, not method ceiling" hypothesis is materially weakened; STOP the campaign and re-open the design (S2 needs a sane teacher; do not proceed).

---

## PHASE S2 — CONSISTENCY DISTILLATION (DSCD nested-mask, CD4LM-faithful, adapted)

**Method choice (verified):** primary recipe = **DSCD nested-mask consistency distillation** (CD4LM, arXiv 2601.02236, code-verified byte-identical). Key verified correction to our old framing: the teacher target is **ONE frozen no-grad forward on a lighter-masked nested view — no rollout**. The expensive careful-decode teacher (~6 s / 95 forwards per turn) is needed ONLY for the offline x0 corpus (S2.1), not per training step. Existence proof: GSM8K 77.4→77.6 at 5.18x wall / 3.35x NFE (distilled student + CAD). SDTT-with-cached-targets is the banked fallback if DSCD-under-LoRA null-results (see kill criteria).

### S2.0 Teacher sanity pre-checks (mandatory, ~1 h)
- Teacher = **frozen copy of the S1 adapter** loaded as a second adapter on the shared NF4 backbone (per-step `set_adapter` swap). **NOT `disable_adapter`** — the raw AR base has never denoised mask tokens; verified risk #1 is garbage KL targets dominating 90% of early gradient.
- Verify teacher produces sane logits on masked inputs and measure **teacher top-1 accuracy on held-out value spans** (nested view, partial span revealed). If teacher can't ground arguments here, the KL signal on the cliff mode is confident noise → fix S1 first.
- Extend `validate_flare_two_stream_forward.py` / `validate_gdn_state_snapshot.py`: assert both nested views seed the noisy GDN block state from the SAME clean block-boundary state, state read-only during denoise, advanced once at commit.

### S2.1 Offline x0 corpus (on-policy anchor, 50/50 hedge)
- Generate with the **S1 checkpoint** at careful decode (6.049 s/turn, 95 forwards/turn measured) over the agentic training prompt pool; **audit-filter** every turn (generated-token audit, exact-args, native format).
- Target **~5,000 audit-clean turns**. At the measured 54% yield: ~9,300 raw turns × ~6 s ≈ **15.5 h; budget 16–17 h**. Batch prompts to keep the 5090 fed (GPU-util standard).
- **Yield floor:** if audit-clean yield < 40%, abandon self-gen x0 and run ground-truth-only (that IS the paper-faithful CD4LM configuration — the self-anchor is our extension, not load-bearing).
- Final S2 mix: ~5,000 self-gen clean turns + the 5,055-sample Run-1 ground-truth/retention mix (50/50, ~10K samples). New builder `scripts/build_s2_dscd_corpus.py`; corpus is teacher-checkpoint-bound — regenerate, never reuse, if the decoder or format changes.

### S2.2 Student loss (exact formula; new flag `FASTDLLM_FLARE_NESTED_DISTILL=1`)
Replace the two COMPLEMENTARY L_diff views with a NESTED pair; keep L_AR clean-stream and GDN boundary seeding unchanged.

- Masking (answer region only; prompt clean): `r_S ~ U(0.40, 0.90)`; `u ~ U(0.30, 0.70)`; `r_T = clip(r_S·u, 0.10, 0.60)`; `M_T = RandomSubset(M_S, n_T)`, `n_T = min(floor(L_y·r_T), n_S)`. Short-span protections (code convention): L<20 → r_S<=0.50; L<=10 → exactly 2 student / 1 teacher masks; min student masks = max(2, ceil(0.10·L)).
- **Block-causality adaptation (GDN-specific, ours):** restrict nesting **within-block** — the teacher view may reveal extra tokens only inside the student's current block or already-committed blocks, never later blocks (avoids off-policy targets the student can never condition on under block-causal decode).
- **Value-span adaptation:** student keeps VALUE_SPAN_MASK_PROB=1.0; the teacher view reveals a partial prefix of each value span — trains "complete a partially committed argument value," which is exactly the tpf-1.23 cliff mode.
- Loss on M_S positions only, answer-region boundaries computed from the native chat_template token structure (NOT string re-tokenization):

```
L_S2 = L_AR + λ(g)·τ²·KL( softmax(z_teacher/τ) ‖ softmax(z_student/τ) )
            + (1−λ(g))·CE(z_student, x0)
```
τ = 2.0 (math config); both distill terms clamped at 5.0; NaN-skip; grad clip 1.0; **softmax/KL in fp32** (bf16 over ~150K vocab at τ=2 is noisy); per-token aggregation (code convention). λ(g): hold 0.9 for first 10% of the round's planned steps, cosine to 0.5 at round end — **schedule λ over each round's actual length**, not a notional 20K horizon.

### S2.3 Rounds + K-reduction schedule
Optimizer: unchanged S1 QLoRA envelope (r=64 student adapter, LR 1e-5, grad clip 1.0, batch 1, accum 2, block 512, torch kernel). Step cost ≈ 6.5–7 s (one extra no-grad teacher forward + adapter swap); profile step 50 — if util is host-bound from adapter swapping, batch student+teacher inputs through the shared quantized backbone.

- **Round 1** (target: K=B → **K=B/2** held): 4,000 micro-steps ≈ **7.5 h**, λ scheduled over 4,000. Init student = S1 adapter copy; teacher = frozen S1 adapter.
- **Round 2** (target: **K=B/4** held): teacher ← frozen copy of the Round-1-best student adapter (SDTT-style hard swap — copy adapters, NEVER merge-requantize into NF4); fresh λ schedule; 4,000 micro-steps ≈ **7.5 h**.
- **Round 3 (optional, only if Rounds 1–2 both gated positive and trend continues):** ~12,000 micro-steps ≈ 22 h toward the paper-faithful ~75K-sample-view dose, targeting K=B/8. A null at the 8K-step dose does NOT falsify the recipe (it is ~1/3 the paper's math budget) — but it does end OUR budget for it.
- **Decode half (CAD), after any passing round, zero training cost:** confidence-adaptive commit inside sequential blocks — commit k = clip(#{c_i >= γ}, 1, k_max) per forward, never remask, native stop-ids only (not the repo's LLaDA list), block stop-ids while inside an open tool-call span instead of the global progress ratio. **Sweep γ ∈ {0.85, 0.90, 0.95, 0.99}, k_max = 2 first** (not 4). Sanity: γ=1.0/k=1 must reproduce the fixed-K baseline exactly before sweeping.

### S2 GATE (per round; evals ~3–4 h/round)
1. **Block-quality curve:** largest K-reduction with GSM8K >= 0.65 held (validated sampler). Round-1 pass = **K=B/2 (>=2x tokens/forward at anchor-held)**; Round-2 pass = K=B/4.
2. **Cliff movement (the crux):** copy-arg exact-match must be nonzero at value_tpf >= 1.5 (Run-1: 41/41 at 1.00, 0/41 at 1.23). RAW/constrained decode only — no value projection.
3. **Retention:** GSM8K careful >= 0.70 (max −0.05 vs S1); frozen battery within noise of S1 (copy-arg heldout >= S1 − 2).
Promotion per the discipline: raw or constrained model-only gains only; CAD wall-clock counts only if NFE reduction shows up as wall-clock at high GPU util.

---

## PHASE S3 STUB — joint modeling for the C(Y|X)>0 residual
**Trigger:** S2 passes >=2x but plateaus below the 2.5–5x band AND error analysis shows residual failures concentrated on correlated spans (paired values, cross-call ids, reasoning) — the architecturally non-parallelizable class. **Approach:** Di4C latent-mixture or CoDD frozen-backbone joint prior, stacked ON the S2 student. Expectation ceiling: ~2x additional; do not design until the S2 error decomposition exists.

## PHASE S4 STUB — speed-RL
**Trigger:** S2 (± S3) holds >=2x raw at anchor and the remaining gap looks like decode-policy robustness rather than modeling. **Approach:** graded reward = audited exact-args + anchor quality under aggressive-step decode, reusing the existing pilot machinery (`rl_pilot_countdown.py`); CD4LM itself names on-policy RL as the only route past the teacher ceiling. Scheduling contends with the Lumo flywheel + quality-RL pilot for the GPU.

---

## RISKS + KILL CRITERIA

| Risk | Phase | Mitigation | Kill criterion |
|---|---|---|---|
| Anchor is a method (not checkpoint) ceiling | S1 | 10x budget + WSD + r=64 | Anchor fails after r=128/3000-step escalation → STOP campaign, reassess |
| r=64/128 VRAM at block 512 unmeasured | S1 | mandatory 3-step smoke | smoke > 30.5 GiB at r=32 fallback too → redesign envelope |
| **LoRA capacity vs full-FT DSCD** (no published LoRA evidence) | S2 | treat Round 1 as feasibility probe, not promotion candidate | Round 1 shows NO block-quality-curve movement at B/2 AND cliff unmoved → kill DSCD; fall back to cached-SDTT (2-step teacher rollout, K=256 top-k targets, reverse-KL caveat) for one probe round; if that also nulls, S2 dead → S4 |
| Teacher garbage targets (mask-naive base / cliff-mode teacher) | S2 | frozen-adapter teacher (never disable_adapter); S2.0 top-1 value-span check | teacher value-span top-1 < ~60% → return to S1 escalation |
| Teacher-cost blowup | S2 | DSCD needs no rollouts; corpus capped 17 h; yield floor 40% → ground-truth-only x0 | corpus overruns 2x budget → ground-truth-only, continue |
| Self-anchor pseudo-labeling reinforces unaudited defects | S2 | 50/50 ground-truth hedge; audit filter; λ makes CE strongest late — inspect samples at round end | qualitative degeneration in round-end samples → drop self-gen half, retrain (cheap) |
| KL-heavy instability at batch-1 (λ=0.9 early) | S2 | clip 1.0, clamp 5.0, fp32 KL, monitor grad-norm | NaN/spikes persist past 200 steps after one LR halving → kill run, restart at LR 5e-6 |
| GDN-specific unknowns (state pollution, nested-view seeding, off-policy later-block reveals) | S2 | within-block nesting; extended bit-exact validators BEFORE round 1; no remasking anywhere | validator mismatch unfixable in a day → hold S2, debug |
| Entrenching the teacher ceiling | S2 | expected: parity + speed, not quality gains; gates are raw exact-args | — (by design) |
| Host-bound stalls (adapter swap, CAD python loop) | S1/S2 | profile step 50; batch through shared backbone; on-device commit selection | GPU-util standard: never accept low util |

---

## TOTAL CAMPAIGN WALL-CLOCK + DECISION POINTS (one 5090)

| Item | GPU hours |
|---|---|
| S1 smoke + train + gate evals | ~0.2 + 5.8 + 4 ≈ **10 h** |
| (S1 escalation, if triggered) | +9–13 h |
| S2.0 sanity + validators | ~2 h |
| S2.1 corpus generation | ~16–17 h |
| S2 Round 1 + evals | ~7.5 + 4 ≈ 11.5 h |
| S2 Round 2 + evals | ~11.5 h |
| CAD sweep | ~2 h |
| **Core S1+S2 total** | **~53 h ≈ 5–6 calendar days** |
| (S2 Round 3, optional) | +26 h |

**Stop/decision points for the monitor/user:** (1) after S1 smoke (VRAM verdict, r=128 bankability); (2) after S1 gate (anchor pass/fail — hard campaign gate); (3) after corpus yield measurement (self-gen vs ground-truth-only); (4) after S2.0 teacher sanity; (5) after each S2 round gate (promote / round 3 / fall back to cached-SDTT / kill to S4); (6) after CAD sweep (constrained-lane serving decision under promotion discipline).

Key paths: `/home/mark/qwen_diffusion/scripts/run_flare_redesign_run1.sh` (template for `run_s1_budget_retrain.sh`), `/home/mark/qwen_diffusion/data/flare_redesign_run1_copy_retention_mix` (S1 data, reused), `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init` (S1 init), `/home/mark/qwen_diffusion/scripts/validate_flare_two_stream_forward.py` + `validate_gdn_state_snapshot.py` (S2 validators), `/home/mark/qwen_diffusion/redesign_plan_and_design_notes.md` §10 (banked program this plan instantiates).