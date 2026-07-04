# REPRODUCE_V2 — Full Recipe for the Dual-Mode (AR+Diffusion) Agentic Qwen-9B

Reproduces the system that scored **130/247 aggregate exact-args (beats stock Qwen3.5-9B AR-guided at 124/247)** on the
audited two-slice battery, at zero conversion tax, built entirely with LoRA on frozen weights, ~10–15 GPU-hours on one
RTX 5090 (32GB). Written by the monitor from the run record; flare's pass adds exact per-stage git hashes
(sampler-pinning rule: every gate records the sampler function + git hash it invoked).

## 0. Base + environment
- Base: Qwen3.5-9B in the Fast-dLLM v2 scaffold → `models/qwen3.5-9b-fastdllm-init` (GDN hybrid: 3-in-4 GatedDeltaNet
  linear-attention layers + 1-in-4 full attention; vocab 248,320). Weights FROZEN throughout; NF4-quantized for
  training (QLoRA), bf16 for serving.
- Env: `.venv-fastdllm` (torch 2.12.1+cu130, triton 3.7.1); GDN kernel = **torch** (FLA is slower at batch-1 here).
- Format rule: the model's NATIVE chat_template / qwen3_xml tool format in ALL stages (gen/train/eval/decoding).

## 1. Stage 1 — Two-stream conversion ("B@1000")
- Objective (the core): ONE forward computes two losses over shared weights —
  L_AR: standard causal next-token CE on the clean stream (byte-identical to the stock AR forward → AR preserved by
  construction); L_diff: complementary noisy views — blocks of positions replaced by [MASK], predict originals,
  **+1-shifted** labels (`noisy_logits[:, :-1]` vs `labels[:, 1:]`).
- GDN discipline (novel, mandatory): noisy block SEEDED from the clean prefix's recurrent state snapshot at the block
  boundary; state READ-ONLY during denoise; advance exactly once on commit; block size aligned to GDN chunk (64).
  Validators: `validate_flare_two_stream_forward.py` (clean stream must be bit-exact vs AR), `validate_gdn_state_snapshot.py`.
- Config: LoRA r=8 (q/k/v/o), 1000 steps, batch 1, block 1024. Data: GSM8K/MBPP-style retention mix.
- Gate: careful-decode GSM8K ≥ 0.65 (validated fullctx/fresh-block/mask-ban sampler ONLY).

## 2. Stage 2 — Copy-grounding ("Run-1")
- Fresh LoRA from init (NOT stacked): r=16, alpha=32, cosine, ~400–600 steps, block 512, seed 71101.
- Two-stream + additions: copy-from-context supervision (value spans masked at VALUE_SPAN_MASK_PROB=1.0, weight 2.0,
  left-to-right within span), conditional-entropy-adaptive schedule (tool-sensitive planner), clipped noise U[0.3,0.8].
- Data: 5055 samples = ~49% tool-call/copy synthetic + ~51% retention (builder `build_flare_redesign_run1_copy_mix.py`).
- ⚠ Scaling law learned the hard way: steps beyond ~400–600 on THIS mix ERODE capability (measured: 2000 steps → 0.45).
  Scale data diversity with budget, never steps alone.
- Gates: GSM8K careful ≥ 0.70 (got 0.70–0.75); copy-arg exactness improved (19→41/52 heldout).
- Entry: `scripts/run_flare_redesign_run1.sh`.

## 3. Stage 3 — Multi-turn GRPO ("v2")
- From the Run-1 adapter. `scripts/run_rl_multiturn_grpo_v2_pilot.sh`: 300 steps, group 4, lr 2e-6, temp 1.0.
- Pool: ~300 leak-checked public multi-turn episodes (ToolACE-derived + synth), MIXED difficulty (easy episodes are a
  load-bearing implicit anchor — an all-hard pool caused KL drift and retention collapse).
- Safety kit (all required): KL-to-base coef 0.05 (0.15 over-anchors and regresses quality); rolling KL early-stop
  (last-50 mean > 0.05 → halt+gate); retention probe every 50 steps; grammar in rollouts for STRUCTURE only; policy
  loss on value/free tokens ONLY (grammar-forced tokens masked out); graded partial-credit reward.
- Gates: retention GSM8K ≥ 0.65 (got 0.65) → matched-20 exact-args (34 → 44/63).

## 4. Serving — hybrid-clean decode (no training)
- Per turn: grammar FSM commits every TRULY-FORCED token (exactly one legal continuation) in bulk with ZERO model
  forwards; every value/non-forced token decoded sequentially (chain rule — measured three ways: values do not survive
  any parallelism). No value projection ever.
- Result: 47/63 matched-20 (+4/−1 paired vs careful), 63/63 valid+seq, 83/184 never-train, 1.36 tok/fwd, 1.71× wall
  vs careful. Reference impl: `scripts/eval_flare_northstar_hybrid_clean.py`.

## 5. Acceptance test — the audit battery (run ALL; any failure voids the claim)
- Generated-token audit: `projected_value_tokens_exact = 0`; `zero_forward_rows = 0`; force counters = 0.
- forwards/turn ≥ 1 per model-chosen token (tpf < 1 ⇒ projection contamination).
- Matched baselines on ONE harness scale; every gate report pins sampler function + git hash.
- Slices: matched-20 (63 turns) + never-train BFCL/API-Bank (184 turns), leak-checked manifests.

## Known limits (honest)
Strict-content parallelism is chain-rule-bound (~1.4–2.5× ceiling via grammar-Amdahl); reasoning content holds 4
tok/fwd natively; wall-clock vs engine-served AR requires the engine port (see `p2_serving_reuse_plan.md` /
`engine_build_status.md`). RL plateaus at v2 across three successor designs — the durable core is stages 1–2 + hybrid
serving. Lab-scale version: fold the two-stream loss into pretraining (MTP-head-style co-training).
