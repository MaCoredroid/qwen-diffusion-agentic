# RL training-data plan — ADOPTED (ultracode workflow wh6mqv02n, 2026-06-30, monitor-red-teamed)

**For the on-policy RL phase (diffu-GRPO/GDPO, decoder-in-the-loop, dual-term loss). PUBLIC verifiable datasets ONLY
(in-house Lumo pack stays held-out eval). Permissive-license-for-training enforced. All picks verified real +
downloadable + non-reference-text reward. Full research output: tasks/wh6mqv02n.output.**

## TIER A — de-risk the RL machinery (grammar+verifier+GRPO+QLoRA loop). DO NOT promote on these.
| # | dataset | link | license | reward |
| A4 | 4×4 Sudoku (d1 CSVs) | github.com/dllm-reasoning/d1 | Apache-2.0 | per-cell exact-match over empty cells vs unique solution → fraction |
| A2 | Countdown (regen via reasoning-gym / TinyZero, license-clean) | github.com/open-thought/reasoning-gym | Apache-2.0 | ast-SAFE eval of eqn, each number used once, abs(res−target)<1e-5 → 1.0/0.1/0.0 |
| A3 | GSM8K (main) | hf.co/datasets/openai/gsm8k | MIT | post-#### integer, normalize, numeric exact-match (SATURATED — small gains, don't over-read) |
| A1 | reasoning-gym (countdown+sudoku+arith generators) | pip install reasoning-gym (2505.24760) | Apache-2.0 | `score_answer→[0,1]` algorithmic; use STRICT score==1.0 (NOT lenient cascade) |
Why: A4 = simplest fixed-shape grammar rehearsal; A2 = arithmetic grammar w/ real RL headroom (+26% in d1) — the
strongest grammar-decoder rehearsal; A3 = fast numeric sanity.

## TIER B — tool-call / function-call argument grounding (THE REAL TARGET)
| # | dataset | link | license | reward | role |
| B1 PRIMARY | **xLAM-function-calling-60k (APIGen)** | hf.co/datasets/Salesforce/xlam-function-calling-60k (**GATED**: accept terms + HF token) | **CC-BY-4.0** (dataset; xLAM *models* are CC-BY-NC — don't conflate) | structural exact-match: emitted `[{name,arguments}]` vs `answers`, set-equality for parallel, **canonicalize args + per-arg partial credit**; GT was exec+semantics-verified at gen (>95% human-checked) | bulk single-turn arg-grounding RL |
| B2 MULTI-TURN | **ToolACE** | hf.co/datasets/Team-ACE/ToolACE (ungated) | Apache-2.0 | structured GT calls, exact/arg-level; **transcode Pythonic-bracket → Qwen-native JSON** + segment scorable turns | multi-turn chaining (permissive alt to APIGen-MT) |
| B3 LIVE (conditional) | **tau2 / τ³-bench** | github.com/sierra-research/tau2-bench (Py≥3.12, uv) | MIT | DB-state hash + action-subsequence + policy nl_assertion, ANDed → 0/1; stateful local env | held-out DOMAIN eval + optional on-policy stretch (SEED/pin the user-sim; stochastic sim biases GRPO group advantages) |
| B4 AUX (conditional) | API-Bank Lv1+Lv2 | github.com/AlibabaResearch/DAMO-ConvAI/.../api-bank (pull raw JSON; HF loader broken) | permissive | exec call + check_api_call_correctness vs GT; EXCLUDE Level-3 (judge-like) | secondary executable signal |

## TIER C — agentic coding / SWE: DEFERRED
No Tier-C verified this pass. SWE-bench Verified (pytest pass/fail = genuine reward) + SWE-Gym are canonical but carry
heavy Docker/repo harness deps (conflict with single-GPU rollout-heavy). Do NOT start until B1/B2 show RAW gains; if
pursued, run a separate verification pass first.

## SEQUENCING
A4 Sudoku → A2 Countdown (prove the GRPO→grammar-vLLM-rollout→verifier→QLoRA loop moves the RAW metric) → A3 GSM8K
sanity → **B1 xLAM-60k** (the real work: per-row grammar from `tools`, exact-arg-match reward w/ canonicalization +
order-insensitive set-match + per-arg partial credit) → B2 ToolACE (multi-turn, after single-turn solid) → B3 tau2
(live, seeded sim, train one domain / hold out another). Align the RL reward with the (binary) eval metric — soft
partial credit is reward-hackable if misaligned.

## REJECTED / DO-NOT (banked so we don't revisit)
- MATH/MATH-500 as TRAIN — license cloud + MATH-500 is an eval subset (contamination). EVAL-ONLY w/ math-aware verifier
  (train mirror nlile/hendrycks-MATH-benchmark, eval on MATH-500). Not the arg-grounding target.
- BFCL executable/REST/live/V4-web — need live API keys → non-deterministic. Use only AST + multi-turn-state subsets.
- API-Bank Level-3 — judge-like, irreproducible. reasoning-gym lenient cascade — use strict==1.0.
- **APIGen-MT-5k — REJECT as picked train source (CC-BY-NC-4.0 + no-compete clause).** Research-only fallback.
- Nexusflow/NexusBench — optional aux EVAL only (small, Pythonic-call, some subsets need live APIs).

## HELD-OUT EVAL (public, distinct from training)
In-house Lumo Codex-Long pack + SWE-bench Verified + BFCL AST/test + tau2 held-out domain + MATH-500 (math-verifier).

## MONITOR CAVEATS
- xLAM-60k GATED → flare accepts terms + sets HF token at use time.
- B1/B2 rewards are STRUCTURAL exact-match (verified at the DATASET's gen time, not live-executed at RL time) — clean
  grounding reward, but it rewards matching gold (outcome-level + on-policy + partial-credit, so still > CE imitation).
  Live execution only at B3 (tau2) / B4 (API-Bank). Keep the dual-term loss (RAW-internalization) so raw improves, not
  just the constrained policy.
