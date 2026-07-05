# REPRODUCE_V3.md

The single self-contained reproduction + claims artifact for the **whole** system:
a preservation-certified recipe that converts an autoregressive (AR) Qwen3.5-9B into a
block-diffusion twin, RL-hardens it for agentic tool use, serves it at batch=1 latency competitive
with stock AR (faster than the `enforce_eager` AR server it was baselined against; at ~parity vs a
fair cudagraph AR) on a byte-parity-audited vLLM engine, and closes the flywheel by re-converting the
RL'd model without losing the gain.

This supersedes **REPRODUCE_V2.md** (kept in-repo, linked here). V2 is the external reproduction
guide for the conversion→RL→hybrid-clean-serving lane (recipe + pins). V3 keeps that lane verbatim,
re-verifies every pin, and adds the three things V2 predates: the **vLLM P2 engine** (batch=1 serving
faster than the eager AR server, ≈parity vs a fair cudagraph AR — §5.1/§5.7; pin chain
`6b81154..0b44dcc`), the **convert-after-RL preservation certificate** (the
flywheel's sharp test), and the **honest speed frontier** (K_max=1.0 today; the 5× goal is an open
training bet, not a shipped result).

Every number below cites its commit/artifact. Refuted priors are marked **REFUTED**. No number
appears without a source. All paths are absolute unless written repo-relative under `$ROOT`.

---

## 0. WHAT THIS SYSTEM IS — AND IS NOT (claims summary)

**IS — a quality-identical, certified agentic serving system on converted weights, at batch=1
latency competitive with stock AR.** Converting Qwen3.5-9B to block-diffusion and serving it on the
P2 vLLM FLARE engine (pin `95d8b47`/`0b44dcc`, batch=1, RTX 5090) posts **exact-args 130/247 == the
HF diffusion reference 130/247 EXACTLY** on the aggregate endgame battery (episode 32/80, valid
247/247, 0 per-turn wins / 0 losses vs HF), and **beats every AR baseline on quality**: +6 vs
stock-bf16-AR (124), +1 vs stock-FP8 (129), +3 vs merged-AR (127). On speed it is **faster than the
`enforce_eager` vLLM AR server it was baselined against** on the identical 247-turn mix: **0.626
s/turn** vs stock-bf16-AR 0.741 (1.18×), stock-FP8 0.910 (1.45×), merged-AR 0.739 (1.18×), and the HF
hybrid stack 2.577 (4.12×). **Speed caveat (load-bearing, per §5.1/§5.7): those AR baselines ran
`enforce_eager` — CUDA graphs OFF (`runs/endgame_stock_qwen35_ar_guided/{bf16,fp8}/server_launch.json`)
— while the engine ran cudagraph; cudagraph is a measured ~1.32× AR speedup. Against a FAIR cudagraph
guided-AR at batch=1 the engine is at ~parity-to-slightly-slower (0.94×, §5.7), so the 1.18× is a win
over the eager server, NOT a fair-harness single-stream speed win.** Conversion + RL did not cost
accuracy — it gained it. (`endgame_table_final.md`, `runs/endgame_scoreboard`,
`p2_engine_battery_v3b_result.md`, `runs/p2_batched_rollout_bench/report.md`.)

**IS — a preservation-certified conversion loop.** Re-running a fresh two-stream conversion on the
*merged RL-v2 weights* — using the original Run-1 mix that **excludes** the RL episode pool, so the
RL-acquired capability is one the conversion was never trained on — **preserves that capability
across two independent seeds**: paired-turn McNemar `b = 0` (C0-right & A_new-wrong) in **both**
seeds across 126 diffusion tool-call turns (zero lost, +3 gained), GSM8K retention combined 26/40 =
0.65 == anchor, all value-projection audits clean. The step-1 preservation mechanisms hypothesized
as *possibly needed* (KL-to-pre-conversion, capability replay, joint convert-and-RL) are **not
required** by this evidence. The flywheel does not eat its own gains at re-diffusionization.
(`convert_after_rl_result.md`, `convert_after_rl_design.md`.) The per-capability conversion tax is
**small and bounded** — no capability class collapses (`conversion_tax_result.md`).

**IS NOT — a rollout-throughput multiplier.** The a-priori thesis that the FLOP-reducing diffusion
twin generates RL rollouts faster by batching is **REFUTED on this hardware/workload.** Against a
*fast* guided-AR baseline the engine is at rollout parity at batch=1 (0.94×) and **loses ground as
batch grows — 0.73× at batch=16** (poor occupancy under FLARE's forced sync scheduler; each forward
~14× costlier even though the hybrid does ~10× fewer forwards). On the actual best-of-N GRPO pattern
(N samples of the same prompt) the engine is **0.67–0.85× AR throughput AND less diverse** — both
paradigms collapse at temp=0.7 on strict tool calls, and where diversity survives AR yields more
correct rollouts, so the engine produces **no extra group signal**. A throughput-bound RL loop
generates rollouts **1.1–1.5× faster with stock guided-AR.** The twin's earned role is the loop's
low-batch/latency serving parity + its conversion/scoring/validity spine — **not** samples/sec.
(`runs/p2_batched_rollout_bench`, `runs/p2_bestofn_grpo`, `methodology_diffusion_accelerated_rl.md`.)

**IS NOT — a solved parallel-content generator.** On genuine reasoning content at B=1 the honest,
audited measurement is **K_max(today) = 1.0 committed token / forward at held GSM8K exactness — there
is NO validated parallel reasoning lane today.** Every model-chosen token is one forward (732
forwards == 732 model-chosen tokens); the only tok/fwd > 1 is the zero-forward grammar scaffold. The
B=1 speed ratio on reasoning content is **0.36× vs a fair AR-cudagraph baseline** (0.86 emitted
tok/fwd × 10.72 AR-ms/tok ÷ 25.8 engine-ms/fwd), leaving a **~14× distance to the 5× north star,
entirely in the K factor.** The prior **"native 4 tok/fwd on GSM8K-class" claim is REFUTED**: the
anchor sampler is ~1.02 tok/fwd at every block width; the only sampler that mechanically reaches 4–8
tok/fwd is the disqualified mutable-remask diagnostic (0.25 at full denoise, fails the ≥0.60 anchor).
Raising reasoning K above 1 at held exactness is the **open L3 bet** (S2 consistency-distillation +
entropy-gated adaptive K), stated plainly as unproven. (`goal_5x_rollout_b1.md`,
`l1_content_mix_result.md`, `l1_baseline_b1_result.md`, `runs/l0l2_final_head_verify`.)

**IS NOT — a token-for-token-certified drop-in (yet).** Strict engine==HF **247/247 byte-parity is
NOT met (233/247)**, so the vLLM engine's FLARE decode **code default stays OFF**. All 14 breaks are
the **same quality-neutral deterministic bf16 GDN-fold fp-residue class, with 0 structural breaks**
(the grammar scaffold always matches; both engine and HF are non-exact on every break — `eng_exact
== hf_exact`; `value_projection_events == 0`; `verify_invariants == ok`), so **exact-args stays
130/247 in every cache configuration** (cold-prefix → 235/247). Byte-parity here is a construction
certificate ("engine == HF by kernels"), not a quality gain — and quality (130/247) is already met
exactly. The residual is the block#0 GDN fold granularity, a kernel-level task explicitly deferred.
(`endgame_table_final.md` §(c), `p2_engine_battery_v3b_result.md`.)

---

## 1. SOURCES OF TRUTH (index — read these; numbers here are transcribed, not re-derived)

| domain | source of truth |
|---|---|
| Conversion recipe + pins (prior artifact) | `REPRODUCE_V2.md` |
| Serving scoreboard (aggregate 247) | `endgame_table_final.md`, `runs/endgame_scoreboard/report.md` |
| Engine chain + build status + all §0.x battery updates | `engine_build_status.md` (pin `6b81154..0b44dcc`) |
| Engine promotion attempt (matched-20 v3b) | `p2_engine_battery_v3b_result.md`, `runs/p2_engine_battery_v3b/` |
| Convert-after-RL preservation (design + verdict) | `convert_after_rl_design.md`, `convert_after_rl_result.md` |
| Per-capability conversion tax | `conversion_tax_result.md`, `runs/conversion_tax/` |
| Loop spec + regime resolution | `methodology_diffusion_accelerated_rl.md` |
| The honest speed frontier (5×-at-B1) | `goal_5x_rollout_b1.md`, `l1_content_mix_result.md`, `l1_baseline_b1_result.md`, `runs/l0l2_final_head_verify/` |
| Rollout-regime disconfirmations | `runs/p2_batched_rollout_bench/report.md`, `runs/p2_bestofn_grpo/report.md` |

---

## 2. HARDWARE, REPOS, ENVIRONMENTS, PINS (all re-verified 2026-07-05)

### 2.1 Validated hardware

```text
GPU: NVIDIA GeForce RTX 5090, 32607 MiB, sm_120 (compute capability [12, 0])
Driver: 595.71.05
```

The 5090 caveat is load-bearing throughout: the measured **stock-FP8-AR path was SLOWER than bf16**
on this card (aggregate speedup 0.814×, seen as stock-FP8 0.910 vs stock-bf16 0.741 s/turn) — a quant
tax, not a speedup. Trust nothing unmeasured on sm_120. (`endgame_table_final.md` §(d),
`runs/endgame_scoreboard` Stock-FP8 block.)

### 2.2 Three repos + re-verified HEADs

| repo | remote | branch | HEAD (verified 2026-07-05) | pushed? |
|---|---|---|---|---|
| A. qwen_diffusion (`$ROOT`) | `MaCoredroid/qwen-diffusion-agentic` | `main` | `1d066be` (this doc adds a commit on top) | yes (origin/main) |
| B. vLLM P2 engine pin | none (local; upstream base `2665ed7`, PR #46838) | `qwen3_5-flare-modelstate` | `0b44dcc` | **no** (local pin) |
| C. flywheel fork (`$FLYWHEEL`) | `MaCoredroid/Lumo_FlyWheel-qwen-diffusion` | `codex/qwen35-9b-feasibility` | `c3a7a753` | **no** |

Paths as run: `$ROOT=/home/mark/qwen_diffusion`, `$FLYWHEEL=/home/mark/shared/lumoFlyWheel_codex_fork`,
vLLM pin `/home/mark/shared/vllm_p2_pr42406` (editable-installed into `$ROOT/.venv-vllm-p2-main`).

**Pin drift vs V2 (re-verified today, documented not silently inherited):** the flywheel fork HEAD is
now `c3a7a753` (V2 pinned `b91184d0`); the only change is one added commit,
`c3a7a753 "flare-hybrid launcher: wire diffusion canvas_length/num_speculative_tokens"`, on top of
`b91184d0`. The export/serving/parity-gate scripts V2 pinned are **byte-identical today** (sha256
table §2.4). The qwen_diffusion HEAD advanced `782b441 → 1d066be` but **every conversion/eval/audit
script V2 pinned is byte-identical today** (§2.4).

### 2.3 The vLLM P2 engine commit chain (pin `6b81154..0b44dcc`, 9 commits)

Upstream base `2665ed7` (PR #46838, at/after the MRV2 align-APC merge). Branch
`qwen3_5-flare-modelstate`, verified with `git -C /home/mark/shared/vllm_p2_pr42406 log --oneline 6b81154^..0b44dcc`:

```text
0b44dcc  FLARE L0 free-text serving: fix CPU-pathological hang + honor EOS like AR   <- FINAL HEAD
95d8b47  FLARE OPT-4 Stage 3: variable-width GPU integration + byte-robust bidir key window (58->62/63)
490e7f3  FLARE OPT-4 Stage 2: per-request variable draft-width scheduler plumbing
8454365  FLARE OPT-4 Stage 1: authoritative 32-absolute variable commit width + single-gate BIDIR_PROBE
e5496cc  FLARE OPT-4 part 2: PIECEWISE cudagraph for the block-diffusion decode
b7d76e2  FLARE OPT-3 parity: windowed-BIDIRECTIONAL denoise read (env-gated foundation) + attribution
d2fccab  FLARE OPT-3: force sync scheduler (kill async-rollback divergence + partial-canvas stall)
58cfe2c  FLARE hybrid_clean OPT-1: GPU-native sampling (kill full-vocab host reduce)
6b81154  FLARE hybrid_clean: single-[MASK] forward VIEW via causal-windowed probe (GAP 5A)   <- chain base
```

The **endgame 247-turn battery** ran on pin **`95d8b47`** (OPT-4 Stage 1+2+3). The **L0 free-text
fix `0b44dcc`** is the FINAL HEAD; it is gated on `grammar.enabled` so the tool-call decode path is
byte-identical to `95d8b47` — the **233/247 certificate is intact across `95d8b47`→`0b44dcc` with
zero source edits** (`runs/l0l2_final_head_verify/summary.json`: `pin_source_edits: 0`,
`certificate_regressed: false`).

### 2.4 Script sha256 pins (recomputed today — do not copy stale ones)

Every hash below was recomputed with `sha256sum` on the working tree on 2026-07-05.

**Repo A — qwen_diffusion (`$ROOT/scripts/…`), all byte-identical to V2:**

| script | sha256 |
|---|---|
| `init_qwen35_fastdllm_candidate.py` | `65b94c94e82d30096222880854cee703e569c2e25e2b16f811417d05074896b3` |
| `materialize_qwen35_fastdllm_weights.py` | `8ee0c6c4e43072f6931dadaf563636a3d69155f790adccb257a4c0e399fb01e4` |
| `check_qwen35_diffusion_readiness.py` | `82cd93ff5cd48af0d30978be7d6aa586248940217a906245acd584fbec25ba8e` |
| `build_flare_stage1_ab_pilot_data.py` | `b49422f7331e2968110b14432d59b7b1fa0365d8fa90806733df145a4e463d61` |
| `run_flare_stage1_ab_pilot_job.sh` | `498c20cb849575c71f0b1fc66bd7c2b8aae5c03281472ce31af139ff9c6fb23c` |
| `eval_flare_stage1_ab_diffusion.py` | `eaa78d7a9abfb32b7ab73c7753cf87026741e372ef13a1c0f8e44ead79b5e503` |
| `build_flare_redesign_run1_copy_mix.py` | `22bb2585448a4fa0f1e9fc4791247b21c9f15846628e8c4db7408180e68a4e7f` |
| `run_flare_redesign_run1.sh` | `bd9ce05f90fbbcca9703dcdcdc6bdc38b6219f639c7898f8bcbb5807876e1579` |
| `eval_flare_redesign_run1.sh` | `d4083959e87b4b87541c4ecd1d6f9d4f5778bc5b204ddccb90b894a90d16118a` |
| `build_rl_multiturn_v2_pool.py` | `9ded5be9b781bf1f0e0c5eb5554c59ff5b77352a4c67ce93c9727245bf568602` |
| `run_rl_multiturn_grpo_v2_pilot.sh` | `f2dcf0b0262a2266893159344bc56055ef9bafe3c1e630ebc91bb38fb8ef888f` |
| `rl_multiturn_grpo_pilot.py` | `ea71fb89a1fa021be34d7aab95996679719a3ebb0e7bb5178331e7c1bbb8355f` |
| `eval_flare_northstar_matched.py` | `4cda3acf752c093a0ee3d3e1208c2cdc5deb064b027d984cdf54e8fa93b6203f` |
| `eval_flare_northstar_hybrid_clean.py` | `a4c66751008390ec44ff4fbb7d025352dc71ba21a005948411883818b908b1f3` |
| `audit_value_projection_tokens.py` | `7b203e3e8e2a7a7bbfa6f831be295543c728b08d9228bd241f0f07e35a620b40` |
| `parity_audit_flare_engine.py` (engine gate) | `49d5326be095d9b25e2d6c8f7da2ee4a2963f0a72ca3df9a1e66c191027520fc` |
| `p2_vllm_smoke.py` (engine smoke) | `c76d56c2a3b48d3f9e0b87de8e78bcc9ed3e54805d1d46035f7d76d231204132` |

**Repo C — flywheel fork (`$FLYWHEEL/scripts/…`):**

| script | sha256 | vs V2 |
|---|---|---|
| `export_qwen35_9b_fastdllm_vllm.py` | `6d507ec9ba3308ff7e0f600bc0b5ec7c4ff96f66eff4e4e92175d42af7a119d5` | identical |
| `qwen35_9b_host_vllm_serve.sh` | `66fe88c7db972435010b1dfb159979de80476e288d3e2b55e9f26d5e7a6e618b` | identical |
| `qwen35_9b_p1a_parity_gate.sh` | `a3420b064488d563bb3a37af64e2ccb9a75865fa10f32c39a7ef21fa5278d232` | identical |
| `qwen35_9b_flare_hybrid_serve.sh` (engine launcher) | `b0f211eafef594da34c0d62e9fb2b1c94fb59130db9391e8a1d2ee6f91cb1acc` | new (at `c3a7a753`) |
| `docker/chat_templates/qwen3-openai-codex.jinja` | present (11526 B) | — |

**Repo B — vLLM pin FLARE source (`/home/mark/shared/vllm_p2_pr42406/…`, pin `0b44dcc`):**

| file | sha256 |
|---|---|
| `vllm/v1/worker/gpu/model_states/qwen3_5_flare.py` | `8ea67e215d0c977bd905e35c64186ffdc85bab56b98ff2f38f443e9bd9c1c790` |
| `vllm/v1/worker/gpu/model_states/qwen3_5_flare_ops.py` | `8605c13148fe785dfc06393c3256f1b9e162bb22c1d5bbfa2ffafcc8a8ebe57f` |
| `vllm/model_executor/models/qwen3_5.py` | `1a4cfdb48ebb9c288db470cb6642e28d62d935cbce9ec71df559e2a67abf0a1b` |

### 2.5 Environments (validated local Python stacks)

```text
.venv-fastdllm:      Python 3.10.20, torch 2.12.1+cu130, CUDA 13.0, transformers 4.53.1, datasets 2.14.6, peft 0.19.1   (train/eval + HF diffusion serving)
.venv-vllm:          Python 3.12.13, torch 2.11.0+cu130, CUDA 13.0, vLLM 0.23.0                                          (AR parity route + stock/merged-AR baselines)
.venv-vllm-p2-main:  vLLM editable-installed from /home/mark/shared/vllm_p2_pr42406 @ 0b44dcc                            (the P2 FLARE engine)
```

Base model snapshot: `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

### 2.6 The Sampler-Pinning Rule (inherited verbatim from V2 §0)

Treat the sampler implementation as part of the model result. Every gate report records: repo git
commit; script sha256; sampler function name; adapter path; dataset path + manifest hash; decode
flags; and the generated-token / value-projection audit file. The GSM8K continuity sampler is
`scripts/eval_flare_stage1_ab_diffusion.py::full_context_sample_one` — **do not substitute**
`scripts/measure_block_quality_curve.py` (a disqualified mutable-remask fixed-K diagnostic; see
§7 and the 4-tok/fwd REFUTED claim). For tool-call serving gates the value-projection invariant is
**hard**: `projected_value_tokens_exact == 0`, `parallel_commit_forced_tokens_counter == 0`,
`wave1_projected_tokens == 0`, `wave1_value_tokens_counter == 0`, `wave2_forced_tokens_counter == 0`,
`zero_forward_rows == 0`.

---

## 3. FULL REPRODUCTION PATH

Order: base conversion → B@1000 two-stream foundation → Run-1 copy-grounding → RL-v2 diffu-GRPO →
vLLM export → engine build → eval batteries. Steps 3.1–3.4 are the V2 lane (commands + exhaustive
flag lists in `REPRODUCE_V2.md §§3–7`); the pins and expected numbers are transcribed here so V3
stands alone. Steps 3.5–3.9 are net-new to V3.

### 3.0 Clone + pin

```bash
export ROOT=/home/mark/qwen_diffusion
export FLYWHEEL=/home/mark/shared/lumoFlyWheel_codex_fork
export VLLM_PIN=/home/mark/shared/vllm_p2_pr42406

git clone git@github.com:MaCoredroid/qwen-diffusion-agentic.git "$ROOT"
git -C "$ROOT" checkout 1d066be           # or later; scripts §2.4 are the invariant
git -C "$ROOT" submodule update --init --recursive

git clone git@github.com:MaCoredroid/Lumo_FlyWheel-qwen-diffusion.git "$FLYWHEEL"
git -C "$FLYWHEEL" checkout c3a7a753

# vLLM pin is a local branch (no upstream on qwen3_5-flare-modelstate). Reconstruct from upstream
# base 2665ed7 (PR #46838) + the 9-commit FLARE chain 6b81154..0b44dcc (§2.3). Editable-install
# into $ROOT/.venv-vllm-p2-main.
```

Environments: build `.venv-fastdllm` and `.venv-vllm` per `REPRODUCE_V2.md §1` (torch/vLLM wheels
pinned there). Install `jq` for the acceptance checks. Download the base snapshot at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

### 3.1 Base conversion (Fast-DLLM candidate + materialized Qwen3.5 weights)

`scripts/init_qwen35_fastdllm_candidate.py` (`--bd-size 32 --mask-token '|<MASK>|'`) →
`scripts/materialize_qwen35_fastdllm_weights.py --write` →
`scripts/check_qwen35_diffusion_readiness.py`. Output `models/qwen3.5-9b-fastdllm-init`.

Expected manifest (`models/qwen3.5-9b-fastdllm-init/conversion_manifest.json`):
`bd_size=32`, `mask_token=|<MASK>|`, **`mask_token_id=248077`**, `bridge_status=implemented`,
`gdn_mode=option_a_causal_gdn_v0`, `has_weights=true`. **STOP** if `mask_token_id != 248077` or
`has_weights != true`. Full commands: `REPRODUCE_V2.md §3`.

### 3.2 B@1000 two-stream conversion foundation

Build Stage-1 A/B pilot data (`build_flare_stage1_ab_pilot_data.py`, gsm8k 160 / mbpp 96, seed
20260701) → train the two-stream **B adapter for 1000 steps** (`run_flare_stage1_ab_pilot_job.sh
two_stream`, block 1024, bd 32, lr 1e-5, LoRA r8/α16, `FASTDLLM_FLARE_TWO_STREAM=1`).

Validated compute (single RTX 5090): `wall_seconds=18945`, `train_runtime=18922.09`,
`gpu_peak_memory_mib=29612`, `train_loss=3.9430`. Export the B@1000 clean stream to the
vLLM-loadable wrapper (`$FLYWHEEL/scripts/export_qwen35_9b_fastdllm_vllm.py`) →
`models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`. Expected export manifest: **`replacement_count=427`,
`lora_merge_count=32`, `lora_scale=2.0`**. **STOP** if `replacement_count != 427` or
`lora_merge_count != 32`.

B@1000 GSM8K continuity gate (pinned legacy sampler
`eval_flare_stage1_ab_diffusion.py::full_context_sample_one`, `FASTDLLM_FLARE_GDN_ROUTE=route_i`,
`FASTDLLM_GDN_KERNEL=torch`, N=20 first20, mask 248077, stop 248046): the current reproducible
legacy continuity point for the exact saved B@1000 adapter is **11/20 = 0.55** (the historical
13/20 target was a drift artifact; see §7). **STOP if B@1000 < 11/20 on the pinned legacy sampler.**
Full commands: `REPRODUCE_V2.md §4`.

### 3.3 Run-1 copy-grounding (campaign base)

Build the copy/retention/public mix (`build_flare_redesign_run1_copy_mix.py`) → train
`run_flare_redesign_run1.sh` (400 steps, block 512, bd 32, lr 1e-5, **LoRA r16/α32**, cosine, targets
`q/k/v/o + in_proj_{qkv,z,b,a} + out_proj`, `VALUE_SPAN_LOSS_WEIGHT=2.0`, seed 71101). Data manifest:
`count=5055` (synthetic-copy 2048 / retention 2560 / public-toolcall-pool 447), `eval_overlap_removed=64`,
`copy_span_count=7363`. Validated compute: `train_runtime=2068.16 s`, `train_loss=4.3907`.

Expected Run-1 gates: **GSM8K corrected legacy strict 14–15/20 = 0.70–0.75** (minimum to proceed
**13/20 = 0.65**); copy-span isolation at careful `value_tpf=1.0` = **41/41 copied args exact**.
**STOP if Run-1 GSM8K corrected legacy strict < 13/20.** Output
`runs/flare_redesign_run1_copy_grounded_qwen35_9b`. Full commands: `REPRODUCE_V2.md §5`.

### 3.4 RL-v2 diffu-GRPO (the promoted agentic adapter)

Build the v2 public pool (`build_rl_multiturn_v2_pool.py`, target 240, leak-filtered vs the frozen
eval batteries → manifest `episode_set_hash=5ffd1ad5…`, all `selected_overlap_counts == 0`) → launch
GRPO from Run-1 (`run_rl_multiturn_grpo_v2_pilot.sh`, group_size 4, max_steps 300, lr 5e-6,
**KL_TO_BASE_COEFF=0.05**, KL on parameter-value/free tokens only, grammar-forced structural tokens
masked out of policy loss, retention probes every 50 steps N=5, seed 20260702).

Validated training accounting: `elapsed_hours≈3.98`, `nonzero_advantage_steps=214/300`, value/free
policy tokens 13858, grammar-forced masked 58886, `mean_reward_last20=0.9796`, `max_KL_to_base=0.0748`,
`trainable_params=18112512`. Expected RL-v2 gates: **GSM8K retention 13/20 = 0.65 (PASS)**;
matched-20 careful exact_args 44/63; episode_exact 11/20; valid 62/63; **value projection 0**.
**STOP if RL-v2 GSM8K retention < 13/20 or matched-20 careful shows nonzero projected value tokens.**
The winning adapter is `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`. Full
commands: `REPRODUCE_V2.md §6`.

### 3.5 vLLM export (RL-v2 → the engine's input weights)

Export the merged RL-v2 clean stream into the vLLM-loadable Qwen3.5 wrapper the FLARE engine loads:

```bash
.venv-fastdllm/bin/python "$FLYWHEEL/scripts/export_qwen35_9b_fastdllm_vllm.py" \
  --official-model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a" \
  --converted-model "$ROOT/models/qwen3.5-9b-fastdllm-init" \
  --adapter "$ROOT/runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model" \
  --output "$ROOT/models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16" \
  --overwrite
```

Verified export manifest (`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16/lumo_export_manifest.json`):
`schema=lumo.qwen35_9b_fastdllm_vllm_export.v1`, `replacement_count=427`, **`lora_merge_count=152`**
(RL-v2's wider target set vs B@1000's 32), `lora_scale=2.0`. This bf16 export (block/canvas 32, mamba
1024, align+APC) is the engine input for §3.8.

### 3.6 Engine build (vLLM pin + flywheel launcher + env gates)

The engine is `Qwen3_5FlareModelState(MambaHybridModelState)` + `Qwen3_5FlareSampler` in the vLLM
pin (`vllm/v1/worker/gpu/model_states/qwen3_5_flare*.py`, registered in
`vllm/model_executor/models/qwen3_5.py`; sha256 §2.4). It grafts the DiffusionGemma canvas/commit
path onto MambaHybrid's align-APC pre/postcopy + GDN attn metadata. The launcher is
`$FLYWHEEL/scripts/qwen35_9b_flare_hybrid_serve.sh` (points at `.venv-vllm-p2-main/bin/vllm`, serves
`models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16`).

**Env gates (the certified endgame configuration):**

```text
VLLM_QWEN3_5_FLARE=1          # route to the FLARE ModelState (else plain MambaHybrid AR)
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_ATTENTION_BACKEND=TRITON_ATTN
VLLM_FLARE_BIDIR_PROBE=1      # reference-exact windowed-BIDIRECTIONAL denoise read (b7d76e2)
VLLM_FLARE_CUDAGRAPH=1        # PIECEWISE CUDA graph for the block-diffusion decode (OPT-4 part 2, e5496cc)
--mamba-cache-mode align  --mamba-block-size 1024  --mamba-ssm-cache-dtype float32
--gdn-prefill-backend triton
VLLM_QWEN3_5_FLARE_BLOCK=<multiple of 64>   # canvas_length; block/chunk must be a 64-multiple (GDN FLA_CHUNK)
```

Two hard scheduler facts (both from the OPT-3 fix, `d2fccab`):
- **The FLARE scheduler is FORCED-SYNC.** The **async scheduler is INCOMPATIBLE** — it produced an
  async-rollback divergence at the first denoise position after a block boundary (pos-33) and a
  partial-canvas forward stall. Forcing the sync scheduler killed both (0/11 breaks at pos-33). Do
  not run FLARE under the async scheduler. (`engine_build_status.md` §0.E.)
- **Per-request mode switching is NOT wired.** Registration is process-global via `VLLM_QWEN3_5_FLARE=1`;
  AR and block-diffusion cannot coexist in one server. (`engine_build_status.md` §1, §4.)

CPU-side pre-flight (run under `.venv-vllm-p2-main`, no GPU needed):
`pytest tests/v1/worker/gpu/test_qwen3_5_flare_state_machine.py` (17) and
`pytest tests/v1/sample/test_hybrid_clean.py` (20); then the engine parity gate
`scripts/parity_audit_flare_engine.py --mode {selftest,ops-parity,state-parity}`
(15/15, 15/15, 4 gates). (`engine_build_status.md` §3 step 1.)

### 3.7 HF hybrid-clean serving (the parity reference + the promoted V2 result)

Hybrid-clean is the constrained lane: FSM grammar-forced structural tokens are bulk-committed without
model forwards; every parameter **value** token is decoded **sequentially, K=1, chain-rule
preserving**; value projection is forbidden. Run matched-20 and never-train with
`scripts/eval_flare_northstar_hybrid_clean.py` (sampler
`diffusion_hybrid_forced_grammar_seq_values`) and audit each with
`scripts/audit_value_projection_tokens.py`. This is the **HF reference row** the engine must
byte-match, and the promoted V2 serving result:

```text
matched-20 (63 turns):   exact_args 47/63 · valid 63/63 · exact_tool_sequence 63/63 · episode 13/20 · 56.83 fwd/turn · 3.904 s/turn
never-train (184 turns): exact_args 83/184 · valid 184/184 · episode 19/60 · 24.62 fwd/turn · 2.123 s/turn
```

Full commands + all flags: `REPRODUCE_V2.md §7`. Adapter used:
`runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model` (`--no-merge-adapter`).

### 3.8 Engine battery — the aggregate 247-turn endgame (pin `95d8b47`)

Serve `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` under the §3.6 env, greedy (temp 0, seed 20260701),
uncapped (`max_tokens = n_ref + 16`), batch=1, RAM-caged. Run the identical 63-turn matched-20 slice
(`runs/p2_engine_battery_v3b/`) and the 184-turn never-train BFCL/API-Bank slice
(`runs/p2_engine_nevertrain/`); record per-turn byte-parity vs the HF reference (§3.7), TRUE denoise
forwards/turn, per-forward ms, `value_projection_events`, `verify_invariants`, and `eng_exact` vs
`hf_exact`. Expected aggregate numbers + tolerances are §5.

### 3.9 Convert-after-RL preservation (the flywheel's sharp test — CLOSE the loop)

This is the affirmative certificate the loop's viability rests on. Design:
`convert_after_rl_design.md` (@`6f5d20f`). Protocol:

1. **STEP 1 — merge (KILL-1 gate):** merge RL-v2 into the base to form `M_{t+1}`; bit-exact
   `init + 2.0·B@A`, mask 248077, bd_size 32 → `models/qwen3.5-9b-fastdllm-mtplus1-merged` (present).
2. **STEP 2 — train A_new:** a **fresh** Run-1-recipe two-stream conversion on `M_{t+1}`, using
   `data/flare_redesign_run1_copy_retention_mix` (5055 rows, **excludes the RL-v2 pool** — so the
   RL gain is a capability the conversion is not trained on), 400 steps, r16/α32, cosine.
   Two independent seeds: 80101, 80102.
3. **STEP 3/4 — eval battery a/b/c/d** paired turn-by-turn against **C0 = init+RL-v2 (no reconvert)**,
   both McNemar (sharpest) and episode bootstrap; retention N=20; value-projection audits.

Verdict: **FLYWHEEL PRESERVES.** McNemar `b = 0` (C0-right & A_new-wrong) in **both** seeds across
126 paired diffusion tool-call turns (net-loss ≤ 0 everywhere; seed-2 +3, a strict superset of C0);
GSM8K combined 26/40 = 0.65 == anchor; all audits clean; no KILL fired. Full table §5.
(`convert_after_rl_result.md`.)

---

## 4. THE AUDIT BATTERY — MANDATORY ACCEPTANCE (not optional; the lessons are half the value)

A result is INVALID unless the audit battery passes. These checks are what separate a real
zero-projection hybrid-clean win from audit theater.

### 4.1 Value-projection audit (every diffusion turns.jsonl)

Run `scripts/audit_value_projection_tokens.py` on each `turns.jsonl` and assert:

```text
zero_projected_value_tokens_verified = 1
projected_value_tokens_exact         = 0
parallel_commit_forced_tokens_counter= 0
wave1_projected_tokens               = 0
wave1_value_tokens_counter           = 0
wave2_forced_tokens_counter          = 0
zero_forward_rows                    = 0
exact_rows_dependent_on_projected_values = 0
verification_mode = no_projection_events
```

Reference expected totals (V2 hybrid-clean): matched-20 `rows=63, exact_args=47,
reported_model_value_tokens=2846, true_xml_value_tokens=2061`; never-train `rows=184, exact_args=83,
reported_model_value_tokens=2932, true_xml_value_tokens=1397`. **Acceptance FAILS if any
value-projection counter is nonzero, if `zero_forward_rows > 0`, or if the sampler path is not
`diffusion_hybrid_forced_grammar_seq_values`.** (`REPRODUCE_V2.md §8`.)

### 4.2 Engine parity + invariants (per turn, every engine battery)

On every engine turn assert: `value_projection_events == 0`, `verify_invariants == ok`, and
`eng_exact == hf_exact`. Byte-parity is recorded but is **not** an exact-args gate — the 14 breaks
are quality-neutral fp-residue (§5, §6). Aggregate expected: `value_projection 0/247`,
`verify_invariants 247/247 ok`, `eng_exact == hf_exact` on all 247. (`endgame_table_final.md` §(c),
`p2_engine_battery_v3b_result.md`.)

### 4.3 Read-only-denoise fingerprint (the GDN-state-discipline crux)

The single GPU-only-unvalidatable crux is whether the read-only-denoise snapshot/restore reconstructs
EXACTLY the physical conv+ssm rows the GDN kernel writes (`engine_build_status.md` §2 CRUX). The
acceptance fingerprint (`runs/l0l2_final_head_verify`): per-forward hash of conv+ssm denoise rows,
**6/6 bit-identical (post-restore == pre-forward), `forward_wrote > 0` on all (restore load-bearing),
0 leaks.** If the restore is not load-bearing or any row leaks, denoise permanently corrupts the
boundary S_t/conv state — the fatal silent failure. Determinism must be 2× identical across fresh
boots on every per-turn field.

### 4.4 Preservation audit KILL criteria (convert-after-RL)

KILL-1 merge gate (bit-exact `init+2.0·B@A`, mask 248077, bd_size 32); KILL-2 GSM8K ≥ 11 both seeds;
KILL-3 audits clean both seeds; KILL-4 not triggered (AR-mode agg ≥ 118 and retention > 11). All
PASS (`convert_after_rl_result.md §3`).

---

## 5. EXPECTED NUMBERS WITH TOLERANCES

### 5.1 The endgame table — aggregate 247 (matched-63 v3b + never-train-184)

Source: `endgame_table_final.md`, `runs/endgame_scoreboard/report.md`. Engine row = vLLM P2 FLARE,
pin `95d8b47`, batch=1, RTX 5090, code default OFF, **PIECEWISE cudagraph ON**. AR rows = vLLM-guided
server, greedy temp 0 seed 20260701, **`enforce_eager` — CUDA graphs OFF** (confirmed
`runs/endgame_stock_qwen35_ar_guided/{bf16,fp8}/server_launch.json`: `enforce_eager: true`; server
logs: "Cudagraph is disabled under eager mode"). **So the s/turn ratios are cudagraph-engine vs
eager-AR-server — NOT a fair-harness comparison; cudagraph is a measured ~1.32× AR speedup (§5.6), and
the fair cudagraph-guided-AR batch=1 comparison is §5.7 (engine 0.94×, i.e. slightly slower).**

| system | exact /247 | episode /80 | valid | s/turn (agg) | fwd-or-tok/turn | parity certificate |
|---|---:|---:|---:|---:|---:|---|
| stock-bf16-AR-guided | 124/247 | 33/80 | 247/247 | 0.741 | 49.06 tok/turn | n/a (AR baseline) |
| stock-FP8-AR-guided | 129/247 | 33/80 | 247/247 | 0.910 | 49.51 tok/turn | n/a (FP8 SLOWER on 5090) |
| merged-AR-guided | 127/247 | 32/80 | 247/247 | 0.739 | 48.89 tok/turn | n/a (AR baseline) |
| OUR HF hybrid-clean (v2) | 130/247 | 32/80 | 247/247 | 2.577 | 32.84 denoise fwd/turn | reference (self) |
| **OUR ENGINE hybrid-clean (v2)** | **130/247** | **32/80** | **247/247** | **0.626** | **32.43 denoise fwd/turn** | **233/247 byte-parity (14 fp-residue, 0 structural)** |

Slice breakdown (engine): matched-20 47/63 @ 1.053 s/turn, 56.86 fwd/turn, byte-parity 62/63 (break
gt44); never-train 83/184 @ 0.480 s/turn, 24.06 fwd/turn, byte-parity 171/184 (13 breaks).

**Tolerances.** exact-args is **exact-match** to HF (130/247; matched-20 exactly 47; never-train
exactly 83) — no tolerance, it is a byte-derived count and the engine ties HF on every turn.
Byte-parity is **233/247 ±2** as a function of cache configuration (cold-prefix → 235/247); it is a
recorded diagnostic, not a gate on exact-args. s/turn is hardware/thermal-sensitive (±~5%); the
**ratios** are the durable claim **against the eager AR server**: engine agg 0.626 beats
stock-bf16-AR 0.741 (1.18×), stock-FP8 0.910 (1.45×), merged-AR 0.739 (1.18×), HF-hybrid 2.577
(4.12×). **These AR rows are `enforce_eager` (see caption); the eager→cudagraph AR speedup is ~1.32×,
so a fair cudagraph AR at batch=1 is FASTER than the engine — measured directly in §5.7 (engine
0.94×). The certified-fair batch=1 result is ≈parity, not a 1.18× win.** GSM8K first20 gates move ±1
row on seed noise — rerun once and report both seeds/artifacts if a gate moves by one row (V2 §10 rule).

### 5.2 The fp-residue break class (documented, quality-neutral)

All 14 aggregate byte-parity breaks (1 on matched-20 = **gt44**; 13 on never-train) are the **same
deterministic bf16 GDN-fold fp-residue class, 0 structural**: the grammar scaffold (`<tool_call>` /
`<function=` / `<parameter=…>` / `>`) always matches; every divergence is a model-chosen value/name
**near-tie** token; `value_projection_events == 0` and `verify_invariants == ok` on all 14; and
`eng_exact == hf_exact` on all 14 (both engine and HF are non-exact there — the fp perturbation flips
an already-wrong near-tie, never the exact_arguments verdict). **So exact-args is 130/247 regardless
of these breaks.** Cold-prefix certificates: gt44 breaks identically fresh-boot (**path-invariant**,
so the documented APC protocol cannot rescue it); 10 of 13 never-train breaks break identically cold;
2 are APC cross-turn near-ties that restore to byte-parity cold (aggregate cold → 235/247); 1 is a
path-sensitive tail. **Root cause:** the block#0 GDN fold-path fp gap — HF folds 32 including the
`prompt%32` leftover; the aligned engine folds `32 − L%32` gen tokens (fp-close, not bit-identical).
Matching HF's fold granularity is a kernel-level task, explicitly deferred. (`endgame_table_final.md`
§(c), `p2_engine_battery_v3b_result.md`.)

### 5.3 Per-forward physics (why single-stream latency has a floor)

Engine per-forward wall **18.52 ms** (matched-20, cudagraph) decomposes by profiler device self-time
as: **weight-stream floor 11.40 ms** (MLP+proj+lm_head GEMM, 63.5% of GPU; arithmetic cross-check
10.77 ms = 19.31 GB bf16 / 1.79 TB/s HBM — **irreducible at batch=1**) + **non-weight GPU compute
6.54 ms** (GDN recurrence / attention / norms — measured **NOT width-reducible**: Stage-3 A/B was
18.52 vs 18.56 ms at variable-vs-fixed-32 width because cudagraph buckets narrow widths back) + host
0.58 ms. The engine is at the **batch-1 physics floor for this weight footprint**; further
single-stream latency is a training problem (fewer forwards) or a precision problem (a low-precision
GEMM that actually wins on sm_120 — FP8 lost), **not** an integration problem.
(`p2_engine_battery_v3b_result.md`, `endgame_table_final.md` §(d).)

### 5.4 Convert-after-RL preservation (two seeds)

Source: `convert_after_rl_result.md`. C0 = init+RL-v2 (no reconvert). A_new = fresh Run-1-recipe
conversion on merged `M_{t+1}`.

| lane | anchor (C0) | A_new seed-1 (80101) | A_new seed-2 (80102) | combined | verdict |
|---|---:|---:|---:|---:|---|
| diffusion hybrid matched-20 (the RL gain) | 47/63 | 47/63 | 50/63 | pooled 97/126 | PASS (both ≥ anchor) |
| diffusion hybrid never-train | 83/184 | 83/184 | 83/184 | — | PASS |
| diffusion aggregate | 130/247 | 130/247 | 133/247 | — | ≥ anchor both |
| AR-guided aggregate | 127/247 | 136/247 | (seed-1 certifies) | — | PASS (+9) |
| GSM8K retention N=20 | 13/20 | 12/20 | 14/20 | 26/40 = **0.65** | PASS (== anchor) |
| value-projection audits | 0 | clean | clean | — | CLEAN both |

Paired-turn McNemar vs C0 (the designated sharpest test): a1 net-loss **b−c = 0** (seed-1) and **−3**
(seed-2, a *gain*); a3 net-loss 0 both seeds; two-sided exact p never significant. `b = 0` for the
RL-gain lane in **both** seeds is the decisive line. (`convert_after_rl_step4_stats.json`.)

### 5.5 Per-capability conversion tax (STOCK-AR → MERGED-AR → ENGINE-DIFFUSION)

Source: `conversion_tax_result.md`. B=1 greedy, identical prompts, strict deterministic scoring.

| class | N | STOCK-AR | MERGED-AR | ENGINE-DIFFUSION | Δ merged−stock | Δ engine−merged |
|---|---:|---:|---:|---:|---|---|
| GSM8K free-CoT | 30 | 29/30 | 27/30 | 26/30 | −2 | −1 |
| CODE / MBPP | 25 | 22/25 | 22/25 | 20/25 | 0 | −2 |
| INSTRUCTION | 25 | 21/25 | 22/25 | 21/25 | +1 | −1 |
| TOOL-CALL (agentic) ¹ | 247 | 124/247 | 136/247 | 130/247 | +12 | −6 |

¹ certified reference row (not re-run). **Note the MERGED-AR tool-call point here is 136/247 — the
promoted `A_new` AR export (§5.4 seed-1), a DIFFERENT operating point from the 127/247 merged-AR-guided
row in §5.1's endgame table (= C0, the init+RL-v2 merge the engine actually serves).** So "engine sits
between stock and merged (130 ∈ [124,136])" uses the A_new merged point; against the C0 merged the
engine serves, engine 130 > C0-AR 127. **Small-N caveat is load-bearing:** the A/B/C N=25–30 Wilson
95% intervals span ≈20 points and all three systems overlap within each class — the table certifies
"no collapse," not a ranked ladder. Only the N=247 tool-call row separates: RL+merge *gains* the
exactness it was trained for (+12). Engine stability held: 0 CPU-pathological hangs,
`value_projection_events == 0`, all `verify.ok == True`.

### 5.6 The honest speed frontier (5×-at-B1) — measured, mostly REFUTED priors

Source: `goal_5x_rollout_b1.md`, `l1_content_mix_result.md`, `runs/l0l2_final_head_verify` (pin
`0b44dcc`). The B=1 equation is `ratio = (committed tok/fwd) × (AR ms/tok ÷ engine ms/fwd)`.

- **Content mix (RLv2 hybrid_clean, reasoning proxy):** grammar-forced 26.5% (0 fwd) · value 54.2%
  (K=1) · structural 19.3% (K=1). `denoise_forwards == model_chosen_tokens` exactly (732 == 732):
  **every model-chosen token is one forward.**
- **K_max(today) = 1.0 tok/fwd** on reasoning at held GSM8K exactness. The legacy anchor sampler is
  ~1.02 tok/fwd at every block width (K=8/16/32 all hold quality, K=4 breaks). **The "native 4
  tok/fwd" prior is REFUTED** — it conflated `small_block_size=8` with tok/fwd; the only sampler
  reaching 4–8 tok/fwd is the disqualified mutable-remask diagnostic (0.25 at full denoise).
- **Corrected 5× equation (final HEAD `0b44dcc`, reasoning content):** `0.86 × (10.72 AR-cudagraph ÷
  25.8 engine-free-text) = 0.36×` (0.47× vs AR-eager). **Distance to the 5× north star ≈ 14×,
  entirely in the K factor.** L2 per-forward parity (25.8→~13 ms) buys at most ~2× and is still
  K-bound; only **L3 (S2 consistency-distillation + entropy-gated adaptive K), unexecuted**, can raise
  reasoning K above 1. The goal-doc "expected 2–3× already" is **REFUTED** on the audited serving path.
- **Where the engine is LEAST behind at B=1:** tool-call-heavy content (higher grammar-forced
  fraction, more free bulk-commits) — the aggregate 0.626 vs the **eager** AR server's 0.741 (1.18×)
  is real, and vs a fair **cudagraph** guided-AR the engine reaches only ~parity (0.94×, §5.7), still
  its best regime. The advantage over the eager server is forced-token bulk commits, not parallel
  reasoning — and it does not survive to a fair-harness win.

### 5.7 Rollout-regime disconfirmations (REFUTED throughput/signal priors)

Source: `runs/p2_batched_rollout_bench/report.md`, `runs/p2_bestofn_grpo/report.md`. Batch-correctness
precondition PASSED (no cross-request contamination; batched path safe).

| batch | eng samp/s | AR samp/s | eng/AR |
|---:|---:|---:|:---:|
| 1 | 1.524 | 1.625 | 0.94× |
| 4 | 3.426 | 4.103 | 0.83× |
| 16 | 5.732 | 7.846 | 0.73× |

The ratio moves the **wrong** way for the throughput thesis (**REFUTED**). Best-of-N GRPO: eng/AR =
0.85× / 0.67× / 0.67× at N=4/8/16 AND AR is more diverse (miss-lane N=16 uniqOut 0.148 vs 0.078) and
converts diversity to more correct rollouts (gt14: AR 6/16 exact vs engine 1/16) — the signal-quality
axis **deepens** the disconfirmation (**REFUTED**). The engine's only edge is 100% valid stops (48/48
groups), but a valid identical rollout is zero-advantage. **Consequence:** rollouts in the loop are
generated by stock guided-AR; the twin is the low-batch serving + conversion/scoring/validity spine.

---

## 6. FAILURE-MODE APPENDIX (the lessons — half the value of this artifact)

Each entry is a real failure that cost real time; documenting them is the point.

1. **Async-scheduler incompatibility (liveness/correctness).** Running FLARE under vLLM's async
   scheduler produced an async-rollback divergence at the first denoise position after a block
   boundary (**pos-33**, recurring) and a partial-canvas forward **stall** (a single denoise forward
   hung >10 min, non-terminating, when staged `valid_len` dropped below block width). Fix: force the
   **sync** scheduler (OPT-3, pin `d2fccab`) — 0/11 breaks at pos-33, all 63 turns complete, zero
   stalls. **Always run FLARE forced-sync.** (`engine_build_status.md` §0.D, §0.E.)

2. **Checkpoint-resume / spec-decode-index traps.** (a) The decode-at-scale CUDA IMA (5B): the
   MambaHybrid align spec-decode copy read the accepted token's GDN state from block-table column
   `src_col + (num_accepted−1)`, assuming speculative checkpoint columns that FLARE (no
   `speculative_config`) never allocates → out-of-bounds IMA at `num_computed≈1272`. Fix: feed the
   align state machine a neutral `num_accepted == 1` (pin `1e32dcd`). (b) The convert-after-RL
   seed-2 training was **chunked into 4 resumable 100-step segments**; the acceptance check is that
   the LR at every 100-step boundary matches the reference cosine to 4 s.f. (8.810e-6 / 5.283e-6 /
   1.581e-6 / 1.639e-10) — i.e. the 4 chunks reproduce a single continuous 400-step horizon.
   (`engine_build_status.md` §0.B; `convert_after_rl_result.md §5`.)

3. **Eval-drift / sampler-mislabeling (the most insidious).** The S1 gate "failure" was a **report
   labeling mistake**: the command path was mutable-remask while the report text called it legacy.
   The B@1000 historical **13/20 target was a drift artifact**; the corrected reproducible legacy
   continuity point for the exact saved adapter is **11/20**. The **"native 4 tok/fwd" claim was a
   sampler conflation** (`small_block_size=8` ≠ tok/fwd). The rule: never compare a score to the
   table until the report identifies the sampler path (function name + script sha256) and the reason
   for any change. (`REPRODUCE_V2.md §0/§4`, `l1_content_mix_result.md §2`.)

4. **The disqualified diagnostic (`measure_block_quality_curve.py`).** It is a **mutable-remask
   fixed-K diagnostic**, NOT the continuity sampler. At full denoise (nominal 1 tok/fwd) it already
   scores only **5/20 = 0.25**, failing the ≥0.60 anchor — so **all** of its higher-tpf points are
   meaningless. Every "parallel speed lane" number that traces back to it is disqualified. Use
   `eval_flare_stage1_ab_diffusion.py::full_context_sample_one` for continuity.
   (`goal_5x_rollout_b1.md`, `qwen35_block_quality_curve_gate_result.md`.)

5. **OOM / MAX_JOBS / GPU cage.** b16 batched rollout OOMs at gmu 0.74 (per-request GDN state caps
   concurrency tighter than AR's flat ~22 GB on a 32 GB card). Every heavy step was
   `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=4G` caged, one model at a time, with
   a `<2 GB` GPU pre-flight wait-loop and every command foreground ≤ 600 s. (`runs/p2_batched_rollout_bench`,
   `convert_after_rl_result.md §6`.)

6. **Audit theater (counters that cannot catch a regression).** The parity harness had a **gate-#3
   tautology**: `two_wave_wave1_projected_tokens` / `parallel_commit_forced_tokens` were hard-coded to
   0, so the "zero projected values" gate could never fail on a projecting engine (fixed `782b441`,
   real counters plumbed + a regression guard that would have wrongly passed before the fix). The
   `[MASK]`-sentinel suppression had been dropped in the port so a value token could argmax to the
   mask id and diverge silently (fixed `6482e1d`). The launcher exported only `FLARE_DECODE_POLICY`
   (zero consumers) → it **silently served the AR path under the diffusion model name** (fixed
   `b91184d0`, now exports `VLLM_QWEN3_5_FLARE`). Lesson: a "0 value projections" claim is evidence
   only if the counter can be shown to fire on a bad engine. (`engine_build_status.md` §2.)

7. **Block/chunk misalignment.** Engine default `_DEFAULT_BLOCK=32` is **half** a GDN chunk
   (FLA_CHUNK 64); a stock export has no `diffusion_config`, so 32 runs and commit boundaries land
   mid-chunk, so the fp32 boundary snapshot is not a clean recurrent checkpoint. Trained
   `canvas_length` must be a multiple of 64; set `VLLM_QWEN3_5_FLARE_BLOCK` accordingly. This is the
   root of the gt44 fp-residue class. (`engine_build_status.md` §2 config hazards, §5.2 above.)

---

## 7. OPEN FRONTIERS (stated plainly, evidence-ranked)

1. **L3 — raise reasoning K above 1 (the 5× bet, the biggest open lever).** North star: avg ≥ 5
   committed tok/fwd at held exactness on rollout reasoning content. The Amdahl arithmetic
   (`avg = 1 / (f_reason/K + f_value/1)`, grammar-forced folded out): for rollout `f_value ≈ 2–5%`,
   avg = 5 needs **reasoning-span K ≈ 5.4–6.3** (K→∞ ceiling 20–50). Distance from K_max(today)=1.0 is
   **~6× reasoning-span tok/fwd at held exactness** (~9× if L2's per-forward penalty is left
   unclosed). The only sampler reaching K≈4–8 today loses **15–45 exactness points** — that entire
   gap is what **S2 consistency-distillation + entropy-gated adaptive K** must erase. Values stay K=1
   forever (chain rule; measured top-1 conditional 0.238). **Unexecuted; feasibility unproven.**
   (`goal_5x_rollout_b1.md §L3`, `l1_content_mix_result.md §4`.)
   **S2 pilot (the cheapest decisive de-risk of this lever): DESIGNED, BLOCKED, UNRUN**
   (`s2_pilot_design.md` @ 9ce9445, `s2_pilot_result.md`). Pre-registered PASS = `tok/fwd ≥ 2.0` at held
   26/30 (net-loss ≤ 2, McNemar p ≥ 0.05) ∧ retention ≥ 13/20 ∧ tool-call 10/10 ∧ audits clean; KILL =
   any of those fail ⇒ K is a wall. Cannot yet be adjudicated **PASS or KILL** — two blockers: (1) the
   trained adapter `A_S2` does not exist because the self-trajectory corpus is only **66 raw / 31
   audit-clean**, below the **700-trajectory yield floor** (data step stalled, gen loop not alive); (2)
   two eval scripts are missing — `eval_flare_freetext_cad.py` (the new CAD sampler, must be authored +
   pass its byte-exact K=1 baseline test) and `export_qwen35_9b_fastdllm_vllm.py` (absent; no file
   matches its pinned sha). Preconditions verified good: 3 pinned eval-script sha256 match §2.4/§6, gate
   set = 30 prompts, KILL-0 base half PASSES (`mtplus1-merged` `mask_token_id=248077`, `bd_size=32`),
   anchor intact. **The 5×-vs-AR claim is UNRESOLVED — not funded, not retired.**

2. **L2 — per-forward parity (deterministic engineering).** Reasoning per-forward 25.8 ms → ~13 ms
   via #37 `fused_recurrent` GDN for 1-token probes + killing residual host overhead. Buys at most
   ~2×, K-bound. Gate: per-forward ≤ 13 ms **with the 233/247 byte-parity certificate not
   regressed.** (`goal_5x_rollout_b1.md §L2`; task #37 still open.)

3. **Strict 247/247 byte-parity (the last kernel-level turn).** Match HF's block#0 GDN fold
   granularity exactly (HF folds 32 incl. `prompt%32`; engine folds `32 − L%32`). Quality-neutral —
   its value is the strict "engine == HF by construction" certificate that flips the code default ON,
   not a quality gain (quality 130/247 is already met). Explicitly deferred as kernel work.
   (`p2_engine_battery_v3b_result.md`, §5.2.)

4. **L4 — NVFP4 / low-precision weight floor cut (ratio-neutral absolute-latency bonus).** Halving
   the 11.40 ms weight-stream floor → ~0.68 / 0.51 s/turn on matched-20 — **but blocked by the
   measured 5090 FP8-slower quant tax** (0.814× aggregate). NVFP4 must beat the bf16 HBM stream in
   practice on sm_120, not on paper, and it is a quality tradeoff. Measure on **this** card; trust
   nothing unmeasured. (`endgame_table_final.md` §(d).)

5. **Off-policy correction for the hybrid-generated rollout fraction only (task #30, RE-SCOPED).** The
   rollout-regime benches moved the *default* rollout path to stock guided-AR (behavior policy ==
   target policy → the diffusion-vs-AR correction is identity, hence unnecessary for AR-generated
   rollouts). The clipped per-token importance term `exp(logp_AR − logp_hybrid)` on value/free tokens
   remains required **only for the hybrid-generated fraction** of a mixed rollout batch; if the loop
   goes AR-only for rollouts, #30 is moot. (`methodology_diffusion_accelerated_rl.md §3`.)

6. **GDN read-only-denoise state discipline (the standing GPU-only crux).** Correctness of the
   snapshot/restore hinges on reconstructing EXACTLY the physical conv+ssm rows the GDN kernel writes;
   the current fingerprint is clean (6/6 bit-identical, §4.3) but the code reconstructs rows
   independently rather than reading the backend's `spec_state_indices_tensor`, and there is no
   assertion pinning layer count/identity. Harden with a hard-fail when readonly is enabled AND
   denoise rows exist AND caches is empty. (`engine_build_status.md` §2 CRUX.)

---

## 8. COMPUTE BUDGET SUMMARY

| stage | GPU | wall time | output |
|---|---|---:|---|
| Base conversion/materialization | CPU/GPU optional | IO-bound, minutes–tens of min | `models/qwen3.5-9b-fastdllm-init` |
| B@1000 two-stream | 1× RTX 5090 32 GB | 18,945 s / 5.26 h (peak 29,612 MiB) | `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` |
| B@1000 vLLM export | CPU + disk IO | tens of min, shard IO-bound | `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16` |
| Run-1 copy-grounding | 1× RTX 5090 32 GB | 2,068 s / 34.5 min | `runs/flare_redesign_run1_copy_grounded_qwen35_9b` |
| RL-v2 GRPO | 1× RTX 5090 32 GB | 14,342 s / 3.98 h | `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model` |
| RL-v2 vLLM export | CPU + disk IO | tens of min | `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` |
| Engine battery (247 turns) | 1× RTX 5090 32 GB | 0.626 s/turn agg | `runs/p2_engine_battery_v3b/` + `runs/p2_engine_nevertrain/` |
| Convert-after-RL (per seed) | 1× RTX 5090 32 GB | ~34 min train + ~20 min evals (≤2 GPU-h) | `models/qwen3.5-9b-fastdllm-mtplus1-merged` + `runs/convert_after_rl/…` |

---

## 9. PROVENANCE / RE-VERIFICATION LOG (this document)

- **Date:** 2026-07-05. Author: V3 assembly sweep (CPU-only; no GPU processes spawned).
- **Pin HEADs recomputed today** via `git rev-parse` / `git log`: qwen_diffusion `1d066be`
  (origin/main in sync + this commit); vLLM pin `qwen3_5-flare-modelstate` HEAD **`0b44dcc`**, chain
  `6b81154..0b44dcc` (9 commits, §2.3), upstream base `2665ed7`; flywheel fork
  `codex/qwen35-9b-feasibility` HEAD **`c3a7a753`** (drift vs V2's `b91184d0` documented §2.2).
- **All script sha256 in §2.4 recomputed today** with `sha256sum` on the working tree. The 15 V2
  qwen_diffusion scripts and the 3 V2 flywheel export/serving/parity-gate scripts are **byte-identical
  to V2**; the flare-hybrid launcher and the two engine-harness scripts + three FLARE source files are
  freshly hashed here.
- **Model artifacts verified present:** `models/qwen3.5-9b-fastdllm-{init,b1000-vllm-bf16,rlv2-vllm-bf16,mtplus1-merged}`;
  RL-v2 export manifest confirms `replacement_count=427`, `lora_merge_count=152`, `lora_scale=2.0`,
  adapter `runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`.
- **No numbers re-derived.** Every value is transcribed from the source-of-truth artifacts in §1;
  each carries its citation inline.
