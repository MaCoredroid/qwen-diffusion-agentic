# SWE-class Agentic End-Goal — Staged Serve → NVFP4 → SWE-Verified Plan

**Author:** monitor synthesis, 2026-07-05. **Task:** #54 (SWE-VERIFIED END-GOAL). **Owner discipline:** [[qwen-diffusion-commit-workflow]] (commit+push each step, narrate reasoning), [[gpu-utilization-standard]], [[diffusion-promotion-discipline]], [[native-function-format-rule]].
**Mode of this doc:** CPU-only synthesis of three audits (serve-path, nvfp4, harness) into a staged, gated, GPU-hour-costed plan. No CUDA was run. All speed/quality numbers below are "measure, never assume."

## 0. What we are actually building (frame — read first)

The 5×-vs-AR north star is **RETIRED** (KILL, `ffaa23b`, K-factor wall). The surviving, funded goal is: **stand the diffusion-9B up as a real OpenAI `/v1` endpoint driving an agentic SWE-bench-Verified loop through Qwen Code, and measure it honestly against the AR-9B of the same weights.** The established result we are extending: engine hybrid-clean = **130/247 exact, quality == HF row exactly, +6 vs stock-AR, speed parity-class vs fair cudagraph-AR (0.94×), 4.12× vs our prior serving** (`endgame_table_final.md`, `43c5bf8`); conversion-after-RL **PRESERVES** (`b019b86`).

Therefore the Stage-C **win condition is not a speed multiplier** — it is: *diffusion resolve-rate ≥ AR resolve-rate on the same SWE subset, at held byte-losslessness across multi-turn context reuse, with honest per-turn economics reported both ways.* Speed is a reported number, not a gate.

## 1. THE GATING DEPENDENCY — lossless-APC: RE-ANCHORED 2026-07-05 (gate battery `8b98aaf` — quality-lossless CERTIFIED, dependency DISSOLVED on the QUALITY axis)

**The blocker this section was written around is GONE for Stage-C's actual win condition.** The lossless-APC gate battery ran live at HEAD `9cb5e7a` (RTX 5090, RAM cage, commit `8b98aaf`, artifacts `runs/lossless_apc/gates2/`) and found: the post-Stage-3 shipped engine is **already effectively cache-lossless on the QUALITY axis, without the seam.** **The GPU is now FREE (0%, 2579 MiB), not owned by an APC build.**

**What the battery certified (the gates this plan gated on):**
- **(a) artifact-census byte-match — CLEARED.** Census turns `{gt20,gt21,gt60}` + `gt130,gt176` all byte-match HF under **warm cache-on** (first_div=None each). These were pre-Stage-3 (`e5496cc`) artifacts that Stage-3 (`95d8b47`) already cleared.
- **(b) full-battery cache-on == fresh — MET on the quality axis.** matched-20 full-63: byte-parity **62/63** vs HF (lone break gt44, path-invariant fp-residue that breaks fresh too), **exact_args EXACTLY 47/63 == HF per-turn (0 wins / 0 losses)**, reproduces `parity_cert_freshboot.jsonl`. never-train full-184: exact **83/184 == HF**; warm cache-on vs cold fresh-proxy **174/184 byte-identical, exact 83==83 (ZERO quality-affecting turns)**. **exact_args is byte-stable and fully APC-invariant across warm/cold/gate-on/off/eager/cudagraph** ⇒ the cache-on quality certificate holds *with* cross-turn reuse.
- **(c) economics input — refold cost measured:** seam inert ⇒ **0 live overhead**; isolated ~**9.0 ms per 1024-crossing** (0.374 ms/layer × 24); **net APC speedup unchanged from the 1.23× lossy number**, amortizes to ~0 for the `<2048`-token turns that dominate.

**What is NOT met — and why it no longer gates this plan:** the canonical-publish seam (W1+W2, `9cb5e7a`) is **STILL LIVE-INERT** (`publish`/`apply` fired 0× everywhere; gate-ON==gate-OFF byte-identical; **NOT promoted, default OFF, not pushed on the pin**). The **only remaining gap is strict BITWISE losslessness on near-tie tokens** — which the inert seam targets but a *chunked* kernel cannot fully close anyway (rootcause Refinement 1: needs the sequential-recurrent republish design). **Stage-C's win condition (§0) is "resolve-rate parity at held byte-losslessness across multi-turn reuse" — that "held losslessness" is the QUALITY axis (exact-args APC-invariant, cache-on == fresh), which is MET.** Strict-bitwise near-tie parity is a nicety, not a Stage-C gate; it is deferred and only worth completing W1+W2 if a future requirement demands it.

**Net effect on this plan:** the ~1.5–2 wk eng + ~20–35 GPU-h long pole is **retired** (the battery is the deliverable; the seam is parked). **Stage A can start now against the Stage-3 shipped engine (default gate OFF); the multi-turn-reuse cert (A7) is now a re-verification of an already-certified quality-lossless property, not a blocked byte-debug campaign.**

## 2. Machine topology (three machines)

| Role | Host | Arch / GPU | Use in this plan |
|---|---|---|---|
| Serving (both endpoints) | 5090 `mark-OMEN` (this box) | x86 / RTX 5090 sm_120, 32 GB | AR-9B (`:9951`) **and** diffusion-9B (`:9952`) `/v1`. **CPU-only now — GPU owned by lossless-APC build.** |
| x86 SWE eval worker | Alienware RTX 5080 | x86 / sm_120, 16 GB, ssh `alienware` `100.83.202.36` | Official `swebench` docker harness (native x86, no aarch64 build friction). Intermittently reserved for user work. |
| Flywheel-native host | GB10 `gx10-edb9` | aarch64 / GB10, 117 GB, `100.103.10.122` | Reference only; the Codex SWE campaigns ran here. Not on this plan's serving path. Cross-boot byte-gate is INVALID on GB10 (forks at tok 11–71) — same-boot oracle only if ever used. |

**Simplification vs the flywheel:** serving is already on x86 (5090), so docker eval runs natively on the 5090 or offloads to alienware — no arch friction either way.

---

## STAGE A — Serve bring-up: certified online `/v1` for BOTH endpoints

**Objective:** the diffusion endpoint served by `AsyncLLM` behaves byte-identically to the certified offline `LLM` path, with OpenAI tool-calling wired to the FLARE grammar, so Qwen Code can drive it.

**De-risking finding (serve-path audit):** the offline `LLM` path is *already* a ZMQ background subprocess (`VLLM_ENABLE_V1_MULTIPROCESSING` default True, `llm_engine.py:157`); server `AsyncLLM` uses the **same** EngineCore subprocess via `AsyncMPClient`. The sync-scheduler force + `is_diffusion` propagation happen engine-config-wide (`config/vllm.py:916-935`), identical in both modes. **The block-commit/denoise/GDN/scheduler loop is byte-identical offline-vs-serve.** Every real gap is front-end wiring, not the engine loop. This substantially de-risks Stage A.

### Work items

| ID | Item | Source | Effort |
|---|---|---|---|
| **A1** | **Launcher selects hybrid_clean.** Add `export VLLM_QWEN3_5_FLARE_DECODE="$DECODE_POLICY"` after `qwen35_9b_flare_hybrid_serve.sh:121`. Today the launcher exports only `FLARE_DECODE_POLICY` (zero consumers) and never sets the decode-mode selector → engine silently boots **canvas** (Gumbel sampler, no grammar/FSM path, `qwen3_5_flare.py:274-276,321-324`) despite the `hybrid-clean` name. Same bug class already fixed for the engine gate, left unfixed for the decode selector. | serve-path G1 | trivial (1 line), load-bearing |
| **A2** | **Bridge OpenAI `tools` → FLARE grammar.** hybrid_clean reads its per-request tool grammar from `SamplingParams.extra_args["tools"]/["grammar_topk"]` (`hybrid_clean.py:29-38`, `qwen3_5_flare.py:1380-1433`); the OpenAI serving layer populates `extra_args` only from `vllm_xargs`+`kv_transfer_params` (`chat_completion/protocol.py:661-664`) — it never copies `request.tools`. Without this, requests hit the free-text/L0 path with `schemas={}` → grammar disabled → the valid-XML + exact-arg-name safety net behind the 47/63 cert is **off**. Fix (pick one): (a) proxy/shim rewrites each request to `extra_body.vllm_xargs={"tools":<tools>,"grammar_topk":256}`; or (b) serving patch: when the FLARE model is loaded, copy `request.tools`→`extra_args["tools"]` in `to_sampling_params`. Verify `parse_hybrid_clean_request` accepts the OpenAI tool-dict shape. | serve-path G2 | small–medium |
| **A3** | **Pin temp-0 greedy for the certified regime.** OpenAI default `temperature=1.0` (`completion/protocol.py:231-249`) exercises the seeded-categorical branch (`qwen3_5_flare.py:1453-1465`), uncertified — byte-parity cert is temp-0 greedy. Set the served default to `temperature=0` (launcher/registry override), and document that temp>0 is functional+seed-reproducible but off-cert. Qwen Code must send temp-0 for the certified comparison arm. | serve-path G3 | trivial |
| **A4** | **Streaming / stop / finish_reason plumbing.** Confirm SSE streaming, stop-token handling, and `finish_reason=length` semantics on the AsyncLLM front-end match offline. From the FR13 config research: treat `finish_reason=length` as **continue-not-giveup** (avoid the "char-8" truncation 400); `reasoning_effort high→medium` for the open model. | serve-path (front-end bucket) + harness FR13 note | small |
| **A5** | **AR endpoint sanity.** AR endpoint (`qwen35_9b_host_vllm_serve.sh`, `.venv-vllm` 0.23, `:9951`, `qwen3_xml` parser, FR13 align-APC) is stock vLLM online serving — already sound. Just confirm it boots current and add the MTP `--speculative-config` only if we want the AR speed arm (optional). | harness §1 | trivial |
| **A6** | **Online-vs-offline byte-parity certificate (the real Stage-A work).** Drive the served diffusion endpoint over the matched-20 + never-train census turns and assert per-turn byte-parity vs the offline `LLM` certificate (`nevertrain_parity_cert_resetapc.jsonl`). Single-turn first (no reuse). | serve-path (architecture finding → must be verified, not assumed) | medium |
| **A7** | **Multi-turn reuse cert — BLOCKED on lossless-APC (a)+(b).** Repeat A6 with APC ON across a growing multi-turn prefix; require cache-on == fresh on every APC-class turn. Cannot pass until §1 lands. | lossless-APC §3(a,b) | gated |

### Stage-A gates
- **A-G1:** launcher-booted engine reports `decode_mode=hybrid_clean` (not canvas) and grammar active (schemas non-empty on a tool request).
- **A-G2:** single-turn online decode byte-identical to offline cert on the 5/5 census turns + a 20-turn spot battery (A6).
- **A-G3 (gated on §1):** multi-turn cache-on online decode byte-identical to fresh (A7).
- **A-G4:** CPU tests green; no host-bound regression on the serving path (util standard).

### Stage-A GPU-hours
Bring-up smoke + A6 single-turn cert, iterated: **~6–12 GPU-h.** A7 multi-turn cert rides lossless-APC gate-(b) reruns (counted there). **Stage-A net new: ~6–12 GPU-h.** Eng ~3–5 days (A1/A3/A5 trivial; A2/A4/A6 the substance).

### Stage-A status — 2026-07-05 (A1/A2/A3/A5/A6/A7 DONE; first agentic-CLI loop CLOSED on the diffusion engine)
- **A1/A2/A3 DONE** (launcher decode-mode fix + OpenAI-tools→FLARE bridge + temp-0 default; flywheel `f063387b`/`07d31dde`, pin `09ab8e4`/`b5fcb3d`). **A6+A7 byte cert DONE** (`3d86df9`, `runs/stage_a_cert/`): AsyncLLM server decodes **byte-identical** to the offline `LLM` cert on all 20 matched turns incl. the lone fp-residue break, token+byte+quality 10/10.
- **A5 + first qwen-code↔diffusion loop DONE** — `task #61`, `runs/stage_a_smoke/report.md` (2026-07-05). Qwen Code `@0.19.2` (already installed; no install needed) drove the SAME planted-bug repo-edit task through **both** endpoints, one server at a time in the RAM cage: **diffusion `:9952` (hybrid_clean) and stock-AR `Qwen/Qwen3.5-9B@c202236` `:9951` (stock vLLM 0.23, cudagraph).** **Both arms COMPLETE the task** (read → edit → run-test loop closes; correct minimal 1-line diff to the expected file; independent tests pass). Diffusion 7 turns / 12.98 s / 23,139 tok; AR 5 turns / 7.70 s / 19,166 tok.
- **Diffusion engine counters CLEAN** (`diffusion_engine_counters.json`): `decode_mode=hybrid_clean` (A-G1 met live), **all 15 hybrid_clean requests on the grammar path** (A2 bridge fired every turn, zero free-text fallback), **`projected_value_tokens_exact=0` on every request** (tripwire held), all `stop_reason=complete_tool_call`, APC hit-rate 78→82.6% (real cross-turn reuse), 0 errors / 0 HTTP-4xx-5xx.
- **A-G1 MET** (live). **A-G2/A-G3** = the A6/A7 byte cert (already MET). **A-G4** = CPU tests green (prior). **Stage A is complete for the serve-bring-up objective.**
- **R4 (qwen-code↔diffusion tool loop) — first-class finding, MEDIUM.** The loop **closes and completes**, but the diffusion arm **never emits a terminating free-text turn**: after tests pass it re-issues the identical verify `run_shell_command` until Qwen Code's always-on loop-detector halts it (CLI exit 1; task still done). **Structural, not prompt-fixable** (an explicit "stop after pass" system prompt reproduced the same 7-turn halt) — the A2 grammar is compiled on every turn that carries `tools`, so a free-text "done" turn is out-of-grammar. Stock-AR terminates cleanly (exit 0). **Stage-C C1 driver must (a) score independent patch/test outcome, not CLI exit, and (b) evaluate a top-level `free-text | tool-call` grammar alternation (or drop `tools` on the post-pass turn) so the diffusion agent can terminate cleanly.**

---

## STAGE B — In-house NVFP4 quantize + quality battery

**Objective:** produce a W4A4 NVFP4 export, serve it on the 5090 with zero new serve-side installs, and **measure** whether it beats bf16 wall-clock on sm_120 — with the FP8-was-SLOWER discipline forcing measurement, not assumption. NVFP4 is a **latency bonus, ratio-neutral** (L4); it is **not a blocker** for Stage C — Stage C runs on bf16 if NVFP4 fails its wall-clock gate.

**Verdict (nvfp4 audit): FEASIBLE, build-don't-buy.** The pin can already *load and serve* a compressed-tensors NVFP4 (W4A4) checkpoint on the 5090 with zero new installs: `compressed_tensors 0.17.0` present in the pin venv, QuTLASS W4A4 kernels compiled (`vllm/_qutlass_C.abi3.so`, 157 syms), `modelopt.py` needs no external `modelopt` to serve. The **only** missing piece is the PTQ/export tool (absent everywhere) — one `pip install`, and it is a **GPU calibration task** (forward passes) that waits behind lossless-APC.

### Work items

| ID | Item | Source | Notes |
|---|---|---|---|
| **B1** | **Install the PTQ tool (GPU).** `pip install llm-compressor` (recommended — emits compressed-tensors NVFP4 and is the **only** path to the QuTLASS Hadamard-transform scheme, group_size 16) or `nvidia-modelopt[torch]`. Absent in every venv today. | nvfp4 §1 | GPU calibration → after lossless-APC frees GPU |
| **B2** | **Author the exclusion recipe.** W4A4 targets: `in_proj_qkvz`, `out_proj` (big GDN GEMMs), full-attn `qkv_proj`/`o_proj`, MLP `gate_up_proj`/`down_proj`. **Exclude (keep bf16):** `in_proj_ba` (β/α delta gates — line 488 "doesn't support blockwise fp8"; W4A4 error compounds over recurrent state; ~0.26M params, near-free to protect); `conv1d` (constructed with no quant_config, conv kernel can't consume FP4 — must be named in the PTQ `ignore` list so the export tool doesn't try); `lm_head` (~1.02B params ≈ 2 GB ≈ 10% of the 19.31 GB stream — exclude from W4A4 for logit fidelity on exact-arg tokens; optionally A/B a W4A16-Marlin lm_head as a stream bonus). `dt_bias`/`A_log`/RMSNorm auto high-precision. | nvfp4 §2 | the load-bearing correctness item |
| **B3** | **Calibrate + export (GPU).** Run PTQ calibration forward passes on a representative agentic/tool-call corpus; emit the compressed-tensors NVFP4 checkpoint + `config.json` + manifest. | nvfp4 §1 | GPU |
| **B4** | **Serve-load the NVFP4 export** through the pin (zero new serve installs) via the Stage-A launcher. | nvfp4 verdict | reuses Stage A |
| **B5** | **Quality battery == bf16.** Re-run the 247-turn battery (matched-20 63 + never-train 184) on the NVFP4 export; require quality within noise of the bf16 130/247 (per [[diffusion-promotion-discipline]]: only ship on held raw/constrained quality). | endgame battery + nvfp4 | GPU |
| **B6** | **Wall-clock A/B — the decisive measurement.** NVFP4 vs bf16 s/turn AND per-forward on the 5090; the bf16 floor is 18.5 ms/forward = 11.40 ms weight-stream (19.31 GB / 1.79 TB/s) + 6.54 ms non-width-reducible. NVFP4 cuts the 11.40 ms stream term ~4× **iff** the QuTLASS W4A4 kernel is faster than bf16 on sm_120 — **measure, FP8 was slower.** | endgame speed physics + nvfp4 caveat | GPU |

### Stage-B gates
- **B-G1:** NVFP4 checkpoint loads + serves on the 5090 with no new serve-side installs.
- **B-G2 (quality):** 247-turn battery within Wilson-CI noise of bf16 130/247, no capability class collapse; exact-arg tokens unharmed (lm_head-exclusion working).
- **B-G3 (speed, PASS/PARK):** NVFP4 s/turn < bf16 0.626 s/turn by a real margin → **PROMOTE** as the Stage-C serving weights. Else **PARK** NVFP4 (bonus not earned on sm_120, exactly like FP8) → Stage C runs bf16. Either outcome is a valid, recorded result.

### Stage-B GPU-hours
Calibration ~2–4 GPU-h; quality battery ~6–10 GPU-h (a full pass is ~2–4 GPU-h, budget iteration); wall-clock A/B ~2–4 GPU-h. **Stage-B total: ~12–20 GPU-h.** Eng ~3–5 days (recipe authoring + calibration debugging). One-time `pip install`.

---

## STAGE C — SWE-bench-Verified: N=5 smoke → N=25–50 subset, through Qwen Code, AR vs diffusion

**Objective:** drive both endpoints through **Qwen Code** on SWE-Verified instances, eval with the official docker harness, and report resolve-rate + per-turn economics AR vs diffusion at held byte-losslessness.

**Harness reality (harness audit):** the flywheel's `run_swe_bench_q36_a.py` (1298 lines) drives **Codex-in-docker** (`codex exec --json`, Responses API via `inference_proxy.py`), **not Qwen Code**. Qwen Code (npm `@qwen-code/qwen-code@0.19.2`, already a dev dep here) speaks **Chat Completions**, so it hits vLLM `/v1/chat/completions` **directly — no Responses-API proxy needed.** We already have `scripts/qwen_code_sglang_proxy.py` + `run_qwen_code_sglang_smoke.sh` as the smoke skeleton. The gap is a **Qwen-Code SWE driver** (the missing piece) reusing the flywheel's workspace-hydrate / `AGENTS.md`-drop / `git diff → patch.diff` / eval-offload plumbing.

### Work items

| ID | Item | Source | Notes |
|---|---|---|---|
| **C1** | **Qwen-Code SWE driver.** Port `run_swe_bench_q36_a.py`'s per-instance orchestration (hydrate at `base_commit` → drop problem_statement → run agent with wall-clock cap → `git diff` → `patch.diff`) to launch **Qwen Code** headless against `/v1/chat/completions` instead of `codex exec`. Keep the state-conditional retry prompts (give-up / empty-patch / setup-loop). Point `--model` at `qwen3.5-9b-flare-hybrid-clean` (`:9952`) or `qwen3.5-9b-...-bf16` (`:9951`) per arm. | harness §1 (new item) | medium (the main Stage-C build) |
| **C2** | **Data + subsets (already committed).** `build_swe_bench_subset.py` over `princeton-nlp/SWE-bench_Verified`, pre-registered lists on disk: **Tier0=20, Tier1=100, Tier2=500** (`docs/reports/auto_research/swe-bench-tier{0,1,2}-verified-instances-*.json`). N=5 smoke = first 5 of Tier0; N=25–50 = Tier0 (20) or a 25–50 stratified slice of Tier1. | harness §1 | done — reuse |
| **C3** | **Eval harness.** `swe_eval_offload.py` + `codex_bench_eval_swe.py` verdict classifier (exit 0/1/2 = resolved/failed/crash) → official `swebench` docker. Run natively on the 5090 or offload to `alienware` (native x86, avoids the ~20–35% aarch64 build-fail rate that forced the flywheel's offload). Respect the alienware reservation notes. | harness §1 | reuse; wire to Qwen-Code patches |
| **C4** | **N=5 smoke, diffusion arm.** 5 instances through Qwen Code → diffusion `:9952` → docker eval. Purpose: prove the *loop closes* (tool calls parse, patches apply, eval runs, no engine crash) — not resolve-rate. | harness §N=5 plan | first GPU milestone |
| **C5** | **N=5 smoke, AR arm.** Same 5 through AR `:9951`. Sanity that both arms drive Qwen Code identically. | harness | pairs with C4 |
| **C6** | **N=25–50 subset, both arms.** Paired run (same instances, both endpoints), temp-0 greedy, native `qwen3_xml` tool format both arms ([[native-function-format-rule]]). Report resolve@1, per-turn s, turns/episode, APC hit-rate at 1024 granularity, and the diffusion-vs-AR resolve delta with paired stats. | harness + endgame framing | the deliverable measurement |
| **C7** | **Losslessness assertion in-loop.** During C6 diffusion arm, assert the online multi-turn cache-on decode stayed byte-lossless (Stage-A A-G3 / lossless-APC gate-c telemetry) — a resolve number under a lossy cache is not creditable. | lossless-APC §3(c) | gate, not optional |

> **⚠ Leakage caveat — belt-lever ENACTED 2026-07-07 (USER greenlit).** The SFT
> data-gen pool (`runs/swe_datagen_s1`) now trains on the **387 Verified-adjacent
> instances** (all 500 SWE-bench_Verified test ids MINUS the 113 still-held eval ids).
> Tier2=500 is **no longer a training firewall** — only the C2 eval rings **inner5 (5)
> ∪ Tier0-20 ∪ Tier1-100 = 113 DISTINCT ids** are held out, hash-asserted (KILL-D1).
> The C-stage eval numbers above (N=5 ⊂ Tier0, N=25–50 ⊂ Tier1) stay on **never-trained**
> instances, so they remain creditable; but the SFT pool is now **repo/era-adjacent to
> Verified**, and any FUTURE eval expansion beyond Tier1-100 would collide with trained
> ids. Full mechanics + trade in `runs/swe_datagen_s1/USER_LEVER_BELT.md`.
>
> **Dual-source scoring caveat (fixed 2026-07-07).** Making the pool dual-source
> surfaced a scoring bug: the first mixed batch (batch_0007) *generated* real patches
> but scored 0 because `datagen_score.sh` fed the merged predictions file to each
> single-source swebench harness (which rejects any prediction id outside its dataset,
> before the `-i` filter) → both aborted. Fixed via per-source filtered prediction
> files; batch_0007/0008 marked `infra_invalid`. **Standing rule:** any dual-source
> change must pass a both-sources LIVE gate (≥2 SWE-Gym + ≥2 Verified real episodes,
> full pull→gen→score). See `USER_LEVER_BELT.md` / REPRODUCE_V3 §3.14.

### Stage-C gates
- **C-G1 (smoke):** N=5 loop closes on **both** arms — patches apply, docker eval returns verdicts, zero engine crash. (Resolve count is informational here.)
- **C-G2 (subset):** N=25–50 completes both arms; **diffusion resolve@1 ≥ AR resolve@1 − (paired-CI margin)** — i.e. parity-or-better, the honest bar (not a speed multiplier).
- **C-G3 (losslessness):** every scored diffusion episode ran under a byte-lossless cache-on cert (C7). A resolve under lossy APC is disqualified.
- **C-G4 (economics, reported):** per-turn s/turn and APC prefill-tokens-saved reported both arms — the Stage-A/lossless-APC (c) payoff number in a real agentic setting.

### Stage-C GPU-hours
SWE episodes are **agent-bound, long, multi-turn** (flywheel used a 25-min wall/instance); GPU is occupied serving the whole episode, arms run sequentially on one GPU. N=5 smoke ×2 arms ≈ **~2–4 GPU-h.** N=25–50 ×2 arms at ~10–25 min wall/episode ≈ **~25–45 GPU-h** (serving occupancy; docker eval on alienware does **not** consume the 5090). **Stage-C total: ~27–49 GPU-h.** Eng ~1–1.5 weeks (C1 driver + C3 wiring + debugging the qwen-code↔diffusion tool loop are the real cost).

---

## 3. Dependency graph + GPU-hour rollup

```
lossless-APC (task #53) ─ quality-lossless CERTIFIED (battery 8b98aaf); GPU FREE
  seam parked (inert, default OFF); strict-bitwise deferred, NOT a Stage-C gate
        │
        ▼
   Stage A (serve bring-up) — UNBLOCKED, starts now
   A1/A3/A5 CPU-side ─┐
   A2/A4/A6 single-turn │ ~6–12 GPU-h
   A7 multi-turn cert ◄─┘  (re-verify already-certified cache-on==fresh quality)
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
   Stage B (NVFP4, PARALLELIZABLE)            (Stage C needs A-G2/A-G3)
   ~12–20 GPU-h, PASS→promote / PARK→bf16
              │                                        │
              └──────────────► Stage C (SWE-Verified) ◄┘
                    N=5 smoke → N=25–50, ~27–49 GPU-h
```

**GPU-hour rollup (5090, greedy, single-stream) — RE-ANCHORED 2026-07-05:**
| Phase | GPU-h | Blocking? |
|---|---|---|
| lossless-APC (task #53) | DONE (battery `8b98aaf`) | ~~owns GPU~~ **freed; quality-lossless certified** |
| Stage A (net new) | 6–12 | A7 now re-verifies an already-certified quality cert (no longer APC-blocked) |
| Stage B (NVFP4) | 12–20 | parallelizable; not a Stage-C blocker |
| Stage C (SWE) | 27–49 | gated on A-G2/A-G3 only |
| **This plan (A+B+C, APC pre-req retired)** | **~45–81 GPU-h** | |

## 4. Realistic calendar (one researcher, consumer GPU, single-stream, measure-not-assume)

Anchored at **2026-07-05**, RE-ANCHORED after the lossless-APC gate battery (`8b98aaf`). **The long pole is retired:** cache-on quality == fresh is CERTIFIED via Stage-3 (the battery is the deliverable), the seam is parked inert, and **the GPU is already free.** Stage A no longer waits ~2 weeks for an APC byte-debug campaign — it starts immediately against the Stage-3 shipped engine (default gate OFF). This pulls the whole calendar forward ~2 weeks. Calendar assumes intermittent alienware reservations and normal debug slippage.

| Window | Milestone |
|---|---|
| **Jul 5 → ~Jul 12** | Stage A opens now: A1/A3/A5 (CPU, trivial) + A2/A4/A6 online single-turn byte-parity cert (A-G2) against the Stage-3 engine. In parallel (CPU): C1 Qwen-Code driver drafted, B2 NVFP4 recipe authored. |
| **~Jul 10 → ~Jul 16** | A7 multi-turn reuse cert (A-G3) — now a **re-verification** of the already-certified cache-on==fresh quality property (not a blocked byte-debug). |
| **~Jul 14 → ~Jul 24** | Stage B in parallel: install PTQ, calibrate, quality battery (B-G2), wall-clock A/B (B-G3 → promote or park). |
| **~Jul 18 → ~Jul 22** | Stage C N=5 smoke, both arms (C-G1) — first real SWE loop closes. |
| **~Jul 22 → ~Aug 4** | Stage C N=25–50 subset, AR vs diffusion (C-G2/G3/G4) — the deliverable measurement. |

**Realistic end-to-end: ~4 weeks (early July → ~early August 2026)** — pulled in from ~6–7 wk because the lossless-APC long pole retired quality-certified. N=5 smoke reachable ~2 weeks in (~Jul 20); the N=25–50 AR-vs-diffusion verdict ~4 weeks in (~early Aug). NVFP4 is a bonus fork that does not move the Stage-C date. **Residual risk moves off APC and onto Stage-A serve-wiring + the qwen-code↔diffusion tool loop (R4).**

## 5. Honest risk register (top 5)
- **R1 — lossless-APC — RETIRED as a schedule risk (2026-07-05, battery `8b98aaf`).** Cache-on quality == fresh is certified via Stage-3; the seam is parked inert; the GPU is free. The residual — strict-bitwise near-tie parity — is NOT a Stage-C gate (Stage-C's "held byte-losslessness" is the quality axis, which holds). The only way this re-emerges as a risk is if a downstream requirement demands strict-bitwise losslessness, which would re-open W1+W2 (hook `_forward_core_decode_non_spec`, drop the chunked-prefill guard) — deferred, not scheduled.
- **R2 — NVFP4 fails wall-clock (B-G3), like FP8 did on the 5090.** Fully expected-possible; PARK and ship bf16. Not a blocker (built into B-G3 as PASS/PARK).
- **R3 — resolve-rate parity not met (C-G2).** 9B on SWE-Verified is a hard task; both arms may resolve few. The creditable claim is the **paired AR-vs-diffusion delta**, not an absolute resolve number — still a valid result even if both are low.
- **R4 — Qwen-Code tool-loop divergence.** Grammar-off free-text path (if A2 shim misfires) silently drops the exact-arg safety net. Mitigation: A-G1 asserts schemas non-empty; native `qwen3_xml` format both arms.
- **R5 — alienware contention.** Eval offload competes with user reservations. Mitigation: docker eval can also run natively on the 5090 (x86), decoupled from serving windows.

## User directive (2026-07-05): Stage C runs through QWEN CODE with LumoFlyWheel as the REFERENCE implementation
The SWE-Verified driver must follow the flywheel's existing SWE machinery, not a from-scratch harness:
`run_swe_bench_q36_a.py` (updated in the 2026-07-05 upstream sync, +72 lines), `scripts/swe_x86_helpers/`
(offload_codex_proxy.sh / relaunch_proxy_remote.sh — the two-machine proxy topology), and the hardened
`inference_proxy.py` (+363 lines). Port the flywheel Codex-orchestrator pattern to Qwen Code as the agent
CLI; keep their episode/eval/reward conventions so results are comparable with the flywheel's own SWE runs.
On any harness wall: pull the flywheel upstream first (standing rule), then adapt.

## Stage-B addendum (2026-07-05): per-forward overhead re-attack is a NVFP4 PREREQUISITE
Reconciliation: engine forward = 17.8-18.3ms (tool-call) / 25.8ms (free-text) vs AR-cudagraph 10.72ms —
per-TOKEN parity on tool-call benches is the grammar-scaffold subsidy (~1.7 tok/fwd), not forward parity.
At bf16 the ~6.4ms non-gemm overhead hides under the 11.4ms weight-stream floor (36% of forward); under
NVFP4 (floor -> ~3-4ms) it becomes DOMINANT and the ratio degrades to ~0.35x unless cut. Therefore Stage B
includes: (B-P0) kernel-level per-forward trace at FP4-projected shares — re-litigate every "non-reducible
at bf16" item (GDN decode-class routing, align-postprocess fusion, launch gaps); (B-P1) free-text decode
policy fix: stop full-canvas re-denoise per committed token + EOS overshoot (~30% on the free-text path,
25.8 -> ~18ms class; SWE turns are free-text-heavy). Both measured before/after with the parity gates.

## Stage-C addendum 2 (user decision, 2026-07-05): LOCAL eval only
SWE-Verified evaluation (patch apply + tests + resolve scoring) runs LOCALLY on this machine. The
alienware x86 offload path (swe_x86_helpers) is excluded by user decision. Local docker/swebench setup is
in-scope work and part of the reproducible recipe.

## Stage-C status — 2026-07-05 (C4+C5 DONE: N=5 paired smoke, first SWE loop closes on the diffusion engine)
- **C4+C5 DONE** — `runs/stage_c_n5/report.md` + `paired_summary.json`. 5 Tier0 SWE-bench_Verified
  instances (`django-11119/12754/13741`, `pytest-8399`, `sympy-13757`) x 2 arms, one server at a time in
  the RAM cage (all 5 on AR `:9951` -> kill -> all 5 on diffusion `:9952` hybrid_clean -> kill). AR arm
  4.5 min wall, diffusion arm 12.9 min. Clean teardown verified both arms. `predictions.jsonl` emitted
  per arm (5+5 rows) for later real resolve@1 scoring.
- **C-G1 (smoke loop closes) MET behaviorally.** Tool calls parse on both arms, real edits land, the
  verdict classifier returns for every instance, **zero engine crash**. Diffusion engine counters CLEAN:
  `decode_mode=hybrid_clean` (A-G1 live), **153 hybrid_clean requests all on the grammar path** (A2 bridge
  fired every turn), **`projected_value_tokens_exact` all-zero, 0 violations**, stop_reasons
  147x`complete_tool_call`, APC hit-rate 88.3->88.9% (real cross-turn reuse), 0 error lines. The 5 HTTP-400s
  are the context-ceiling limit below, not engine faults.
- **Verdicts are MOCK, not docker resolve@1** — docker+swebench absent on the 5090, alienware unreachable
  this session (and offload out-of-scope per addendum 2). Mock = extracted-patch-lines superset gold-lines
  (strict; a genuine-but-different fix scores `failed`). Rollup: **AR mock-resolved 1/5, made a real edit
  3/5, exited clean 5/5; diffusion mock-resolved 0/5, made a real edit 1/5, exited clean 2/5.** Real
  resolve@1 waits on local docker/swebench (in-scope recipe work, C3).
- **R4 REPRODUCES at SWE scale and sharpens benign->malign.** Two diffusion episodes (`pytest-8399`,
  `sympy-13757`) halt on qwen's `consecutive_identical_tool_calls` guard (exit 1) **before landing any
  edit** (empty patch) — vs Stage-A where it fired *after* a correct edit. One (`django-11119`) hits the
  50-turn `FatalTurnLimitedError` (exit 53) but produces a 977B patch. Same structural root cause: the A2
  grammar requires a tool call every turn, so diffusion never emits the terminating free-text turn AR uses
  to exit clean (AR exit-0 5/5 vs diffusion 2/5; turn-count asymmetry reproduces and amplifies, e.g.
  django-11119 diff 50 vs AR 8).
- **NEW SHARED BLOCKER — 32,768 context ceiling (both arms).** `max_model_len=32768` + proxy
  `max_tokens=2048` -> usable input ~30,720; long episodes 400 out (AR 3x400, diff 5x400). Qwen Code has no
  compaction. Confounds the long episodes on *both* arms; **not** an engine defect.
- **Go/no-go for C6 (N=25-50): CONDITIONAL GO, gated on three fixes first** — (1) raise `max_model_len` to
  40-48k and/or enable qwen-code compaction (arm-neutral context fix); (2) land the R4 free-text|tool-call
  grammar alternation (or drop `tools` post-work) so diffusion can terminate; (3) stand up local
  docker/swebench for real resolve@1. Running C6 today for a resolve verdict would score a diffusion
  termination artifact, not capability. **Do not launch N=25-50 for a verdict until (1)+(2)+(3) land.**

## Stage-C addendum 3 (user, 2026-07-05): runtime alignment is OURS to fix
The flywheel agent runtime misses official SWE-Verified per-instance dependencies (agent cannot run tests
in-episode). Fix on our end (do not wait for the flywheel team): episodes run inside the official
per-instance swebench images (runtime + scoring aligned in one move). N=5 absolute numbers carry the
arm-invariant confound until the re-run; the three-arm attribution remains differentially valid.

## Stage-C status — 2026-07-05 (v2 CLEAN 3-arm DONE: aligned runtime + OFFICIAL docker scoring = the FIRST real SWE resolve table)
The deprecated `stage_c_n5/` re-run is COMPLETE and SCORED. `runs/stage_c_n5v2/report.md` +
`report.json`. Every episode ran **inside the official per-instance swebench docker image**
(`runtime=container`; acceptance 5/5, task #64); resolve is the **OFFICIAL** `swebench.harness.run_evaluation`
docker harness (`logs/pipeline.log` `score rc=0`), not a mock. Verified against primary artifacts
(official verdict JSON + per-instance report.json + usage.jsonl exit-proof + applied patch.diff + official
test_output).
- **RESOLVE@1 (official): stock-AR 4/5 · merged-AR (RL-v2 as AR) 2/5 · diffusion (same RL-v2 weights) 1/5.**
  Exits: stock {ok:3, turn-limit:2, loops:0}; merged {ok:3, turn-limit:1, loop-halt:1}; diffusion
  {loop-halt:2, turn-limit:3, ok:0}. Spot-checked one resolved patch per arm — all real (applied diff,
  official FAIL_TO_PASS green).
- **ATTRIBUTION: weights effect -2 (stock->merged, both AR); paradigm effect -1 (merged->diffusion, same
  weights).** A loop-halt appears in **merged-AR** (RL-v2 loops even as AR) ⇒ **looping is substantially
  weights-driven**, the diffusion paradigm compounds it. The old "looping = broken-env artifact" reading
  is **RETIRED** (env now aligned; looping persists).
- **BINOMIAL HONESTY (n=5): a ranking, NOT a verdict.** Wilson CIs all overlap (stock [0.38,0.96], merged
  [0.12,0.77], diff [0.04,0.62]); the widest contrast (4/5 vs 1/5) is Fisher p=0.206 — **not significant.**
  A ~0.2–0.3 SWE-scale gap needs ~80–90/arm for power; **N=25–50 is the pragmatic go/no-go tier** (surfaces
  large effects + ranks; a small paradigm tax stays inside the CIs).
- **RL-v2 IS THE WRONG PAYLOAD FOR SWE (the load-bearing finding).** RL-v2 was GRPO-trained on short
  structured tool-call episodes, not SWE-style long-horizon repo edits; as an AR payload it is **-2 vs
  stock on SWE**. The diffusion-vs-AR question is therefore contaminated on RL-v2 weights.
- **N=25–50 change list:** (1) **add a 4th arm = diffusion-on-STOCK-conversion** (B@1000 recipe, no RL-v2,
  ~2–3 h) to complete the {weights}x{paradigm} 2x2 and decide paradigm-vs-weights at scale — highest-value
  marginal GPU-hour; (2) **raise the turn cap 50->75** (VERIFIED **6/15** episodes hit the 50-turn
  `FatalTurnLimitedError`, and one *resolved at the cap* — the limit truncates real work; ~1.5x wall/token
  cost on the affected episodes); (3) keep aligned runtime + official scoring.
- **GO/NO-GO: GO on N=25–50**, but as the **4-arm 2x2 paradigm-vs-weights disambiguation**, not a
  diffusion-vs-AR verdict on RL-v2 weights. Do not over-invest GPU in RL-v2 arms; the actionable next-cycle
  investment is **SWE-style RL data** via the certified convert->RL->re-convert loop (preservation-certified,
  `convert_after_rl_result.md`) so the payload matches the eval distribution.

## Stage-C status — 2026-07-06 (LOOP-HALT PASS-GATE = FAIL; pp REJECTED; N=50 config FROZEN + boot-probed + priced)
Full loop-halt PASS-GATE ran: 2 arms (stock-AR, diffusion) x 3 seeds x 5 Tier0, ENVELOPE +
**presence_penalty=1.5** (arm-neutral) + re-drive=1, one server at a time, OFFICIAL docker scoring.
Report: `runs/loop_halt_polish/report.md`, `runs/stage_c_n5v3_gate/gate_report{.json,_table.txt}`.
- **VERDICT: FAIL (a=F, b=F, c=P).** (a) diffusion loop-halt rate 7/15 = **46.7%** (< 10% target; v3 was 60%);
  (b) diffusion resolve **3/5 (v3) -> mean 0.67/5** (per-seed [2,0,0]) = degradation of >1 instance-equiv;
  (c) post-resolve halts **0** on both arms.
- **THE FINDING: `presence_penalty` is applied arm-neutrally but is NOT neutral in effect — REJECT it.**
  stock-AR is robust under pp=1.5 (**3/5 resolved every seed**, same instances as v3, 1/15 loops); the
  diffusion FLARE wave sampler's **exact grounding collapses** (seeds 2345/3456 = 5/5 EMPTY patches, re-drives
  also empty). A strong presence penalty perturbs the joint canvas logits and breaks the copy/argument-
  grounding SWE patches need — the project crux. It buys a marginal loop reduction (60->47%) for a
  catastrophic resolve loss on the arm under test. The "probe had 0 loop-halts" premise was a **misread** of
  the `--eval-mode skip` `verdict=skipped` marker; re-audit shows the probe itself had 2/5 loop-halts, 1/5
  resolved (reproduced by the gate).
- **FROZEN N=50 SAMPLING = v3 ENVELOPE (temp0.6/top_p0.95/top_k20) + re-drive=1, presence_penalty DROPPED.**
  Loop-halts ship as a **measured covariate** (pre-registered FAIL branch), not masked by a grounding-hurting
  knob. resolve@1 (paired McNemar) is the primary endpoint (concurrency-invariant, per-episode seeds fixed).
- **FROZEN DIFFUSION SERVING = concurrency 4 @ gpu_memory_utilization 0.74, max_model_len 32768** (boot-probed
  at max long-context load). The **MAMBA-CACHE RULE, measured:** idle footprint 28,631 MiB >> the 0.74 pool
  budget because the GDN/mamba align-checkpoint cache + per-request denoise snapshots (~4.5 GiB) sit OUTSIDE
  the KV pool; the un-profiled **FLARE wave lm_head transient** (block x vocab x fp32, scales with concurrency)
  OOM'd the naive **c=8 @ gmu 0.82** in warmup — never copy AR's gmu. Frozen setting: peak 30,431 MiB,
  **headroom 2,176 MiB (6.7%)**, KV pool 152,917 tok (**4.67x full-32k => zero preemption at the 32k cap**),
  no allocation failure. c=6 @ gmu 0.70 is a probed HBM-safe throughput upgrade (2,048 MiB headroom) but only
  3.54x full-32k (preempts under all-long load) — select only if contexts stay < ~18k. AR arm: c=4 @ native
  gmu 0.85 (no mamba cache, ~22 GiB flat, wide headroom). `runcage_ar.sh --max-num-seqs` parametrized (gap-fix).
- **PRICED N=50 (Tier1, 2 arms, c=4):** model-GPU **~3-4 GPU-h** (diffusion ~20 eps/GPU-h, AR ~55 eps/GPU-h);
  Tier1 image pulls ~200 GB / 2-4 h (one-time, disk-checked); official docker scoring ~3-6 h (CPU/RAM-caged);
  **total wall ~1 day**. Speed reported as THROUGHPUT; b=1 latency refs (this gate): AR median 87s, diffusion
  median 250s per episode. **LAUNCH-READY** after wiring MAX_NUM_SEQS=4 + the two gmu's + a concurrency
  fan-out + Tier1 pull-stage into the N=50 orchestrator, and dropping the presence-penalty proxy force.

## Stage-C status — 2026-07-06 (W2 N=50 BUILT + DE-RISKED + PRE-REGISTERED; THE RUN is PENDING — no verdict yet)
The N=50 pipeline is fully built and frozen but **has not executed**: `runs/w2_n50/{ar,diffusion}/`
hold **zero scored episodes**, the GPU is idle, no orchestrator is live. The only episode artifacts are
a 4-instance AR memory/concurrency de-risk smoke (`_derisk/`, all `verdict=skipped`, 0 resolved). **No
2×50 table, no McNemar, no parity verdict exists yet** — the pre-run report (returned as text; the
pipeline's `build_report.py` owns the on-disk `runs/w2_n50/report.md`, generated post-scoring) registers
the analysis and presents the table with every result cell pending. **Report authoring fabricates nothing.**
- **BLOCKER FOUND + FIXED (structural).** The prior run agent died mid-debug on a worker-claim **race**:
  the shared `mkdir`-based claim was non-atomic (succeeds on `EEXIST`) ⇒ two slots double-ran one instance,
  which would corrupt the paired McNemar design (`runs/w2_n50/RACE_DIAGNOSIS_HANDOFF.md`; reproduced 3/5
  trials). **Fix = disjoint sharding, race-free by construction:** `gen_shard_plan.py` pre-partitions the 50
  instances into C fixed lists (round-robin over pool order → balanced repo mix per shard), `run_arm.sh`
  drives each shard with `--only <its ids>`. No shared queue, no claim. Both arms use the SAME
  instance→shard assignment + per-shard base seed (paired validity; seeds spaced 100 000 apart).
  Committed `shard_plan.json` (C=4, disjoint [13,13,12,12], `assignment_sha256 520d8204…`). The AR de-risk
  confirmed **4 distinct claims** (no double-run) + host-RAM fit (server-idle 7 357 MiB, **peak 8 886 MiB**
  with server + 4 concurrent episode containers). **Use `runs/w2_n50/run_all.sh` (sharded, race-free), NOT
  the older `w2_orch.sh` which chains the racy `w2_arm.sh`.**
- **PRE-REGISTERED PRIMARY (encoded in `build_report.py`).** Paired **resolve@1 McNemar**, exact two-sided
  binomial on the discordant pairs b=AR-only, c=diffusion-only; net = c − b (diffusion − AR).
  **PARITY = ( |net| ≤ 2 ) AND ( McNemar exact 2-sided p ≥ 0.05 )** — the honest "parity-or-better" bar
  (Stage-C **C-G2**), NOT a speed multiplier. |net|>2 with significance = a diffusion win (c>b) or tax (b>c).
- **SECONDARY, pre-registered:** throughput (eps/GPU-h at c=4, single RTX 5090 — honest framing: diffusion
  ~20 vs AR ~55 eps/GPU-h ⇒ ~2.7× **slower**; this is a resolve-quality parity test, not a speed claim);
  tokens; loop-halt covariate with **post-resolve vs pre-resolve** split (post-resolve halts are benign;
  the gate found 0/both at N=5); per-repo resolved (composition frozen: django22 sympy8 sphinx5 mpl3
  sklearn3 astropy2 xarray2 pytest2 seaborn1 pylint1 requests1).
- **ARM WEIGHTS = the clean paradigm test.** AR = stock `qwen3.5-9b-ar`; diffusion = `flare-hybrid-clean`
  **stock-conversion (NOT RL-v2)** — same stock weights, AR vs diffusion. The N=5 priors (v2 1/5, v3 3/5)
  were on **RL-v2** weights (shown to be the wrong SWE payload) and do **not** predict this arm.
- **READ-TIME GUARD (defect in `build_report.py`):** on empty data (b=c=0) it computes net 0, p=1.0 ⇒
  prints **`PARITY: YES`**. A parity claim is only valid when the ANOMALIES block is empty (both `scoring`
  present, 50/50 episodes/arm). Never read parity from a run with `scoring MISSING` or `<50 episodes`.
- **NEXT ACTION (the only one):** `setsid bash runs/w2_n50/run_all.sh >runs/w2_n50/logs/run_all.console 2>&1 &`
  (detached, self-bounded; ~1 day wall — model-GPU 3–4 h, official scoring 3–6 h, 50 images already pulled).

## Stage-C status — 2026-07-06 (W2 N=50 EXECUTED + SCORED + SERVING-VERIFIED → VERDICT BANKED: NOT SWE-parity)
**THE RUN is DONE and the verdict is banked.** Both arms 50/50, official docker scoring, serving
verified clean from artifacts. `runs/w2_n50/report.{json,md}` (report.md §ADJUDICATION is the full
banked record). C-G2 (parity-or-better) is **FAILED — decisively.**

- **PRIMARY (paired resolve@1 McNemar): stock-AR 19/50 vs diffusion 2/50.** both=2, AR-only(b)=17,
  diffusion-only(c)=**0**, neither=31, net (diff−AR) = **−17**, **McNemar exact 2-sided p ≈ 0.0000,
  PARITY = FALSE** (`|net|≤2 AND p≥0.05`). data_sufficient=true. Pre-registered detectable effect
  (|net|>2) **satisfied, 17 ≫ 2.** Every diffusion win (sympy-16886, django-14373) is also an AR win ⇒
  **no diffusion-only resolve.** Pool `fe1973937dfb500b…` == frozen manifest (n=50); paired valid.
- **SERVING VERIFIED (rule out silent regression) — CLEAN, verdict VALID.** Per-request from
  `logs/diffusion_server.log`: `decode_mode=hybrid_clean` on the serve banner **and** engine state;
  **2852/2852** completions on the FLARE grammar path (2684 `complete_tool_call`), 840
  `DiffusionDecoding metrics` lines ⇒ genuine diffusion decode, **no canvas-silent-serve, no AR
  fallback**; the A1 launcher bug did not fire. 100% of 2926 diffusion + 2364 AR requests used the
  frozen **v3 envelope temp0.6/top_p0.95/top_k20** (per-request seeds). AR symmetric + provably pure AR
  (0 FLARE lines, stock `Qwen/Qwen3.5-9B`). Diffusion model = `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` =
  **byte-identical path to the Tier0 v3 gate**; 3 episode dumps + a Tier0 dump spot-checked coherent.
  The 12:26Z boot-probe failure was a transient GPU-settle-between-arms artifact — the safety gate
  correctly SKIPPED, and `diff_launch.sh` relaunched the arm on a settled/healthy server (KV 152,917 tok,
  4.67× conc). HTTP 400s appear in both arms (2.5%/1.7%), benign.
- **CAVEAT (disclosed):** the 2026-07-06 pre-run paragraph above named the diffusion arm as the **b1000
  stock-conversion (NOT RL-v2)**; the frozen serve script, the v3 gate cert, and this run all served the
  **rlv2** twin. The run matches its own certification (so serving is valid and directly comparable to
  the v3 prior), but the executed comparison is **stock-AR vs RL-v2-diffusion**, not the pristine
  same-weights paradigm test that paragraph described. Net effect is conservative for the tax story: the
  twin had an agentic-RL advantage and still posted 2 vs 19.
- **N=5 Tier0 was small-sample + easy-pool, never a verdict.** `runs/stage_c_n5` (mock, AR-edit 3/5 /
  diff-edit 1/5, 0 docker-resolves) and `runs/stage_c_n5v3_gate` (3-seed) were the *first 5* Tier0
  instances with overlapping Wilson CIs — a ranking. At N=50 diverse with official scoring the tie
  dissolves into −17.
- **FAILURE SIGNATURE = loop-before-edit on unfamiliar repos.** Exits diffusion: **26 loop-detector /
  13 turn-limit / 10 clean / 1 budget** (AR: 26 clean / 24 turn-limit / 0 loop). **35/50 empty patches**
  (AR 4). Of 26 loop-halts, **18 emit no patch at all, 25 end unresolved** (pre-resolve 25 / post-resolve
  1). On Tier0 loops fired post-edit; here they fire pre-edit — coherent-but-repetitive tool-calling that
  never commits, median empty-patch wall 673 s vs AR 229 s. Per-repo the gap is broad, worst on django
  (11/22 → 1/22), matplotlib (2/3 → 0/3), sklearn (2/3 → 0/3). Throughput 21.4 vs 99.6 eps/GPU-h (4.65×
  slower) — a reported covariate, not the finding.
- **STRATEGIC CONSEQUENCE → the fork.** (a) **Primary: the SWE-tuning campaign premise RETURNS, now
  powered.** The missing ingredient is **SWE-trajectory training**, not decode engineering: general
  agentic RL-v2 does not transfer to SWE resolve. The certified loop already yields the data (data-gen
  yield 0.25, GO-priced) and a verified-clean serving spine — **train the base on SWE trajectories via
  the certified loop (SFT on successful SWE episodes ± SWE-reward RL), then re-run W2 N=50 as the gate.**
  (b) **Parallel engineering track: decode-policy work on the loop pattern** (edit-biased planning,
  loop-detector stall-vs-explore discrimination, forced-edit budget, empty-patch retry) — symptom
  mitigation, cheap, but not the cure. (c) **Do not re-litigate serving or re-run the same rlv2 twin** —
  it will reproduce ~2/50; the next model must be SWE-tuned. **Recommended fork for the user: fund (a),
  run (b) in parallel as a cheap hedge.**
