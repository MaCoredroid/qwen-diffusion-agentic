# SWE-class Agentic End-Goal — Staged Serve → NVFP4 → SWE-Verified Plan

**Author:** monitor synthesis, 2026-07-05. **Task:** #54 (SWE-VERIFIED END-GOAL). **Owner discipline:** [[qwen-diffusion-commit-workflow]] (commit+push each step, narrate reasoning), [[gpu-utilization-standard]], [[diffusion-promotion-discipline]], [[native-function-format-rule]].
**Mode of this doc:** CPU-only synthesis of three audits (serve-path, nvfp4, harness) into a staged, gated, GPU-hour-costed plan. No CUDA was run. All speed/quality numbers below are "measure, never assume."

## 0. What we are actually building (frame — read first)

The 5×-vs-AR north star is **RETIRED** (KILL, `ffaa23b`, K-factor wall). The surviving, funded goal is: **stand the diffusion-9B up as a real OpenAI `/v1` endpoint driving an agentic SWE-bench-Verified loop through Qwen Code, and measure it honestly against the AR-9B of the same weights.** The established result we are extending: engine hybrid-clean = **130/247 exact, quality == HF row exactly, +6 vs stock-AR, speed parity-class vs fair cudagraph-AR (0.94×), 4.12× vs our prior serving** (`endgame_table_final.md`, `43c5bf8`); conversion-after-RL **PRESERVES** (`b019b86`).

Therefore the Stage-C **win condition is not a speed multiplier** — it is: *diffusion resolve-rate ≥ AR resolve-rate on the same SWE subset, at held byte-losslessness across multi-turn context reuse, with honest per-turn economics reported both ways.* Speed is a reported number, not a gate.

## 1. THE GATING DEPENDENCY — lossless-APC (in-flight, task #53)

Every Stage-C SWE turn re-sends a growing prefix; the diffusion engine only wins multi-turn if its prefix cache is **lossless** (cache-on decode byte-identical to fresh-context decode). Today's align-APC is functional but **lossy** on the `{20,21,60}`+`{16,130}` artifact class under cross-turn reuse (`lossless_apc_design.md`, root cause `cc3a422`, design `98576fc`). The fix is **designed, not implemented** — it owns the GPU right now (this box is CPU-only because of it).

**This plan cannot reach its multi-turn Stage-C gate until lossless-APC clears its gates (a)+(b):**
- (a) artifact-census byte-match: 5/5 census turns byte-identical to cold-prefix under cache-on.
- (b) full-battery cache-on == fresh certificate: 63 + 184 turns, per-turn byte-parity on every APC-class turn (upgrades `nevertrain_parity_cert_resetapc.jsonl` to a cache-on cert).
- (c) is the multi-turn agentic speedup number — it *is* Stage-C's economics input.

Lossless-APC cost (from its own §4): **~1.5–2 weeks eng + ~20–35 GPU-h**, bring-up byte-debugging dominant. **Stage A single-turn bring-up can be built CPU-side / overlap the tail; Stage A's multi-turn-reuse cert and all of Stage C are hard-blocked on (a)+(b).**

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
lossless-APC (task #53, IN FLIGHT)  ──gates(a,b)──┐
  ~1.5–2 wk eng, ~20–35 GPU-h                     │
        │                                          ▼
        │                         Stage A (serve bring-up)
        │                         A1/A3/A5 CPU-side now ─┐
        │                         A2/A4/A6 single-turn   │ ~6–12 GPU-h
        └──(a,b)──► A7 multi-turn reuse cert ◄───────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
   Stage B (NVFP4, PARALLELIZABLE)            (Stage C needs A-G2/A-G3)
   ~12–20 GPU-h, PASS→promote / PARK→bf16
              │                                        │
              └──────────────► Stage C (SWE-Verified) ◄┘
                    N=5 smoke → N=25–50, ~27–49 GPU-h
```

**GPU-hour rollup (5090, greedy, single-stream):**
| Phase | GPU-h | Blocking? |
|---|---|---|
| lossless-APC (pre-req, task #53) | 20–35 | yes — owns GPU now |
| Stage A (net new) | 6–12 | A7 gated on APC (a,b) |
| Stage B (NVFP4) | 12–20 | parallelizable; not a Stage-C blocker |
| Stage C (SWE) | 27–49 | gated on A-G2/A-G3 |
| **Total incl. pre-req** | **~65–116 GPU-h** | |
| **This plan (A+B+C, excl. APC)** | **~45–81 GPU-h** | |

## 4. Realistic calendar (one researcher, consumer GPU, single-stream, measure-not-assume)

Anchored at **2026-07-05**. Lossless-APC is at DESIGN today; its ~1.5–2 wk eng + byte-debug is the long pole and it owns the GPU. Stages overlap where CPU-side work (A1/A3/A5, C1 driver, B2 recipe) can proceed before the GPU frees. Calendar assumes the intermittent alienware reservations and normal debug slippage.

| Window | Milestone |
|---|---|
| **Jul 5 → ~Jul 20** | Lossless-APC lands: gates (a)+(b) byte-parity certs green. GPU frees. In parallel (CPU): A1/A3/A5 done, C1 Qwen-Code driver drafted, B2 NVFP4 recipe authored. |
| **~Jul 18 → ~Jul 28** | Stage A: A2/A4/A6 online single-turn cert (A-G2), then A7 multi-turn reuse cert (A-G3) on the fresh APC. |
| **~Jul 26 → ~Aug 6** | Stage B in parallel: install PTQ, calibrate, quality battery (B-G2), wall-clock A/B (B-G3 → promote or park). |
| **~Aug 1 → ~Aug 5** | Stage C N=5 smoke, both arms (C-G1) — first real SWE loop closes. |
| **~Aug 5 → ~Aug 18** | Stage C N=25–50 subset, AR vs diffusion (C-G2/G3/G4) — the deliverable measurement. |

**Realistic end-to-end: ~6–7 weeks (early July → ~mid-to-late August 2026)**, hard-gated on lossless-APC landing first. N=5 smoke is reachable ~4 weeks in (~Aug 1); the N=25–50 AR-vs-diffusion verdict ~6–7 weeks in (~mid-Aug). NVFP4 is a bonus fork that does not move the Stage-C date.

## 5. Honest risk register (top 5)
- **R1 — lossless-APC slips.** It is the long pole and blocks A7+all of C. If (a)/(b) fight fp-determinism (R1 in its own doc), Stage C multi-turn credit slips with it. Mitigation: Stage-C N=5 smoke (C-G1) only needs single-turn correctness and can run before A7 to de-risk the loop; full N=25–50 waits for A-G3.
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
