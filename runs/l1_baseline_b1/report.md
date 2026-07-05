# L1 — honest B=1 baseline on reasoning-heavy content (engine hybrid_clean vs guided-AR)

_Assembled 2026-07-05. Regime: B=1, greedy (temperature 0, top_p 1, seed 20260701)._
_Campaign: `goal_5x_rollout_b1.md`, lever L1 (content-mix, measure-first)._

## TL;DR — the honest number, not spun

On genuine reasoning content at B=1 the engine's **avg committed tok/fwd = 1.36** (24 well-behaved
GSM8K turns), and **every model-chosen/reasoning token is K=1 (one forward each)** — the 0.36 above 1.0 is
entirely the ~26 % **tool-call grammar scaffold** that commits with zero forwards. The wall-clock ratio vs
stock guided-AR is:

| framing (shared 24 prompts, >1 = engine faster) | ratio |
|---|---:|
| **reasoning-token-only throughput** (the honest useful-content rate) | **0.75x** |
| all-committed-token throughput (credits free grammar scaffold) | 1.03x |
| per-**forward** speed (engine 18.7 ms vs AR 14.1 ms/tok) | **0.75x (engine 1.33x slower/forward)** |

**Verdict: on reasoning content the engine has NO B=1 speed advantage today — it is ~0.75x (slower) on
useful tokens, ~1.0x only if you count tool-call XML padding as output.** This lands exactly in the
task's predicted 0.7-1.0x band and confirms the campaign's KNOWN FACT (hybrid_clean is 1 tok/fwd on all
model-chosen tokens; zero-forward commits only on forced grammar). The goal doc's "expected 2-3x already"
is **not** met on this content — the diffusion engine only wins on tool-call-heavy content (see contrast),
and only L3 (adaptive-K training) can raise reasoning tok/fwd.

## What was measured

- **Engine** = P2 **hybrid_clean**, the converted **RLv2** block-diffusion Qwen3.5-9B, vLLM pin `95d8b47`,
  `VLLM_FLARE_BIDIR_PROBE=1` + `VLLM_FLARE_CUDAGRAPH=1`, cudagraph (PIECEWISE), export
  `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16`, RTX 5090, batch 1. TRUE denoise forwards captured per turn
  (byte-identical counter machinery to the endgame v3b battery).
- **AR** = **stock Qwen3.5-9B** snapshot `c202236` (the scoreboard's stock-bf16 baseline), offline vLLM
  0.23 (`.venv-vllm`), bf16, **enforce_eager**, mamba-cache align, gdn triton — **schema-free plain greedy
  generation, no grammar, no tools** (AR = 1 token/forward by construction).
- **Reasoning content** = GSM8K test first-30, 5-shot chain-of-thought, **schema-free**. The census prompts
  carried the model's native chat template with a trailing empty `<think></think>` scaffold; that scaffold
  was **stripped** so the final turn matches the in-context CoT exemplars (standard GSM8K few-shot
  continuation). Identical `prompt_ids` fed to both engine and AR.
- **Proxy disclosure (per task):** the flywheel SWE-style episodes live on a separate machine; GSM8K-class
  is the reasoning-heavy **proxy** here. It stands in for rollout reasoning content.

## CONTENT CAVEAT (decisive) — this agentic model does not free-CoT on GSM8K

The converted **RLv2** model does **not** emit free-form chain-of-thought single-turn. Given a math
question it wraps its reasoning inside a **hallucinated tool call** —
`<tool_call><function=think><parameter=thought> ... </parameter></function></tool_call>` (sometimes
`calculate`/`tool`) — and then stops, expecting a tool loop. The reasoning inside is frequently **correct**
(idx0: "Remaining eggs = 16 - 7 = 9 ... 9 * 2 = 18", gold 18), but single-turn generation never emits the
scored `#### N`, so **strict GSM8K exact-match is 0/24 for the engine — a harness/format artifact of
single-turn eval on an agentic model, not a reasoning failure.** The reasoning text lands in the tool-call
**value span**, which hybrid_clean decodes strictly at **K=1**. So even the reasoning proxy is
tool-flavored for this model, and the reasoning tokens are strictly sequential. (Stock AR, by contrast,
free-CoTs and scores 29/30 — see below.) **Consequence: "at held exactness" cannot be asserted on this
content/model; GSM8K-single-turn is a weak exactness gate for the RLv2 model.**

## (1) Engine hybrid_clean — reasoning-content metrics (24 well-behaved turns)

| metric | value |
|---|---:|
| well-behaved turns (fin=stop) | 24 / 30 attempted |
| **aggregate tok/fwd** | **1.361** (mean 1.46, median 1.50) |
| per-forward wall | 18.7 ms (matches endgame 18.5 ms bs=1 floor) |
| committed tokens / s | 72.6 |
| **model-chosen (reasoning) tokens / s** | **53.4** |
| median s/prompt | 0.42 (short — tool-wraps then stops) |
| strict GSM8K correct | **0 / 24** (single-turn tool-wrap artifact) |
| token mix | grammar-scaffold (0-forward) **264 (26 %)** · model-chosen K=1 **732 (74 %)** |

`denoise_forwards == model_chosen_tokens` exactly on every turn (732 == 732): **K=1 on all reasoning
tokens, confirmed**. The only tok/fwd > 1 is the zero-forward grammar scaffold. `value_projection_events
== 0` and `verify.ok == True` on all 24.

## (2) Guided-AR stock baseline (schema-free) + (3) the ratio TODAY

Stock Qwen3.5-9B, plain greedy, same clean GSM8K prompts. It **does free-CoT and solves them**:

| metric | engine (reasoning) | stock AR |
|---|---:|---:|
| tok / s (aggregate) | 72.6 committed / **53.4 reasoning** | 70.8 |
| ms / token | 13.8 committed / **18.7 per forward** | 14.1 |
| avg tok / fwd | 1.36 | 1.00 |
| median s / prompt | 0.42 | 1.89 |
| strict GSM8K correct | 0 / 24 | **29 / 30** |
| stability | 24/30 clean (6 hang/degenerate) | 30/30 clean |

**Ratio TODAY (24 shared prompts, engine/AR wall-clock token throughput, >1 = engine faster):**
- **reasoning-token-only: 0.75x** (engine slower at producing useful content)
- all-committed-token: 1.03x (only because ~26 % free grammar scaffold offsets the per-forward penalty)
- per-forward penalty: engine 18.7 ms vs AR 14.1 ms = engine **1.33x slower per forward** (diffusion
  GDN-recurrence + block overhead; and AR here is *enforce_eager*-handicapped, so with cudagraph AR would
  be faster still -> the true engine/AR ratio is **<= reported**).

Note the median-s/prompt column is **not** a speed win: the engine's 0.42 s is lower only because it emits
a short tool-call and quits without answering (0/24 correct), while AR spends 1.89 s producing a complete
correct CoT. Per-token normalization is the honest metric; per-prompt is not.

## Instability finding — hangs + degenerate loops off-distribution

Attempting all 30 prompts on a **freshly booted engine** (a SIGALRM abort corrupts the engine, so each
prompt was run on a clean engine via an exit-on-first-hang + reboot loop) exposed a **~20 % instability
rate** for this agentic model on single-turn math:

- **CPU-pathological HANGS** (100 % CPU / 0 % GPU, killed by a 40 s per-turn watchdog) at **idx 9, 11, 21**
  — the hybrid_clean grammar/FSM spins on CPU without issuing GPU forwards (idx9: hung after 26 forwards;
  idx11: hung at fwd=0 during prompt/grammar setup). One (idx9) hung for >14 min before being killed in
  the first run.
- **Degenerate grammar loops** at **idx 7, 15, 18**: the model opens a tool call then emits a runaway of
  whitespace/newline tokens that the FSM force-commits for free, hitting the 384-token cap (fin=length).
  idx7 = 372 forced tokens over 12 forwards -> a **spurious "32 tok/fwd" that is NOT throughput** (it is
  repetition the grammar commits at zero cost, and it is excluded from the 24-turn numbers above).

Breakdown: **24 clean (80 %), 3 degenerate (10 %), 3 hang (10 %)**. Stock AR had **0** runaways / hangs.

## Tool-call CONTRAST — where the engine has its current edge

Engine on the 184-turn never-train BFCL/API-Bank battery (`runs/p2_engine_nevertrain/`, same pin/config):

- **aggregate tok/fwd = 1.60** (per-turn mean 1.75, median 1.76) vs reasoning **1.36**.
- 5 representative turns [episode · n_gen/fwd = tok/fwd · ms/f · exact]:
  - `bfcl_mt_multi_turn_long_context_0005`: 42/28 = 1.50 · 19.1 ms · exact
  - `bfcl_mt_multi_turn_miss_func_0002`: 42/26 = 1.62 · 19.3 ms
  - `apibank_AddAgenda-level-1-1_0`: 37/21 = 1.76 · 21.6 ms · exact
  - `bfcl_mt_multi_turn_miss_func_0007`: 26/14 = 1.86 · 20.9 ms
  - `bfcl_mt_multi_turn_miss_func_0003`: 26/13 = 2.00 · 20.9 ms

Tool-call content has a higher grammar-forced fraction (arg names + XML scaffold ~37 % zero-forward vs
~26 % on the reasoning wrapper), so tok/fwd is higher on tool-calls (~1.6) than reasoning (~1.36). This is
why the engine beats AR on the tool-call-heavy endgame battery (0.626 vs 0.741 s/turn = 1.18x) but **not**
on reasoning content: the win is forced-token bulk commits, and reasoning has fewer of them. **Both are far
below the north-star 5.0.** Value/reasoning tokens are K=1 forever (chain rule).

## Verdict for the 5x-at-B1 campaign (L1)

1. **Reasoning content, B=1: engine ~0.75x (reasoning tokens) to ~1.0x (all-committed) vs stock AR — no
   speed advantage today**, exactly as the campaign's KNOWN FACTS predicted, and squarely in the expected
   0.7-1.0x band. The goal-doc "expected 2-3x already" is **refuted** for this hybrid_clean serving path.
2. **Reasoning tokens are strictly K=1** (732 forwards == 732 model-chosen tokens). This is the chain-rule
   wall. **L2 (per-forward parity, 18.7->~13 ms) cannot change tok/fwd — only ms/forward**; it would take
   the 0.75x toward ~1.0x (closing the 1.33x per-forward penalty) but not past AR. **Only L3 (S2 consistency
   distillation + entropy-gated adaptive K) raises reasoning tok/fwd** — the sole lever to >1x.
3. **The prior "engine faster" result was tool-call content.** The 4-tok/fwd "native GSM8K" figure was a
   *different* decode mode (fixed-K block sampler on the converted BASE model, per
   `measure_block_quality_curve.py`), **not** the hybrid_clean serving path on the RLv2 model — hybrid_clean
   does K=1 on reasoning today.
4. **Content honesty:** the RLv2 agentic model does not free-CoT single-turn on GSM8K (tool-wraps -> 0
   scorable) and is ~20 % unstable off-distribution (hangs + degenerate loops), while stock AR free-CoTs
   at 29/30. A clean L3 reasoning gate should use the converted BASE model's native block decode or a
   multi-turn harness that lets the tool loop close — not RLv2 single-turn GSM8K exact-match.

## Artifacts (absolute paths)

- Report: `/home/mark/qwen_diffusion/runs/l1_baseline_b1/report.md`
- Machine-readable summary: `/home/mark/qwen_diffusion/runs/l1_baseline_b1/summary.json`
- Engine per-turn JSONL (30 attempts incl. hang/degenerate rows): `/home/mark/qwen_diffusion/runs/l1_baseline_b1/engine_gsm8k_clean.jsonl`
- AR per-turn JSONL (30): `/home/mark/qwen_diffusion/runs/l1_baseline_b1/ar_gsm8k_clean.jsonl`
- Clean (stripped-think) prompts: `/home/mark/qwen_diffusion/runs/l1_census/gsm8k_prompts_clean.json`
- Tool-call contrast source: `/home/mark/qwen_diffusion/runs/p2_engine_nevertrain/nevertrain_turns.jsonl`
- Runners: `run_engine_hardened.py` (per-turn SIGALRM watchdog), `reboot_loop.sh` (fresh-engine sweep),
  `run_ar_gsm8k.py` (AR), `compute_report.py` (metrics) — all under `runs/l1_baseline_b1/`.
