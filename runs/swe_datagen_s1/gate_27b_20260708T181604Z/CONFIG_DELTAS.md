# CONFIG_DELTAS — Qwen3.6-27B-NVFP4 datagen-teacher serving audit vs official spec

**Scope:** audit the ACTUAL serving chain (runcage_27b.sh → vLLM 0.23; proxy
`LUMO_PROXY_FORCE_*` envelope; qwen-code driver; gate-2 battery) against the
Qwen3.6-27B official sampling / thinking-mode / template guidance. CPU/network
only; read-only on the live serving chain; all corrections **STAGED, not applied**.

## TL;DR
The gate-2 raw `exact_args = 0.30` is **NOT a sampling-config defect and NOT a
capability defect** — it is a **battery/metric artifact**. The same run's
**source-verbatim grounding = 63/64 (0.9844), zero malformed**, and gate-3 resolve
= 4/4 real patches. The NVFP4-calibration copy crux (the only thing this gate was
built to protect) is already answered and is **mode/sampler-insensitive**.

There IS a real config defect, but it bites **gate-3 resolve / teacher quality**,
not `exact_args`: the whole envelope + thinking decision was **inherited wholesale
from the NON-thinking Qwen3.5-9B AR teacher**, producing a **mode/sampler chimera**
on the 27B (a thinking model): it runs **thinking-OFF** (proxy) with a
**thinking-mode coding sampler** (0.6/0.95/20). Additionally the **gate-2 battery
was run thinking-ON** (default) while **production datagen runs thinking-OFF** — so
the gate did not test the config the teacher ships in.

---

## How the chain actually samples (verified on disk)

| Hop | Mode | Sampler | Set where |
|---|---|---|---|
| **Production datagen** (driver→proxy→vLLM) | **thinking OFF** (`enable_thinking:False` forced UNCONDITIONALLY in `qwen_code_sglang_proxy.py`) | **0.6 / 0.95 / 20**, min_p·pp unset, per-req seed | `datagen_gen.sh:48-51` → proxy `_ENVELOPE_PINS` |
| **Gate-2 battery** (`gate2_grounding.py`, direct to vLLM:9952, **bypasses proxy**) | **thinking ON** (no `enable_thinking` passed → template else-branch `<think>\n`) | **0.6 / 0.95 / 20**, seed 1234, max_tokens 8192 | hardcoded in the battery payload |
| **ckpt default** (`generation_config.json`) | thinking ON (template default) | **1.0 / 0.95 / 20** | model card / NVFP4 ckpt |
| **Qwen3.6 official — thinking general** | ON | 1.0 / 0.95 / 20 / min_p 0 / pp 0 | HF card |
| **Qwen3.6 official — precise coding** | ON | 0.6 / 0.95 / 20 | HF card |
| **Qwen3.6 official — non-thinking/instruct** | OFF | 0.7 / **0.80** / 20 / **pp 1.5** | HF card |
| **NVIDIA NVFP4 tool-use bench (tau2)** | OFF | **0.0 / 1.0 (greedy)** | NVFP4 card |

Both the codex template (`qwen3-openai-codex.jinja`) and the ckpt-native template
gate thinking identically: `enable_thinking is defined and false` → emit empty
`<think>\n\n</think>` (hard-off); else → `<think>\n` (thinking on). Confirmed by
gate-2 latencies (4.9–43.6 s/turn = real reasoning generation).

---

## DELTAS

### D1 — MODE MISMATCH gate-2 vs production  · severity HIGH (test validity)
- **Current:** gate-2 measured **thinking ON**; production datagen runs **thinking OFF**.
- **Official:** a promotion gate must test the mode the teacher ships in.
- **Impact on gate-2:** the `exact_args`/verbatim numbers were produced in a mode
  the teacher will not run in. The 0.9844 verbatim result is reassuring but was
  measured thinking-ON; it has **not been confirmed in the production (thinking-OFF)
  mode**. This is the one genuine reason to re-run before promotion.

### D2 — MODE/SAMPLER CHIMERA on the datagen path · severity HIGH (gate-3 / teacher quality)
- **Current:** thinking **OFF** + thinking-mode coding sampler **0.6 / 0.95 / 20**,
  min_p & presence_penalty unset.
- **Official:** either **Regime T** (thinking ON → 1.0/0.95/20/pp0) — the
  officially-tuned agentic path, == ckpt `generation_config` defaults, and what the
  codex template's interleaved-thinking machinery is built for — **or Regime I**
  (thinking OFF → **0.7/0.80/20/pp1.5**). Current config is neither.
- **Root cause:** inherited verbatim from the Qwen3.5-9B AR teacher (a non-thinking
  model on the flywheel reference envelope). `MTP_GATE_PLAN.md:84` and the driver
  comment (`run_swe_bench_qwen_code.py:83`, "inject enable_thinking=false") confirm
  the inheritance, not a 27B-specific decision.
- **Impact on gate-2 `exact_args`:** **low** — verbatim copy is already 0.9844 and
  is mode/temp-insensitive; raising temp 0.6→1.0 would if anything **worsen**
  multi-call planning divergence (see D9). Impact is on **gate-3 resolve** and
  overall teacher trajectory quality, where thinking is officially load-bearing.

### D3 — TEMPERATURE 0.6 · severity LOW (for exact_args)
- **Current:** 0.6 everywhere.
- **Official:** 0.6 is the Qwen3.5-9B thinking ref AND the Qwen3.6 *precise-coding*
  value; the Qwen3.6 thinking-**general** default is **1.0**; a pure exactness read
  would use **greedy 0.0** (non-thinking, NVIDIA tau2).
- **Impact:** 0.6 is defensible for a coding/exactness read and is **not** the cause
  of 0.30. `DO NOT use greedy in thinking mode` (Qwen hard rule) — the current chain
  is never greedy, so no violation. No change strictly required for the crux.

### D4 — presence_penalty / min_p never set · severity LOW
- **Current:** unset (vLLM default 0.0 / 0.0).
- **Official:** 27B thinking pp=**0.0** (correct — do NOT copy the 35B-A3B's 1.5);
  non-thinking/instruct pp=**1.5**. So pp is only wrong **if** Regime I is chosen.
- **Impact:** negligible on gate-2.

### D5 — CHAT TEMPLATE codex-override vs ckpt-native · severity NONE (research hypothesis REFUTED)
- **Current:** `--chat-template qwen3-openai-codex.jinja` (codex fork).
- **Research hypothesis:** "for Qwen3.6 the ckpt-native template is presumably
  correct for qwen-code."
- **Evidence:** **REFUTED** by the gate-1 format cert (`gate_verdict.json`):
  codex template renders **identical qwen3_xml** on the 27B tokenizer (vocab
  248320), **0 schema mismatches** vs a production keeper. Both templates use the
  same `<tool_call>/<function=>/<parameter=>` XML and the same thinking gating.
- **Decision:** **KEEP codex** (format-equivalence-by-construction with 9B keepers;
  a switch buys nothing and breaks equivalence). No change.

### D6 — proxy `enable_thinking=False` is UNCONDITIONAL · severity MED (config rigidity)
- **Current:** `qwen_code_sglang_proxy.py:100-103` hard-sets `enable_thinking:False`
  on every chat request; not env-gated. You **cannot select Regime T without a code
  change**.
- **Staged fix:** gate it on an env var, e.g.
  `enable_thinking = os.environ.get("LUMO_ENABLE_THINKING","false").lower()=="true"`
  and only inject the key when explicitly set. 1-line change; wired to
  `LUMO_ENABLE_THINKING` in `envelope_corrected.env`.

### D7 — proxy `--max-tokens` cap (2048) · severity MED **iff** Regime T
- **Current:** `DEFAULT_PROXY_MAX_TOKENS=2048` clamps every turn (proxy also has a
  512 class default, overridden by the driver). Gate-2 bypassed this (8192 direct).
- **Impact:** fine for non-thinking; **truncates** thinking traces + patch under
  Regime T. Raise to ≥8192 if thinking is enabled.

### D8 — served-model-name swap · severity MED (swap-time integration, not gate-2)
- **Current:** `datagen_gen.sh:133` sends `--model qwen3.5-9b-ar`; the 27B serves
  `qwen3.6-27b-nvfp4`. Must be reconciled when `RUNCAGE_SCRIPT` is swapped to the 27B.

### D9 — the 0.90 gold-match bar itself · severity HIGH (mis-operationalization)
- **Current bar:** `exact_args ≥ 0.90` vs a **9B-era synthetic multicall battery**
  (`flare_scaleup_native_58.jsonl`, source `heldout_seed_multicall_clean`).
- **Why it can't be met by any sampling:** the strict full-call-list-equality metric
  conflates three things, and ~3–4/10 episodes fail on gold that is **not
  reproducible verbatim** — no envelope fixes these:
  - **non-verbatim synthetic gold**: `0023` (slug ids `nina_simone`, ISO timestamps,
    JSON blobs; 0 verbatim copy targets), `0048` (fabricated `path/to/*.json`
    placeholders → model correctly emits **no call**);
  - **multi-call planning divergence**: `0004` emits 7 well-grounded calls vs 3 gold
    (per-arg 21/21, copy 19/19) — over-planning, not copy damage;
  - **complex-JSON canonicalization/whitespace**: `0000/0026/0055` miss only a nested
    JSON/long-text arg;
  - **entity selection**: `0015` picks `device_001` vs gold `device_002` — byte-exact
    digits, wrong-of-3 selection (a reasoning miss, made **in thinking mode**).
- **Correct operationalization for a stock 27B teacher:** **(a)** source-verbatim
  grounding ≥ 0.98 (the NVFP4 crux; measured **0.9844**), **(b)** zero malformed
  (measured **0**), **(c)** gate-3 **resolve** on real SWE instances (measured 4/4).
  Retire `exact_args ≥ 0.90` as a promotion gate.

---

## RECOMMENDATION

**1. Do NOT chase 0.30 → 0.90 with sampling.** It is a battery/metric artifact
(D9); the crux (verbatim copy) is already 0.9844 and mode-insensitive. Retire the
`exact_args ≥ 0.90`-vs-9B-synthetic-gold bar; promote on **verbatim-grounding ≥ 0.98
+ zero malformed + gate-3 resolve**.

**2. RE-RUN gate-2 — but for MODE, not for the score.** The one honest gap is D1:
gate-2 was measured **thinking-ON**, production runs **thinking-OFF**. Before
promoting the teacher: **decide the production mode first**, then re-run the
**verbatim-grounding** check (`gate2_verbatim.py`) + a small **resolve spot-check**
in **that** mode. If production stays non-thinking, confirm the 0.98 verbatim result
holds with `enable_thinking=False`. A full-battery re-run for `exact_args` is not
warranted.

**3. Fix the chimera (D2) — a user-owned cost/quality call.** Recommended:
**Regime T (thinking ON, 1.0/0.95/20/pp0)** — the officially-tuned agentic path for
this checkpoint, == its own `generation_config`, and what the codex interleaved-
thinking template is built for; most likely to lift **gate-3 resolve** (the real
teacher objective). Requires D6 (env-gate thinking) + D7 (raise max-tokens) + a
thinking-mode re-gate (point 2). If thinking-token cost/latency is unacceptable,
fall back to **Regime I (0.7/0.80/20/pp1.5, thinking OFF)** — still a strict
improvement over the current chimera. Staged in `envelope_corrected.env` /
`runcage_27b.sh.corrected` (serving flags unchanged — they audited clean).

## Staged artifacts (this dir)
- `runcage_27b.sh.corrected` — runnable-equivalent to the original; serving flags
  unchanged; corrected provenance + MODE/ENVELOPE decision block. **Not applied.**
- `envelope_corrected.env` — corrected `LUMO_PROXY_FORCE_*` block (both regimes,
  Regime T active) for `datagen_gen.sh:47-51`. **Not applied.**
- Proxy D6/D7 fixes described above (1-line env-gate + max-tokens bump) — **not applied.**
