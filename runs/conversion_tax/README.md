# Per-capability conversion-tax battery (#28)

**Goal.** Quantify what the diffusion conversion + RL costs/gains **per capability
class**, not just on tool-calls. Three systems × three capability classes, identical
prompts across systems, greedy B=1, strict deterministic scoring, engine-side audits
(`value_projection_events == 0`).

## Systems under test (columns)

| col | system | weights | serve | venv / pin |
|---|---|---|---|---|
| STOCK-AR | pre-conversion baseline | stock Qwen3.5-9B `c202236` | offline vLLM, bf16, `enforce_eager`, `mamba-cache align`, `gdn-prefill triton`, plain greedy | `.venv-vllm` (vLLM 0.23) |
| MERGED-AR | RL-v2 merged, served AR | `models/qwen3.5-9b-fastdllm-mtplus1-Anew-vllm-bf16` (the **136/247** export; native `Qwen3_5ForConditionalGeneration`) | **same** offline-vLLM harness/flags as STOCK-AR (only the weight dir changes) | `.venv-vllm` |
| ENGINE-DIFFUSION | diffusion twin | `models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16` | vLLM pin `0b44dcc` (L0 free-text fix) hybrid_clean, `VLLM_FLARE_BIDIR_PROBE=1 VLLM_FLARE_CUDAGRAPH=1`, free-text (`tools=[]`), grammar inert | `.venv-vllm-p2-main` |

STOCK-AR and MERGED-AR use the **identical** offline-vLLM AR path (`run_ar_gsm8k.py`
style), differing only in the weight directory — the cleanest apples-to-apples for the
"what did RL+merge do to capability, served the normal way" axis. ENGINE-DIFFUSION is
the same RL-v2 weights served through the block-diffusion engine (the "what did the
diffusion conversion cost on top" axis).

## Capability classes (rows)

### A — GSM8K free-CoT strict (30 clean L1 prompts)
- Prompts: `runs/l1_census/gsm8k_prompts_clean.json` (GSM8K test first-30, 5-shot CoT,
  bare Qwen chat turns, thinking-off).
- Scoring: last `#### <number>` == gold (`scoring.py::score_gsm8k`). Strict.

### B — CODE (MBPP-sanitized, 25 problems)
- Source: `google-research-datasets/mbpp` config `sanitized`, **local HF cache, offline**
  (`.../mbpp/sanitized/0.0.0/4bb6404f…`). No network download.
- Few-shot = the dataset's designated `prompt` split, task_ids **[2, 3, 4]** (3-shot).
- Eval = first 25 `test`-split problems by ascending task_id: **[11,12,14,16,17,18,19,20,
  56,57,58,59,61,62,63,64,65,66,67,68,69,70,71,72,74]**.
- Prompt scaffold: bare Qwen chat turns (no system prompt, thinking-off); each user turn
  gives the task + the problem's `test_list`; the assistant answers with one ```python
  fenced block. `code_prompts.json`.
- Scoring (`scoring.py::score_code`): extract the **first ```python fenced block**; exec
  `test_imports + code + test_list asserts` in a fresh subprocess (5 s wall timeout);
  pass = subprocess return code 0. Validated: all 25 MBPP reference solutions pass this
  harness (25/25).

### C — INSTRUCTION-FOLLOWING (25 verifiable-constraint prompts, constructed locally)
- Constructed here (IFEval-style), `build_sets.py::INSTR` → `instr_prompts.json`.
  Zero-shot, bare Qwen chat turn.
- Each item carries **one deterministic machine check** (`check.type`), scored on the
  stripped completion by `scoring.py::score_instruction`. 18 check types spanning:
  exact word/sentence/line counts, keyword count/presence/absence, lipogram (no letter),
  no-comma, JSON-key set, line-prefix lists, all-caps, starts/ends-with, digit presence.
- Scoring is **strict on the full stripped completion** (a "Sure, here is…" preamble
  fails a strict-format item — that is the point). Same scoring for all three systems.

## Shared decode / prompt-token invariants
- Prompt token ids built once with the stock Qwen3.5-9B tokenizer; prompt vocab is shared
  by all three systems (diffusion mask id 248077 only occupies the generation canvas,
  never the prompt), so one id sequence feeds all three — exactly as class A was built.
- `block_size=32`, `mask_id=248077`, `grammar_topk=256`,
  `stop_token_ids=[248044,248045,248046,248059]` (`<|endoftext|> <|im_start|> <|im_end|>
  </tool_call>`), greedy (temp 0), seed 20260701, `max_tokens` 384 (A/B) / 256 (C).

## Reuse policy (existing rows)
- **A / STOCK-AR** = reuse `runs/l1_baseline_b1/ar_gsm8k_clean.jsonl` (29/30) — same
  prompts, same offline-vLLM greedy config.
- **A / ENGINE** = reuse `runs/l0l2_final_head_verify/engine_gsm8k_clean_head.jsonl`
  (26/30, pin 0b44dcc, all fin=stop) — the L0-fixed free-text head.
- Everything else is run fresh here.

## Files
- `build_sets.py` — builds `code_prompts.json` + `instr_prompts.json` + manifest.
- `scoring.py` — the three deterministic scorers (single source of truth).
- `code_prompts.json`, `instr_prompts.json`, `prompt_sets_manifest.json`.
- `run_ar_cell.py` — offline-vLLM AR runner (STOCK-AR / MERGED-AR), any class.
- `run_engine_cell.py` + `reboot_cell.sh` — hybrid_clean engine runner (reboot-safe).
- `aggregate.py` + `summary.json` + `report.md` — the 3×3 table.
