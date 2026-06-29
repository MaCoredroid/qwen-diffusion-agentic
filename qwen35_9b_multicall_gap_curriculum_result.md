# Qwen3.5-9B Multi-Call Gap Curriculum Result

Date: 2026-06-27.

## Scope

This is a CPU-side curriculum build for the remaining public multi-call failure
classes after contextual scalar projection:

- missing or reordered calls in a multi-call chain
- complex array/object payload extraction
- omitted required payloads that cannot be fixed by scalar projection alone

It uses public training multi-call records only. It does not use public eval gold
from `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`.

## Builder

Script:

```bash
.venv-fastdllm/bin/python scripts/build_toolcall_multicall_gap_curriculum.py
```

Output:

- dataset:
  `data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.json`
- audit:
  `data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.audit.jsonl`
- manifest:
  `data/qwen35_9b_toolcall_multicall_gap_curriculum/train_agentic_mix.manifest`

The emitted dataset is a Fast-DLLM conversation dataset with
`system`/`tools`/`messages`, not a plain `text` dataset.

## Result

- public multi-call source records: `56`
- candidates: `199`
- accepted rows: `181`
- rejected rows: `18`
- skipped candidates with no complex top-level payloads: `111`
- block size: `896`
- truncation side: `right`
- full assistant-label retention required: `true`
- max request excerpt: `1200` chars

Accepted row mix:

- missing-call recovery rows: `137`
- complex array/object extraction rows: `44`

Accepted rendered-token stats:

- length min/p50/p90/max: `405 / 728 / 842 / 894`
- kept assistant labels min/p50/p90/max: `25 / 45 / 98 / 194`
- accepted zero-label rows after truncation: `0`
- accepted partial-label rows after truncation: `0`

## Interpretation

Positive data-path result. Shortening missing-call prompts to call-local request
excerpts improved accepted rows from `160` to `181` and reduced rejections from
`39` to `18`, while keeping full label retention for all accepted rows.

This is not a promoted model result. The current promoted public multi-call path
is still the active checkpoint-275 constrained draft plus contextual projection:
`7/12` exact sequence and `7/12` exact arguments on the 12-row public multi-call
slice.

## Next Use

Use this curriculum as a staged repair/extraction lane first, not as a direct
main-generator mix. The safest next experiment is a small one-step/nonzero-label
gate, then a short adapter trained only for:

- recovering the omitted single call from a draft chain
- extracting one exact complex payload from request evidence

Do not mix this heavily into the main generator until an early public multi-call
sequence gate shows it does not regress the current `7/12` sequence path.

## First Adapter Probe

Date: 2026-06-27.

Added tooling:

- `scripts/build_toolcall_multicall_gap_eval_cases.py`
  - builds a held-out public multi-call gap eval from
    `data/toolcall_eval/public_multicall_hermes_smoke.jsonl`
  - output:
    `data/toolcall_eval/public_multicall_gap_eval.jsonl`
  - summary:
    `data/toolcall_eval/public_multicall_gap_eval.summary.json`
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`
  - now exposes `LORA_MODEL_PATH`, which lets a run continue from an existing
    PEFT adapter such as the active checkpoint-275 adapter.

Held-out gap eval:

- source cases: `12` public multi-call cases
- gap rows: `38`
- missing-call rows: `31`
- complex-extraction rows: `7`
- skipped complex candidates with no complex top-level payload: `24`

Training checks:

- one-step standalone gate from diffusion init:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_argspanw1p5_b896_step1_gate`
  - labels: pre-MDM `[29]`, post-MDM `[25, 4]`
  - train loss: `9.249107360839844`
- 50-step standalone adapter from diffusion init:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_argspanw1p5_b896_step50`
  - train loss: `7.142361106872559`
  - runtime: `107.7891s`
- one-step continuation gate from active checkpoint-275:
  - `LORA_MODEL_PATH`:
    `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_from_ckpt275_argspanw1p5_b896_step1_gate`
  - labels: pre-MDM `[29]`, post-MDM `[25, 4]`
  - train loss: `3.570006847381592`
- 50-step continuation from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_from_ckpt275_argspanw1p5_b896_step50`
  - train loss: `2.550190029144287`
  - runtime: `107.9005s`

Eval setup:

- evaluator:
  `scripts/eval_fastdllm_toolcall_cases.py`
- settings:
  `--full-context-sampling --max-new-tokens 160 --constrained-tool-decoding --constrained-max-calls 1`
- baseline output:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_eval/ckpt275/public_multicall_gap_eval.jsonl`
- continued-adapter output:
  `runs/fastdllm_qwen35_9b_toolcall_multicall_gap_eval/gap_from_ckpt275_step50/public_multicall_gap_eval.jsonl`

Overall held-out result:

| Model | Raw valid JSON | Raw exact sequence | Raw exact args | Constrained exact sequence | Constrained exact args |
| --- | ---: | ---: | ---: | ---: | ---: |
| active checkpoint-275 | 5/38 | 13/38 | 9/38 | 37/38 | 26/38 |
| checkpoint-275 + 50 gap steps | 8/38 | 11/38 | 6/38 | 38/38 | 23/38 |

By gap kind:

| Model | Kind | Rows | Raw exact sequence | Raw exact args | Constrained exact sequence | Constrained exact args |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| active checkpoint-275 | missing-call | 31 | 12/31 | 8/31 | 30/31 | 24/31 |
| active checkpoint-275 | complex-extract | 7 | 1/7 | 1/7 | 7/7 | 2/7 |
| checkpoint-275 + 50 gap steps | missing-call | 31 | 10/31 | 5/31 | 31/31 | 20/31 |
| checkpoint-275 + 50 gap steps | complex-extract | 7 | 1/7 | 1/7 | 7/7 | 3/7 |

Interpretation:

- Negative promotion result. The 50-step continuation improves raw JSON validity
  and slightly improves constrained complex-payload extraction, but it regresses
  the main missing-call lane and overall exact arguments.
- Do not promote `gap_from_ckpt275_argspanw1p5_b896_step50`.
- Do keep the eval builder and `LORA_MODEL_PATH` wrapper support. Both are
  useful for future staged repair experiments.
- Next gap experiment should split the curriculum by kind. A complex-payload
  only probe may be worth a small run; missing-call recovery should not be
  trained as free-form generation at this ratio because it damages exact call
  selection.

## Complex-Only Split Probe

Date: 2026-06-27.

After the mixed gap continuation regressed missing-call recovery, I split the
complex-payload extraction lane from the missing-call lane.

Complex-only train build:

```bash
.venv-fastdllm/bin/python scripts/build_toolcall_multicall_gap_curriculum.py \
  --no-include-missing-call \
  --out-dir data/qwen35_9b_toolcall_multicall_complex_extract_curriculum
```

Result:

- accepted rows: `44`
- rejected rows: `0`
- source records: `56` public train multi-call records
- source family: `public_train_multicall:complex_extract`
- block size: `896`
- accepted rendered length min/p50/p90/max: `405 / 633 / 753 / 877`
- accepted kept assistant labels min/p50/p90/max: `31 / 57 / 115 / 148`
- accepted zero-label rows after truncation: `0`
- accepted partial-label rows after truncation: `0`

Complex-only held-out eval build:

```bash
.venv-fastdllm/bin/python scripts/build_toolcall_multicall_gap_eval_cases.py \
  --no-include-missing-call \
  --out-jsonl data/toolcall_eval/public_multicall_complex_extract_eval.jsonl
```

Result:

- source cases: `12`
- held-out complex-extraction rows: `7`
- skipped candidates with no complex top-level payload: `24`

Training:

- one-step continuation gate from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_complex_extract_from_ckpt275_argspanw1p5_b896_step1_gate`
  - labels: pre-MDM `[55]`, post-MDM `[50, 5]`
  - train loss: `4.448453426361084`
- 25-step continuation from active checkpoint-275:
  - run:
    `runs/fastdllm_qwen35_9b_toolcall_complex_extract_from_ckpt275_argspanw1p5_b896_step25`
  - train loss: `3.2026946640014646`
  - runtime: `54.1621s`

Eval setup:

- evaluator:
  `scripts/eval_fastdllm_toolcall_cases.py`
- settings:
  `--full-context-sampling --max-new-tokens 160 --constrained-tool-decoding --constrained-max-calls 1`
- baseline output:
  `runs/fastdllm_qwen35_9b_toolcall_complex_extract_eval/ckpt275/public_multicall_complex_extract_eval.jsonl`
- complex-only adapter output:
  `runs/fastdllm_qwen35_9b_toolcall_complex_extract_eval/complex_from_ckpt275_step25/public_multicall_complex_extract_eval.jsonl`

Held-out result:

| Model | Raw valid JSON | Raw exact sequence | Raw exact args | Constrained exact sequence | Constrained exact args | Constrained schema-valid | Constrained required-present |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active checkpoint-275 | 1/7 | 1/7 | 1/7 | 7/7 | 2/7 | 3/7 | 3/7 |
| checkpoint-275 + 25 complex steps | 1/7 | 1/7 | 0/7 | 7/7 | 3/7 | 4/7 | 5/7 |

Interpretation:

- Mixed result. The split complex-only continuation improves constrained
  complex-payload extraction by one row and improves constrained schema/required
  coverage, but raw exact arguments regress from `1/7` to `0/7`.
- Do not promote this as a first-pass model update.
- The useful signal is narrower: complex-payload extraction appears to benefit
  from a small amount of targeted training when paired with constrained
  projection. The model still does not emit enough raw valid exact complex
  payloads without projection.
- Next complex-payload work should focus on generation-time constrained
  array/object decoding or a per-field extraction accept policy rather than
  more free-form continuation steps.

## Complex Context Projection v3

Date: 2026-06-27.

After the complex-only adapter produced only a small constrained gain, I moved
the useful part into deterministic constrained decoding:

- `scripts/eval_fastdllm_toolcall_cases.py` now attempts conservative
  request-context extraction for array/object arguments before accepting
  malformed generated JSON fragments.
- Supported evidence shapes include markdown tables, bullet lists, simple
  inline lists, schema-aware scalar coercion, plural key variants, and fuzzy
  table-header matching.
- The rule is only used for constrained tool-call reconstruction. Raw strict
  model metrics are unchanged.

Complex held-out eval:

| Model | Before constrained args | v3 constrained args | v3 schema-valid | v3 required-present |
| --- | ---: | ---: | ---: | ---: |
| active checkpoint-275 | 2/7 | 7/7 | 7/7 | 7/7 |
| checkpoint-275 + 25 complex steps | 3/7 | 7/7 | 7/7 | 7/7 |

Artifacts:

- active checkpoint-275:
  `runs/fastdllm_qwen35_9b_toolcall_complex_extract_eval/ckpt275/public_multicall_complex_extract_eval_complex_projection_v3.jsonl`
- checkpoint-275 + 25 complex steps:
  `runs/fastdllm_qwen35_9b_toolcall_complex_extract_eval/complex_from_ckpt275_step25/public_multicall_complex_extract_eval_complex_projection_v3.jsonl`

Public multi-call active path:

- sequence-preserving constrained output with complex extraction:
  `7/12` exact sequence, `4/12` exact arguments, `11/12` schema/required
- plus contextual scalar projection:
  `7/12` exact sequence, `7/12` exact arguments, `12/12` schema/required
- artifact:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_ckpt275_agentic_multicall_eval384_modelrepair/public_multicall_12_sequence_preserve_complex_contextual_v4.jsonl`

Cross-slice check:

| Slice | Before constrained args | v3 constrained args | Direction |
| --- | ---: | ---: | --- |
| public one-call | 5/8 | 5/8 | neutral |
| Qwen3.6 teacher train one-call | 5/12 | 5/12 | neutral |
| Qwen3.6 teacher heldout one-call | 3/8 | 4/8 | positive |
| synthetic tool-result | 8/10 | 8/10 | neutral |
| OpenAI-style tool-result | 9/10 | 9/10 | neutral |

Decision:

- Promote the complex context projection as a deterministic constrained-decoder
  improvement.
- Do not promote the complex-only trained adapter. It is no longer needed for
  the held-out complex lane and still regresses raw exact arguments.
- The next model-training work should target missing-call sequence recovery and
  raw valid complex-payload emission, while keeping this constrained decoder in
  the eval/promotion suite.

## Guarded Sequence-Planner Projection

Date: 2026-06-27.

Added a second CPU-side constrained-decoding diagnostic:
`scripts/rescore_toolcall_sequence_planner_projection.py`.

It uses request evidence only, not eval gold:

- numbered and bulleted task lists
- markdown table sections
- tool names, descriptions, required properties, enum/property evidence
- a guard that keeps one-call/tool-result outputs unchanged unless the input
  already contains at least two tool calls

Public multi-call result on the active checkpoint-275 path:

| Projection | Exact sequence | Exact args | Schema-valid | Required-present | Extra / missing / repeated |
| --- | ---: | ---: | ---: | ---: | ---: |
| complex + contextual scalar projection | 7/12 | 7/12 | 12/12 | 12/12 | 1 / 3 / 0 |
| guarded sequence-planner projection + segment-local args | 11/12 | 10/12 | 12/12 | 12/12 | 1 / 1 / 0 |

Cross-slice guard result:

- public one-call remains `8/8` sequence and `5/8` arguments
- Qwen3.6 teacher train one-call remains `10/12` sequence and `5/12`
  arguments
- Qwen3.6 teacher heldout one-call remains `8/8` sequence and `4/8`
  arguments
- synthetic tool-result remains `10/10` sequence and `8/10` arguments
- OpenAI-style tool-result remains `10/10` sequence and `9/10` arguments

Remaining public multi-call planner failures:

- one semantic ambiguity: the request provides voice-command arguments for
  security cameras while a direct `activate_security_cameras` tool also exists
- one scalar code assignment mismatch after sequence reordering in the smart
  home security case

Decision:

- Promote the guarded sequence planner as a deterministic diagnostic and a
  generation-time constrained-decoding blueprint.
- Do not treat it as a model-only score.
- The next missing-call training target should distill planner order into the
  student, while the next decoder target should continue tightening
  segment-local scalar extraction after any planned reorder.
