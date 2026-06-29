# Qwen3.6 Teacher Heldout Multi-Call Result

Date: 2026-06-28

## Purpose

Measure whether the local Qwen3.6-27B teacher can replace the weak
request-derived planner on the clean heldout multi-call seed slice.

The target slice is:

```text
data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl
```

It has `13` clean Hermes multi-call rows with `2` to `3` gold calls each.

## Serving Profile

Attempted first:

```text
Qwen3.6-27B NVFP4, 8k context, MTP/NEXTN enabled, radix cache disabled,
CUDA graph disabled, MEM_FRACTION_STATIC=0.84 then 0.86
```

Both MTP attempts failed during SGLang memory-pool setup because this SGLang
runtime now counts draft weights in the KV-cache memory check:

```text
Loaded weights leave no GPU memory for the KV cache ...
If using speculative decoding, draft weights are now counted.
```

Working profile for this run:

```bash
PROFILE=nvfp4 \
HOST=127.0.0.1 \
PORT=30000 \
CONTEXT_LENGTH=8192 \
CHUNKED_PREFILL_SIZE=1024 \
MEM_FRACTION_STATIC=0.84 \
MAX_RUNNING_REQUESTS=1 \
MAX_TOTAL_TOKENS=8192 \
DISABLE_RADIX_CACHE=1 \
CUDA_GRAPH_BACKEND_DECODE=disabled \
CUDA_GRAPH_BACKEND_PREFILL=disabled \
ENABLE_MTP=0 \
scripts/serve_sglang_qwen36_teacher.sh
```

The no-MTP profile served successfully:

```text
GET /v1/models -> qwen3.6-27b-teacher, max_model_len=8192
```

Observed local GPU memory while live:

```text
about 24.9 GiB / 32.6 GiB
```

The server was stopped after the eval to free the RTX 5090.

## Eval Commands

Native required tool calls:

```bash
.venv-fastdllm/bin/python scripts/eval_openai_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl \
  --out-jsonl runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_required.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --message-field prompt_messages \
  --tool-choice required \
  --allow-text-fallback \
  --max-tokens 900 \
  --timeout 180
```

Auto/text fallback:

```bash
.venv-fastdllm/bin/python scripts/eval_openai_toolcall_cases.py \
  --input-jsonl data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl \
  --out-jsonl runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_auto.jsonl \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.6-27b-teacher \
  --message-field prompt_messages \
  --tool-choice auto \
  --allow-text-fallback \
  --max-tokens 900 \
  --timeout 180
```

## Results

Required/native:

- records: `13`
- native tool-call responses: `13`
- valid tool JSON: `13/13`
- exact tool-name set: `10/13`
- exact tool sequence: `9/13`
- exact tool-name multiset: `9/13`
- same tool-call count: `9/13`
- exact arguments: `6/13`
- schema valid: `11/13`
- required args present: `11/13`
- extra-call records: `1`
- missing-call records: `3`
- repeated-call records: `1`
- elapsed: `72.1s`

Auto/text fallback:

- records: `13`
- native tool-call responses: `0`
- text fallback responses: `13`
- valid tool JSON: `13/13`
- exact tool-name set: `9/13`
- exact tool sequence: `8/13`
- exact tool-name multiset: `8/13`
- same tool-call count: `8/13`
- exact arguments: `6/13`
- schema valid: `13/13`
- required args present: `13/13`
- extra-call records: `0`
- missing-call records: `5`
- repeated-call records: `1`
- elapsed: `83.0s`

## Failure Shape

Required/native is the stronger teacher setting for this slice. Its failures:

- `heldout_seed_multicall_0001`: teacher calls the three gold
  `record_project_expense` calls, but also follows the prompt's explicit
  categorize/report instructions. Gold omits those later requested actions.
- `heldout_seed_multicall_0002`: exact sequence, but percentage/risk values
  are normalized differently from gold (`0.15` style rates, `Value at Risk`,
  `0.3333` weights).
- `heldout_seed_multicall_0003`: exact sequence, but first device add chooses
  the smart lock while gold chooses the smart light, and discovery omits
  `network_id`.
- `heldout_seed_multicall_0004`: combines two demographic campaigns into one
  `create_campaign` call, while gold uses two separate calls.
- `heldout_seed_multicall_0007`: creates the trivia game but omits the stream
  setup call.
- `heldout_seed_multicall_0009`: exact sequence, but event/refund arguments
  differ from gold.
- `heldout_seed_multicall_0010`: generates recommendations but omits the
  update-preferences call.

Auto/text fallback undercalls more often, so it is not the preferred planner
target mode for this slice.

## Interpretation

Qwen3.6 is much stronger than the deterministic empty-request planner
(`9/13` exact sequence and `6/13` exact arguments versus `3/13` and `0/13`),
but it does not close the heldout live-planner gap.

This result also exposes seed-label ambiguity: at least one heldout prompt asks
for actions that the gold intentionally or accidentally omits. Therefore, the
next planner target should not blindly optimize to the seed gold. It needs an
explicit decomposition policy:

- decide whether to follow the full user request or the seed gold subset;
- decide when repeated list items should become separate calls versus one call
  with an array payload;
- normalize percentages, risk-model aliases, IDs, and policy thresholds;
- preserve tool-result dependencies such as `use_id_from_previous_call`.

For the current behavior-preserving diffusion route, the gold-span protected
replay remains the clean sampler proof. The live planner still needs a
teacher/decomposition sidecar or curated planner distillation before the
heldout route can be counted as end to end.
