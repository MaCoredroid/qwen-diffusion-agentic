# AR-9B Agentic Baseline Result

Date: 2026-06-30 local

## Status

PASS: the real AR `Qwen/Qwen3.5-9B` baseline loads under SGLang and generates a
valid Qwen-native tool call.

## Launch

Command used:

```bash
PROFILE=bf16 PORT=30000 HOST=127.0.0.1 CONTEXT_LENGTH=8192 \
  MEM_FRACTION_STATIC=0.84 MAX_RUNNING_REQUESTS=1 \
  SERVED_MODEL_NAME=qwen3.5-9b-ar \
  scripts/serve_sglang_qwen35_9b_ar.sh
```

Observed backend:

- SGLang `0.5.14`
- model `Qwen/Qwen3.5-9B`
- served name `qwen3.5-9b-ar`
- dtype/quant for this verified baseline: bf16, no runtime weight quantization
- context length: 8192
- native parsers: `--tool-call-parser qwen`, `--reasoning-parser qwen3`
- SGLang allocated `MambaRadixCache` / hybrid GDN cache and reported server
  ready on `http://127.0.0.1:30000`

The first startup attempt used `MEM_FRACTION_STATIC=0.70` and failed after weight
load because the hybrid state-cache memory pool had negative rest memory. The
successful run used `0.84` and `MAX_RUNNING_REQUESTS=1`, which is appropriate on
this desktop session because GNOME was already using about 5.9 GiB.

## Smoke

Command:

```bash
scripts/smoke_openai_native_tool_call.py \
  --endpoint http://127.0.0.1:30000/v1 \
  --model qwen3.5-9b-ar \
  --out-json runs/agentic_eval/ar9b_baseline_smoke.json
```

Result:

- `/v1/models` returned `qwen3.5-9b-ar` with `max_model_len=8192`.
- The model emitted valid Qwen-native tool-call text:

```text
<tool_call>
<function=lookup_order>
<parameter=order_id>
ORD-419
</parameter>
<parameter=customer>
Acme Labs
</parameter>
</function>
</tool_call>
```

- The smoke validator parsed the Qwen-native content and verified required args
  `order_id` and `customer`.
- SGLang did not normalize this response into `message.tool_calls`; for the
  eval harness, parse Qwen-native content as the canonical format unless a proxy
  layer normalizes both backends identically.

## Quantization Note

This baseline is bf16 because the validated diffusion HF serving path loads the
bf16 Fast-dLLM base plus the LoRA adapter rather than a runtime NF4 server. If
the diffusion eval path changes to runtime 4-bit/NF4 serving, rerun this AR
baseline with `PROFILE=bnb4` and discard the bf16 comparison for matched-quant
claims.

No multi-turn harness was built in this step.
