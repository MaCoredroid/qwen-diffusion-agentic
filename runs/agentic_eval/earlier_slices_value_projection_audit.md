# Earlier Slice Value-Projection Audit

Verdict: FAIL for the requested "genuinely 0 projected VALUE tokens" check.

The old `wave1_value_tokens=0` counter did not prove value-safe projection for Qwen native XML. It only counted tokens whose schedule interval was labeled `argument_value`; before the Qwen XML planner fix, `<parameter>...</parameter>` bodies could be mislabeled as `json_structure` and then grammar-projected by wave 1.

| Slice | Rows/turns | wave1_value_tokens counter | wave1_projected_tokens | true XML value tokens | reported model value tokens | projected true-value token lower bound | rows with projected-value lower bound | exact rows dependent, lower bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matched-20ep northstar diffusion | 63 | 0 | 4157 | 1966 | 621 | 1345 | 63 | 55 |
| scale-up-58 per-call tau095 | 58 | 0 | 5999 | 4879 | 5194 | 18 | 3 | 0 |

One-line answer: matched-20ep was value-projection contaminated; scale-up-58 was not a strict zero either, though the lower-bound projected-value leakage was small and did not support any exact-passing row.

Method: tokenize each generated assistant output with `models/qwen3.5-9b-fastdllm-b1000-vllm-bf16`, count tokens overlapping Qwen XML parameter bodies, then compute a conservative lower bound of projected value tokens per row as:

`denoise_forwards_total == 0 ? true_xml_value_tokens : max(0, true_xml_value_tokens - reported_model_value_tokens)`, capped by `two_wave_wave1_projected_tokens`.

Inputs:

- `runs/agentic_eval/northstar_matched_ar_vllm_vs_diffusion/diffusion/turns.jsonl`
- `runs/flare_scaleup_eval/percall_waves_tau095/scaleup_native_58.jsonl`
