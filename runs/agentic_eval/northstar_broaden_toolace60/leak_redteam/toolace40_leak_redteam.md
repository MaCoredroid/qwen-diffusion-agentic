# ToolACE-40 Leak Red-Team

Verdict: **no-leak-but-easy**

## Provenance

- toolcall_seed: `data/toolcall_seed/qwen_toolcall_seed.jsonl`
  Upstream: {'hermes': 'NousResearch/hermes-function-calling-v1 train streaming', 'glaive': 'glaiveai/glaive-function-calling-v2 train streaming', 'toolace': 'Team-ACE/ToolACE train streaming'}
  Script: `scripts/prepare_toolcall_seed_data.py`
- fastdllm_toolcall_train: `data/fastdllm_toolcall_train/train_toolcall.json`
  Upstream: data/toolcall_seed/qwen_toolcall_seed.jsonl; manifest shows selected train_source_counts hermes=96
- public_train_multicall_gold_cases: `data/toolcall_eval/public_train_multicall_gold_cases.jsonl`
  Upstream: data/fastdllm_toolcall_train/train_toolcall.json via materialize_conversation_toolcall_cases.py
- public_train_multicall_no_public_smoke_cases: `data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl`
  Upstream: train_toolcall.json after removing public_multicall_hermes_smoke overlaps
- flare_agentic_mix_v2_native: `data/flare_agentic_mix_v2_native/train_agentic_mix.json`
  Upstream: includes 28 public_multicall_gold_native and 28 public_multicall_no_public_native records
- run1_copy_retention_mix: `data/flare_redesign_run1_copy_retention_mix/train_agentic_mix.json`
  Upstream: uses flare_agentic_mix_v2_native as native pool and lists both public-train multicall JSONLs in pool/exclude-eval lists
- B@1000 two-stream adapter: `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000`
  Upstream: scripts/run_flare_stage1_ab_pilot_job.sh with DATASET_DIR=data/flare_stage1_ab_pilot_train, GSM8K/MBPP retention only
- ToolACE-40 broaden eval: `data/toolcall_eval_native/flare_broaden_public_toolace60.jsonl rows 20-59`
  Upstream: Team-ACE/ToolACE train split, first accepted 40 rows from scripts/build_flare_broaden_public_eval.py

## Explicit Overlap Check

- Eval ids checked: 40
- Train files checked: 92
- Train rows checked: 24505
- Train tool calls indexed: 14276
- Canonical text overlaps: 0
- User prompt overlaps: 0
- Full tool-signature overlaps: 0
- Same-tool/all-arg-value overlaps: 0
- Descriptive tool-name overlaps: 0

## Difficulty

| Slice | episodes | turns | calls/turn | args/call | avg value len | copy args | derived/constant args |
|---|---:|---:|---:|---:|---:|---:|---:|
| ToolACE-derived | 40 | 128 | 1.000 | 1.398 | 11.061 | 104 | 77 |
| our-synthetic | 20 | 63 | 1.000 | 2.492 | 13.133 | 222 | 54 |

Interpretation: ToolACE-40 uses eval rows from the ToolACE train split and is therefore not a clean external heldout source. The explicit row-level overlap checks against local train mixes found no hard overlap, but the slice is structurally easier: one call per generated turn and mostly short scalar arguments copied verbatim from the active user context.

Full JSON: `/home/mark/qwen_diffusion/runs/agentic_eval/northstar_broaden_toolace60/leak_redteam/toolace40_leak_redteam.json`
Per-episode overlap rows: `/home/mark/qwen_diffusion/runs/agentic_eval/northstar_broaden_toolace60/leak_redteam/toolace40_overlap_rows.jsonl`
Per-episode difficulty rows: `/home/mark/qwen_diffusion/runs/agentic_eval/northstar_broaden_toolace60/leak_redteam/toolace40_difficulty_rows.jsonl`
