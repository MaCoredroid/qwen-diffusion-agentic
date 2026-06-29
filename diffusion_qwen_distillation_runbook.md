# Diffusion-Qwen Distillation Runbook

Concrete train → verify → serve recipe for adapting **Qwen3.6-27B** (autoregressive) into a
**block-diffusion** model. The procedure is "fork an existing template and add the part it doesn't
cover."

- **Template:** Fast-dLLM v2 (NVlabs/Fast-dLLM, `v2/`) — adapts a pretrained AR model into a
  block-diffusion decoder with ~1B tokens of fine-tuning, already shown on Qwen2.5-Instruct with
  released 1.5B / 7B checkpoints. [R1, R2]
- **The gap you fill:** Qwen2.5 is pure attention; Qwen3.6-27B is GDN-heavy (3 of every 4 sublayers
  are Gated DeltaNet linear-attention). The template has no GDN handling — that is the novel work.
- **Target use:** agentic coding / SWE-bench Verified, via a patched coding-agent harness.

Architecture recap: 64 layers = 16 × (3×[Gated DeltaNet → FFN] + 1×[Gated Attention → FFN]); GDN
= gated linear-recurrence, causal, constant-size state, no KV cache; ships MTP (multi-step) heads. [R12]

---

## Phase 1 — TRAIN (adaptation / distillation)

1. **Fork + surgery.** Start from `Fast-dLLM/v2`. Load Qwen3.6-27B weights. Add an absorbing
   `[MASK]` token (reuse a reserved vocab id; init its embedding from the mean token embedding).
   Install the block-causal attention mask on the Gated Attention layers: **bidirectional within a
   block, causal across blocks** (clean previous blocks visible, future blocks masked). Switch the
   head to **shifted prediction** (Dream) so the AR next-token head maps onto fill-this-position
   with no new head weights. [R1, R3, R4]

2. **GDN handling — the only non-template step.**
   - **Option A (v0, cheapest):** leave the forward GDN chunk-scan kernel unchanged; use GDN as the
     *cross-block causal state carrier*. Snapshot the constant-size state `S` at each block boundary;
     re-scan only the current block each denoising pass. Within-block bidirectionality then comes
     entirely from the 1-in-4 Gated Attention layers.
   - **Option B (upgrade):** bidirectional GDN — forward scan (carries state across blocks +
     through block) plus a **backward scan within the block only** (resets at each block boundary so
     it can't leak future blocks); fuse the two outputs. New reversed-scan kernel + fusion, ~2×
     within-block GDN cost. Switch to B only if verification shows within-block under-mixing.

3. **Objective.** Block masked-diffusion: within the current block, mask a subset per a noise
   schedule, predict the originals, cross-entropy on masked positions only; previous blocks are
   clean context. The complementary mask preserves the AR objective so pretrained knowledge is not
   destroyed. [R1]

4. **Curriculum + scale.**
   - **Block-size warmup:** start at block size ≈1 (model is in-distribution, barely moves), grow
     1 → 4 → 16 → … → 256. This is what makes the transition from sequential to parallel cheap. [R5, R6]
   - **Low LR** (~1e-5, cosine) — Dream found LR critical to preserve AR-inherited knowledge. [R3]
   - **~1B tokens**, weighted toward code / repo edits / infill (target is SWE-bench), plus some
     general text to avoid forgetting. [R1]
   - 27B at ~1B tokens ≈ hundreds of GPU-hours, not a pretraining run. bf16 + fp32 master is the
     safe default; fp8 optional if your stack supports it.

## Phase 2 — VERIFY (cheap → expensive)

5. **Numerics.** Denoising loss / perplexity on held-out vs the AR teacher (expect a gap — it's a
   speed model). Sweep (denoising steps × block size × confidence threshold τ) to map the
   speed/quality Pareto.

6. **Diffusion-strength checks.** Infilling, code-in-the-middle, structured-format correctness.
   These should be *strengths*; if not, the conversion is broken.

7. **Capability test — read the caution first.** Wire into the patched codex; run a SWE-bench
   Verified slice (DiffusionGemma stand-in first, then Diffusion-Qwen), comparing vs AR Qwen3.6.
   Before this, read the agentic reality-check paper [R7] — agentic multi-step is where diffusion
   LMs have shown weakness, and it tells you what to watch. The GDN Option-A-vs-B ablation lives
   here: if large blocks smear, you need the bidirectional scan.

8. **Coverage check (RL-bridge prerequisite).** Measure the gap between Diffusion-Qwen samples and
   AR gold logprobs. Even if you only ship the product, this quantifies drift and gates the
   downstream calibrated RL bridge.

## Phase 3 — SERVE / INFERENCE

9. **Decode loop (block by block).** For block *b*: fix prefix `x_<b`, start from an all-`[MASK]`
   block, then iteratively refine — for each masked position compute confidence
   `c_i = max_v p(x_i = v | x_<b, x_b)`, unmask tokens with `c_i ≥ τ`, and always unmask at least the
   single highest-confidence token to guarantee progress. Repeat K passes (K = speed/quality dial),
   commit the block, advance. [R7]

10. **Caching (throughput win).** Fast-dLLM v2 ships hierarchical caching — a block-level cache for
    cross-block context plus a sub-block dual-cache for within-block parallel decoding. **Your
    addition:** snapshot the constant-size GDN state alongside the attention KV at each block
    boundary. That snapshot is what keeps long context cheap and is absent from the template. [R1]

11. **Engine + quant.** Extend vLLM's diffusion path (DiffusionGemma's day-0 vLLM support is a
    scaffold to build on) or SGLang, with a block scheduler + GDN state cache. Quantize to 4-bit to
    fit and to chase throughput on a 5090. Knobs exposed to the agent: denoising steps, τ, block
    size, per-position temperature. [R8]

12. **Harness glue.** Re-do tool-call boundary and stop-condition detection for block decoding (the
    block resolves all at once; you can't scan left-to-right for an end marker). Debug against the
    DiffusionGemma stand-in before Diffusion-Qwen exists.

### Tool-Sensitive Block Boundaries

The agentic target should not use one uniform block policy everywhere. Prose and
reasoning can use larger blocks, while tool-call syntax, function names, schema
keys, scalar argument values, and stop boundaries should use tiny or small
blocks with stronger constraints and more denoising steps. This is the direct
bridge from the current post-hoc protected path to generation-time protection.

Current prototype:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl data/toolcall_eval/public_onecall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_onecall_hermes_gold_blocks.jsonl \
  --limit 8
```

The prototype emits an auditable span plan over known assistant/tool-call text:

- prose: large blocks, light or no constraints
- `<tool_call>` tags: literal tiny blocks
- function names: tool-name enum tiny blocks
- JSON keys: schema-key enum tiny blocks
- argument values: small blocks with schema and request-evidence constraints
- JSON containers/punctuation: small grammar-constrained blocks

With `--tokenizer-path models/qwen3.5-9b-fastdllm-init`, the same script also
emits `token_blocks`: non-overlapping Qwen tokenizer ranges assigned to the
character span they overlap most. This is not yet wired into the sampler. Use
it first to measure sensitive-span distribution across the six split-route
scorecard lanes, then consume `token_blocks` in a sampler dry-run before
changing generation.

Sampler dry-run handoff:

```bash
.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl
```

The schedule chunks prose, JSON structure, tool names, schema keys, tool tags,
and argument values with separate block-size and denoising-step policies. It is
the non-generating trace format to wire into Fast-DLLM before changing actual
sampling.

Fast-DLLM absolute-position trace:

```bash
.venv-fastdllm/bin/python scripts/trace_tool_sensitive_fastdllm_schedule.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_fastdllm_trace.jsonl \
  --conversation-template fast_dllm_v2 \
  --block-size 32 \
  --small-block-size 8
```

The trace adds prompt-token offsets and splits scheduled chunks at Fast-DLLM
block and small-block boundaries. On the public multi-call slice, raw live
output crosses `28` Fast-DLLM block boundaries and `80` small-block boundaries;
protected live output improves this to `15` and `61`, while gold is `9` and
`54`. Use this as the first risk metric before enabling real sampler overrides.

First opt-in sampler override:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  ...
```

This keeps Fast-DLLM block commits aligned but replaces the inner uniform
`small_block_size` windows with scheduled tool-sensitive intervals. A one-case
public multi-call smoke completed and recorded `132` scheduled interval visits,
but it did not improve strict tool-call quality because the model still entered
a thinking/prose trajectory. Treat schedule-aware windows as mechanical
infrastructure; the next sampler change should force or heavily bias the first
tool-tag interval and suppress thinking/prose when the task requires only tool
calls.

First-prefix format control:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --force-tool-call-prefix \
  ...
```

This fixes `<tool_call>\n` as assistant context before sampling while preserving
that prefix in the decoded assistant output. On the same one-case public
multi-call smoke, it removed the thinking/prose trajectory and generated
plausible tool-call blocks, but strict scoring remained `0/1`: the second call
had malformed JSON around a key/value separator and the sample continued into
an extra `<tool_call>\nuser...` fragment after the intended calls. The next
generation-time control should therefore target JSON separator grammar and
stop-boundary enforcement after the planned number of `</tool_call>` blocks.

Stop-boundary and sequence-preserving protected replay:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule.jsonl \
  --force-tool-call-prefix \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-assume-utc-z \
  --constrained-max-calls 3 \
  ...
```

The one-case public multi-call smoke with these flags writes
`runs/tool_sensitive_block_plans/public_multicall_forced_prefix_stopguard_seqproject_utc_smoke1.jsonl`.
It is still a protected path: raw strict scoring is `0/1`, while constrained
projection recovers `1/1` valid tool JSON, `1/1` exact tool sequence, and
`1/1` exact arguments. The previous non-UTC protected replay remains useful as
an ablation because it showed the isolated timestamp suffix failure.

Schedule target-token forcing:

```bash
.venv-fastdllm/bin/python scripts/plan_tool_sensitive_blocks.py \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v3.jsonl \
  --text-field sequence_planner_assistant \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized_with_ids.jsonl \
  --limit 1 \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-token-ids

.venv-fastdllm/bin/python scripts/emit_tool_sensitive_sampler_schedule.py \
  --input-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_blocks_tokenized_with_ids.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --include-token-ids
```

The evaluator can then hard-fill selected scheduled spans:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  ...
```

Structural-only forcing is not enough: the first one-case smoke forces `77`
structural/tool-name/key tokens but still fails raw strict scoring because an
unforced argument value emits an early `</tool_call>` fragment. The oracle
all-schedule force (`tool_tag,json_key,json_structure,tool_name,argument_value,prose`)
forces `137` scheduled tokens and reaches raw `1/1` valid JSON, exact sequence,
and exact arguments. Treat that as an alignment/upper-bound diagnostic. The
next non-oracle step is constrained value decoding, especially banning structural
delimiters inside scalar argument values unless the value grammar permits them.

Argument-value delimiter guard:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_1.jsonl \
  --force-schedule-token-kinds tool_tag,json_key,json_structure,tool_name \
  --force-argument-boundary-target-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-json-boundary-tokens \
  --ban-argument-newline-tokens \
  ...
```

The first one-case smoke with these flags writes
`runs/tool_sensitive_block_plans/public_multicall_schedule_forced_structure_argban_jsonnewline_smoke1.jsonl`.
It forces `77` structural/tool/key tokens and `4` argument-boundary tokens, then
bans delimiter/newline candidates across `167` masked value positions. It still
fails raw strict scoring, but the protected projection recovers exact tool
sequence. Remaining failure is value-copy corruption, especially datetime and
camera-ID extraction in the malformed third call.

Candidate-value diagnostic:

```bash
.venv-fastdllm/bin/python scripts/diagnose_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_meta_1.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_value_candidate_diagnostic_1.jsonl
```

The first diagnostic shows deterministic extraction is exact for only `2/7`
argument values, but the target value is present in the evidence candidate set
for `7/7` values. That is the right shape for the next sampler experiment:
do not ask the regex extractor to decide; use it to build candidate sets, then
let the model choose under a constrained value-span vocabulary.

Candidate-constrained value decoding:

```bash
.venv-fastdllm/bin/python scripts/augment_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_ids_meta_1.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_candidates_1.jsonl
```

For protected sidecar/target schedules, do not let the deterministic evidence
extractor define `selected_candidate`; that can overwrite a correct target span
with a nearby but wrong value from the request. Use:

```bash
.venv-fastdllm/bin/python scripts/augment_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_with_ids.jsonl \
  --cases-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-target-candidate \
  --selected-candidate-mode target \
  --out-jsonl runs/tool_sensitive_block_plans/synthetic_selector_sidecar_sampler_schedule_augmented_targetselected.jsonl
```

This makes `selected_candidate` mean "the protected schedule target" unless a
later selector-sidecar injection narrows a span. The synthetic sidecar scheduled
gate then reaches raw exact sequence `8/8`, exact arguments `8/8`, and valid
JSON `8/8` using checkpoint-275 as the main generator. Treat this as protected
runtime evidence, not learned model behavior.

The stronger non-oracle ablation leaves `--force-selected-candidate-tokens` off
and relies on `--force-best-candidate-sequence` to choose whole argument values.
With selector-injected restrictions it reaches raw sequence/arguments `8/8` /
`8/8`; with selector injection removed it still reaches `8/8` / `8/8` on the
synthetic analogue slice. Therefore the local sampler bottleneck is not
argument-span forcing once target-containing candidate sequences exist. The next
harder gate is candidate proposal and public multi-call generalization.

Full public multi-call target-candidate gate, 2026-06-28:

```bash
.venv-fastdllm/bin/python scripts/augment_schedule_value_candidates.py \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_ids_meta_12.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --tokenizer-path models/qwen3.5-9b-fastdllm-init \
  --include-target-candidate \
  --selected-candidate-mode target \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_candidates_targetselected_v4_12.jsonl
```

The rebuilt schedule has `100` argument blocks augmented and `90` argument
blocks with sequence candidates. On the 12-case public multi-call slice:

- model-ranked values with `--force-selected-candidate-tokens` off:
  raw `11/12` exact sequence, `9/12` exact arguments, `11/12` valid JSON.
- target-selected upper bound with `--force-selected-candidate-tokens` on:
  raw `12/12` exact sequence, `12/12` exact arguments, `12/12` valid JSON.

The result note is
`qwen35_public_multicall_targetcandidate_sampler_result.md`. The remaining
model-ranked misses are time/value ranking and long table/array consistency,
not inability to force spans once candidates exist.

V5 update: decimal numeric target spans are now represented by using target
token IDs directly when the candidate value equals the target. This raises
argument sequence coverage from `90/100` to `100/100` and improves the public
model-ranked sampler from raw `11/12` sequence and `11/12` valid JSON to
`12/12` sequence and `12/12` valid JSON. Exact arguments remain `9/12`.

Miss-audit helper:

```bash
.venv-fastdllm/bin/python scripts/analyze_toolcall_candidate_misses.py \
  --eval-jsonl runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_generation_v4_12.jsonl \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_candidates_targetselected_v4_12.jsonl \
  --out-jsonl runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v4_12.jsonl
```

It finds `3` failed records: two scalar time-ranking errors where the gold value
is already in the candidate set, and one invalid `process_invoices` table body.

After the v5 decimal-span fix, the long finance row becomes valid and exact
sequence. The remaining five focused candidate-ranking misses are:

- thermostat `schedule_time`: `19:00` over `11:00`
- fridge `start_time`: `23:00` over `22:00`
- finance `invoice_data[1].client_id`: `CLI-102` over `CLI-103`
- finance `invoice_data[1].invoice_id`: `INV-302` over `INV-301`
- finance `invoice_data[2].client_id`: `CLI-103` over `CLI-101`

Focused target:

```bash
.venv-fastdllm/bin/python scripts/filter_candidate_examples_by_miss_audit.py \
  --examples-jsonl data/candidate_ranking/public_multicall_targetcandidate_ranking_v5_12.jsonl \
  --audit-jsonl runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v5_12.jsonl \
  --out-jsonl data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl \
  --out-train-json data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.train.json
```

Checkpoint-275 scores `0/5` on this focused target in both prefix-only and
full-gold masked value-span ranking modes, so this is now the smallest public
value-ranking pressure test.

Focused value-ranker diagnostic, 2026-06-28:

- result note:
  `qwen35_public_multicall_focused_value_ranker_result.md`
- curriculum:
  `data/qwen35_9b_public_multicall_v5_focused_miss_value_span_diag_curriculum`
- training:
  `runs/fastdllm_qwen35_9b_public_multicall_v5_focused_miss_valuespan_diag_from_ckpt275_step10_plain`
- the label-only value-span variant failed before training because
  `FASTDLLM_VALUE_SPAN_TOKEN_IDS` cannot be derived from standalone value-span
  answers.
- the plain 10-step value-span conversation SFT trained successfully but kept
  focused masked candidate-ranking at `0/5` for both checkpoint-5 and
  checkpoint-10.

Do not scale this exact plain SFT recipe. The next candidate/value work should
use an explicit sidecar classifier, a pairwise ranking loss, or table-row-local
candidate grouping.

Focused index-sidecar diagnostic, 2026-06-28:

- result note:
  `qwen35_public_multicall_focused_index_sidecar_result.md`
- curriculum:
  `data/qwen35_9b_public_multicall_v5_focused_miss_index_diag_curriculum`
- conservative training:
  `runs/fastdllm_qwen35_9b_public_multicall_v5_focused_miss_index_diag_from_ckpt275_step10_plain`
- stronger 5090-fit training:
  `runs/fastdllm_qwen35_9b_public_multicall_v5_focused_miss_index_diag_from_ckpt275_step20_b1536_lr1e5`
- new generation evaluator:
  `scripts/eval_fastdllm_candidate_index_generation.py`
- direct masked index ranking starts at `2/5` for checkpoint-275, because the
  two time fields are already correct.
- both the 10-step `1e-6` run and the 20-step `1e-5`, `1536`-block,
  `DISABLE_GROUP_TEXTS=1` run remain `2/5`; the three finance table ID choices
  keep preferring earlier nearby candidates.
- diffusion generation of the numeric index is not usable yet: checkpoint-275
  and the stronger step-20 adapter both produce `0/5` in-range index answers.
- the attempted `2048` block per-example run OOMed on the RTX 5090 during the
  first backward pass by a small margin. Use `1536` as the practical 9B QLoRA
  block-size ceiling unless memory-saving kernels/offload are added.

Do not scale plain numeric-index chat SFT. The next selector should remove
index-position bias: pairwise candidate comparison, randomized candidate order,
table-row tuple scoring, or a small external sidecar classifier with heldout
movement before distillation back into the diffusion model.

Path-aware pairwise selector sidecar, 2026-06-28:

- result note:
  `qwen35_public_multicall_pairwise_path_sidecar_result.md`
- scripts:
  `scripts/build_candidate_pairwise_curriculum.py`
  `scripts/eval_fastdllm_candidate_pairwise_ranking.py`
  `scripts/eval_fastdllm_candidate_pairwise_tournament.py`
  `scripts/inject_pairwise_tournament_schedule_choices.py`
- focused path-aware A/B gate:
  `runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_path_rank.summary.json`
  reaches `60/60`.
- focused path-aware tournament:
  `runs/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets_ckpt275_pairwise_path_tournament.summary.json`
  reaches `5/5`.
- adding `JSON path` is the decisive change. Pathless prompts cannot reliably
  distinguish repeated array fields such as `invoice_data[1].client_id` versus
  `invoice_data[2].client_id`.
- injecting the first five path-aware choices as singleton candidate sequences
  improves the public v5 sampler to raw `12/12` exact sequence and `11/12`
  exact arguments, exposing one additional span:
  `payment_data[0].invoice_id`.
- adding that sixth path-aware tournament choice reaches raw `12/12` exact
  sequence, `12/12` exact arguments, and `12/12` valid JSON:
  `runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_generation_v5_12.summary.json`.
- final audit:
  `runs/tool_sensitive_block_plans/public_multicall_pairwise_path_singleton_structguard6_ckpt275_candidate_miss_audit_v5_12.summary.json`
  has `0` failed records and `0` mismatches.

Path-aware phrase selector heldout gate, 2026-06-28:

- result note:
  `qwen35_pathaware_phrase_selector_gate_result.md`
- candidate extractor upgrade:
  `scripts/diagnose_schedule_value_candidates.py` now adds conservative
  single-quoted string spans and digit-bearing model/product phrases. This
  turns repeated-substring failures like `Main Control Control`,
  `YRD256 Yale Assure Lock Lock`, and `ChemSimulationSimulation` into explicit
  user-evidence candidate choices.
- train-only phrase-aware ranking examples:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase.jsonl`
  contains `367` rows, `328` usable rows, and `173` usable argument-value rows.
- promotion-eligible train-only pairwise curriculum:
  `data/qwen35_9b_public_train_pairwise_pathaware_phrase_curriculum`
  has `376` accepted rows, `0` rejected labels,
  `contains_eval_slice=false`, and `promotion_allowed=true`.
- heldout public-12 phrase-aware tournament:
  `runs/candidate_ranking/public_multicall_pathaware_phrase12_ckpt275_pairwise_tournament.summary.json`
  reaches `98/99` overall, `68/68` argument values, and `30/31` tool names.
- injecting only argument-value choices:
  `runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_path_arg_choices_pathaware_phrase_12.summary.json`
  applies `68` correct argument selectors to `74` schedule chunks with `0`
  candidate misses and filters out all `31` tool-name selectors.
- protected sampler gate:
  `runs/tool_sensitive_block_plans/public_multicall_pathaware_phrase12_argselector_structguard_ckpt275_generation.summary.json`
  reaches raw `12/12` exact sequence, `12/12` exact arguments, and `12/12`
  valid JSON.
- final audit:
  `runs/tool_sensitive_block_plans/public_multicall_pathaware_phrase12_argselector_structguard_ckpt275_candidate_miss_audit.summary.json`
  has `0` failed records and `0` mismatches.

Interpretation: the argument selector route is now strong enough to use as a
distillation target, but it is still a protected runtime scaffold. Do not count
it as model promotion until train-only selector behavior improves raw or
constrained-decoder metrics on heldout public/teacher gates.

Train-only phrase pairwise SFT follow-up, 2026-06-28:

- result note:
  `qwen35_public_train_pairwise_phrase_sft_result.md`
- run:
  `runs/fastdllm_qwen35_9b_public_train_pairwise_pathaware_phrase_from_ckpt275_step10_lr1e6`
- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- settings: `10` steps, `1536` block size, left truncation,
  `DISABLE_GROUP_TEXTS=1`, learning rate `1e-6`, LoRA `r=8/alpha=16`, Qwen3.5
  attention plus GDN projection target modules.
- training completes on the RTX 5090 in `156.2s`, saves checkpoint-5 and
  checkpoint-10, and has train loss `4.9155`.
- heldout selector gate is unchanged:
  - checkpoint-275: `98/99` overall, `68/68` argument values, `30/31` tool names
  - checkpoint-5: `98/99` overall, `68/68` argument values, `30/31` tool names
  - checkpoint-10: `98/99` overall, `68/68` argument values, `30/31` tool names
- row-level predictions changed on `0/99` heldout rows.

Interpretation: this is a no-regression SFT smoke, not model promotion. Do not
replace checkpoint-275 with this run. The next training target should either
address the known tool-name selector miss or pursue raw selected-value
span/copy movement.

Train-only tool-name pairwise SFT follow-up, 2026-06-28:

- result note:
  `qwen35_public_train_toolname_pairwise_sft_result.md`
- train-only curriculum:
  `data/qwen35_9b_public_train_pairwise_toolname_pathaware_phrase_curriculum`
  has `240` accepted rows, `0` rejected labels,
  `contains_eval_slice=false`, and `promotion_allowed=true`.
- run:
  `runs/fastdllm_qwen35_9b_public_train_pairwise_toolname_pathaware_phrase_from_ckpt275_step10_lr1e6`
- settings match the phrase pairwise SFT: `10` steps, `1536` block size, left
  truncation, `DISABLE_GROUP_TEXTS=1`, learning rate `1e-6`, LoRA
  `r=8/alpha=16`, Qwen3.5 attention plus GDN projection target modules.
- training completes on the RTX 5090 in `156.3s`, saves checkpoint-5 and
  checkpoint-10, and has train loss `5.0694`.
- heldout selector gate is unchanged:
  - checkpoint-275: `98/99` overall, `68/68` argument values, `30/31` tool names
  - checkpoint-5: `98/99` overall, `68/68` argument values, `30/31` tool names
  - checkpoint-10: `98/99` overall, `68/68` argument values, `30/31` tool names
- row-level predictions changed on `0/99` heldout rows.

Interpretation: more SFT on the current tool-name pairwise prompt is neutral.
The next tool-name selector experiment should change the prompt to include
same-call argument keys/values or sequence-plan context before training.

Tool-name same-call argument-sketch selector gate, 2026-06-28:

- result note:
  `qwen35_toolname_argsketch_selector_result.md`
- code paths:
  `scripts/build_candidate_ranking_examples.py`,
  `scripts/build_candidate_pairwise_curriculum.py`, and
  `scripts/eval_fastdllm_candidate_pairwise_tournament.py`
- heldout examples:
  `data/candidate_ranking/public_multicall_toolname_argument_ranking_pathaware_phrase_argsketch_12.jsonl`
  contain `99` usable rows: `68` argument-value rows and `31` tool-name rows.
- train-only arg-sketch examples:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking_pathaware_phrase_argsketch.jsonl`
  contain `367` rows and `328` usable rows.
- promotion-eligible curricula:
  `data/qwen35_9b_public_train_pairwise_toolname_pathaware_phrase_argsketch_curriculum`
  has `240` accepted tool-name rows, and
  `data/qwen35_9b_public_train_pairwise_pathaware_phrase_argsketch_curriculum`
  has `616` accepted tool-name plus argument-value rows.
- heldout checkpoint-275 tournament:
  `runs/candidate_ranking/public_multicall_pathaware_phrase_argsketch12_ckpt275_pairwise_tournament.summary.json`
  reaches `99/99` overall, `68/68` argument values, and `31/31` tool names.
  It ran in `248.6s`, with `18.47 GiB` max allocated and `27.46 GiB` max
  reserved VRAM.

Interpretation: the remaining tool-name miss was a context/boundary problem.
Adding same-call argument evidence fixed it without further SFT. For the
behavior-preserving diffusion recipe, tool-name blocks should receive local
call sketches or be committed after argument evidence is available. Do not
count this as raw model promotion; count it as a stronger selector/boundary
component gate.

Arg-sketch tool+argument selector sampler gate, 2026-06-28:

- result note:
  `qwen35_argsketch_toolargselector_sampler_result.md`
- injected schedule:
  `runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_argsketch_choices_12.summary.json`
- injection consumes all `99` correct selector rows from the arg-sketch
  tournament, restricts `226` token-level schedule items, and has `0`
  candidate misses.
- generation:
  `runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_generation.summary.json`
  reaches `12/12` valid JSON, `12/12` exact tool sequence, `12/12` exact
  arguments, and `12/12` schema-valid calls.
- final audit:
  `runs/tool_sensitive_block_plans/public_multicall_argsketch_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json`
  has `0` failed records, `0` mismatches, `0` missing/extra calls, and `0`
  invalid tool blocks.

Interpretation: selector-owned semantic spans now cover both tool names and
argument values for the protected public multi-call gate. This is still a
gold-tokenized schedule with deterministic structural and stop guards. The next
engineering target is a non-gold/live schedule format where the planner
proposes call slots and argument sketches, the selector chooses tool/value
candidates, and the diffusion sampler commits the protected spans.

Live-planner arg-sketch selector sampler gate, 2026-06-28:

- result note:
  `qwen35_live_planner_argsketch_sampler_result.md`
- live planner source:
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v5_voice_safe.jsonl`
- regenerated block plan:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_blocks_tokenized_with_ids.summary.json`
  covers `12` records, `31` tool calls, `31` tool-name blocks, and `100`
  argument-value blocks with path and `tool_call_index` metadata.
- evidence-only candidate coverage:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_evidence.summary.json`
  has sequence candidates for `69/100` planned argument-value blocks.
- planner-target-included selector examples:
  `data/candidate_ranking/public_multicall_live_v5_sequence_planned_toolname_argument_ranking_targetincluded.summary.json`
  has `131/131` usable selector rows.
- selector tournament:
  `runs/candidate_ranking/public_multicall_live_v5_sequence_planned_targetincluded_ckpt275_pairwise_tournament.summary.json`
  reaches `131/131` overall, `100/100` argument values, and `31/31` tool
  names.
- injected schedule:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_pairwise_choices.summary.json`
  consumes `131` correct selectors, restricts `267` schedule items, and has
  `0` candidate misses.
- generation:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_argsketch_toolargselector_structguard_ckpt275_generation.summary.json`
  reaches `12/12` valid JSON, `12/12` exact tool sequence, `12/12` exact
  arguments, and `12/12` schema-valid calls.
- audit:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_argsketch_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json`
  has `0` failed records, `0` mismatches, `0` missing/extra calls, and `0`
  invalid tool blocks.
- code fix:
  `scripts/inject_pairwise_tournament_schedule_choices.py` now keeps
  empty-string selector predictions; `""` is a valid argument value.

Interpretation: this removes the gold-schedule dependency for the protected
sampler replay. The planner now proposes the text/spans, the pairwise selector
chooses tool/value candidates, and the sampler commits protected semantic
spans. The remaining engineering gap is evidence-only coverage: remove
planner-target inclusion by improving candidate extraction for the `31` planned
argument-value blocks that currently lack sequence candidates. Those misses
are concentrated in table/list row fields (`date`, `category`, `description`,
`due_date`, `date_received`), copied phrases/IDs, enum/boolean values, room or
door phrases, and one empty-string argument.

Live-planner evidence-only selector sampler gate, 2026-06-28:

- result note:
  `qwen35_live_planner_evidence_selector_sampler_result.md`
- code change:
  `scripts/diagnose_schedule_value_candidates.py` now resolves nested schemas,
  supports empty strings, extracts ISO dates/booleans/snake-case/symbolic
  language tokens/location phrases/command phrases/capitalized target phrases,
  and uses path-aware markdown table row extraction.
- evidence-only candidate schedule:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_candidates_evidence_v5.summary.json`
  covers `100/100` argument-value blocks and `31/31` tool-name blocks.
- selector examples:
  `data/candidate_ranking/public_multicall_live_v5_sequence_planned_toolname_argument_ranking_evidence_v5.summary.json`
  has `131/131` usable rows and `0` missing targets.
- selector tournament:
  `runs/candidate_ranking/public_multicall_live_v5_sequence_planned_evidence_v5_ckpt275_pairwise_tournament.summary.json`
  reaches `131/131`, including `100/100` argument values and `31/31` tool
  names. Row-local table pruning cuts pair comparisons from the broad
  evidence run's `1480` to `690`.
- injected schedule:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_sequence_planned_sampler_schedule_with_evidence_pairwise_choices.summary.json`
  consumes `131` correct selectors, restricts `267` schedule items, and has
  `0` candidate misses.
- generation:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_generation.summary.json`
  reaches `12/12` valid JSON, `12/12` exact tool sequence, `12/12` exact
  arguments, and `12/12` schema-valid calls.
- audit:
  `runs/tool_sensitive_block_plans/public_multicall_live_v5_evidence_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json`
  has `0` failed records, `0` mismatches, `0` missing/extra calls, and `0`
  invalid tool blocks.
- replay route:
  `scripts/run_qwen35_live_evidence_selector_route.py` writes
  `runs/tool_sensitive_block_plans/live_v5_evidence_selector_route/route_plan.json`
  and `route_plan.sh`; `--verify-existing` writes
  `route_plan_verification.json` and currently passes `16` checks with `0`
  missing artifacts and `0` failures.

Interpretation: the live planner semantic candidate path no longer needs
planner-target inclusion on this slice. The current protected recipe is:
planner proposes spans, evidence extractor builds candidates, pairwise selector
chooses tool names/argument values, and the diffusion sampler commits those
singleton semantic spans under structural and stop guards. Next, promote this
into a reusable route and test on teacher-heldout/fresh Qwen3.6 multi-call
cases before training the selector/boundary adapter.

Synthetic analogue evidence-selector replay, 2026-06-28:

- result note:
  `qwen35_synthetic_evidence_selector_route_result.md`
- reusable runner:
  `scripts/run_qwen35_evidence_selector_route.py`
- default cases:
  `data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl`
- default planner/span source:
  `runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl`
- route artifacts:
  `runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan.json`
  and `route_plan.sh`

Command:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_evidence_selector_route.py \
  --execute \
  --verify-existing
```

Result:

- evidence candidate coverage: `60/60` argument-value blocks and `24/24`
  tool-name blocks, with `0` missing selector targets.
- selector tournament:
  `runs/candidate_ranking/synthetic_multicall_failure_evidence_selector_ckpt275_pairwise_tournament.summary.json`
  reaches `84/84`, including `60/60` argument values and `24/24` tool names.
- injected schedule:
  `runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/sampler_schedule_with_evidence_pairwise_choices.summary.json`
  consumes `84` correct selectors and has `0` candidate misses.
- generation:
  `runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/evidence_toolargselector_structguard_ckpt275_generation.summary.json`
  reaches `8/8` valid JSON, `8/8` exact tool sequence, `8/8` exact
  arguments, and `8/8` schema-valid calls.
- audit:
  `runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/candidate_miss_audit.summary.json`
  has `0` failed records, `0` mismatches, `0` missing/extra calls, and `0`
  invalid tool blocks.
- generic route verifier:
  `runs/tool_sensitive_block_plans/synthetic_multicall_failure_evidence_selector/route_plan_verification.json`
  passes `20` checks with `0` missing artifacts and `0` failures.

Interpretation: this moves the evidence-only protected route off the public
multi-call smoke and onto the synthetic analogue slice for the two active
failure families. It is still protected runtime evidence: structure, stop
boundaries, planner spans, and semantic singleton choices are scaffolded. The
next route gate should be a larger fresh/teacher multi-call slice, followed by
a model-side selector/boundary/value adapter experiment whose raw or
constrained-decoder metrics improve.

Heldout seed evidence-selector preflight, 2026-06-28:

- result note:
  `qwen35_heldout_seed_evidence_selector_preflight_result.md`
- heldout cases:
  `data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl`
  contains `13` clean Hermes multi-call rows with `2` to `3` calls each and no
  exact overlap with the filtered train/public/synthetic eval sources.
- planner preflight:
  `runs/heldout_seed_multicall_2to3_clean/sequence_planner_from_empty.summary.json`
  reaches `13/13` valid planned JSON but only `3/13` exact tool sequence and
  `0/13` exact arguments.
- gold-span evidence preflight:
  `runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/sampler_schedule_with_candidates_evidence.summary.json`
  covers `140/140` argument blocks and `32/32` tool-name blocks; the diagnostic
  target coverage is `144/144` argument values.
- selector tournament:
  `runs/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_ckpt275_pairwise_tournament.summary.json`
  reaches `157/172` overall, with `125/140` argument values and `32/32` tool
  names.
- miss shape:
  all `15` misses are argument-value row/list alignment errors: construction
  expense rows, IoT device rows, ad campaign schedule rows, and one refund
  policy choice.
- peer-context follow-up:
  `scripts/build_candidate_ranking_examples.py` now keeps `json_path` in
  argument-value group keys and emits non-leaking local peer argument sketches.
  `scripts/build_candidate_pairwise_curriculum.py` consumes those sketches and
  adds request snippets anchored on peer values/path terms.
- best follow-up tournament:
  `runs/candidate_ranking/heldout_seed_multicall_gold_evidence_selector_peerctx_rules_snippets_ckpt275_pairwise_tournament.summary.json`
  reaches `174/176` overall, with `142/144` argument values and `32/32` tool
  names.
- remaining misses:
  final rounded portfolio residual `0.334` versus `0.333`, and
  refund-policy threshold selection (`20` days before event should be `full`).
- derived-rule sidecar:
  `scripts/apply_derived_rule_selector_sidecar.py` applies auditable
  equal-weight residual, percentage-range midpoint, and refund-policy threshold
  rules after the model tournament. On this heldout gate it applies exactly
  `2` corrections and raises selector accuracy from `174/176` to `176/176`.
- derived selector injection:
  `runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/sampler_schedule_with_derived_pairwise_choices.summary.json`
  consumes `176/176` correct selectors, restricts `341` schedule items, and
  has `0` candidate misses.
- protected generation replay:
  `runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/derived_toolargselector_structguard_ckpt275_generation.summary.json`
  reaches `13/13` valid JSON, `13/13` exact tool sequence, `13/13` exact
  arguments, and `13/13` schema-valid calls.
- final audit:
  `runs/tool_sensitive_block_plans/heldout_seed_multicall_gold_evidence_selector/derived_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json`
  has `0` failed records, `0` mismatches, `0` missing/extra calls, and `0`
  invalid tool blocks.

Interpretation: the evidence path has heldout coverage, and tool-name
selection is no longer the first issue on this slice. Local peer/path/snippet
context removes the broad row/list-local argument failure family, and the
derived-rule sidecar closes the remaining gold-span selector gap. The gold-span
protected replay now passes this clean heldout multi-call slice. The planner
still needs teacher/decomposition targets; the empty request-derived planner is
not strong enough for diverse seed rows.

Qwen3.6 teacher heldout planner/eval, 2026-06-28:

- result note:
  `qwen36_teacher_heldout_multicall_result.md`
- target:
  `data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl`
- serving:
  Qwen3.6-27B NVFP4 no-MTP 8k profile serves locally on the RTX 5090. The
  MTP/NEXTN profile failed this turn because SGLang now counts draft weights in
  the KV-cache memory check. No-MTP used about `24.9 GiB / 32.6 GiB`.
- required/native teacher:
  `runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_required.summary.json`
  reaches `13/13` valid tool JSON, `9/13` exact tool sequence, `6/13` exact
  arguments, `11/13` schema-valid, and `11/13` required-args-present.
- auto/text fallback:
  `runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_auto.summary.json`
  reaches `13/13` valid tool JSON, `8/13` exact tool sequence, `6/13` exact
  arguments, `13/13` schema-valid, and `13/13` required-args-present.

Interpretation: Qwen3.6 is the stronger live planner baseline, but it is not
enough to close the heldout route by itself. It also exposes seed-label
ambiguity: some prompts ask for more actions than the gold contains, and other
rows require a policy for split calls versus array payloads. The next live
planner target should be a teacher/decomposition sidecar with an explicit
action-selection policy, not simply "use teacher output" or "fit the seed gold"
uncritically.

Heldout planner decomposition policy, 2026-06-28:

- result note:
  `qwen35_heldout_planner_decomposition_policy_result.md`
- analyzer:
  `scripts/analyze_planner_decomposition_policy.py`
- policy analysis:
  `runs/planner_decomposition/heldout_seed_multicall_policy_analysis.summary.json`
  splits the `13` heldout rows into `6` clean teacher/gold targets, `3`
  teacher-sequence/value-sidecar rows, `3` gold decomposition targets where the
  teacher undercalls prompt-supported actions, and `1` rejected
  full-request-vs-seed-gold ambiguity.
- materializer:
  `scripts/materialize_planner_policy_targets.py`
- accepted targets:
  `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`
  contains `12` rows; rejected target:
  `heldout_seed_multicall_0001`.
- verification:
  the accepted policy targets score `12/12` valid JSON, `12/12` exact tool
  sequence, `12/12` exact arguments, `12/12` schema valid, and `12/12`
  required args present.

Interpretation: this is the planner-side scaffold for the next live route. It
does not solve raw planning, but it prevents the experiment from silently
optimizing against contradictory labels. The next replay should use the
12-row policy target set for planner-span scheduling and keep the rejected
construction-expense row separate until the action-selection policy is decided.

Heldout policy-target evidence-selector route, 2026-06-28:

- result note:
  `qwen35_heldout_policy_target_evidence_selector_route_result.md`
- target:
  `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`
  with `policy_planner_assistant` as the route/span source.
- selector examples:
  `data/candidate_ranking/heldout_seed_policy_evidence_selector_toolname_argument_ranking_evidence.jsonl`
  contains `152` examples: `123` argument values, `29` tool names, and `0`
  missing selector targets.
- raw selector tournament:
  `runs/candidate_ranking/heldout_seed_policy_evidence_selector_ckpt275_pairwise_tournament.summary.json`
  reaches `150/152` overall, `121/123` argument values, and `29/29` tool
  names.
- derived sidecar:
  `runs/candidate_ranking/heldout_seed_policy_evidence_selector_derived_sidecar.summary.json`
  applies the known equal-weight residual and refund-policy threshold rules,
  raising the selector to `152/152`.
- injected schedule:
  `runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/sampler_schedule_with_derived_pairwise_choices.summary.json`
  consumes `152/152` correct selectors, restricts `302` schedule items, and
  has `0` candidate misses.
- protected generation:
  `runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_toolargselector_structguard_ckpt275_generation.summary.json`
  reaches `12/12` valid JSON, `12/12` exact tool sequence, `12/12` exact
  arguments, `12/12` schema-valid calls, and `12/12` required args present.
- final audit:
  `runs/tool_sensitive_block_plans/heldout_seed_policy_evidence_selector/derived_toolargselector_structguard_ckpt275_candidate_miss_audit.summary.json`
  has `0` failed records, `0` mismatches, `0` missing/extra calls, and `0`
  invalid tool blocks.

Interpretation: the protected route now passes end to end on the accepted
heldout planner-policy rows. This is the clean oracle ceiling for the next
model-side experiment: distill planner-policy, selector, and boundary behavior
until raw or constrained-decoder metrics move, while keeping this protected
route as a debugging target rather than counting it as model promotion.

Heldout policy derived-pairwise diagnostic, 2026-06-28:

- result note:
  `qwen35_heldout_policy_derived_pairwise_diagnostic_result.md`
- builder updates:
  `scripts/build_synthetic_multicall_planner_distill_curriculum.py` now accepts
  `--planner-text-field`, `--contains-eval-slice`, and `--diagnostic-only`;
  `scripts/build_candidate_pairwise_curriculum.py` now supports focused filters
  by id, JSON path, and JSON key.
- planner-policy diagnostic corpus:
  `data/qwen35_9b_heldout_policy_planner_distill_diagnostic_curriculum`
  has `15` accepted rows from `12` policy cases, `9` label-rejected
  full/compact candidates, no accepted partial labels, and
  `promotion_allowed=false`.
- focused derived-rule pairwise corpus:
  `data/qwen35_9b_heldout_policy_derived_pairwise_diagnostic_curriculum`
  has `120` rows for only `portfolio[2].weight` and `refund_policy`, with
  `repeat=20`, both A/B orders, no label truncation, and
  `promotion_allowed=false`.
- focused SFT output:
  `runs/fastdllm_qwen35_9b_heldout_policy_derived_pairwise_from_ckpt275_step10_diag`
  trained for `10` steps from checkpoint-275 and saved checkpoints 5/10;
  train loss was `3.6861`.
- selector eval:
  `runs/candidate_ranking/heldout_seed_policy_evidence_selector_derived_pairwise_diag_ckpt10_tournament.summary.json`
  ties checkpoint-275 at `150/152` overall, `121/123` argument values, and
  `29/29` tool names.
- row-level comparison:
  `0/152` selector predictions changed versus checkpoint-275. The two misses
  remain final portfolio residual `0.334` and refund-policy threshold `full`.

Interpretation: the derived-rule decisions did not move even under focused
in-sample pairwise SFT. Do not keep scaling this exact objective. Treat
derived arithmetic and policy-threshold choices as a separate value-reasoning
component: explicit verifier/logit-margin training, a learned value adapter
with numeric/policy features, or a generation-time sidecar scorer are better
next probes than more A/B text SFT.

Heldout policy planner-distill diagnostic, 2026-06-28:

- result note:
  `qwen35_heldout_policy_planner_distill_diagnostic_result.md`
- corpus:
  `data/qwen35_9b_heldout_policy_planner_distill_diagnostic_curriculum`
  has `15` accepted rows from the `12` heldout policy targets, no accepted
  partial labels, and `promotion_allowed=false`.
- training:
  `runs/fastdllm_qwen35_9b_heldout_policy_planner_from_ckpt275_step25_diag`
  trains `25` QLoRA steps from checkpoint-275 with block size `1024`,
  `ARGUMENT_SPAN_LOSS_WEIGHT=1.5`, and `train_loss=2.8638`.
- baseline generation eval:
  `runs/fastdllm_qwen35_9b_heldout_policy_planner_from_ckpt275_step25_diag/ckpt275_baseline_policy_targets_forcedprefix.summary.json`
  reaches raw valid/exact sequence/exact args `0/12` / `0/12` / `0/12`; the
  constrained sequence/args counters are `5/12` / `0/12`.
- checkpoint-25 generation eval:
  `runs/fastdllm_qwen35_9b_heldout_policy_planner_from_ckpt275_step25_diag/checkpoint25_policy_targets_forcedprefix.summary.json`
  reaches raw valid/exact sequence/exact args `1/12` / `0/12` / `0/12`; the
  constrained sequence/args counters are `6/12` / `0/12`.
- row-level shape:
  six records change. Extra and repeated raw calls improve (`7 -> 2` and
  `7 -> 1`), but missing calls slightly worsen (`22 -> 23`) and one
  constrained-exact sequence row regresses.

Interpretation: direct planner-policy SFT produces weak model-side movement,
unlike the focused derived-pairwise SFT, but it does not solve planning or
arguments. Treat it as evidence that planner-policy pressure should be mixed
with retention and selector/value supervision; do not train directly on this
heldout-derived corpus for promotion.

Planner/selector/retention train-only mix, 2026-06-28:

- result note:
  `qwen35_planner_selector_retention_mix_result.md`
- builder:
  `scripts/build_qwen35_planner_selector_retention_mix.py`
- clean corpus:
  `data/qwen35_9b_planner_selector_retention_mix_nooverlap_curriculum`
- composition:
  `377` rows total: `187` route-delta retention, `30` sequence planner, and
  `160` pairwise selector.
- overlap handling:
  first draft had `5` public multi-call exact/user overlaps from retention;
  `--exclude-eval-jsonl` now removes eval prompt/full-output overlaps before
  writing the train JSON.
- clean overlap audit:
  `runs/planner_selector_retention_mix_nooverlap_audit/train_vs_public_synthetic_heldout.json`
  reports `0` exact overlaps and `0` user-prompt overlaps across public
  multi-call, synthetic analogues, heldout seed multi-call, and heldout policy
  target eval files.
- label audit:
  no zero-label or partial-label accepted rows at block size `1536`, left
  truncation.
- one-step gate:
  `runs/fastdllm_qwen35_9b_planner_selector_retention_mix_nooverlap_from_ckpt275_step1_gate`
  trains from checkpoint-275 and saves with `train_loss=1.4096` in `16.23s`.
- short sweep:
  `runs/fastdllm_qwen35_9b_planner_selector_retention_mix_nooverlap_from_ckpt275_step10`
  trains checkpoints `5` and `10` from checkpoint-275 on all `377` rows,
  block size `1536`, with final `train_loss=0.5359` in `156.10s`.
- heldout policy-target eval:
  checkpoint-5 ties checkpoint-275 constrained exact sequence at `5/12` and
  keeps exact arguments at `0/12`; checkpoint-10 regresses to `3/12`
  constrained exact sequence and `0/12` arguments. Neither checkpoint improves
  raw valid JSON, raw exact sequence, or exact arguments.
- raw shape:
  checkpoint-5 reduces extra/repeated raw calls from checkpoint-275's `7/7` to
  `1/1`, but missing calls move from `22` to `23`, so it is not a planning
  solution.

Interpretation: this is a clean and trainable substrate, but the first short
sweep is not a model win. Do not scale this exact balance unchanged. The next
recipe should add stronger explicit tool-sequence/planner pressure and separate
argument-value grounding while keeping the overlap audit and retention gates.

Planner-heavy follow-up substrate, 2026-06-28:

- corpus:
  `data/qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_curriculum`
- balance:
  `357` rows total: `187` retention, `90` sequence planner, and `80` pairwise
  selector.
- overlap audit:
  `runs/plannerheavy_selectorlight_retention_mix_nooverlap_audit/train_vs_public_synthetic_heldout.json`
  reports `0` exact overlaps and `0` user-prompt overlaps against the same
  public/synthetic/heldout eval files.
- label audit:
  no rejected, zero-label, or partial-label rows at block size `1536`.
- one-step gate:
  `runs/fastdllm_qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_from_ckpt275_step1_gate`
  trains from checkpoint-275 and saves with `train_loss=1.3002` in `16.23s`.
- short sweep:
  `runs/fastdllm_qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_from_ckpt275_step10`
  trains checkpoints `5` and `10` with final `train_loss=0.4283` in
  `156.14s`.
- heldout policy-target eval:
  checkpoint-5 reaches constrained exact sequence `4/12` and constrained exact
  arguments `1/12`; checkpoint-10 reaches constrained exact sequence `4/12`
  and exact arguments `0/12`. Raw valid JSON, raw exact sequence, and raw exact
  arguments remain `0/12` for both.
- comparison:
  checkpoint-275 remains better on constrained sequence (`5/12`), but
  planner-heavy checkpoint-5 is the first train-only promotable-branch result
  to move heldout constrained exact arguments above `0/12`.

Interpretation: this is the next clean substrate for testing the data-balance
hypothesis, but the `5/10` sweep is still not a quality result. The signal is
"argument grounding can move," not "the model is better." The next branch needs
sequence anti-regression plus a separate value/argument objective rather than
more planner repetition alone.

Sequence/value/retention mix, 2026-06-28:

- result note:
  `qwen35_sequence_value_retention_mix_result.md`
- builder:
  `scripts/build_qwen35_sequence_value_retention_mix.py`
- corpus:
  `data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum`
- balance:
  `387` rows total: `154` route-delta retention, `173` explicit value-span,
  and `60` sequence planner.
- overlap handling:
  `8` eval-overlap rows were removed before writing training data, all from
  retention.
- independent overlap audit:
  `runs/sequence_value_retention_mix_nooverlap_audit/train_vs_public_synthetic_heldout.json`
  reports `0` exact overlaps and `0` user-prompt overlaps against public
  multi-call, synthetic analogues, heldout seed multi-call, and heldout policy
  target eval files.
- label audit:
  no rejected, zero-label, or partial-label rows at block size `1536`.
- one-step gate:
  `runs/fastdllm_qwen35_9b_sequence_value_retention_mix_nooverlap_from_ckpt275_step1_gate`
  trains from checkpoint-275 and saves with `train_loss=1.2486` in `16.22s`.
- short sweep:
  `runs/fastdllm_qwen35_9b_sequence_value_retention_mix_nooverlap_from_ckpt275_step10`
  trains checkpoints `5` and `10` with final `train_loss=0.5634` in
  `156.12s`.
- heldout policy-target eval:
  checkpoint-5 reaches raw valid JSON `1/12`, constrained exact sequence
  `5/12`, and constrained exact arguments `0/12`; checkpoint-10 reaches raw
  valid JSON `1/12`, constrained exact sequence `4/12`, and exact arguments
  `0/12`.

Interpretation: this is the next prepared branch because it removes pairwise
selector mass while preserving sequence pressure and explicit value grounding.
It is not a model-quality result. Checkpoint-5 is useful because raw-valid JSON
moved without aggregate constrained-sequence regression, but exact arguments
remain stuck. The next step should isolate or compose positive deltas instead
of adding another broad SFT mixture.

Working sampler recipe:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  --full-context-sampling \
  --sampler-schedule-jsonl runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_pairwise_path_choices_v5_12.jsonl \
  --force-schedule-token-kinds json_key,json_structure,tool_tag \
  --force-argument-boundary-target-tokens \
  --ban-argument-boundary-tokens \
  --ban-argument-newline-tokens \
  --force-best-candidate-sequence \
  --force-best-tool-name-sequence \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving \
  --constrained-max-calls 3 \
  ...
```

Do not use `--force-selected-candidate-tokens` for this v5 path. It is unstable
on the public schedule. The stable mechanism is to narrow selected spans to a
singleton `candidate_sequence_values` list and let whole-candidate sequence
forcing handle the denoising interval.

The first candidate-set-only smoke writes
`runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_constrained_smoke1.jsonl`.
It reaches raw `1/1` valid JSON and exact tool sequence but not exact
arguments: `stream_quality` becomes `100pp`, a per-position recombination of
allowed candidate tokens.

The selected-candidate smoke writes
`runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_selected_smoke1.jsonl`.
It adds `--force-selected-candidate-tokens` for the two spans where the
extractor has a concrete selected value, and reaches raw `1/1` valid JSON,
exact tool sequence, exact arguments, and schema validity. This is still
protected/runtime behavior, not learned model behavior. The next non-oracle
sampler upgrade should enforce whole-candidate sequence consistency instead of
independent per-position token masks.

Whole-candidate sequence consistency is now implemented behind
`--force-best-candidate-sequence`. It scores complete candidate strings for an
argument span and forces one sequence, instead of letting per-position masks
recombine candidates. The sequence-only smoke writes
`runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_sequence_smoke1.jsonl`:
raw valid JSON `1/1`, exact sequence `1/1`, schema valid `1/1`, but exact
arguments `0/1` because the model chooses
`end_time=2023-04-22T15:00:00Z` instead of `17:00:00Z`. This is the current
model-ranking gap.

The protected selector variant adds a paired start/end datetime extractor and
uses `--force-selected-candidate-tokens` before model-ranked sequence choice.
It writes
`runs/tool_sensitive_block_plans/public_multicall_schedule_candidate_sequence_selected_smoke1.jsonl`
and reaches raw valid JSON `1/1`, exact sequence `1/1`, exact arguments `1/1`,
and schema validity `1/1`. Treat this as the runtime scaffold target, not as
proof that the LoRA adapter learned semantic candidate ranking.

Tool-name candidate sequence constraints are now implemented behind
`--force-best-tool-name-sequence`. The augmented schedule
`runs/tool_sensitive_block_plans/public_multicall_live_sequence_planned_sampler_schedule_with_toolname_candidates_2.jsonl`
adds available-tool candidate sequences to `tool_name` rows. Because the current
sampler has fixed token spans, only length-compatible tool names are legal; the
Qwen token block also absorbs the `","` suffix after the function name.

The non-oracle tool-name smoke writes
`runs/tool_sensitive_block_plans/public_multicall_schedule_toolname_candidate_sequence_selected_smoke2.jsonl`.
It removes `tool_name` from `--force-schedule-token-kinds` and uses model-ranked
available-tool sequences instead. Result: raw valid JSON `1/1`, schema valid
`1/1`, but exact sequence `0/1`; the model repeats
`get_camera_live_feed` for the third call instead of choosing
`get_recorded_feed`. This isolates function-candidate ranking as the next
learned behavior target.

The first concrete candidate-ranking artifact is now built from the 12-case
public multi-call gold slice:

- examples:
  `data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.jsonl`
- conversation-format train file:
  `data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.train.json`
- source schedule:
  `runs/tool_sensitive_block_plans/public_multicall_gold_sampler_schedule_with_toolname_candidates_v3_12.jsonl`
- coverage: `86/86` usable examples, with `31` tool-name examples and `55`
  argument-value examples
- target missing from candidate set: `0`

This is intentionally a small executable target, not a final training corpus.
It makes the next objective precise: train/evaluate a candidate-ranker or
adapter loss over `(prompt, call index, schema/key, candidate set) -> correct
tool/value index`, then use that signal to reduce raw model-ranked misses like
`get_camera_live_feed` vs `get_recorded_feed` and `15:00` vs `17:00`.

Masked-span ranker baseline for the current checkpoint-275 adapter:

- evaluator:
  `scripts/eval_fastdllm_candidate_ranking.py`
- output:
  `runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12.jsonl`
- overall accuracy: `80/86` (`93.0%`)
- multi-candidate accuracy: `52/58` (`89.7%`)
- tool-name accuracy: `31/31`
- argument-value accuracy: `49/55`

Interpretation: when the surrounding gold assistant trace is fixed and only the
candidate span is masked, this adapter already ranks all tool names correctly.
The earlier full-generation repeated-tool failure was traced to sampler
commitment timing: the sampler chose a whole tool candidate on the first
one-token chunk even when several compatible candidates shared that prefix.
After deferring commitment until the candidate set is unambiguous, the one-case
tool-name candidate smoke reaches raw valid JSON `1/1`, exact sequence `1/1`,
and exact arguments `1/1`; artifact:
`runs/tool_sensitive_block_plans/public_multicall_schedule_toolname_candidate_sequence_selected_deferred_smoke1.jsonl`.

The same baseline was rerun with stricter context modes:

- `full_gold`: `80/86`
- `prefix_only`: `80/86`
- `future_masked`: `80/86`

All three modes preserve `31/31` tool-name accuracy and fail on the same six
argument-value spans. This rules out future-token leakage as the reason for the
masked-span score and keeps the next learned-model pressure on row/time
argument-value alignment.

Diffusion-init comparison, 2026-06-27:

- evaluator update: `scripts/eval_fastdllm_candidate_ranking.py` supports
  `--no-adapter`, so masked candidate ranking can be measured on the converted
  diffusion base without a PEFT adapter.
- diffusion-init prefix-only baseline:
  `runs/candidate_ranking/public_multicall_qwen35_diffusion_init_masked_span_rank_v3_12.summary.json`
- delta report:
  `qwen35_candidate_ranking_delta_result.md`
- diffusion init: `78/86` overall, `31/31` tool names, `47/55` argument values
- checkpoint-275: `80/86` overall, `31/31` tool names, `49/55` argument values
- row-level delta: checkpoint-275 improves `3` argument spans, regresses `1`,
  and leaves `6` argument spans failing

Interpretation: checkpoint-275 gives a small real model-side lift on
candidate/value ranking, but the remaining gap is still row/time alignment.
This argues for a targeted value-ranking or row-grounding objective rather than
more generic tool-call syntax replay.

Candidate-ranker diagnostic curriculum, 2026-06-27:

- builder: `scripts/build_candidate_ranking_curriculum.py`
- curriculum:
  `data/qwen35_9b_candidate_ranker_public12_diagnostic_curriculum`
- result note:
  `qwen35_candidate_ranker_diagnostic_curriculum_result.md`
- output checkpoint:
  `runs/fastdllm_qwen35_9b_candidate_ranker_public12_diag_from_ckpt275_step1_gate`
- one-step gate from checkpoint-275 trains and saves successfully with loss
  `1.0847731828689575`
- masked candidate-ranking eval is unchanged from checkpoint-275: `80/86`
  overall, `31/31` tool names, `49/55` argument values, with `0` improved and
  `0` regressed rows

Interpretation: an index-selection conversation SFT objective is trainable, but
one step does not transfer to the masked-span value-ranking score. Treat this
as a diagnostic sidecar path. The next generator-side objective should be
closer to the actual failure surface: masked selected-value span CE,
row/table-grounding labels, or a separately evaluated ranker/verifier head.

Public-train candidate-ranking artifact, 2026-06-27:

- result note: `qwen35_public_train_candidate_ranking_result.md`
- materializer: `scripts/materialize_conversation_toolcall_cases.py`
- train cases:
  `data/toolcall_eval/public_train_multicall_gold_cases.jsonl`
- candidate-ranking examples:
  `data/candidate_ranking/public_train_multicall_toolname_argument_ranking.jsonl`
- non-eval curriculum:
  `data/qwen35_9b_candidate_ranker_public_train_curriculum`
- coverage: `56` train multi-call records, `155` tool calls, `338` ranking
  examples, `299` usable examples
- checkpoint-275 train-slice masked rank: `295/299` overall, `155/155` tool
  names, `140/144` argument values
- one-step train-slice candidate-ranker continuation saves successfully with
  loss `1.3103340864181519`
- heldout public-12 masked candidate ranking is unchanged from checkpoint-275:
  `80/86` overall, `31/31` tool names, `49/55` argument values

Interpretation: the non-eval ranker/data pipeline is now available for longer
or sidecar experiments, but the one-step index-selection recipe is neutral on
heldout. Do not promote it yet. Promote only after heldout public-12 argument
ranking moves above `49/55` without regressing tool names, and then confirm
with the protected/raw tool-call scorecard.

Public-train value-span candidate objective, 2026-06-27:

- result note: `qwen35_public_train_candidate_value_span_result.md`
- builder update: `scripts/build_candidate_ranking_curriculum.py` now supports
  `--answer-mode target_text` and `--include-kinds`
- curriculum:
  `data/qwen35_9b_candidate_value_span_public_train_curriculum`
- coverage: `173` accepted argument-value rows, `0` rejected labels, p50 kept
  assistant labels `8`, p90 `11`
- one-step continuation from checkpoint-275 saves successfully with loss
  `3.019835948944092`
- heldout public-12 masked candidate ranking is unchanged from checkpoint-275:
  `80/86` overall, `31/31` tool names, `49/55` argument values

Interpretation: selected-value span SFT is more aligned than index SFT, but one
step is still neutral on heldout. The next escalation should be either a short
multi-step value-span sweep with heldout gates, or explicit row/table grounding
examples for the recurring time/ID row-alignment failures.

Short value-span sweep, 2026-06-27:

- run:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10`
- result note: `qwen35_public_train_candidate_value_span_result.md`
- checkpoint-5 heldout public-12 candidate ranking: `81/86` overall, `31/31`
  tool names, `50/55` argument values
- checkpoint-10 heldout public-12 candidate ranking: `80/86` overall, `31/31`
  tool names, `49/55` argument values
- checkpoint-5 improves one public-12 invoice-ID row and regresses none
- checkpoint-5 cheap public one-call generation gate keeps raw `3/8` sequence
  and `2/8` arguments while improving first-pass constrained arguments from
  checkpoint-275's `5/8` to `8/8`

Interpretation: checkpoint-5 is the first value-span continuation that improves
a heldout model-side metric without a cheap one-call regression. It is not a
full promotion yet. The next gate is a broader split-route/tool-call scorecard
or a focused public multi-call generation check before replacing checkpoint-275
anywhere.

Focused public multi-call checkpoint-5 gate, 2026-06-27:

- first-pass artifact:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_nomodelrepair.summary.json`
- projection artifacts:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_contextual_projection.summary.json`
  and
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/public_multicall_12_sequence_planner_projection.summary.json`
- raw remains weak: `1/12` sequence and `0/12` arguments, matching the active
  checkpoint-275 raw public multi-call line
- direct constrained improves from active `7/12` sequence and `4/12` arguments
  to `8/12` and `5/12`
- contextual projection improves from active `7/12` sequence and `7/12`
  arguments to `8/12` and `8/12`
- guarded sequence-planner projection ties the active protected route at
  `11/12` sequence and `10/12` arguments
- row-level delta: direct constrained and contextual projection improve the
  public invoice row `adc48a37-6341-4ea6-972a-8ec2b5421321`; the guarded
  planner has no row-level improvements or regressions versus the active route

The scalar contextual projection script now also cleans malformed leading/trailing
quote artifacts only when the cleaned scalar appears in prompt context, while
preferring ID-specific context selection for ID-like fields. This fixed a camera
row where the first two calls needed `"front_door` -> `front_door`, but the
third call still needed `front_garden`.

Interpretation: checkpoint-5 is no longer just a one-call positive. It is a
public multi-call sidegrade/early improvement: better at the constrained and
contextual stages, tied after the guarded planner, but still not promoted as the
main route because raw generation did not move.

Remaining split-route lanes for checkpoint-5, 2026-06-27:

- row-level delta report:
  `qwen35_public_train_candidate_value_span_route_delta.md`
- machine-readable delta:
  `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/route_delta_vs_current_routed_target.json`
- teacher-train one-call:
  raw `2/12` sequence and `2/12` arguments, constrained `10/12` sequence and
  `6/12` arguments
- teacher-heldout one-call:
  raw `1/8` sequence and `0/8` arguments, constrained `7/8` sequence and
  `5/8` arguments
- synthetic text tool-result:
  raw `5/10` sequence and `3/10` arguments, constrained `10/10` sequence and
  `9/10` arguments
- OpenAI-style tool-result after raw-assistant projection:
  raw `7/10` sequence and `7/10` arguments, constrained `10/10` sequence and
  `8/10` arguments

Decision: do not promote checkpoint-5 into the split-route target. It is weaker
than staged checkpoint-24 on the public/teacher one-call generator role, weaker
than the current routed teacher-heldout protected gate, and weaker than
checkpoint-275 on OpenAI-style protected exact arguments. Use its positive
public multi-call constrained/contextual deltas as training signal, not as a
route replacement.

Route-delta train-only mix, 2026-06-27:

- result note: `qwen35_route_delta_trainonly_mix_result.md`
- builder: `scripts/build_qwen35_route_delta_trainonly_mix.py`
- curriculum:
  `data/qwen35_9b_route_delta_trainonly_mix_curriculum`
- rows: `335` accepted, `0` rejected
- provenance: `contains_eval_slice=false`, `promotion_allowed=true`
- sources: `173` public-train value-span rows, `64` Fast-DLLM train tool-call
  rows, `48` synthetic one-call train rows, `20` synthetic text tool-result
  rows, and `30` synthetic OpenAI-style tool-result train rows
- one-step gate from checkpoint-275:
  `runs/fastdllm_qwen35_9b_route_delta_trainonly_mix_from_ckpt275_step1_gate`
- gate result: train loss `0.2080969214439392`, adapter saved

Interpretation: this mix is trainable and provenance-clean, but the one-step
adapter is not promoted. The next sweep should stop early unless teacher-heldout
one-call recovers `8/8` / `6/8` protected exactness and OpenAI-style tool-result
recovers `10/10` / `9/10`.

Route-delta follow-up, 2026-06-28:

- A context-first constrained decoder fix for explicit tool-result evidence now
  maps conservative aliases such as `email_subject -> subject`,
  `email_body -> body`, `callback_date -> date`, and `callback_time -> time`.
  It is gated on tool-result evidence so ordinary one-call prompts keep the
  previous decoder behavior.
- With that patch, the one-step route-delta adapter recovers the earlier cheap
  blocker lanes: public one-call `8/8` / `8/8`, teacher-heldout `8/8` / `6/8`,
  synthetic text tool-result `10/10` / `10/10`, and OpenAI-style tool-result
  `10/10` / `9/10`.
- Public multi-call still blocks promotion. Direct constrained remains
  `7/12` / `4/12`, contextual projection is `7/12` / `7/12`, and only the
  guarded planner reaches the active protected target of `11/12` / `10/12`.
- Decision: keep the decoder fix as protected-path infrastructure, but do not
  promote the adapter or launch a longer sweep of the same mix as the default.
  The next learned target should be narrower planner/value selection for the
  public multi-call failures, while keeping tool-result context-copy replay as
  an anti-regression lane.

Public-train overlap audit, 2026-06-28:

- `qwen35_public_eval_overlap_audit_result.md` shows that
  `data/fastdllm_toolcall_train/train_toolcall.json` contains `11/12` public
  multi-call smoke rows verbatim.
- The two current public multi-call planner failure families, voice-command
  security cameras and motion-detector installation code, only appear locally as
  exact public eval rows.
- Added `scripts/audit_toolcall_eval_overlap.py` and
  `scripts/filter_toolcall_eval_overlap.py`.
- Added filtered source:
  `data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json`
  with `85/96` rows kept and `0` exact/user overlaps against public multi-call
  smoke.
- `scripts/build_toolcall_sequence_planner_distill_curriculum.py` now supports
  `--exclude-eval-jsonl`; the clean compact/instruction planner curriculum has
  `45` remaining multi-call records, `22/45` planner exact sequence, `1/45`
  planner exact arguments, and `15` accepted rows after token-label filtering.
- Future public-train-derived planner/value curricula should include an overlap
  audit artifact before being treated as promotion-eligible evidence.

### Current 9B Pilot Bridge

The Qwen3.5-9B pilot is the local execution bridge before spending the compute
on the full Qwen3.6-27B target. The current best result is not a single
promoted adapter; it is a split-route sidecar target:

- scorecard:
  `qwen35_9b_split_route_sidecar_scorecard.md`
- executable gates:
  `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.json`
- implementation manifest:
  `runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json`

Run:

```bash
.venv-fastdllm/bin/python scripts/write_qwen35_split_route_sidecar_scorecard.py --check
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --check-only --strict-replayable
```

The manifest records the base model, staged checkpoint-24 generator adapter,
active checkpoint-275 protection adapter, per-slice input cases, routed eval
summaries, and post-processing chains. It is the concrete handoff for the first
runtime router/sidecar implementation described in Phase 3: load the shared
base model, select the adapter role by prompt/eval lane, and apply the recorded
protection chain for multi-call and OpenAI-style tool-result lanes. This is a
serving/harness bridge, not yet the final Qwen3.6 diffusion closeout.

To regenerate the six routed lanes, emit the replay shell plan:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --strict-replayable
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py --verify-outputs --plan-json runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json
```

This writes:

- `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.json`
- `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/route_runner_plan.sh`

The current plan has `6` routes, `10` replayable steps, and `0` unknown
post-processing steps. Generation commands are wrapped in `systemd-run --user`
with memory limits.

The verifier reads the final summary for each route in the plan and fails
nonzero if any route gate regresses or a planned summary is missing. A
historical-output verification plan is kept at
`runs/qwen35_9b_split_route_sidecar_eval/historical_verify_plan/route_runner_plan_verification.json`
and currently passes all six route gates without launching model generation.

For controlled live replay, restrict the runner by slice:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --strict-replayable \
  --slice public_one_call \
  --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall \
  --execute
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --verify-outputs \
  --plan-json runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/route_runner_plan.json
```

The current live public-onecall smoke passes: raw `4/8` sequence, raw `3/8`
arguments, and protected `8/8` / `8/8`.

The current live OpenAI-style tool-result smoke also passes, exercising the
active checkpoint-275 protection route:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --strict-replayable \
  --slice openai_style_tool_result \
  --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult \
  --execute
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --verify-outputs \
  --plan-json runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/route_runner_plan.json
```

It reproduces raw `6/10` sequence and arguments, and protected `10/10`
sequence / `9/10` arguments.

The current live public multi-call planner smoke passes as well, exercising the
full protection chain:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --strict-replayable \
  --slice public_multi_call_planner \
  --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner \
  --execute
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --verify-outputs \
  --plan-json runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/route_runner_plan.json
```

It runs generation, sequence-preserving rescore, contextual projection, and
sequence-planner projection, then verifies protected `11/12` sequence and
`10/12` arguments.

The staged checkpoint-24 text tool-result route is also live-verified:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --strict-replayable \
  --slice synthetic_text_tool_result \
  --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult \
  --execute
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --verify-outputs \
  --plan-json runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/route_runner_plan.json
```

It verifies raw `6/10` sequence, raw `4/10` arguments, and protected `10/10`
sequence / `9/10` arguments.

The teacher one-call staged checkpoint-24 lanes are also live-verified:

```bash
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --strict-replayable \
  --slice teacher_train_one_call \
  --slice teacher_heldout_one_call \
  --out-root runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall \
  --execute
.venv-fastdllm/bin/python scripts/run_qwen35_split_route_sidecar_manifest.py \
  --verify-outputs \
  --plan-json runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/route_runner_plan.json
```

Teacher-train verifies protected `11/12` sequence / `6/12` arguments;
teacher-heldout verifies raw `2/8` sequence, raw `1/8` arguments, and protected
`8/8` sequence / `6/8` arguments. With this, all six split-route lanes have
live replay evidence.

## Knob cheat-sheet

| Knob | Start | Note |
|---|---|---|
| diffusion `bd_size` B | warmup 8/16/32 → larger | actual denoising block target; use `TRAIN_BD_SIZE` or `TRAIN_BD_SIZE_CHOICES` |
| data `BLOCK_SIZE` | 896/1536 in local 9B runs | max token window/chunk length; not the same as diffusion `bd_size` |
| denoising steps K | sweep | fewer = faster, lower quality |
| confidence threshold τ | ~0.9 | higher = more passes, safer; always unmask top-1 |
| learning rate | ~1e-5, cosine | low; critical to preserve AR knowledge |
| fine-tune tokens | ~1B | more for quality |
| precision | bf16 (+fp32 master) | fp8 optional |
| GDN mode | Option A | switch to B if within-block under-mixes |

Local 2026-06-28 note: Qwen3.5-9B now has fixed and dynamic `bd_size`
training controls. See `qwen35_blockdiffusion_target_ablation_result.md`.
The first useful heldout signal is fixed `bd_size=16`: it improves
policy-target constrained exact sequence to `6/12`, while dynamic `8,16,32`
stays at `5/12`. Direct arg/value pressure creates the first raw exact
argument hit but destabilizes call order. A lower-pressure arg/value dose curve
reduces extra/repeated-call damage but still leaves exact arguments at `0/12`
and drops constrained sequence to `3/12` by checkpoint-10. LoRA delta blending
collapses the sequence gain; scalar repair sidecar transfer improves
schema/required-arg validity but leaves exact arguments at `0/12`. Treat this
as evidence for tool-sensitive skeleton/infill plus grammar-completable
generation, not a reason to keep scaling uniform scalar repair or uniform
argument-span masking.

Tool-call JSON completability is now measured directly with
`scripts/diagnose_toolcall_json_completability.py`. On the same 12-row heldout
policy-target slice, raw fixed `bd_size=16`, dynamic `8,16,32`, and
low-pressure arg/value outputs all contain unrecoverable JSON-prefix errors in
every row before repair/projection. Projection and scalar sidecars can make the
strings syntactically valid, but they still leave exact arguments at `0/12`.
The next runbook step should therefore be generation-time grammar-completable
tool-call commits, with tiny blocks around sentinels, JSON keys, and scalar
values, before broader curriculum scaling. Detailed result:
`qwen35_toolcall_json_completability_diagnostic_result.md`.

The first opt-in generation-time guard is now live in
`scripts/eval_fastdllm_toolcall_cases.py` as `--guard-tool-json-prefix`. It
checks scheduled JSON/tool-call intervals before commit and keeps those commits
left-to-right. A one-row public multi-call tool-tag-only smoke shows the guard
moving raw output from invalid JSON and `0/1` exact sequence to valid JSON and
`1/1` exact sequence; exact arguments remain `0/1`, so this is a grammar/route
primitive, not value grounding. Detailed result:
`qwen35_toolcall_json_prefix_guard_smoke_result.md`.

The mode/sentinel companion is also live as `--guard-tool-call-mode`. It
hard-fills only scheduled `tool_tag` target tokens and records separate
mode-force counters, preventing prose/thinking from bypassing the active JSON
checker. On the same one-row smoke, mode + prefix protection produces `3/3`
complete raw JSON tool-call segments and raw exact sequence `1/1`; exact
arguments remain `0/1`. Detailed result:
`qwen35_toolcall_mode_guard_smoke_result.md`.

Operational next step:

```bash
# After building or selecting a token-sensitive schedule:
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  ... \
  --full-context-sampling \
  --sampler-schedule-jsonl <schedule.jsonl> \
  --guard-tool-call-mode \
  --guard-tool-json-prefix \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving
```

Then add schema-aware key/value masks and value-candidate infill before the
next 12-case public or heldout scorecard.

The first named value/name candidate guards are now live:
`--guard-tool-value-candidates` and `--guard-tool-name-candidates`. Public
multi-call 12 with mode + JSON-prefix + value candidates reaches raw valid JSON
`12/12`, raw exact sequence `11/12`, and raw exact arguments `11/12`. Adding
tool-name candidates fixes the remaining tool-name set, but exposes a new
mechanical bug: closing `</tool_call>` can be forced while the active JSON body
is incomplete-but-completable. Detailed result:
`qwen35_tool_value_name_guard_scorecard_result.md`.

The close-tag completeness guard is now part of `--guard-tool-call-mode`.
Scheduled tool-call sentinel forcing is deferred when an active `<tool_call>`
body has started JSON but that JSON body is not yet complete. Re-running public
multi-call 12 with mode + JSON-prefix + name + value + close protection reaches
raw valid JSON `12/12`, exact tool sequence `12/12`, and exact arguments
`11/12`; the completability diagnostic reports `31/31` raw complete JSON
segments and zero invalid segments. The only remaining raw miss is a
benchmark-exact value-grounding mismatch (`location: ""` vs `"home"`), not a
tool route or JSON closure failure.

Updated command shape for the next scorecard:

```bash
.venv-fastdllm/bin/python scripts/eval_fastdllm_toolcall_cases.py \
  ... \
  --full-context-sampling \
  --sampler-schedule-jsonl <schedule.jsonl> \
  --guard-tool-call-mode \
  --guard-tool-json-prefix \
  --guard-tool-name-candidates \
  --guard-tool-value-candidates \
  --stop-after-schedule-tool-calls \
  --constrained-tool-decoding \
  --constrained-sequence-preserving
```

Use this stack for the next heldout policy-target scheduled route and six-lane
split-route scorecard. The next training branch should target evidence-grounded
value infill and on-policy AR-teacher correction rather than another structural
close-boundary fix.

Heldout policy-target follow-up:

- result note: `qwen35_heldout_policy_close_guard_scorecard_result.md`
- lean named stack on
  `runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl`:
  raw valid JSON `11/12`, exact sequence `11/12`, exact arguments `11/12`
- miss: `heldout_seed_multicall_0004`, nested `create_campaign` JSON skeleton
  drift inside `campaign_details`
- target fallback ablation: still `3/4` on the first four rows; it fires zero
  times because the prefix is already unrecoverably broken
- structural ceiling:
  add `--force-schedule-token-kinds json_key,json_structure` while keeping
  named mode/name/value guards; heldout reaches raw valid JSON `12/12`, exact
  sequence `12/12`, exact arguments `12/12`, and `29/29` complete raw JSON
  segments

Use the structural ceiling as a debugging oracle. Do not count it as raw model
promotion. The model-side route should distill this into skeleton stability,
dynamic block boundaries, or a learned skeleton/value adapter.

Skeleton-conditioned value-infill artifact builder:

```bash
.venv-fastdllm/bin/python scripts/build_skeleton_value_infill_artifacts.py \
  --schedule-jsonl <candidate-augmented-schedule.jsonl> \
  --cases-jsonl <cases.jsonl> \
  --out-dir <out-dir> \
  --provenance-label <label> \
  [--promotion-allowed]
```

Current outputs:

```text
qwen35_skeleton_value_infill_artifacts_result.md
data/skeleton_value_infill/public_train_no_public_smoke/
data/skeleton_value_infill/heldout_policy_diagnostic/
```

The trainable clean set uses
`data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json`,
materialized as
`data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl`, then
planned into:

```text
runs/tool_sensitive_block_plans/public_train_no_public_smoke_blocks_tokenized_with_ids.jsonl
runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_ids.jsonl
runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_candidates_targetincluded.jsonl
```

It has `45` clean records, `331` usable value slots, `711` candidate rows,
`4667` boundary labels, and `331` value-infill train instances. Provenance
audit against public and heldout eval slices reports `0` exact overlaps and
`0` user overlaps. The older `public_train_multicall_gold_cases.jsonl` should
not be used for this target because it overlaps `11/12` public multi-call eval
rows.

First skeleton value-infill QLoRA gate, 2026-06-28:

- result note: `qwen35_skeleton_value_infill_training_gate_result.md`
- staging dataset:
  `data/qwen35_9b_skeleton_value_infill_no_public_smoke_curriculum/`
- one-step fit gate:
  `runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step1_gate`
- 75-step sweep:
  `runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75`
- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- saved checkpoint adapters: `checkpoint-25`, `checkpoint-50`,
  `checkpoint-75`
- final train loss: `2.6479`
- status: trainable on the local RTX 5090 without OOM; not promotion evidence
  until checkpoint gates are scored.

Checkpoint eval result:

- result note: `qwen35_skeleton_value_infill_checkpoint_eval_result.md`
- public closeguard eval:
  `runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_public_closeguard_eval`
- heldout lean closeguard eval:
  `runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_heldout_lean_closeguard_eval`
- one-call sweep:
  `runs/fastdllm_qwen35_9b_skeleton_value_infill_from_ckpt275_step75_onecall_checkpoint_sweep_eval96_modelrepair_max1`
- status: do not promote. All three checkpoints tie active checkpoint-275 on
  public and heldout multi-call guard gates; checkpoint-25 has only a small
  one-call raw/model-repair lift.

Schedule-state selector curriculum, 2026-06-28:

- result note: `qwen35_schedule_state_selector_curriculum_gate_result.md`
- builder: `scripts/build_schedule_state_selector_curriculum.py`
- dataset:
  `data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum/`
- objective: emit a compact JSON schedule decision with `candidate_index`,
  `span_kind`, `protection`, `block_size`, `denoise_steps`,
  `force_candidate_sequence`, `require_json_prefix_safe`, and
  `close_tool_call_only_when_json_complete`
- accepted train instances: `539`
- rejected train instances: `0`
- one-step gate:
  `runs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step1_gate`
- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- train loss: `4.8791`
- status: trainable on the local RTX 5090; next step is a short checkpoint
  sweep plus selector-decision evaluator before wiring it into sampler gates.

Schedule-state selector free-generation sweep, 2026-06-28:

- evaluator: `scripts/eval_fastdllm_schedule_state_selector.py`
- training output:
  `runs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step75`
- training log:
  `logs/fastdllm_qwen35_9b_schedule_state_selector_no_public_smoke_from_ckpt275_step75.log`
- final train loss: `4.5926`
- runtime: `745.59s`
- saved checkpoints: `25`, `50`, `75`
- eval output:
  `runs/schedule_state_selector/no_public_smoke_step75_selector_sweep_eval16/`
- eval log:
  `logs/fastdllm_qwen35_9b_schedule_state_selector_sweep_eval16.log`
- eval slice: first `16` ambiguous selector examples
- result:
  - active checkpoint-275: valid JSON `0/16`, candidate index `3/16`,
    exact decision `0/16`
  - selector checkpoint-25: valid JSON `0/16`, candidate index `1/16`,
    exact decision `0/16`
  - selector checkpoint-50: valid JSON `0/16`, candidate index `3/16`,
    exact decision `0/16`
  - selector checkpoint-75: valid JSON `0/16`, candidate index `2/16`,
    exact decision `0/16`
- status: do not promote free-generation selector JSON. Keep the
  schedule-state data, but move the next experiment to constrained JSON
  prefix-forcing or masked/pairwise candidate scoring.

Constrained schedule-state candidate-index ranking, 2026-06-28:

- evaluator: `scripts/eval_fastdllm_schedule_state_selector_ranking.py`
- mode: `score_mode=index_only`
- method: force-prefix `{"candidate_index":`, score candidate index values
  `0..N-1` by masked likelihood, then inject the fixed protection-policy JSON
  suffix into the sampler control state.
- 64-row checkpoint sweep output:
  `runs/schedule_state_selector/no_public_smoke_step75_selector_indexonly_rank64/`
- 64-row sweep log:
  `logs/fastdllm_qwen35_9b_schedule_state_selector_indexonly_rank64.log`
- result on first `64` ambiguous rows:
  - active checkpoint-275: `59/64` top-1, `63/64` target top-2
  - selector checkpoint-25: `59/64` top-1, `63/64` target top-2
  - selector checkpoint-50: `59/64` top-1, `63/64` target top-2
  - selector checkpoint-75: `59/64` top-1, `63/64` target top-2
- full active-checkpoint ambiguous output:
  `runs/schedule_state_selector/no_public_smoke_ckpt275_indexonly_rank_all_ambiguous.jsonl`
- full active-checkpoint ambiguous summary:
  `runs/schedule_state_selector/no_public_smoke_ckpt275_indexonly_rank_all_ambiguous.summary.json`
- full active-checkpoint result: `312/349` top-1 (`89.40%`), `334/349`
  target top-2 (`95.70%`), `0` runtime errors, `18.43 GiB` max allocated GPU
  memory.
- full-template sanity check:
  `runs/schedule_state_selector/no_public_smoke_ckpt275_fulldecision_rank16.jsonl`
  scores only `5/16`; do not score whole JSON from an empty continuation.
- status: promote the constrained scorer/injector pattern, not the selector
  SFT checkpoints. Next gate is to inject scored `candidate_index` choices into
  the protected sampler and evaluate public/heldout tool-call generation with
  top-1 and top-2 fallback reported separately.

Schedule-state selector ranking injection smoke, 2026-06-28:

- bridge: `scripts/inject_schedule_state_selector_ranking_choices.py`
- injected schedules:
  `runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank1.jsonl`
  and
  `runs/tool_sensitive_block_plans/public_train_no_public_smoke_sampler_schedule_with_schedule_state_selector_rank2.jsonl`
- injection summary: `163` argument schedule items restricted across `30`
  records; `0` candidate-order mismatches in the preflight audit.
- four-case smoke case file:
  `data/toolcall_eval/public_train_multicall_no_public_smoke_cases_selector_rank_smoke4.jsonl`
- rank-1 generation:
  `runs/tool_sensitive_block_plans/public_train_no_public_smoke_schedule_state_selector_rank1_smoke4_generation.summary.json`
- rank-2 generation:
  `runs/tool_sensitive_block_plans/public_train_no_public_smoke_schedule_state_selector_rank2_smoke4_generation.summary.json`
- guard stack: full-context scheduled sampler, `--guard-tool-value-candidates`,
  `--guard-tool-call-mode`, `--guard-tool-json-prefix`,
  `--stop-after-schedule-tool-calls`, sequence-preserving constrained
  projection for reporting.
- result:
  - rank-1: raw valid JSON `4/4`, exact sequence `4/4`, exact arguments
    `3/4`; miss is `search_deals.limit` predicted `1` vs gold `5`.
  - rank-2: raw valid JSON `4/4`, exact sequence `4/4`, exact arguments
    `2/4`; additional miss is `campaign_end_date` predicted `2023-06-01`
    vs gold `2023-08-31`.
- status: bridge works. Rank-1 hard restriction is the current default;
  rank-2 fallback is not automatically beneficial and needs a separate
  rerank/repair policy.

## What is novel vs the template

Everything except two things is standard Fast-dLLM v2: (1) GDN handling — as cross-block state
carrier (A) or bidirectional within-block scan (B), and the GDN state snapshot in the cache; and
(2) scale (27B vs the template's 1.5B/7B). The rest — block-causal mask, masked-diffusion objective,
block-size warmup, confidence-threshold decode, hierarchical cache — is the template.

---

## References

**Template & method**
- [R1] Fast-dLLM v2: Efficient Block-Diffusion LLM. arXiv:2509.26328 — https://arxiv.org/abs/2509.26328
- [R2] NVlabs/Fast-dLLM (code, v1 + v2; checkpoint `Efficient-Large-Model/Fast_dLLM_v2_1.5B`) —
  https://github.com/NVlabs/Fast-dLLM

**AR → diffusion adaptation**
- [R3] Dream 7B: Diffusion LLMs (AR-init from Qwen2.5, shifted prediction, noise rescheduling, LR
  sensitivity). arXiv:2508.15487 — https://arxiv.org/abs/2508.15487 ; blog: https://hkunlp.github.io/blog/2025/dream/
- [R4] Block Diffusion: Interpolating Between Autoregressive and Diffusion LMs (Arriola et al.).
  arXiv:2503.09573 — https://arxiv.org/abs/2503.09573
- [R5] Autoregressive-to-Diffusion VLMs / A2D-VL (progressive prediction-window curriculum) —
  https://runwayml.com/research/autoregressive-to-diffusion-vlms
- [R6] Stop Training for the Worst: Progressive Unmasking (PUMA; block-size warmup, AR-init).
  arXiv:2602.10314 — https://arxiv.org/abs/2602.10314
- [R9] Scaling Diffusion LMs via Adaptation from AR Models (DiffuGPT / DiffuLLaMA).
  arXiv:2410.17891 — https://arxiv.org/abs/2410.17891

**Inference / decoding + agentic caution**
- [R7] The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality
  Check (also documents block-diffusion confidence-threshold decoding). arXiv:2601.12979 —
  https://arxiv.org/abs/2601.12979

**Existence proof (throughput)**
- [R8] DiffusionGemma (26B MoE, ~4B active; >1,000 tok/s H100, ~700 tok/s RTX 5090; day-0 vLLM) —
  https://deepmind.google/models/gemma/diffusiongemma/ ; docs: https://ai.google.dev/gemma/docs/diffusiongemma

**Target model**
- [R12] Qwen/Qwen3.6-27B (64 layers, 3:1 Gated DeltaNet : Gated Attention, MTP, 262K ctx) —
  https://huggingface.co/Qwen/Qwen3.6-27B

**Method primitive**
- Gated DeltaNet (the GDN layer: gated delta-rule linear-attention recurrence). Underlying method
  behind Qwen3.6's linear-attention sublayers; see [R12] for Qwen's exact head/dim configuration.

> Note: the downstream **calibrated RL bridge** (using Diffusion-Qwen as a rollout sampler) is a
> separate document. Its references (off-policy correction, fp8 RL, on-policy distillation) live
> there, not here.
