# Agentic Diffusion-Qwen Plan

Date: 2026-06-25

## Ultimate Goal

Build a converted block-diffusion Qwen-family model that can perform agentic coding
and tool workflows, not merely reproduce Fast-dLLM.

Current goal, restated 2026-06-28: start with a strong agentic autoregressive
Qwen model, then make a diffusion/block-diffusion version that preserves its
tool calling, coding, planning, argument grounding, and stop-boundary behavior.
The desired gain is diffusion test-time compute: block size control, selective
re-denoising, confidence thresholds, grammar-aware constrained sampling, and
eventually dynamic block boundaries. Raw model behavior, constrained-decoding
behavior, and protected/sidecar behavior must be tracked separately.

Goal clarification, 2026-06-28: protected tool calling is not the end goal. It
is scaffolding and an oracle for where the diffusion sampler/model is failing.
The end goal is behavior-preserving conversion from a good agentic AR Qwen model
to a diffusion Qwen model, with protected decisions converted over time into
generation-time constraints, dynamic block policies, or learned value/boundary
adapters. A checkpoint is only a real model-side step forward when raw or
constrained-decoder metrics improve without hiding the error in post-processing.

Goal clarification, 2026-06-28 late: the clean project statement is:

```text
Given a strong agentic AR Qwen model, make a diffusion/block-diffusion Qwen
version that preserves the AR model's tool, code, planning, argument-grounding,
and stop behavior, while using diffusion test-time compute to improve latency,
parallelism, infill/editing, and selective re-denoising on fragile spans.
```

Therefore, the central recipe is not "train a diffusion model and hope it
becomes agentic." The central recipe is behavior-preserving AR-to-diffusion
conversion. Tool-aware dynamic block boundaries are part of that recipe:
ordinary prose can use larger blocks, while tool names, JSON keys, scalar
arguments, code hunk boundaries, and stop points need small protected blocks,
grammar/candidate constraints, and later a learned boundary/value selector.

Refined objective, 2026-06-27: start from a strong agentic autoregressive Qwen
model and convert it into a behavior-preserving block-diffusion model. The
success target is not just diffusion training loss; it is preserving the AR
model's tool-use, coding, planning, and stop-boundary behavior while unlocking
diffusion-style test-time compute knobs such as block size, denoising steps,
confidence thresholds, constrained decoding, and selective re-denoising.
The working recipe and research map are tracked in
`behavior_preserving_agentic_diffusion_recipe.md`.
The component taxonomy and dynamic tool-sensitive boundary research note are
tracked in `agentic_diffusion_boundary_protection_research.md`.

Further refinement, 2026-06-27: the project goal is a behavior-preserving
conversion recipe, not merely an agentic diffusion model trained from data. We
assume a good agentic AR model exists first. The experiment is whether we can
make a diffusion version that keeps that behavior while gaining diffusion
test-time compute controls. Raw model behavior, constrained decoding behavior,
and protected/sidecar behavior must be reported separately.

The term "protected tool calling" is intentionally a system score. It may use a
learned adapter, but it also includes deterministic projection, schema checks,
stop guards, and sidecar repair/routing. Promotion requires model-side movement
on raw or constrained-decoder metrics, not only a better post-processed score.

Fast-dLLM v2 is the scaffold. The actual target is a dLLM that can:

- generate valid tool/function calls
- preserve JSON/schema constraints
- perform multi-step reasoning without action-loop collapse
- edit code in repositories
- run through coding-agent harnesses with stable stop/tool boundaries
- eventually serve fast enough locally to matter

Closeout metrics are tracked separately in
`qwen36_diffusion_closeout_metrics.md`. In short, the final Qwen3.6 diffusion
target must be compared against both the local AR Qwen3.6 baseline and the
diffusion-init pretraining baseline, with SWE-bench Verified as the expensive
final gate.

## Available Hardware

### Local Workstation

- GPU: RTX 5090
- VRAM: about 32 GB
- Best role:
  - primary fast iteration machine
  - SGLang serving/eval for quantized Qwen3.6-27B teacher/reference at reduced context
  - Qwen3.5-9B LoRA/QLoRA pilots
  - sampler/harness development
  - NVFP4/FP8 serving experiments
- Weak role:
  - full 27B training
  - long-context 27B FP8 serving with large caches

### Alienware over Tailscale

- Tailscale SSH configured; exact private address omitted in the public notes.
- GPU: RTX 5080
- VRAM: about 16 GB
- Best role:
  - small eval worker
  - 1.5B/4B smoke jobs
  - data preprocessing
  - lightweight LoRA/QLoRA if memory fits
- Weak role:
  - 27B serving/training
  - larger block/canvas sweeps

### GX10 / GB10 over Tailscale

- Tailscale SSH configured; exact private address omitted in the public notes.
- Hardware: GB10-class system
- RAM/unified memory seen by Linux: about 117 GB
- Best role:
  - memory-heavy model loading
  - 9B/27B correctness experiments
  - quant/export experiments
  - long-context smoke tests
  - jobs where capacity matters more than raw GDDR bandwidth
- Weak role:
  - high-throughput training compared with 5090 GDDR7
  - multi-node training unless networking/software is explicitly validated

## Current State

We have:

- Fast-dLLM v2 repo cloned and patched locally.
- Local hybrid Fast-dLLM/Qwen2.5-1.5B base:
  `/home/mark/qwen_diffusion/models/qwen2.5-1.5b-fastdllm-init`
- Completed Alpaca LoRA run:
  `/home/mark/qwen_diffusion/runs/fastdllm_qwen25_1p5b_alpaca_lora_full`
- Adapter-aware `lm-eval` path in `fast-dllm/v2/eval.py`.
- Isolated eval environment:
  `/home/mark/qwen_diffusion/.venv-lmeval`
- Checkpoint sweep runner:
  `/home/mark/qwen_diffusion/scripts/run_fastdllm_checkpoint_sweep.py`
- DiffusionGemma / agentic dLLM research notes:
  `/home/mark/qwen_diffusion/diffusiongemma_agentic_research_notes.md`
- Local Qwen3.6-27B NVFP4 teacher serving result:
  `/home/mark/qwen_diffusion/qwen36_teacher_serving_result.md`
- Qwen3.6 teacher argument-level tool-call eval result:
  `/home/mark/qwen_diffusion/qwen36_teacher_toolcall_arg_eval_result.md`
- Qwen3.6 teacher 5090 MTP/CUDA graph fit result:
  `/home/mark/qwen_diffusion/qwen36_teacher_mtp_cuda_5090_result.md`
- Qwen3.6 teacher public multi-call eval result:
  `/home/mark/qwen_diffusion/qwen36_teacher_multicall_eval_result.md`
- Qwen3.6 teacher synthetic tool-result eval result:
  `/home/mark/qwen_diffusion/qwen36_teacher_toolresult_eval_result.md`
- Qwen3.6 teacher strict OpenAI tool-result eval result:
  `/home/mark/qwen_diffusion/qwen36_teacher_openai_toolresult_eval_result.md`
- Qwen3.5-9B AR baseline result:
  `/home/mark/qwen_diffusion/qwen35_9b_ar_baseline_result.md`
- Qwen3.5-9B diffusion pilot readiness result:
  `/home/mark/qwen_diffusion/qwen35_9b_diffusion_pilot_readiness.md`
- Qwen3.5-9B diffusion checkpoint-275 agentic scorecard:
  `/home/mark/qwen_diffusion/qwen35_9b_diffusion_ckpt275_agentic_scorecard.md`
- Qwen3.5 candidate-agreement diagnostic:
  `/home/mark/qwen_diffusion/qwen35_candidate_agreement_diagnostic_result.md`
  - added `scripts/eval_qwen_ar_diffusion_candidate_agreement.py`
  - supports AR scoring, Fast-DLLM masked scoring, Fast-DLLM causal scoring,
    and score-file comparison
  - true local AR Qwen3.5 scoring is currently blocked by the installed
    Transformers build not recognizing `model_type=qwen3_5`
  - `fastdllm_causal` mode now gives a local AR-proxy path through the
    converted Qwen3.5 text model's causal eval path
  - smoke with converted Fast-DLLM init as reference shows fixed `bd_size=16`
    checkpoint-5 improves a tiny synthetic voice-command candidate slice from
    `2/4` to `3/4`, with `0` regressions and `1` improvement
  - heldout policy candidate smoke over the first `40` nontrivial choices ties
    converted init and fixed `bd_size=16` checkpoint-5 at `38/40` accuracy with
    `40/40` prediction agreement; the two shared misses are derived numeric
    choices on `heldout_seed_multicall_0002`
  - full `88`-example nontrivial heldout policy candidate sweep shows a small
    positive masked-choice signal: fixed `bd_size=16` checkpoint-5 improves
    converted init from `84/88` to `85/88`, with `0` regressions, `1`
    argument-value improvement, and tool names tied at `13/13`
  - the same full sweep in causal Fast-DLLM mode matches the masked-choice
    result exactly on predictions, so the remaining generation gap is probably
    sampler/serialization/commit behavior rather than local candidate
    preference alone
- Qwen3.5-9B multi-call scalar curriculum result:
  `/home/mark/qwen_diffusion/qwen35_9b_multicall_scalar_curriculum_result.md`
- Qwen3.5-9B multi-call scalar adapter result:
  `/home/mark/qwen_diffusion/qwen35_9b_multicall_scalar_adapter_result.md`
- Qwen3.5-9B model-repair scalar-mix result:
  `/home/mark/qwen_diffusion/qwen35_9b_modelrepair_scalar_mix_result.md`
- Qwen3.5-9B multi-call gap curriculum result:
  `/home/mark/qwen_diffusion/qwen35_9b_multicall_gap_curriculum_result.md`
- Local text-only Qwen3.5-9B Fast-DLLM candidate:
  `/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init`
  - bridge status: implemented
  - GDN mode: `option_a_causal_gdn_v0`
  - remapped candidate weights: materialized
  - latest readiness gate: `ready=true`
  - 9B QLoRA loss-smokes: nonzero loss/gradient and adapter save verified on
    the local RTX 5090, including a real agentic-curriculum 512-token
    left-truncated run
  - first tiny strict diffusion eval now runs on the 9B init and adapter
- Qwen Code official harness smoke:
  `/home/mark/qwen_diffusion/qwen_code_official_harness_result.md`
- Qwen Code tiny repo-edit eval result:
  `/home/mark/qwen_diffusion/qwen_code_repo_edit_eval_result.md`
- Qwen3.6 teacher codegen eval result:
  `/home/mark/qwen_diffusion/qwen36_teacher_codegen_eval_result.md`
- Local 1.5B Fast-dLLM strict tool-call baseline result:
  `/home/mark/qwen_diffusion/qwen25_1p5b_diffusion_baseline_result.md`
- Behavior-preserving AR-to-diffusion recipe:
  `/home/mark/qwen_diffusion/behavior_preserving_agentic_diffusion_recipe.md`
- Tool-sensitive dynamic block-boundary prototype result:
  `/home/mark/qwen_diffusion/tool_sensitive_block_boundary_result.md`
- Candidate-ranking supervision artifact for the next learned selector target:
  `/home/mark/qwen_diffusion/data/candidate_ranking/public_multicall_toolname_argument_ranking_v3_12.train.json`
  - source: 12-case public multi-call gold slice
  - coverage: 86/86 usable ranker examples
  - split: 31 tool-name examples, 55 argument-value examples
- Masked-span candidate-ranking baseline for the active Qwen3.5-9B diffusion
  adapter:
  `/home/mark/qwen_diffusion/runs/candidate_ranking/public_multicall_qwen35_ckpt275_masked_span_rank_v3_12.summary.json`
  - overall: 80/86
  - tool names: 31/31
  - argument values: 49/55
  - `full_gold`, `prefix_only`, and `future_masked` context modes match at
    80/86, so the remaining masked-span failures are stable argument-value
    alignment misses rather than future-token leakage
- Diffusion-init vs checkpoint-275 candidate-ranking delta:
  `/home/mark/qwen_diffusion/qwen35_candidate_ranking_delta_result.md`
  - evaluator now supports `--no-adapter`
  - diffusion init: `78/86` overall, `31/31` tool names, `47/55` arguments
  - checkpoint-275: `80/86` overall, `31/31` tool names, `49/55` arguments
  - model-side lift is real but small: `3` argument spans improved, `1`
    regressed, and `6` argument spans remain wrong
- Candidate-ranker diagnostic curriculum and one-step gate:
  `/home/mark/qwen_diffusion/qwen35_candidate_ranker_diagnostic_curriculum_result.md`
  - builder: `scripts/build_candidate_ranking_curriculum.py`
  - diagnostic corpus: `131` instances, `0` rejected labels, hard remaining
    failures repeated, and `promotion_allowed=false` because it uses the public
    12-case eval slice
  - one-step continuation from checkpoint-275 trains and saves with loss
    `1.0847731828689575`
  - masked candidate-ranking remains unchanged at `80/86` overall and `49/55`
    argument values, so index-selection SFT is not enough evidence for masked
    span/value grounding
- Public-train candidate-ranking artifact and one-step heldout gate:
  `/home/mark/qwen_diffusion/qwen35_public_train_candidate_ranking_result.md`
  - materialized `56` non-eval public-train multi-call cases from
    `data/fastdllm_toolcall_train/train_toolcall.json`
  - built `338` ranking examples, `299` usable: `155` tool-name and `144`
    argument-value examples
  - checkpoint-275 train-slice masked rank: `295/299` overall, `155/155`
    tool names, `140/144` arguments
  - packaged `329` accepted curriculum rows with `promotion_allowed=true`
    by data provenance
  - one-step continuation from checkpoint-275 trains and saves with loss
    `1.3103340864181519`, but public-12 heldout ranking remains unchanged at
    `80/86` overall and `49/55` arguments
- Public-train candidate value-span gate:
  `/home/mark/qwen_diffusion/qwen35_public_train_candidate_value_span_result.md`
  - `scripts/build_candidate_ranking_curriculum.py` now supports
    `--answer-mode target_text` and `--include-kinds`
  - built `173` argument-value target-span rows with `0` rejected labels and
    p50 kept labels `8`
  - one-step continuation from checkpoint-275 trains and saves with loss
    `3.019835948944092`, but public-12 heldout ranking remains unchanged at
    `80/86` overall and `49/55` arguments
  - 10-step continuation finds an early positive checkpoint: checkpoint-5 gets
    public-12 heldout candidate ranking `81/86` overall and `50/55` arguments,
    with tool names still `31/31`; checkpoint-10 falls back to `80/86` and
    `49/55`
  - checkpoint-5 public one-call first-pass generation keeps raw `3/8` sequence
    and `2/8` arguments while improving constrained arguments from checkpoint-
    275's `5/8` to `8/8`
  - checkpoint-5 public multi-call gate keeps raw `1/12` sequence and `0/12`
    arguments, improves direct constrained from active checkpoint-275's
    `7/12` / `4/12` to `8/12` / `5/12`, improves contextual projection from
    `7/12` / `7/12` to `8/12` / `8/12`, and ties guarded planner projection
    at `11/12` / `10/12`; treat this as a broader positive sidegrade, not a
    full route promotion yet
  - remaining split-route lanes confirm no promotion: checkpoint-5 is weaker
    than the current routed target on teacher-heldout protected one-call
    (`7/8` / `5/8` versus `8/8` / `6/8`) and OpenAI-style protected
    tool-result arguments (`8/10` versus `9/10`), while tying synthetic text
    tool-result protected `10/10` / `9/10`
  - row-level route delta report:
    `/home/mark/qwen_diffusion/qwen35_public_train_candidate_value_span_route_delta.md`
  - route-delta train-only mix:
    `/home/mark/qwen_diffusion/qwen35_route_delta_trainonly_mix_result.md`
    builds `335` train-only rows from public-train value spans,
    Fast-dLLM tool-call train rows, synthetic one-call train rows, and
    synthetic tool-result train rows converted into both text and OpenAI-style
    targets. A cgroup-protected one-step gate from checkpoint-275 trains and
    saves with loss `0.2080969214439392`. Follow-up eval with the patched
    context-first tool-result constrained decoder recovers public one-call
    `8/8` / `8/8`, teacher-heldout `8/8` / `6/8`, synthetic text tool-result
    `10/10` / `10/10`, OpenAI-style tool-result `10/10` / `9/10`, and guarded
    public multi-call planner `11/12` / `10/12`; however, direct/contextual
    public multi-call remains `7/12` / `4/12` and `7/12` / `7/12`, below the
    checkpoint-5 sidegrade. Do not promote this one-step adapter or run a
    longer sweep of the same mix as the next default.
- Public eval overlap audit:
  `/home/mark/qwen_diffusion/qwen35_public_eval_overlap_audit_result.md`
  - `data/fastdllm_toolcall_train/train_toolcall.json` contains `11/12`
    public multi-call smoke rows verbatim.
  - the current voice-command camera and motion-detector installation-code
    failure families have no non-eval analogue in that file.
  - new scripts `scripts/audit_toolcall_eval_overlap.py` and
    `scripts/filter_toolcall_eval_overlap.py` create an auditable filtered
    source:
    `data/fastdllm_toolcall_train/train_toolcall_no_public_multicall_smoke.json`
  - future public-train-derived planner/value experiments must use explicit
    overlap filtering or synthetic/teacher-generated analogues before they are
    eligible for promotion evidence.
- Synthetic multi-call failure analogues:
  `/home/mark/qwen_diffusion/synthetic_multicall_failure_analogue_result.md`
  - built `8` non-eval analogue rows for the two active public multi-call
    failure families: voice-command camera routing and security installation
    code scoping.
  - pure request-derived planner from empty drafts reaches `8/8` exact sequence
    and `8/8` exact arguments.
  - bad-draft conservative projection repairs only `4/8`: it fixes same-sequence
    code-copy mistakes but refuses same-count wrong tool sequences.
  - new opt-in `--use-plan-on-sequence-mismatch` repairs the bad drafts to
    `8/8` exact sequence and `8/8` exact arguments.
  - the public multi-call safety ablation regresses the active guarded planner
    from `11/12` sequence and `10/12` arguments to `9/12` and `8/12`, and it
    still does not fix the public voice-command camera row because the planner
    over-weights the later `status/mode` argument list. Keep the flag as a
    debug option only until a confidence gate and voice-command conflict
    resolver are added.
  - follow-up implemented both guards: `--use-safe-plan-on-sequence-mismatch`
    for score/margin-gated same-length replacements, plus a targeted camera
    voice-command conflict resolver and anchored nearest-code extraction. The
    synthetic bad-draft safe gate reaches `8/8` exact sequence and `8/8` exact
    arguments; the public multi-call protected diagnostic now reaches `12/12`
    exact sequence and `12/12` exact arguments. Treat the public number as a
    protected regression diagnostic, not model-promotion evidence.
- Synthetic multi-call planner distillation:
  `/home/mark/qwen_diffusion/qwen35_synthetic_multicall_planner_distill_result.md`
  - built a train-only `24` row curriculum from `8` synthetic analogue cases
    repeated three times, with `0` target rejects, `0` label rejects, and `0`
    public multi-call eval overlap.
  - one-step QLoRA continuation from active checkpoint-275 trains and saves
    with loss `1.4955649375915527`.
  - generation eval is unchanged versus checkpoint-275 on the synthetic
    analogue slice: raw `1/8` sequence and `0/8` arguments; constrained `2/8`
    sequence and `0/8` arguments. Do not promote this adapter; use it as a
    clean plumbing gate and as evidence that one-step synthetic planner SFT is
    too weak to move model-side behavior.
  - serial candidate-index diagnostics show a small checkpoint-275 lift over
    diffusion init, but no step-1 lift: diffusion init scores `6/8` overall
    and `2/4` on voice-command camera tool-name choices; checkpoint-275 and
    the synthetic planner step-1 adapter both score `7/8` overall, `3/4` on
    voice-command camera tool-name choices, and `4/4` on security-code
    argument-value choices. The only shared adapter miss predicts
    `set_thermostat` instead of `activate_voice_command` on
    `synthetic_voice_command_camera_003`, so the one-step continuation did not
    move the simpler masked selection preference either.
- Synthetic candidate-index leave-one-out selector probe:
  `/home/mark/qwen_diffusion/qwen35_synthetic_candidate_index_leaveone_result.md`
  - builder:
    `/home/mark/qwen_diffusion/scripts/build_synthetic_candidate_index_leaveone_curriculum.py`
  - held out the exact remaining miss,
    `synthetic_voice_command_camera_003`, and trained on the other seven
    synthetic selector examples repeated to `84` rows.
  - 30-step continuation from checkpoint-275 trained cleanly with loss
    `1.013732647895813`; checkpoints 10/20/30 all improve masked selector
    ranking from checkpoint-275's `7/8` to `8/8`, including the heldout row.
  - generation gates do not promote it: checkpoint-10 regresses raw synthetic
    generation from `1/8` to `0/8` exact sequence while only tying constrained
    sequence at `2/8`; checkpoint-20 regresses constrained sequence to `1/8`.
    Exact arguments stay `0/8`. Treat selector training as a side objective or
    sidecar signal, not a standalone main-generator recipe.
- Synthetic selector plus replay mix:
  `/home/mark/qwen_diffusion/qwen35_synthetic_selector_replay_mix_result.md`
  - builder:
    `/home/mark/qwen_diffusion/scripts/build_synthetic_selector_replay_mix.py`
  - mixed `84` full planner replay rows with `14` selector-index rows while
    holding out `synthetic_voice_command_camera_003`.
  - 20-step continuation from checkpoint-275 trained cleanly with loss
    `2.079222249984741`, but both checkpoint-10 and checkpoint-20 stay at
    `7/8` masked selector accuracy and miss the heldout row.
  - checkpoint-10 generation ties checkpoint-275 raw sequence at `1/8` but
    regresses raw valid JSON from `4/8` to `1/8` and constrained sequence from
    `2/8` to `0/8`; do not promote. This argues for a separate selector/value
    sidecar or generation-time candidate scorer rather than ordinary mixed SFT
    in the main generator adapter.
- Selector sidecar projection:
  `/home/mark/qwen_diffusion/qwen35_selector_sidecar_projection_result.md`
  - script:
    `/home/mark/qwen_diffusion/scripts/apply_synthetic_selector_sidecar_projection.py`
  - uses the selector-only leave-one checkpoint as a separate scorer over
    candidate-index decisions, then applies only the selected ambiguous
    tool-name/value replacement to `bad_draft_assistant`.
  - bad drafts start at `4/8` exact sequence and `0/8` exact arguments.
    checkpoint-275 selector sidecar reaches `7/8` and `7/8`; selector-only
    leave-one checkpoint-20 sidecar reaches `8/8` and `8/8`, including the
    heldout `synthetic_voice_command_camera_003` row.
  - this is protected-path evidence for a separate selector/value sidecar or
    generation-time candidate scorer. It is not raw generator promotion because
    the projection still applies deterministic local evidence extraction after
    the sidecar chooses the candidate.
- Selector sidecar scheduled sampler:
  `/home/mark/qwen_diffusion/qwen35_selector_sidecar_scheduled_sampler_result.md`
  - script:
    `/home/mark/qwen_diffusion/scripts/inject_selector_sidecar_schedule_choices.py`
  - moves the selector-only leave-one checkpoint-20 sidecar from post-hoc
    projection into the generation-time sampler schedule while keeping the main
    generator at checkpoint-275.
  - the first evidence-selected metadata run improved tool sequence but kept
    exact arguments at `0/8`, revealing that argument spans were being
    overwritten by extractor-selected evidence candidates.
  - `scripts/augment_schedule_value_candidates.py` now supports
    `--include-target-candidate` and `--selected-candidate-mode target`.
  - with target-selected metadata, synthetic analogue scheduled generation
    reaches `8/8` raw exact sequence, `8/8` raw exact arguments, and `8/8`
    valid JSON; max reserved VRAM was `17.95 GiB`.
  - turning off selected-candidate forcing still reaches `8/8` raw sequence and
    arguments with model-ranked whole-candidate value choices. Removing selector
    injection as well also reaches `8/8` on this synthetic slice, with `26`
    argument candidate-sequence choices and `12` tool-name sequence choices.
  - this is a positive protected-sampler handoff and argument-span forcing gate,
    not model promotion. The next bottleneck is no longer local argument-span
    forcing on this synthetic slice; it is proposing target-containing
    candidate sets and validating model-ranked choices on public/harder
    multi-call schedules.
- Deferred whole-candidate sampler fix:
  `/home/mark/qwen_diffusion/runs/tool_sensitive_block_plans/public_multicall_schedule_toolname_candidate_sequence_selected_deferred_smoke1.summary.json`
  - fixes premature commitment on shared tool-name token prefixes
  - one-case public multi-call smoke: raw exact sequence 1/1, exact arguments
    1/1
- Public multi-call target-candidate sampler:
  `/home/mark/qwen_diffusion/qwen35_public_multicall_targetcandidate_sampler_result.md`
  - rebuilt the full 12-case public gold schedule with
    `--include-target-candidate` and `--selected-candidate-mode target`.
  - schedule coverage improves to `100` argument blocks augmented and `90`
    argument blocks with sequence candidates.
  - with selected-candidate forcing off, model-ranked whole-candidate sampling
    reaches raw `11/12` exact sequence, `9/12` exact arguments, and `11/12`
    valid JSON, up from the older public schedule's `11/12`, `3/12`, `11/12`.
  - with target-selected forcing on, the same sampler reaches raw `12/12` exact
    sequence, `12/12` exact arguments, and `12/12` valid JSON.
  - the remaining model-ranked misses are two time/value ranking mistakes and
    one long table/array JSON consistency failure. This keeps the next
    protected-sampler target focused on candidate proposal/ranking, especially
    row-local value scoring, not on basic argument-span forcing.
  - miss audit:
    `/home/mark/qwen_diffusion/runs/tool_sensitive_block_plans/public_multicall_targetcandidate_modelranked_ckpt275_candidate_miss_audit_v4_12.summary.json`
    shows `3` failed records, `2` scalar argument mismatches where the gold
    value was already in sequence candidates, and `1` invalid middle
    `process_invoices` block on the long finance/table row.
  - v5 schedule fix: `scripts/augment_schedule_value_candidates.py` now uses
    target token IDs directly when a candidate equals the target, preserving
    decimal numeric spans such as `1500.0`. This raises public schedule
    argument sequence coverage to `100/100` and improves the model-ranked gate
    to raw `12/12` exact sequence, `9/12` exact arguments, and `12/12` valid
    JSON.
  - focused miss target:
    `/home/mark/qwen_diffusion/data/candidate_ranking/public_multicall_targetcandidate_v5_miss_targets.jsonl`
    contains `5` usable examples. Checkpoint-275 scores `0/5` on these in both
    prefix-only and full-gold candidate-ranking modes, so the remaining public
    gap is a stable value-ranking target rather than a schedule coverage bug.
- Focused public multi-call value-ranker diagnostic:
  `/home/mark/qwen_diffusion/qwen35_public_multicall_focused_value_ranker_result.md`
  - built a diagnostic-only value-span curriculum from the five public v5 miss
    targets, repeated to `120` accepted rows with `0` rejects.
  - a first label-only attempt failed before training because value-span token
    id extraction expects tool-call JSON fragments, not standalone value-span
    answers.
  - a plain 10-step value-span conversation SFT from checkpoint-275 trained and
    saved checkpoint-5/checkpoint-10 with train loss `2.4059`.
  - focused masked candidate-ranking remained `0/5` at checkpoint-5 and
    checkpoint-10. Do not scale this exact recipe; use the five-example target
    to test an explicit ranker/classifier sidecar or pairwise ranking loss.
- Focused public multi-call index-sidecar diagnostic:
  `/home/mark/qwen_diffusion/qwen35_public_multicall_focused_index_sidecar_result.md`
  - built a diagnostic-only numeric-index curriculum from the same five public
    v5 miss targets, repeated to `120` accepted rows with `0` rejects.
  - checkpoint-275 direct masked index ranking starts at `2/5`, better than the
    value-span ranker's `0/5`; the two time fields are correct, while the
    three finance table ID fields remain wrong.
  - conservative 10-step `1e-6` SFT and stronger 20-step `1e-5` SFT from
    checkpoint-275 both remain `2/5`; margins move slightly but rank order does
    not flip on the hard finance rows.
  - a `2048` block per-example run OOMs on the RTX 5090 during backward by a
    small margin; `1536` block with `DISABLE_GROUP_TEXTS=1` runs and is the
    current practical 9B QLoRA full-context-ish ceiling.
  - new script `scripts/eval_fastdllm_candidate_index_generation.py` shows
    diffusion generation of the numeric index is not usable yet: `0/5` in-range
    answers for checkpoint-275 and the stronger step-20 adapter.
  - do not scale plain numeric-index chat SFT. Next selector work should remove
    index-position bias with pairwise ranking, candidate-order randomization,
    table-row tuple scoring, or an external sidecar classifier before
    distilling back into the diffusion model.
- Path-aware pairwise selector sidecar:
  `/home/mark/qwen_diffusion/qwen35_public_multicall_pairwise_path_sidecar_result.md`
  - new scripts:
    `scripts/build_candidate_pairwise_curriculum.py`,
    `scripts/eval_fastdllm_candidate_pairwise_ranking.py`,
    `scripts/eval_fastdllm_candidate_pairwise_tournament.py`, and
    `scripts/inject_pairwise_tournament_schedule_choices.py`.
  - key diagnosis: value-selector prompts need the full structural JSON path,
    not only the scalar key. `client_id` is ambiguous across table rows, while
    `invoice_data[1].client_id` is actionable.
  - checkpoint-275 path-aware pairwise A/B reaches `60/60` on the focused
    five-miss target; path-aware tournament reaches `5/5`.
  - injecting those five choices into the v5 public sampler improves raw public
    multi-call from `12/12` sequence and `9/12` arguments to `12/12` and
    `11/12`; the remaining exposed span is `payment_data[0].invoice_id`.
  - adding that sixth path-aware tournament choice reaches raw `12/12` exact
    sequence, `12/12` exact arguments, and `12/12` valid JSON on the public
    12-case multi-call slice, with a final miss audit of `0` failed records.
  - working sampler mechanism: singleton `candidate_sequence_values` plus the
    existing structural guards and `--force-best-candidate-sequence`. Direct
    `--force-selected-candidate-tokens` is unstable on this v5 schedule.
  - this is protected-sampler evidence from public/gold artifacts. Next
    promotion-eligible work is train-only or synthetic path-aware selector data
    and heldout gates, then distill the selector behavior into model-side
    boundary/value adapters.
- Path-aware phrase selector heldout gate:
  `/home/mark/qwen_diffusion/qwen35_pathaware_phrase_selector_gate_result.md`
  - path metadata is now propagated through planner, schedule, ranking
    examples, pairwise prompts, tournament rows, and schedule injection.
  - candidate extraction now includes conservative single-quoted strings and
    digit-bearing model/product phrases, which fixes copied free-form values
    such as `Main Control Group`, `YRD256 Yale Assure Lock SL`, and
    `ChemSimulationProject`.
  - train-only phrase-aware ranking examples:
    `367` rows, `328` usable, with `173` usable argument-value rows.
  - promotion-eligible train-only pairwise curriculum:
    `data/qwen35_9b_public_train_pairwise_pathaware_phrase_curriculum`,
    `376` accepted rows, `0` rejected labels, `contains_eval_slice=false`,
    `promotion_allowed=true`.
  - heldout public-12 path-aware phrase tournament from checkpoint-275 reaches
    `98/99` overall, `68/68` argument values, and `30/31` tool names.
  - injecting only argument-value choices into the protected sampler reaches
    raw `12/12` exact tool sequence, `12/12` exact arguments, and `12/12`
    valid JSON, with a final miss audit of `0` failed records and `0`
    mismatches.
  - interpretation: the current selector route is strong for argument values,
    but tool names remain a separate selector/model-side target. This is still
    protected-sampler evidence; model promotion requires distilling the
    train-only selector behavior and proving heldout movement in raw or
    constrained-decoder metrics.
- Train-only path-aware phrase pairwise SFT:
  `/home/mark/qwen_diffusion/qwen35_public_train_pairwise_phrase_sft_result.md`
  - continuation from checkpoint-275 on the promotion-eligible phrase-aware
    pairwise curriculum completes on the RTX 5090 at block size `1536` with
    `DISABLE_GROUP_TEXTS=1`.
  - checkpoint-5 and checkpoint-10 both match checkpoint-275 exactly on the
    heldout phrase selector gate: `98/99` overall, `68/68` argument values,
    and `30/31` tool names.
  - row-level predictions changed on `0/99` heldout rows. This is a
    no-regression training smoke, not a promoted model-side improvement.
  - next model-side work should target the separate tool-name miss, raw
    selected-value span/copy movement, or a larger synthetic/teacher heldout
    selector gate where argument selection is not saturated.
- Train-only tool-name pairwise SFT:
  `/home/mark/qwen_diffusion/qwen35_public_train_toolname_pairwise_sft_result.md`
  - built a promotion-eligible train-only tool-name pairwise curriculum with
    `240` accepted rows, `0` rejects, and `contains_eval_slice=false`.
  - 10-step continuation from checkpoint-275 completes on the RTX 5090 at block
    size `1536`, saves checkpoint-5/checkpoint-10, and trains with loss
    `5.0694`.
  - heldout predictions are unchanged on `0/99` rows: checkpoint-5 and
    checkpoint-10 remain `98/99` overall, `68/68` argument values, and `30/31`
    tool names.
  - conclusion: the remaining tool-name selector miss is likely prompt/context
    under-specification. The next selector experiment should add same-call
    argument keys/values or sequence-plan context before doing more SFT.
- Tool-name same-call argument-sketch selector gate:
  `/home/mark/qwen_diffusion/qwen35_toolname_argsketch_selector_result.md`
  - selector prompts now include same-call argument keys/values for
    `tool_name` rows.
  - heldout public-12 tournament from checkpoint-275 improves from `98/99` to
    `99/99` overall, with `68/68` argument values and `31/31` tool names.
  - the previous miss choosing `activate_security_cameras` over
    `set_thermostat` is fixed once the prompt includes `temperature: 72` and
    `location: "living room"`.
  - implication: tool-name blocks need local call evidence. For diffusion
    decoding, either expose a planned argument sketch before committing the
    tool-name span or delay that decision until argument evidence is visible.
    This is component-gate progress, not raw model promotion.
- Arg-sketch tool+argument selector sampler gate:
  `/home/mark/qwen_diffusion/qwen35_argsketch_toolargselector_sampler_result.md`
  - injected the `99/99` arg-sketch tournament into the protected public
    multi-call sampler for both `tool_name` and `argument_value` spans.
  - injection restricts `226` schedule items from `99` correct selector rows
    with `0` candidate misses.
  - generation reproduces raw protected `12/12` exact tool sequence,
    `12/12` exact arguments, `12/12` valid JSON, and `12/12` schema-valid
    calls; final candidate/miss audit has `0` failed records.
  - sampler counters show `0` model-choice events for semantic candidate
    sequences because selector injection reduced tool-name and argument-value
    spans to singleton candidate sequences. This is the desired runtime
    component split for now, but it still uses a gold-tokenized schedule and
    deterministic structural/stop guards. Next step is the same selector API
    on non-gold/live planner schedules.
- Live-planner arg-sketch selector sampler gate:
  `/home/mark/qwen_diffusion/qwen35_live_planner_argsketch_sampler_result.md`
  - regenerated the public multi-call v5 `sequence_planner_assistant` as a
    non-gold tokenized block/sampler schedule with current path and
    `tool_call_index` metadata.
  - live planner row-level metrics are `12/12` valid JSON, `12/12` exact tool
    sequence, and `12/12` exact arguments before sampler replay.
  - evidence-only candidate extraction covers `69/100` planned argument-value
    spans; target-included planner candidates cover `100/100`.
  - checkpoint-275 pairwise selector gate on the target-included live planner
    spans reaches `131/131`: `100/100` argument values and `31/31` tool names.
  - injected live schedule consumes `131` correct selectors, restricts `267`
    schedule items, and has `0` candidate misses.
  - generation with deterministic structural/stop guards reaches `12/12` exact
    tool sequence, `12/12` exact arguments, and a clean final mismatch audit.
  - code fix: selector injection now preserves empty-string predictions rather
    than treating them as missing. Next target is closing the evidence-only
    candidate gap so live replay no longer needs planner-target inclusion.
    The missing evidence-only blocks are concentrated in table/list row fields,
    date fields, copied phrases/IDs, enum/boolean values, and one empty-string
    argument.
- Live-planner evidence-only selector sampler gate:
  `/home/mark/qwen_diffusion/qwen35_live_planner_evidence_selector_sampler_result.md`
  - improved request/schema evidence extraction with nested-schema resolution,
    empty-string support, ISO dates, booleans, snake-case values, symbolic
    languages, room/location/door phrases, command phrases, capitalized target
    phrases, and path-aware markdown table row extraction.
  - evidence-only candidate coverage now reaches `100/100` planned
    argument-value blocks and `31/31` tool-name blocks, with `0` missing
    selector targets.
  - row-local markdown table pruning cuts selector pair comparisons from
    `1480` to `690` and fixes the three financial table row-alignment misses.
  - checkpoint-275 selector tournament reaches `131/131`, injected live
    schedule restricts `267` items with `0` candidate misses, and generation
    reaches `12/12` exact tool sequence and `12/12` exact arguments with a
    clean final audit.
  - this removes planner-target inclusion from the semantic candidate path on
    the public multi-call slice. Remaining caveats: planner/structural/stop
    protection are still runtime sidecars, and the gate is small.
  - replay route added:
    `scripts/run_qwen35_live_evidence_selector_route.py` writes
    `runs/tool_sensitive_block_plans/live_v5_evidence_selector_route/route_plan.json`
    and `route_plan.sh`; `--verify-existing` passes `16/16` checks with `0`
    missing artifacts and `0` failed checks.
- Synthetic analogue evidence-selector route:
  `/home/mark/qwen_diffusion/qwen35_synthetic_evidence_selector_route_result.md`
  - added reusable route runner:
    `scripts/run_qwen35_evidence_selector_route.py`.
  - default route targets the `8` non-public synthetic multi-call analogue
    rows for voice-command camera routing and security installation-code
    scoping, using
    `runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl`
    as the planner/span source.
  - evidence-only candidate coverage reaches `60/60` argument-value blocks
    and `24/24` tool-name blocks, with `0` missing selector targets.
  - checkpoint-275 pairwise selector reaches `84/84`, including `60/60`
    argument values and `24/24` tool names.
  - scheduled protected generation reaches `8/8` valid JSON, `8/8` exact tool
    sequence, `8/8` exact arguments, and a clean final audit; the generic route
    verifier passes `20/20` checks with `0` missing artifacts and `0` failures.
  - this is the first non-public analogue replay of the evidence-selector
    route, but it remains protected runtime evidence, not raw model-side
    promotion. Next route target should be a larger fresh/teacher multi-call
    slice before training selector or boundary adapters.
- Heldout seed evidence-selector preflight:
  `/home/mark/qwen_diffusion/qwen35_heldout_seed_evidence_selector_preflight_result.md`
  - built `13` clean heldout Hermes multi-call cases with `2` to `3` calls
    each, excluding filtered-train/public/synthetic exact overlaps.
  - request-derived planner from empty text is weak on this broader slice:
    `13/13` valid planned JSON, but only `3/13` exact tool sequence and
    `0/13` exact arguments.
  - using `gold_assistant` as the span source isolates the semantic route:
    evidence extraction covers `140/140` argument blocks and `32/32` tool-name
    blocks, with diagnostic target coverage `144/144`.
  - checkpoint-275 pairwise selector reaches `157/172`: tool names are
    saturated at `32/32`, but argument values are `125/140`.
  - all `15` misses are row/list-local argument alignment errors in four
    records: construction expense rows, IoT device rows, ad schedule rows, and
    refund-policy selection.
  - follow-up: argument-value grouping now includes `json_path`, and pairwise
    prompts include non-leaking local peer arguments plus request snippets
    anchored on peer values/path terms.
  - best follow-up selector gate reaches `174/176`: `142/144` argument values
    and `32/32` tool names. This removes the broad row/list-local failure
    family.
  - remaining misses are derived-rule cases: final rounded portfolio weight
    `0.334` versus `0.333`, and refund-policy threshold application
    (`20` days before event should choose `full`).
  - added `scripts/apply_derived_rule_selector_sidecar.py` with auditable
    equal-weight residual, percentage-range midpoint, and refund-policy
    threshold rules. On this heldout selector gate it applies exactly `2`
    corrections and raises the selector to `176/176`.
  - derived sidecar injection consumes `176/176` correct selectors, restricts
    `341` schedule items, and has `0` candidate misses.
  - gold-span protected generation replay reaches `13/13` valid JSON,
    `13/13` exact tool sequence, `13/13` exact arguments, and a clean final
    audit with `0` mismatches.
  - immediate implication: the gold-span protected route now passes this clean
    heldout multi-call slice, but it is still not raw model promotion and not a
    live-planner pass. Improve planner targets for this broader heldout slice
    before claiming end-to-end behavior preservation.
- Qwen3.6 teacher heldout multi-call planner/eval:
  `/home/mark/qwen_diffusion/qwen36_teacher_heldout_multicall_result.md`
  - local Qwen3.6-27B NVFP4 no-MTP 8k serving works on the 5090 after the MTP
    profile fails memory-pool setup because draft weights are counted.
  - required/native teacher mode reaches `13/13` valid tool JSON, `9/13` exact
    tool sequence, and `6/13` exact arguments on the clean heldout seed slice.
  - auto/text fallback is weaker: `8/13` exact sequence and `6/13` exact
    arguments.
  - teacher is much stronger than the empty deterministic planner (`3/13`
    sequence, `0/13` arguments), but it does not close the live-planner gap.
  - important diagnosis: some heldout seed rows are label-ambiguous. Example:
    the construction-expense prompt explicitly asks to categorize and generate
    a report, while gold only contains the three expense-record calls. The next
    planner target needs an explicit decomposition policy rather than blindly
    optimizing to every seed gold.
- Heldout planner decomposition policy:
  `/home/mark/qwen_diffusion/qwen35_heldout_planner_decomposition_policy_result.md`
  - added `scripts/analyze_planner_decomposition_policy.py` to compare seed
    gold, Qwen3.6 required/auto outputs, and deterministic planner output.
  - policy analysis splits the 13 heldout rows into `6` clean teacher/gold
    targets, `3` teacher-sequence/value-sidecar rows, `3` gold decomposition
    rows where the teacher undercalls prompt-supported actions, and `1`
    full-request-vs-seed-gold ambiguity.
  - added `scripts/materialize_planner_policy_targets.py`.
  - materialized `12` accepted planner policy targets and rejected
    `heldout_seed_multicall_0001` pending adjudication.
  - accepted targets verify at `12/12` valid JSON, exact tool sequence, exact
    arguments, schema valid, and required args present. This gives a clean
    planner target set for the next live-route replay without silently training
    or evaluating against contradictory labels.
- Heldout policy-target evidence-selector route:
  `/home/mark/qwen_diffusion/qwen35_heldout_policy_target_evidence_selector_route_result.md`
  - replayed the protected route on the `12` accepted planner-policy targets,
    using `policy_planner_assistant` as the span source.
  - candidate coverage is clean: `152` selector examples, `123` argument-value
    examples, `29` tool-name examples, and `0` missing selector targets.
  - checkpoint-275 selector reaches `150/152` raw; the two misses are the known
    equal-weight residual and refund-policy threshold cases.
  - derived-rule sidecar raises the selector to `152/152`; injected schedule
    consumes `152/152` correct selectors with `0` candidate misses.
  - protected scheduled generation reaches `12/12` valid JSON, `12/12` exact
    tool sequence, `12/12` exact arguments, `12/12` schema valid, and a clean
    audit with `0` mismatches.
  - implication: the planner-policy route is now an end-to-end protected
    replay pass on the accepted heldout rows. It is still an oracle/protected
    ceiling, so the next promotion gate must distill this behavior into raw or
    constrained-decoder model-side movement.
- Heldout policy derived-pairwise diagnostic:
  `/home/mark/qwen_diffusion/qwen35_heldout_policy_derived_pairwise_diagnostic_result.md`
  - added diagnostic/provenance controls to the planner and pairwise builders:
    configurable planner target field, eval-slice flags, and focused pairwise
    filters by record id / JSON path / JSON key.
  - built a non-promotable planner-policy distill corpus from
    `policy_planner_assistant`: `15` accepted rows from `12` policy cases,
    with `9` label-rejected full/compact candidates at block size `1024`.
  - built a full non-promotable heldout selector diagnostic corpus with `420`
    pairwise rows and a focused derived-rule corpus with `120` rows for the two
    sidecar misses.
  - a 10-step focused QLoRA run from checkpoint-275 trained cleanly
    (`train_loss=3.6861`) but did not move the selector gate: checkpoint-10
    remains `150/152`, with `121/123` argument values and `29/29` tool names.
  - row-level selector predictions changed on `0/152` rows; the two misses
    remain final portfolio residual and refund-policy threshold.
  - implication: do not scale more of the same short pairwise SFT for derived
    arithmetic/policy choices. Keep the sidecar for now and test a different
    value-reasoning objective or learned value/scorer adapter.
- Heldout policy planner-distill diagnostic:
  `/home/mark/qwen_diffusion/qwen35_heldout_policy_planner_distill_diagnostic_result.md`
  - trained a non-promotable 25-step QLoRA continuation from checkpoint-275 on
    the 15-row heldout policy-planner diagnostic corpus.
  - training completed cleanly with `train_loss=2.8638`, block size `1024`,
    and about `26.3 GiB` peak observed GPU memory.
  - baseline checkpoint-275 on the 12 policy targets with forced tool-call
    prefix reaches raw valid/exact sequence/exact args `0/12` / `0/12` /
    `0/12`, and constrained exact sequence/args `5/12` / `0/12`.
  - checkpoint-25 moves raw valid JSON to `1/12` and constrained exact sequence
    to `6/12`, but raw exact sequence remains `0/12` and exact arguments
    remain `0/12`.
  - extra/repeated raw calls decrease (`7 -> 2` extra, `7 -> 1` repeated), but
    missing calls slightly increase (`22 -> 23`) and one constrained sequence
    row regresses.
  - implication: direct planner-policy SFT has weak model-side movement where
    pairwise derived-rule SFT had none, but it is unstable and non-promotable.
    The next trainable route should mix planner-policy pressure with retention
    and selector/value supervision, then gate on separate heldout rows.
- Planner/selector/retention train-only mix:
  `/home/mark/qwen_diffusion/qwen35_planner_selector_retention_mix_result.md`
  - added `scripts/build_qwen35_planner_selector_retention_mix.py` to combine
    retention, sequence-planner, and pairwise-selector rows with token-label
    audits and eval-overlap filtering.
  - initial draft had `5` public multi-call exact/user overlaps, all from the
    retention source; the cleaned build removes them before writing training
    rows.
  - clean corpus:
    `data/qwen35_9b_planner_selector_retention_mix_nooverlap_curriculum` with
    `377` rows: `187` route-delta retention, `30` sequence planner, and `160`
    pairwise selector.
  - clean overlap audit against public multi-call, synthetic analogues,
    heldout seed multi-call, and heldout policy targets is `0` exact overlaps
    and `0` user-prompt overlaps.
  - one-step QLoRA trainability gate from checkpoint-275 at block size `1536`
    saves successfully with `train_loss=1.4096` and `16.23s` runtime.
  - short sweep trained checkpoints `5` and `10` from checkpoint-275 on the
    clean `377`-row mix. Training completed with `train_loss=0.5359` in
    `156.10s`, using about `30.5 GiB` observed training VRAM.
  - heldout policy-target eval with forced prefix and constrained
    sequence-preserving projection shows no promotion: checkpoint-5 ties
    checkpoint-275 constrained exact sequence at `5/12` with `0/12` exact
    arguments, while checkpoint-10 regresses to `3/12` constrained sequence and
    `0/12` arguments.
  - implication: the clean mix is useful infrastructure, but this exact
    planner/selector/retention balance should not be scaled unchanged. It
    reduces extra/repeated raw calls, but it does not fix missing calls or
    argument grounding.
  - follow-up substrate:
    `data/qwen35_9b_plannerheavy_selectorlight_retention_mix_nooverlap_curriculum`
    shifts the same clean sources to `187` retention, `90` sequence-planner,
    and `80` selector rows. Overlap audit is again `0` exact/user overlaps
    against public/synthetic/heldout eval files, and the one-step QLoRA gate
    saves with `train_loss=1.3002`.
  - planner-heavy `5/10` sweep trained cleanly with final `train_loss=0.4283`.
    Heldout policy-target eval does not promote: checkpoint-5 reaches
    constrained sequence `4/12` and exact args `1/12`; checkpoint-10 reaches
    constrained sequence `4/12` and exact args `0/12`. Both keep raw valid JSON
    and raw exact sequence at `0/12`.
  - implication: stronger planner balance exposes one argument-grounding hit
    but loses sequence retention. The next branch needs sequence
    anti-regression plus a separate value/argument objective, not just more
    planner repetition.
- Sequence/value/retention train-only mix:
  `/home/mark/qwen_diffusion/qwen35_sequence_value_retention_mix_result.md`
  - added `scripts/build_qwen35_sequence_value_retention_mix.py` to combine
    route-delta retention rows, explicit candidate value-span rows, and
    train-only sequence-planner rows while filtering eval overlaps.
  - clean corpus:
    `data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum` with
    `387` rows: `154` retention, `173` value-span, and `60` sequence-planner.
  - retained sources exclude the value-span rows embedded in route-delta
    retention, so value grounding is represented explicitly rather than
    accidentally duplicated.
  - independent overlap audit against public multi-call, synthetic analogues,
    heldout seed multi-call, and heldout policy targets is `0` exact overlaps
    and `0` user-prompt overlaps.
  - one-step QLoRA gate from checkpoint-275 saves successfully with
    `train_loss=1.2486` and `16.22s` runtime at block size `1536`.
  - `5/10` sweep trained cleanly with final `train_loss=0.5634`.
    Heldout policy-target eval does not promote: checkpoint-5 reaches raw valid
    JSON `1/12`, constrained sequence `5/12`, and constrained args `0/12`;
    checkpoint-10 reaches raw valid JSON `1/12`, constrained sequence `4/12`,
    and constrained args `0/12`.
  - implication: this is the best raw-valid JSON movement so far without
    aggregate constrained-sequence regression at checkpoint-5, but argument
    grounding did not move. The next branch should avoid more broad mixing and
    instead isolate/compose the positive deltas.

The Alpaca LoRA is functional but far behind the released Fast-dLLM v2 1.5B
checkpoint. That is expected; it is a plumbing proof, not the final training
recipe.

## Principle

Do not optimize for benchmark reproduction alone. Optimize for agentic failure
modes:

- malformed tool calls
- invalid JSON
- wrong function choice
- repeated action loops
- failure to stop
- lossy reasoning under large blocks
- code edits that do not apply
- output that looks fluent but violates schema

## Phase 1: SGLang Teacher / Reference Serving

Goal: put a strong Qwen3.6-family AR model behind an OpenAI-compatible local
endpoint so it can act as:

- label generator
- repair/verifier for tool-call data
- AR quality baseline
- logit/behavior teacher for later distillation
- reference implementation for Qwen tool-call formatting

Preferred server stack:

- SGLang first. Local notes and upstream support suggest Qwen3.6 support is
  better there than in our current vLLM path.
- Current local `.venv-sglang` is `sglang==0.5.9`; Qwen3.6 serving should use
  `sglang>=0.5.10` before serious 27B work.

Teacher selection policy:

Use **Qwen3.6-27B** as the teacher/reference for the eval/data loop. The teacher
should be served in whichever Qwen3.6-27B precision/profile gives the best
quality-throughput-memory tradeoff on the RTX 5090:

- FP8 first if it fits with reduced context and acceptable cache headroom.
- NVFP4/Q4 fallback is acceptable and likely practical for local 5090 serving.
- MTP/speculative decoding should be enabled once validated for this model/server
  path.
- Fast attention/GEMM backends should be used when stable on Blackwell.
- GX10/GB10 is the backup for capacity-heavy checks, but the preferred teacher
  loop should run on the 5090 if quality and speed are acceptable.

Teacher profile priority:

1. `Qwen/Qwen3.6-27B-FP8` on the RTX 5090 with reduced context.
2. Qwen3.6-27B NVFP4 / Q4 variant if FP8 is too tight.
3. Same Qwen3.6-27B profile with MTP/speculative and fast attention enabled.
4. GX10/GB10 for capacity-heavy 27B checks if local 5090 serving is unstable.

Speed knobs to expose:

- MTP/speculative decoding when supported by the model/server path.
- `--attention-backend fa3` or another proven fast backend on Blackwell.
- `--fp8-gemm-backend auto` or a backend validated on 5090.
- `--fp4-gemm-backend auto` / NVFP4 backend for Q4 fallback.
- reduced `--context-length` first, then increase only after stable load.
- constrained JSON/tool-call parser options for agentic evals.
- small `max-running-requests` and conservative memory fraction until stable.

Exit gate:

- SGLang serves Qwen3.6-27B-class teacher locally or on GX10.
- A simple OpenAI-compatible chat request succeeds.
- Tool-call formatting works with the Qwen parser/template.
- Throughput and VRAM/memory use are recorded.

Current result:

- Local RTX 5090 NVFP4 serving works with SGLang `0.5.14`, Triton attention,
  CUTLASS FP4 GEMM, MTP/NEXTN enabled, one running request, radix cache
  disabled, and CUDA graph disabled. The 4k profile is enough for tool-call
  slices; the Qwen Code repo-edit harness needs the 8k profile with
  `MAX_TOTAL_TOKENS=8192`.
- The endpoint returns `/v1/models` and chat completions when Qwen thinking is
  disabled through `chat_template_kwargs.enable_thinking=false`.
- The teacher gets 48/48 exact tool-name selection on the synthetic one-call
  held-out probe.
- Argument-level scoring is now wired. Current teacher result:
  - synthetic one-call held-out: 48/48 exact arguments and schema-valid
  - public Hermes one-call slice: 21/24 exact tool sequence, 18/24 exact
    arguments, 23/24 schema-valid
- Public Hermes multi-call slice, 12 cases with 2-3 calls each:
  - 12/12 valid tool-call emissions
  - 11/12 exact tool sequence and exact tool-name multiset
  - 10/12 exact arguments
  - 0 repeated-call loops, 1 extra tool, 1 missing tool
- Synthetic two-step tool-result slice, 10 cases:
  - 10/10 valid tool-call emissions
  - 10/10 exact next-tool sequence and exact arguments
  - 0 repeated, extra, or missing calls
- Strict OpenAI `assistant.tool_calls` plus `role=tool` tool-result slice,
  10 cases:
  - generated native gold calls validate 10/10 exact sequence, exact arguments,
    and schema validity
  - Qwen3.6 with `tool_choice=auto`: 0/10 native tool-call responses, but 10/10
    exact sequence and exact arguments through Qwen text fallback
  - Qwen3.6 with `tool_choice=required`: 10/10 native tool-call responses,
    10/10 exact sequence, 8/10 exact arguments/schema
  - native failures are empty-argument calls, not wrong function choices
- Qwen3.5-9B AR baseline runs on Alienware RTX 5080 in 4-bit bitsandbytes:
  - synthetic one-call: 48/48 exact sequence and exact arguments
  - public Hermes one-call: 17/24 exact sequence, 13/24 exact arguments
  - public Hermes multi-call: 11/12 exact sequence, 10/12 exact arguments
  - synthetic tool-result: 10/10 exact sequence and exact arguments
  - public one-call weakness: 7 records with extra calls, including 1 repeated
    action-loop style failure
- Qwen Code official harness is installed locally and can drive the live
  Qwen3.6 SGLang teacher through `scripts/qwen_code_sglang_proxy.py`.
- Qwen Code tiny repo-edit slice, 5 cases:
  - 5/5 cases start with failing tests
  - 5/5 independent final test pass after Qwen Code edits
  - 5/5 changed only the expected source file
  - 0/5 unexpected file changes
  - current forced native-tool workaround makes Qwen Code exit on tool budget
    after success, so final test pass is the primary patch metric
- Synthetic Python codegen slice, 10 cases:
  - Qwen3.6 teacher extracted code in 10/10 cases
  - 8/10 passed static checks
  - 7/10 passed unit tests
  - failures: two no-import instruction violations, one word-wrap logic bug
- Local 1.5B Fast-dLLM strict diffusion baselines are now measured on the same
  tool-call slices:
  - base diffusion init: 0 strict hits across synthetic one-call, public
    one-call, public multi-call, and synthetic tool-result
  - public-data LoRA: 0 strict hits across the same slices
  - synthetic one-call LoRA, train-style prompt: 17/48 exact sequence and 10/48
    exact arguments on synthetic one-call, but 0 exact sequence on public
    one-call, public multi-call, and synthetic tool-result
  - appended strict instruction increases wrapper emission but adds
    extra/repeated calls and drops synthetic exact arguments to 4/48
- MTP now fits on the 5090. CUDA graph plus MTP also fits at batch 1, but
  SGLang `0.5.14` fails in the hybrid-attention speculative verification path
  on first generation, so CUDA graph is a runtime compatibility blocker rather
  than a VRAM blocker.

## Phase 2: Agentic Eval and Data Loop on Qwen3.5/3.6

Goal: build eval/data plumbing around the actual target family, not only the
Qwen2.5/Fast-dLLM lab model.

Primary model set:

- SGLang-served Qwen3.6-27B AR teacher/reference.
- Qwen3.5-9B as first real GDN diffusion target.
- Qwen3.5-4B only as an architecture/debug smoke target.
- Local Fast-dLLM/Qwen2.5-1.5B remains a cheap sampler/objective lab, not the
  main target.

Eval set:

1. **Strict JSON/tool-call formatting**
   - model must emit exactly one JSON object
   - schema validation
   - no prose before/after
   - nested arguments and string escaping

2. **Function-choice tests**
   - choose correct tool from 3-8 tools
   - include only required args
   - avoid hallucinated args

3. **Multi-step tool traces**
   - two or three sequential calls
   - previous observation must affect next call
   - detect loops/repeated calls

4. **Code generation**
   - HumanEval/MBPP or small local tests
   - exact runnable code, not just explanation

5. **Patch generation**
   - small repo edit tasks
   - diff applies cleanly
   - tests pass

6. **Existing generic checks**
   - GSM8K limited
   - IFEval limited
   - unresolved mask count
   - repetition/truncation rate
   - tokens/s

Exit gate:

- Qwen3.6 teacher/reference passes the local tool-call eval and can label/repair
  public examples.
- Qwen3.5-9B AR baseline is measured on the same eval.
- Failure modes are categorized enough to train against.
- Qwen3.6 closeout metrics are initialized:
  - `AR_Q36`: local AR Qwen3.6-27B teacher/reference
  - `DIFF_INIT_Q36`: converted diffusion Qwen3.6 before diffusion training
  - `DIFF_TRAIN_Q36`: trained diffusion checkpoints
- SWE-bench Verified slices are not run until tool-call and patch-harness gates
  are stable.

Scheduling rule:

- Default to async teacher/student flow. Use Qwen3.6-27B as an offline labeler,
  repairer, and reference scorer that writes JSONL artifacts. The 27B teacher
  and 9B student do not need to be alive at the same time unless a future
  experiment explicitly needs online KL/logit distillation or live judge calls.

## Phase 3: Agentic/Code Data Instead of Alpaca

Goal: train on the behavior we actually need.

Public candidate data:

- Hermes function-calling v1: open, schema/tool-call examples.
- Glaive function-calling v2: open, mixed tool/no-tool examples.
- ToolACE: open, multi-turn tool-use traces.
- ToolBench / ToolLLM: larger real-API tool-use trajectories.
- xLAM function-calling 60K: useful, but gated in the current HF environment;
  use only when authenticated access is available.
- BFCL: evaluation gate, not a training set.

Generated/teacher data:

- Qwen3.6 teacher rewrites public examples into Qwen tool-call chat format.
- Qwen3.6 teacher repairs invalid JSON/tool calls.
- Qwen3.6 teacher generates “think / tool / observation / final” traces where
  allowed by the target format.
- Hard negatives from failed local eval cases.

Data requirements:

- JSON-schema constrained outputs
- function-call / tool-call conversations
- coding instruction data tied to tools
- repo-edit traces
- patch generation examples
- some general instruction data to prevent narrow collapse

Do not rely on Alpaca as the main corpus. Alpaca is useful only as a plumbing
smoke test.

## Phase 4: Better Objective

Fast-dLLM’s core recipe is AR initialization plus masked-token CE on ground-truth
tokens. For agentic behavior, we should test adding explicit AR-teacher
distillation:

- masked-token cross-entropy against gold tokens
- KL/logit distillation from frozen AR Qwen teacher on masked positions
- extra weighting for structural tokens:
  - `{`, `}`, `[`, `]`, `:`, `,`
  - quote tokens
  - tool/function names
  - stop and boundary tokens
- optional sequence-level checks for JSON validity and call format

Hypothesis:

Agentic tasks need symbolic precision and causal ordering. Ground-truth CE alone
may not preserve enough AR behavior after block diffusion conversion.

## Phase 5: Block-Size Curriculum

Goal: avoid jumping straight into large-block denoising that smears action order.

Initial curriculum:

- block size 1 or 4: near-AR behavior
- 8 / 16: early parallelism
- 32: current Fast-dLLM default
- 64 / 128 only after structured-output metrics are stable

Track metrics per block size:

- task score
- invalid JSON rate
- repeated action rate
- unresolved mask rate
- denoising steps
- tokens/s

## Phase 6: GDN Qwen3.5 / Qwen3.6 Target Sequence

The main target loop should move to Qwen3.5/3.6 as soon as the eval/data plumbing
exists. The 1.5B path remains useful for cheap sampler debugging, but it should
not dominate the roadmap.

Target sequence:

1. Qwen3.6-27B AR teacher/reference via SGLang.
2. Qwen3.5-9B AR baseline and first real GDN LoRA/QLoRA target.
3. Qwen3.5-4B only when a cheap GDN architecture smoke test is needed.
4. Qwen3.6-27B diffusion LoRA/selective adapter after the 9B loop proves out.

Why 9B before 4B:

- 4B is too weak to be a serious teacher or quality target.
- 9B is still plausibly trainable with LoRA/QLoRA on the available hardware.
- 9B exercises the same GDN/full-attention hybrid family as the 27B target.
- 9B quality is more likely to make agentic eval trends meaningful.

GDN starting strategy:

- Option A first:
  - keep GDN causal
  - use it as cross-block state carrier
  - bidirectionality comes from full-attention layers inside block
  - snapshot GDN state at block boundaries
- Option B only if needed:
  - add backward within-block GDN scan
  - more expensive and more implementation risk

GDN diffusion research stance:

- There is no mature, well-studied recipe for converting Qwen3.5/Qwen3.6-style
  Gated DeltaNet hybrids into diffusion LMs. Treat Qwen2.5/Fast-DLLM papers and
  repos as objective/sampler references, not as an implementation recipe for
  the target architecture.
- The current v0 bridge is a baseline hypothesis, not the final method:
  full-attention layers receive block-diffusion masking, while GDN layers stay
  causal and carry state. Its value is that it trains and evaluates; it should
  be challenged by GDN-specific ablations.
- Do not spend most 5090 time on larger data mixes until at least one
  architecture or sampler ablation improves a full-schema gate. The recent data
  replay runs show that small curriculum changes can preserve projected metrics
  but rarely improve model-only raw behavior.

GDN ablation ladder:

1. LoRA target ablations:
   - attention-only adapters
   - GDN-only adapters: `in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj`
   - mixed attention + GDN adapters, current default
   - optional structural embedding/head update for `|<MASK>|`, `<tool_call>`,
     JSON punctuation, and stop-boundary tokens
2. Masking/state ablations:
   - current Option A causal GDN plus diffusion-masked full attention
   - causal GDN with noisy-block stop-gradient or clean-prefix state isolation
   - local dual-pass GDN inside a denoising block: forward prefix scan plus
     reverse/block-local scan, evaluated without serving cache first
   - full Option B bidirectional GDN only after local dual-pass shows value
3. Objective ablations:
   - block-size `704/896/1024` for tool-call spans
   - structural-token and argument-span loss weights
   - planner/repair rows as auxiliary low-ratio data, not main signal
   - teacher-forced Qwen3.6 outputs only after raw public one-call and
     tool-result gates do not regress
4. Sampler/cache ablations:
   - full-context sampling remains correctness baseline
   - implement GDN boundary-state cache only after a raw/model-only quality gain
     appears
   - compare denoising steps and block sizes against repeated-call and
     unresolved-mask failures, not just tokens/s

Promotion rule for GDN innovation:

- A GDN-specific change must beat or tie active checkpoint-275 on the full
  one-call, public multi-call projection, and tool-result sweep, and improve at
  least one model-only/raw metric such as public one-call raw exact arguments,
  public multi-call raw same-call-count, repeated-call rate, or tool-result raw
  exact arguments.
- If a change only preserves the deterministic projected top line, document it
  as infrastructure or data-shaping progress, not as a promoted diffusion model.

Scaling-aware evidence policy:

- Treat one-step and very short `5/25`-step runs as fit, memory, and regression
  gates only. They can reject a specific pressure setting for promotion, but they
  do not prove that a mechanism is a dead end.
- A mechanism should only be called weak after at least one dose curve over
  checkpointed training steps and update pressure, for example LR
  `5e-6/1e-6/5e-7`, checkpoints `25/50/75/100`, and the same public/train/heldout
  one-call eval slices.
- A mechanism should only be called unpromising after a broader data-scale check:
  the small grounded `16`-row slice, then a `100+` row teacher/grounded slice, then
  agentic public multi-call and tool-result gates. The current small slice is
  enough to catch obvious damage, not enough to estimate asymptotic quality.
- Promotion still needs raw/model-only improvement. Longer training that merely
  fits the projection target or preserves constrained postprocessing remains a
  useful scaffold, not the agentic diffusion model we want.
- This policy follows the general LLM scaling lesson from Kaplan-style neural
  language model scaling laws, Chinchilla compute/data scaling, and Fast-DLLM
  v2's much larger AR-to-diffusion adaptation budget: quality claims need
  training-token and compute context, not isolated tiny-step anecdotes.

First GDN-specific ablation gate:

- `qwen35_gdn_lora_ablation_gate_result.md` verifies that GDN-only,
  attention-only, and mixed LoRA branches can all instantiate, train one step,
  and save adapters from the base Qwen3.5 diffusion candidate without loading
  the active mixed checkpoint-275 adapter.
- A follow-up 25-step base-start branch gate also ran under the same 5090
  cgroup cap. Mixed reached the best training loss and best raw public one-call
  sequence count (`2/8`), attention-only reached the best constrained sequence
  count (`7/8`), and all three stayed at `0/8` raw exact arguments and `1/8`
  constrained exact arguments. None is close to active checkpoint-275
  (`8/8` constrained sequence, `5/8` constrained arguments on the same public
  one-call slice), so no branch is promoted.
- First state/masking probe is also complete: `option_a_noisy_block_isolation_v0`
  resets the noisy MDM half at diffusion-block boundaries while keeping the
  clean `x_0` half causal. It is selectable with
  `FAST_DLLM_QWEN3_5_GDN_MODE=option_a_noisy_block_isolation_v0` and leaves
  generation/eval causal. A 25-step mixed LoRA run reached `2/8` raw exact
  sequence and `7/8` constrained sequence on the same public one-call slice,
  but train loss was much worse (`10.9004`) and arguments stayed at `0/8` raw,
  `1/8` constrained. Do not promote or extend as-is.
- Clean-state injection probe is complete:
  `FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_injection_v0` records clean
  `x_0` recurrent state at diffusion-block boundaries and initializes each
  noisy block from the previous clean boundary state. It is a better
  state-handling hypothesis than hard reset: one-step loss improved to
  `7.3922`, 25-step train loss was `8.0212`, and the cheap public one-call gate
  reached `7/8` constrained sequence. It still did not move argument exactness
  (`0/8` raw, `1/8` constrained), so do not promote.
- Clean-state plus mild structural objective probe is complete: structural
  weight `1.25` plus argument-span weight `1.5` trained cleanly with loss
  `8.2204`, but did not improve the public one-call gate (`1/8` raw sequence,
  `0/8` raw arguments, `7/8` constrained sequence, `0/8` constrained
  arguments). Do not scale naive structural-token loss pairing.
- Clean-state local dual-pass GDN probe is complete:
  `FAST_DLLM_QWEN3_5_GDN_MODE=option_a_clean_state_dualpass_v0` adds a reverse
  local noisy-block pass and averages it with the clean-state forward pass. It
  trained and saved, but regressed: train loss `8.9085`, public raw sequence
  `0/8`, constrained sequence `6/8`, constrained arguments `1/8`. Do not scale
  this reverse-pass averaging variant.
- Clean-state value-copy objective hook is complete:
  `scripts/fastdllm_value_copy_token_ids.py` derives scalar argument-value token
  IDs from the selected curriculum and the launcher exposes
  `VALUE_COPY_LOSS_WEIGHT`. On the model-repair curriculum it found `227` tool
  calls, `869` scalar values, `241` unique scalar values, and `410` token IDs.
  A clean-state value-copy weight-`2.0` plus argument-span-`1.5` 25-step gate
  trained cleanly but regressed train loss to `8.8879` and did not improve the
  cheap public one-call gate (`0/8` raw sequence, `0/8` raw arguments, `7/8`
  constrained sequence, `1/8` constrained arguments). Keep the hook, but do not
  scale naive token-ID value weighting as-is.
- Clean-state aligned value-span objective hook is complete:
  the model now also supports `FASTDLLM_VALUE_SPAN_LOSS_WEIGHT` and
  `FASTDLLM_VALUE_SPAN_TOKEN_IDS`, and the launcher exposes
  `VALUE_SPAN_LOSS_WEIGHT`, `VALUE_SPAN_TOKEN_IDS`, and
  `VALUE_SPAN_TOKEN_MANIFEST`. This uses the same scalar argument-value token
  extraction as value-copy, but only boosts those labels inside the derived
  `arguments ... </tool_call>` span. A one-step smoke confirmed `41`
  argument-span labels and `18` aligned value-span labels in the first batch.
  The 25-step gate trained cleanly with loss `8.1482`, improving over global
  value-copy loss but not behavior: public one-call raw exact sequence was
  `1/8`, raw exact arguments `0/8`, constrained exact sequence `7/8`, and
  constrained exact arguments `0/8`. Keep the hook as infrastructure, but do
  not scale scalar-token weighting further without teacher-KL or full-span
  reconstruction.
- Clean-state argument-span mask-forcing hook is complete:
  the model now also supports `FASTDLLM_ARGUMENT_SPAN_MASK_PROB`, and the
  launcher exposes `ARGUMENT_SPAN_MASK_PROB`. The hook reuses the existing
  `arguments ... </tool_call>` span boundary IDs and forces a sampled subset of
  argument-span labels into the masked MDM branch before the complementary
  branch is constructed. A p=`1.0` smoke forced `41/41` argument-span labels and
  trained, but the 25-step gate was too harsh: train loss `9.9601`, public
  one-call raw exact sequence `0/8`, constrained exact sequence `6/8`, and
  constrained exact arguments `0/8`. A p=`0.5` base-start gate forced `24/41`
  argument-span labels in the smoke and reached train loss `7.7339`, but only
  matched the clean-state public gate (`1/8` raw exact sequence, `0/8` raw
  arguments, `7/8` constrained sequence, `1/8` constrained arguments). A
  25-step p=`0.5` continuation from active checkpoint-275 trained cleanly with
  loss `2.1003`, but regressed public constrained exact arguments from the
  active baseline's `5/8` to `2/8`. Keep the hook as infrastructure; do not use
  this continuation setting as a promotion path. A gentler p=`0.1` continuation
  from active checkpoint-275 at LR `5e-6` forced `5/41` argument-span labels in
  the smoke, trained cleanly with loss `2.2841`, and improved constrained exact
  arguments versus p=`0.5` to `3/8`, but still missed the active `5/8` argument
  baseline and dropped constrained exact sequence from `8/8` to `7/8`. Stop
  hard-mask sweeps here and move to teacher-KL/full-span targets.
- Smaller hard clean-repair/full-span replay from active checkpoint-275 is also
  complete and negative. A capped curriculum
  `data/qwen35_9b_toolcall_model_repair_clean_hard24_curriculum` contains 246
  rows: 147 original label-aware rows, 80 prior model-repair rows, and 19
  accepted clean-repair rows. A 25-step continuation at LR `5e-6` from
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
  trained cleanly to loss `2.0963`, but public one-call regressed to raw exact
  sequence / arguments `2/8` / `1/8` and constrained exact sequence /
  arguments `7/8` / `3/8` versus active `8/8` / `5/8`. Do not scale
  repair/full-span replay as the next main-generator recipe; use repair rows
  as diagnostics or a separate repair adapter candidate unless paired with a
  teacher-KL or span-local objective.
- Grounded one-call constrained projection is promoted as a sampler-side
  diagnostic while the active trained adapter stays checkpoint-275. The
  projector now extracts request-evidence weekly schedules, ID-like strings,
  and a few low-confidence contextual strings before trusting corrupted parsed
  scalars. Rescoring the active checkpoint improves public one-call constrained
  sequence / arguments from `8/8` / `5/8` to `8/8` / `8/8`; Qwen3.6
  teacher-train one-call improves from `10/12` / `5/12` to `10/12` / `6/12`;
  teacher-heldout improves from `8/8` / `4/8` to `8/8` / `6/8`; synthetic and
  OpenAI-style tool-result remain `10/10` / `8/10` and `10/10` / `9/10`; and
  public multi-call constrained-draft sequence-preserve stays `7/12` / `4/12`.
  This is not model-only learning, but it gives a concrete target for
  generation-time constrained span filling and is a better next systems lever
  than another broad repair replay.
- Grounded span-fill curriculum gate is complete:
  `scripts/build_toolcall_grounded_spanfill_curriculum.py` converts exact
  grounded one-call projection rows into supervised repair/span-fill examples.
  The block-`1024` train-slice build
  `data/qwen35_9b_toolcall_grounded_spanfill_teacher_train_b1024_curriculum`
  accepted `16` rows with full label retention, recovering longer schedule rows
  that block `896` rejected. A one-step continuation from active checkpoint-275
  at LR `5e-6` trained cleanly on the local RTX 5090 under the 28G cgroup cap
  with train loss `3.069`. Eval ties the active grounded constrained top line
  but does not improve raw model-only behavior: public one-call remains raw
  `3/8` sequence and `2/8` arguments, teacher-train remains raw `2/12` /
  `2/12`, and teacher-heldout remains raw `1/8` / `0/8`; constrained scores
  are `8/8` / `8/8`, `10/12` / `6/12`, and `8/8` / `6/8`. Do not promote this
  adapter. Result note:
  `qwen35_9b_grounded_spanfill_curriculum_result.md`.
- Value-span mask forcing is now available as a sharper objective hook:
  `FASTDLLM_VALUE_SPAN_MASK_PROB` in the local Qwen3.5 bridge and
  `VALUE_SPAN_MASK_PROB` in the QLoRA launcher. Unlike prior whole
  argument-span mask forcing, it only forces labels that are inside the derived
  `arguments ... </tool_call>` span and also match dataset-derived scalar
  value token IDs. The first one-step grounded b1024 continuation from
  checkpoint-275 forced `17` value labels, not the whole `41`-label argument
  span, and trained cleanly to loss `3.5501`. Eval tied the active grounded
  scores on public one-call (`3/8` / `2/8` raw, `8/8` / `8/8` constrained),
  teacher-train (`2/12` / `2/12` raw, `10/12` / `6/12` constrained), and
  teacher-heldout (`1/8` / `0/8` raw, `8/8` / `6/8` constrained). Do not
  promote; keep as a cleaner span-local ablation hook.
- The guarded value-span-only 25-step continuation is complete and negative.
  With argument-span loss neutral at `1.0`, value-span loss `2.0`,
  `VALUE_SPAN_MASK_PROB=1.0`, LR `5e-6`, and block `1024`, the first debug
  batch forced only `17` value labels and weighted only `17` labels, then
  trained to loss `2.0682`. Public one-call raw regressed to `2/8` sequence and
  `1/8` arguments while constrained stayed `8/8` / `8/8`; teacher-train raw
  regressed to `1/12` / `1/12` while constrained args rose to `7/12`;
  teacher-heldout constrained regressed to `7/8` sequence and `5/8` arguments.
  Do not promote or scale `p=1.0` value-span masking as-is.
- True value-span label-only training is now available as a safer span-local
  objective hook: `FASTDLLM_VALUE_SPAN_LABEL_ONLY` in the local Qwen3.5 bridge
  and `VALUE_SPAN_LABEL_ONLY` in the launcher. It keeps the full rendered
  tool-call sequence as context but drops non-value assistant labels before MDM
  masking, so only grounded scalar value tokens contribute loss. A one-step
  continuation from checkpoint-275 with `VALUE_SPAN_LABEL_ONLY=1`,
  `VALUE_SPAN_MASK_PROB=1.0`, neutral argument/value weights, LR `5e-6`, and
  block `1024` reduced the first batch from `55` assistant labels to `17`
  value labels, forced those `17` labels, and trained to loss `0.7236`. Eval
  tied the active grounded scores on public one-call (`3/8` / `2/8` raw,
  `8/8` / `8/8` constrained), teacher-train (`2/12` / `2/12` raw, `10/12` /
  `6/12` constrained), and teacher-heldout (`1/8` / `0/8` raw, `8/8` / `6/8`
  constrained). Do not promote the one-step adapter, but this hook is the next
  safer objective candidate.
- The value-span label-only 25-step continuation is complete and mixed, but not
  promotable. With neutral argument/value loss weights, `VALUE_SPAN_LABEL_ONLY=1`,
  `VALUE_SPAN_MASK_PROB=1.0`, LR `5e-6`, and block `1024`, it trained to loss
  `0.3459`. Public one-call raw tied active at `3/8` sequence and `2/8`
  arguments, but constrained exact arguments regressed from `8/8` to `7/8`.
  Teacher-train raw regressed to `1/12` / `1/12`, while constrained sequence
  improved to `11/12` and constrained arguments stayed `6/12`. Teacher-heldout
  raw improved to `2/8` sequence and `1/8` arguments, while constrained stayed
  `8/8` / `6/8`. Do not promote step 25; keep the hook and next test earlier
  checkpoints or lower update pressure.
- The value-span label-only 5-step continuation is also negative. With the same
  settings as step 25, it trained to loss `0.4866`, but public one-call raw
  regressed to `2/8` sequence and `1/8` arguments while constrained stayed
  `8/8` / `8/8`; teacher-train raw regressed to `1/12` / `1/12`; and
  teacher-heldout constrained regressed to `7/8` / `5/8`. This rules out simply
  stopping earlier at the same full-strength value-mask pressure. Next tests
  should lower update pressure or move to teacher-KL/span distillation.
- The lower-pressure value-span label-only dose curve is complete:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_from_ckpt275_b1024_lr1e6_step100`
  trained from active checkpoint-275 at LR `1e-6`, block `1024`, `100` steps,
  with checkpoints `25/50/75/100` archived and evaluated on the same one-call
  slices. Train loss was `0.3614`. Checkpoint `25` still regressed heldout
  constrained recovery to `7/8` / `5/8`, but checkpoints `50` and `75` preserved
  the active constrained top line across public, teacher-train, and
  teacher-heldout. Checkpoint `75` is the best dose point: raw metrics tie active
  (`3/8` / `2/8` public, `2/12` / `2/12` train, `1/8` / `0/8` heldout) while
  teacher-train constrained improves to `11/12` sequence and `7/12` arguments.
  Checkpoint `100` keeps exact counts but raw-valid JSON drifts down (`1/8`
  public, `0/12` train). Do not promote because there is still no raw/model-only
  gain, but treat checkpoint `75` as the current scaling candidate for broader
  eval and larger data, not as evidence that value-span label-only is dead.
- The first larger-data scaling probe for that recipe is complete:
  active checkpoint-275 was evaluated on a 48-row synthetic one-call slice, then
  `44` exact grounded constrained rows were converted into
  `data/qwen35_9b_toolcall_grounded_spanfill_synthetic_onecall48_b1024_curriculum`.
  A value-span label-only continuation from active checkpoint-275 at LR `1e-6`,
  block `1024`, and `75` steps trained to loss `0.3350` in
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_from_ckpt275_b1024_lr1e6_step75`.
  Checkpoint `75` is not promotable because teacher-heldout constrained recovery
  regressed from active `8/8` / `6/8` to `7/8` / `5/8`. But it is the first
  trained branch to improve raw/model-only behavior: public one-call rises from
  active `3/8` sequence and `2/8` arguments to `4/8` and `3/8`, and
  teacher-heldout rises from `1/8` and `0/8` to `2/8` and `1/8`. This supports
  the scaling hypothesis for the cleaner objective while preserving the rule
  that promotion requires no heldout constrained regression. Result note:
  `qwen35_9b_grounded_spanfill_curriculum_result.md`.
- The cheaper synthetic checkpoint-50 follow-up is also complete and not
  promotable. It ties active raw on public (`3/8` / `2/8`) and teacher-train
  (`2/12` / `2/12`) and preserves public constrained `8/8` / `8/8`, but
  teacher-heldout constrained still regresses to `7/8` / `5/8` and raw heldout
  falls back to `1/8` / `0/8`. The useful step-75 raw gain and the preservation
  behavior do not coincide in this single-objective run.
- The first replay/preservation mix is complete:
  `scripts/build_toolcall_grounded_replay_mix.py` built a 76-row mix with `44`
  synthetic grounded span-fill rows plus `32` repeated original grounded
  teacher-train rows. A 100-step value-span label-only continuation at LR
  `1e-6` trained cleanly in
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_synth48_replay_teacher2_from_ckpt275_b1024_lr1e6_step100`
  with train loss `0.3505`. No checkpoint is promotable. Checkpoint `50`
  preserves the raw gain on public (`4/8` / `3/8`) and heldout (`2/8` / `1/8`),
  but regresses public constrained arguments to `7/8` and heldout constrained to
  `7/8` / `5/8`. Checkpoint `100` restores heldout constrained to active
  `8/8` / `6/8` and improves teacher-train constrained args to `7/12`, but raw
  public/train/heldout regress. This confirms the current CE replay objective is
  trading raw copying against constrained retention rather than solving both.
- The staged retention follow-up is complete:
  starting from the replay-mix raw-gain checkpoint `50`, a lower-LR
  teacher-train-only retention continuation trained for 50 steps in
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_from_synth48replay_ckpt50_b1024_lr5e7_step50`.
  Checkpoint `24` is the first broader-eval candidate: public raw improves over
  active checkpoint-275 from `3/8` / `2/8` to `4/8` / `3/8`, public constrained
  remains `8/8` / `8/8`, teacher-heldout constrained remains active `8/8` /
  `6/8`, and teacher-train constrained sequence improves from `10/12` to
  `11/12` while arguments tie at `6/12`. It is not yet the active model because
  raw-valid formatting is still weak (`1/8` public, `0/12` teacher-train) and
  it has not passed multi-call/tool-result slices. Checkpoints `40` and `50`
  show that the retention stage overshoots: by checkpoint `50`, raw public falls
  to `1/8` / `0/8` and teacher-train raw falls to `0/12` / `0/12`, even though
  constrained teacher-train improves to `12/12` / `7/12`.
- The broader checkpoint-24 eval is complete:
  `runs/fastdllm_qwen35_9b_toolcall_grounded_valuespan_labelonly_staged_retention_ckpt24_broad_eval96_modelrepair_agentic`.
  It keeps the one-call gain and heldout constrained guard, but does not replace
  active checkpoint-275 globally. Public multi-call guarded sequence-planner
  projection reaches `11/12` sequence and `9/12` arguments, one argument point
  behind active `11/12` / `10/12`. Text-compatible synthetic tool-result
  constrained recovery is `10/10` / `9/10`, one argument point above active
  `10/10` / `8/10`, but OpenAI-style tool-result constrained recovery is
  `10/10` / `8/10`, one argument point below active `10/10` / `9/10`. Use
  checkpoint `24` as a seed for the next gentle scaling run, not as the new
  active model.
- The first checkpoint-24 anti-regression continuation is complete and should
  not be promoted. `scripts/build_toolcall_checkpoint24_antiregression_mix.py`
  built a 127-row b1024 mix with synthetic grounded rows, teacher retention
  rows, sequence-planner retention rows, text tool-result rows, and native
  OpenAI-style tool-result rows. A low-LR continuation from staged checkpoint
  `24` at LR `2e-7` trained in
  `runs/fastdllm_qwen35_9b_toolcall_checkpoint24_antiregression_mix_from_staged24_b1024_lr2e7_step80`.
  The one-call dose sweep was stopped after checkpoints `10`, `20`, and `40`
  because all three lost the checkpoint-24 public raw gain (`4/8` / `3/8`
  falls back to `3/8` / `2/8`). Checkpoint `40` improves teacher-train
  constrained recovery to `11/12` / `7/12`, but regresses teacher-heldout
  constrained recovery to `7/8` / `5/8`. This says broad anti-regression rows
  should not be pushed through the same value-span-label-only generator update.
  Keep checkpoint `24` as the seed and move protection into a separate
  repair/projection path, two-adapter routing, or a smaller sidecar objective.
- The split-route sidecar scorecard is now captured in
  `qwen35_9b_split_route_sidecar_scorecard.md`. It is not a deployed router or
  promoted single adapter; it is the current upper-bound target from existing
  artifacts. `scripts/write_qwen35_split_route_sidecar_scorecard.py` now also
  writes the executable gate artifacts
  `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.json` and
  `runs/qwen35_9b_split_route_sidecar_eval/route_scorecard.tsv`, plus the
  implementation manifest
  `runs/qwen35_9b_split_route_sidecar_eval/route_manifest.json`. The manifest
  names the shared base model, staged checkpoint-24 generator adapter, active
  checkpoint-275 protection adapter, per-slice input case files, routed summary
  artifacts, and post-processing chains. Run the writer with `--check` to fail
  on route-gate regression. The routed gate currently passes all six
  slices: staged checkpoint `24` handles public/teacher one-call and text
  tool-result (`4/8` / `3/8` public raw, `10/10` / `9/10` text tool-result
  protected), while active checkpoint-275 handles public multi-call (`11/12` /
  `10/12`) and OpenAI-style tool-result (`10/10` / `9/10`). The next
  implementation gate is a runtime router or sidecar repair/projection path
  that reproduces those executable route lines before any further broad
  generator training.
- The manifest replay runner is initialized:
  `scripts/run_qwen35_split_route_sidecar_manifest.py`. It validates the route
  manifest and emits a replay plan under
  `runs/qwen35_9b_split_route_sidecar_eval/replay_plan/`. Current dry-run
  validation is `6` routes, `10` replayable steps, `0` unknown steps, with
  generation commands wrapped in a user `systemd-run` memory scope. This is the
  first concrete runner handoff for regenerating the routed eval lanes without
  manually reconstructing adapter and projection commands. The runner also has
  `--verify-outputs --plan-json <plan.json>` to gate replay outputs. A
  historical-output verification plan has already passed all six route gates in
  `runs/qwen35_9b_split_route_sidecar_eval/historical_verify_plan/route_runner_plan_verification.json`.
- A partial live replay has passed for the smallest route:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice public_one_call --execute`
  regenerated the public one-call staged checkpoint-24 lane in
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_onecall/`.
  The output verifier passed with raw `4/8` sequence, raw `3/8` arguments, and
  protected `8/8` / `8/8`, matching the route gate. This proves the manifest
  runner can regenerate and verify at least one live route, not just inspect
  historical summaries.
- A second partial live replay has passed for the active protection path:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice openai_style_tool_result --execute`
  regenerated the OpenAI-style tool-result route in
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_openai_toolresult/`.
  The generation plus rescore path reproduced raw `6/10` sequence and
  arguments, and protected `10/10` sequence / `9/10` arguments. This proves
  both adapter roles in the split route can be regenerated and gated live.
- The full public multi-call protection chain has also passed live replay:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice public_multi_call_planner --execute`
  regenerated the active checkpoint-275 route in
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/`.
  This exercised generation, sequence-preserving rescore, contextual projection,
  and sequence-planner projection. The verifier passed with protected `11/12`
  sequence and `10/12` arguments, matching the routed multi-call gate.
- The staged checkpoint-24 text tool-result route has passed live replay:
  `scripts/run_qwen35_split_route_sidecar_manifest.py --slice synthetic_text_tool_result --execute`
  regenerated the route in
  `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_synthetic_text_toolresult/`.
  The verifier passed with raw `6/10` sequence, raw `4/10` arguments, and
  protected `10/10` sequence / `9/10` arguments. Staged checkpoint-24 is now
  live-verified on both one-call and text tool-result lanes.
- The remaining teacher one-call staged checkpoint-24 routes have passed live
  replay in `runs/qwen35_9b_split_route_sidecar_eval/live_smoke_teacher_onecall/`.
  Teacher-train verified raw `2/12` sequence and arguments, with protected
  `11/12` sequence / `6/12` arguments. Teacher-heldout verified raw `2/8`
  sequence, raw `1/8` arguments, and protected `8/8` sequence / `6/8`
  arguments. At this point all six split-route scorecard lanes have live replay
  evidence, not only historical-summary evidence.
- The result is a partial architecture signal: LoRA target-family choice alone
  is not enough, hard noisy-block reset is too blunt, clean-state injection is
  viable, structural-token weighting is unhelpful, simple reverse-pass averaging
  is too disruptive, and scalar-token value weighting is too blunt even when
  aligned to argument spans. Aggressive argument-span mask forcing damages the
  promoted adapter, and a lower-LR p=`0.1` replay still fails to preserve the
  active one-call top line. Argument-copy likely needs teacher-KL over argument
  spans or explicit span denoising targets rather than more token-ID reweighting
  or hard mask forcing alone.
- Current innovation stance: because no mature GDN diffusion conversion recipe
  exists, the next useful experiments are not bigger blind replays. Prefer
  controlled ablations that isolate one mechanism at a time: lower-pressure
  value-span label-only denoising over grounded argument values, teacher-KL over
  argument spans, two-stage tool-sequence planning plus argument filling, and
  GDN state handling only when paired with a sharper span objective. Promotion
  still requires a raw/model-only gain, not merely a deterministic projection
  tie.
- Evidence policy: tiny `1/5/25`-step runs are fit and regression gates, not
  proof that a mechanism is dead. A defensible negative result needs a dose curve
  and a data-scale probe unless the setting immediately damages protected
  metrics. The synthetic-48 run gives a positive raw scaling signal but fails the
  promotion guard. Since both synthetic checkpoints `50` and `75` regress
  teacher-heldout constrained recovery, the replay mix was the right next test.
  Its result says not to keep increasing simple CE replay. The staged-retention
  result adds a positive scaling signal because checkpoint `24` keeps the
  heldout constrained guard while improving public raw behavior. The broad eval
  shows the remaining constraint: preserve active multi-call planner arguments
  and OpenAI-style tool-result arguments. The first broad anti-regression dose
  curve failed this by erasing the raw gain and damaging heldout constrained
  recovery at checkpoint `40`. Next branch: separate generator learning from
  repair/projection protection instead of adding more broad CE rows to the same
  adapter.

Primary scaling context:

- `https://arxiv.org/abs/2001.08361`
- `https://arxiv.org/abs/2203.15556`

## Phase 7: Serving and Quantization

Training format:

- keep first training runs simple: bf16 base plus LoRA/QLoRA where needed
- do not make NVFP4 training the first systems problem

Serving/export:

- use SGLang FP8 as reference/quality format when memory allows
- use NVFP4/Q4 as the practical 5090 deployment fallback
- expose generation knobs:
  - block size
  - denoising steps
  - threshold or entropy bound
  - temperature
  - top-p
  - cache on/off

## Immediate Next Goal

Revise the Qwen3.5-9B model-side recipe after the balanced, planner-heavy, and
sequence/value mixes all failed promotion on the heldout policy gate. The
positive signals are now separated: planner-heavy checkpoint-5 briefly moves
constrained exact arguments to `1/12`, while sequence/value checkpoint-5 moves
raw valid JSON to `1/12` and preserves aggregate constrained sequence at
`5/12`. The next branch should isolate or compose these deltas rather than
adding more broad SFT rows. The Qwen3.6 teacher path remains the reference/data
source for larger follow-on loops.

Current first-version status:

- Done:
  - SGLang launch script for Qwen3.6-27B teacher/reference.
  - NVFP4/Q4 fallback path; the verified 5090 profile is NVFP4, MTP, 4k
    context, CUDA graph disabled.
  - speed knobs exposed: MTP/speculative options, attention backend, GEMM
    backend, reduced context, memory fraction, radix/Mamba cache, tool parser.
  - data prep script for Hermes/Glaive/ToolACE seed data, with xLAM optional
    when HF access is available.
  - synthetic one-call strict/function-choice slice.
  - public Hermes one-call and multi-call slices.
  - 10 synthetic two-step tool-result traces.
  - Qwen3.5-9B AR baseline on the one-call, multi-call, and tool-result slices.
  - 10 small Python code-generation tasks and Qwen3.6 teacher baseline.
  - Qwen Code official coding-agent harness smoke against the local teacher.
  - Qwen Code tiny repo-edit tasks with patch/test-pass metrics:
    5/5 final tests pass, 0 unexpected file changes.
  - local 1.5B Fast-dLLM diffusion baselines on the shared tool-call slices,
    including base init, public-data LoRA, and synthetic one-call LoRA.
  - stricter OpenAI `assistant.tool_calls` plus `role=tool` tool-result harness
    and Qwen3.6 teacher baseline.
  - first Qwen3.5-9B agentic diffusion pilot curriculum:
    303 de-duplicated examples from public tool calls, synthetic one-call,
    synthetic tool-result, and Qwen3.6 teacher repo-edit diffs.
  - guarded Qwen3.5-9B diffusion/QLoRA launch script and readiness preflight.
    The local candidate config/tokenizer now loads in the Fast-DLLM training env
    with `model_type=Fast_dLLM_Qwen3_5`, `bd_size=32`, and `|<MASK>|` as one
    special token. The readiness gate now passes.
  - Qwen3.5/GDN Fast-DLLM bridge v0:
    `FAST_DLLM_QWEN3_5_BRIDGE_STATUS="implemented"` and
    `FAST_DLLM_QWEN3_5_GDN_MODE="option_a_causal_gdn_v0"`. Full-attention layers
    receive the block-diffusion mask; GDN layers remain causal and process noisy
    and clean halves independently during MDM training to avoid target-stream
    leakage.
  - raw Qwen3.5 checkpoint index analysis and weight-remap/materialization:
    427 text/LM-head keys kept, 333 vision keys dropped, 15 MTP keys dropped
    for the first text-only pilot. The candidate now has four remapped
    safetensor shards and an exact 427/427 meta key audit.
  - local RTX 5090 QLoRA smoke on the Qwen3.5-9B diffusion candidate:
    - agentic curriculum load/optimizer smoke works and saves adapters, but
      `BLOCK_SIZE=32` windows are prompt-only and produce zero loss
    - short supervised loss-smoke reaches train loss `7.07149658203125` with
      final logged grad norm `62.266544342041016`, proving backward/update/save
      on the current bridge
    - 512-token grouped pilot initially exposed missing gradient-checkpointing
      integration; after patching the model loop, the same shape fits on the
      RTX 5090
    - `DISABLE_GROUP_TEXTS=1` plus `TRUNCATION_SIDE=left` gives the first real
      agentic-curriculum learning signal: 5 steps, train loss
      `5.71199951171875`, final grad norm `19.994108200073242`
    - 100-step mixed agentic pilot completes on the local RTX 5090:
      train loss `5.716379795074463`, runtime `139.1164s`, throughput
      `0.719` steps/s, adapter saved under
      `runs/fastdllm_qwen35_9b_agentic_qlora_pilot_b512_left_step100`
  - first 9B diffusion strict evals:
    - diffusion init, synthetic one-call 2-case slice: 0/2 exact sequence,
      0/2 exact arguments, 0/2 valid tool JSON, 0 unresolved masks, 0 errors
    - 5-step left-truncated adapter, same slice: 0/2 exact sequence,
      0/2 exact arguments, 0/2 valid tool JSON, 0 unresolved masks, 0 errors
    - 100-step mixed adapter:
      - synthetic one-call 8-case slice: 0/8 exact sequence, 0/8 exact
        arguments, 0/8 valid tool JSON, 0 unresolved masks, 0 errors
      - public one-call 8-case slice: 0/8 exact sequence, 0/8 exact arguments,
        0/8 valid tool JSON, 0 unresolved masks, 0 errors
      - public multi-call 4-case slice: 0/4 exact sequence, 0/4 exact
        arguments, 0/4 valid tool JSON, 0 unresolved masks, 0 errors
      - synthetic tool-result 4-case slice: 0/4 exact sequence, 0/4 exact
        arguments, 0/4 valid tool JSON, 0 unresolved masks, 0 errors
    - result is a train/eval plumbing milestone, not an agentic-quality result
  - synthetic-only structural warmup probe:
    - raw and rebuilt synthetic examples have assistant labels in-window
    - direct one-batch model forward at 704 tokens gives nonzero loss
      `5.870865345001221`
    - root cause of the zero-loss synthetic-only runs was Hugging Face JSON
      schema widening for nested `tools`, which inserted many `None` fields and
      bloated rendered prompts until right truncation removed all assistant
      labels
    - LMFlow decoder tokenization now recursively drops `None` fields from
      `messages` and `tools` before chat-template rendering
    - one-step label gate now passes:
      pre-MDM `valid_labels=[51]`, post-MDM `valid_labels=[21, 30]`, train
      loss `5.339972972869873`
    - corrected 192-example synthetic one-call run:
      `runs/fastdllm_qwen35_9b_synthetic_onecall_b704_pruned_step100`, train
      loss `5.1511626625061036`, runtime `172.2698s`, throughput `0.58`
      steps/s
    - cached Fast-dLLM sampler remains invalid for this Qwen3.5 bridge because
      the bridge does not yet implement layer KV cache, so prompt context is
      mostly lost after prefill
    - `scripts/eval_fastdllm_toolcall_cases.py --full-context-sampling` is now
      the correctness eval path for Qwen3.5 bridge probes
    - full-context eval on the 192-example run:
      holdout synthetic one-call `0/8` valid JSON and `0/8` exact sequence;
      train-slice synthetic one-call `2/8` valid JSON and `1/8` exact sequence
    - tiny 8-example overfit run:
      `runs/fastdllm_qwen35_9b_synthetic_onecall_tiny8_b704_pruned_step200`,
      train loss `1.537280468940735`; full-context same-train eval reaches
      `3/8` valid JSON and `2/8` exact sequence, but `0/8` exact arguments
    - format-first curriculum:
      `scripts/build_toolcall_format_curriculum.py` builds
      `data/qwen35_9b_toolcall_format_curriculum` from synthetic one-call data
      with original, single-tool, and explicit-format variants
    - format-first 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_format_b704_step300`, train loss
      `1.2748866283893585`, runtime `521.1449s`, throughput `0.576` steps/s
    - `scripts/eval_fastdllm_toolcall_cases.py --repair-mode schema` now
      reports separate repaired metrics without changing strict scores
    - first nonzero strict exact-argument results for Qwen3.5 diffusion:
      shared synthetic one-call smoke 16-case full-context eval reaches `8/16`
      valid JSON, `6/16` exact sequence, `4/16` exact arguments, and `13/16`
      repaired exact arguments
    - public one-call remains weak after format-only training:
      8-case full-context eval reaches `1/8` valid JSON, `0/8` exact sequence,
      `0/8` exact arguments, while schema repair finds the intended tool in
      `7/8`
    - public-mix curriculum:
      `scripts/build_toolcall_format_public_mix.py` combines 92 compact
      format-curriculum rows, 40 public one-call rows, and 10 deduped exact
      Qwen3.6 teacher public one-call rows into
      `data/qwen35_9b_toolcall_format_public_curriculum`
    - public-mix 896-token one-step label gate:
      `runs/fastdllm_qwen35_9b_toolcall_format_public_b896_debug_step1`,
      pre-MDM `valid_labels=[12]`, post-MDM `valid_labels=[7, 5]`, train loss
      `7.1235`
    - public-mix 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_format_public_b896_step300`, train
      loss `1.8588300728797913`, runtime `646.6325s`, throughput `0.464`
      steps/s
    - public-mix full-context eval:
      Qwen3.6 teacher train `1/12` strict valid JSON, `0/12` strict exact
      sequence, `8/12` repaired exact sequence, `1/12` repaired exact
      arguments; Qwen3.6 teacher heldout `1/8` strict valid JSON, `0/8`
      strict exact sequence, `6/8` repaired exact sequence, `0/8` repaired
      exact arguments
    - public-mix public/synthetic comparison:
      public one-call remains `1/8` strict valid JSON, `0/8` strict exact
      sequence, `1/8` repaired exact arguments; synthetic one-call reaches
      `4/16` strict valid JSON, `3/16` strict exact sequence, `3/16` strict
      exact arguments, `16/16` repaired exact sequence, and `14/16` repaired
      exact arguments
    - label-aware public-mix curriculum:
      `scripts/build_toolcall_labelaware_public_mix.py` renders candidate rows
      through the actual Qwen3.5 Fast-DLLM tokenizer and `fast_dllm_v2`
      template, then keeps only variants whose assistant labels fully survive
      the 896-token right-truncated training window
    - label-aware dataset:
      `data/qwen35_9b_toolcall_labelaware_public_curriculum` has 141 deduped
      rows from 148 raw candidates; accepted rendered length min/p50/p90/max
      is `239 / 450 / 779 / 890`, accepted kept assistant labels
      min/p50/p90/max is `24 / 38 / 74 / 315`, with `0` zero-label and `0`
      partial-label accepted rows
    - label-aware 896-token one-step gate:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_public_b896_debug_step1`,
      pre-MDM `valid_labels=[33]`, post-MDM `valid_labels=[15, 18]`, train
      loss `5.563186168670654`
    - label-aware 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_public_b896_step300`,
      train loss `1.860784117380778`, runtime `646.7168s`, throughput
      `0.464` steps/s
    - label-aware full-context eval:
      Qwen3.6 teacher train `1/12` strict exact sequence, `0/12` strict exact
      arguments, `8/12` repaired exact sequence, `1/12` repaired exact
      arguments; Qwen3.6 teacher heldout remains `0/8` strict exact sequence
      and `0/8` strict exact arguments, with `6/8` repaired exact sequence
    - label-aware public/synthetic comparison:
      public one-call improves to `1/8` strict exact sequence but remains
      `0/8` strict exact arguments; synthetic one-call drops to `1/16` strict
      exact sequence and `0/16` strict exact arguments, while schema repair
      reaches `16/16` exact sequence and `16/16` exact arguments
    - argument-focused curriculum:
      `scripts/build_toolcall_argument_curriculum.py` adds explicit
      public/Qwen3.6 teacher argument variants on top of label-aware originals:
      exact function-call copy, request plus selected function plus arguments
      JSON, and argument key-value reconstruction
    - argument-focused dataset:
      `data/qwen35_9b_toolcall_argument_curriculum` has 280 deduped rows:
      147 label-aware originals plus 141 accepted argument variants, all with
      full assistant-label retention at 896 tokens; rendered length
      min/p50/p90/max is `239 / 459 / 750 / 890`
    - argument-focused 896-token one-step gate:
      `runs/fastdllm_qwen35_9b_toolcall_argument_b896_debug_step1`,
      pre-MDM `valid_labels=[46]`, post-MDM `valid_labels=[37, 9]`, train
      loss `7.094879627227783`
    - argument-focused 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_argument_b896_step300`, train loss
      `1.872867987950643`, runtime `646.5903s`, throughput `0.464` steps/s
    - argument-focused full-context eval:
      Qwen3.6 teacher train remains `1/12` strict exact sequence and `0/12`
      strict exact arguments, while repaired exact sequence rises to `10/12`
      and repaired exact arguments drops to `0/12`; Qwen3.6 teacher heldout
      reaches `1/8` strict exact sequence but still `0/8` exact arguments
    - argument-focused public/synthetic comparison:
      public one-call regresses to `0/8` strict exact sequence and `0/8`
      strict/repaired exact arguments, with `6/8` repaired exact sequence;
      synthetic one-call reaches `2/16` strict exact sequence, `1/16` strict
      exact arguments, `16/16` repaired exact sequence, and `15/16` repaired
      exact arguments
    - sampled argument failures:
      missing colons, duplicated substrings, malformed nested arrays, dropped
      `arguments` keys, and occasional `<think>` leakage. Explicit argument-copy
      rows alone are not enough; the model needs constrained decoding and/or
      structural-token/argument-span loss weighting.
    - constrained tool-call decoding:
      `scripts/eval_fastdllm_toolcall_cases.py` now supports
      `--constrained-tool-decoding`, and
      `scripts/rescore_fastdllm_toolcall_outputs.py` can apply repair and
      constrained metrics to existing raw eval JSONL outputs without re-running
      GPU generation
    - constrained decoder scope:
      uses generated text, prompt messages, and available tool schemas only;
      it does not use gold tool calls or gold arguments. It reports separate
      `constrained_*` metrics so strict model quality remains visible.
    - constrained public one-call results:
      label-aware checkpoint improves from strict/repaired exact args
      `0/8` / `0/8` to constrained exact args `2/8`, with constrained exact
      sequence `7/8`; argument-focused checkpoint reaches constrained exact
      args `1/8` and constrained exact sequence `6/8`; public-mix checkpoint
      reaches constrained exact args `2/8` and constrained exact sequence `7/8`
    - constrained teacher results:
      label-aware checkpoint reaches constrained exact args `3/12` on teacher
      train and `1/8` on teacher heldout; argument-focused checkpoint reaches
      constrained exact args `1/12` on teacher train and `1/8` on teacher
      heldout
    - constrained synthetic sanity:
      constrained exact sequence remains `16/16` on the synthetic smoke slices,
      with constrained exact args `14/16-15/16` on the recent public/argument
      checkpoints
    - structural-token weighting probe:
      `models/qwen3.5-9b-fastdllm-init/modeling.py` now has an opt-in weighted
      CausalLM loss path driven by `FASTDLLM_STRUCTURAL_LOSS_WEIGHT` and
      `FASTDLLM_STRUCTURAL_TOKEN_IDS`; the launcher derives token IDs with
      `scripts/fastdllm_structural_token_ids.py` only when enabled
    - structural-token one-step gate:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw_b896_step1_gate`
      used weight `2.0` on the label-aware public curriculum, saw
      pre/post-MDM valid labels `[27] -> [8, 19]`, weighted `9/27`
      structural labels, and produced train loss `8.206977844238281`
    - structural-token 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_b896_step300`,
      block size `896`, max steps `300`, LR `3e-5`, train loss
      `2.6473990964889524`, runtime `646.309s`, throughput `0.464` steps/s
    - structural-token comparable eval at `max_new_tokens=96`:
      public one-call constrained sequence/args `7/8` / `1/8`; Qwen3.6
      teacher train constrained sequence/args `9/12` / `2/12`; Qwen3.6
      teacher heldout constrained sequence/args `8/8` / `1/8`
    - structural-token interpretation:
      heldout constrained sequence improved from the prior label-aware
      checkpoint's `6/8` to `8/8`, but public and teacher-train constrained
      argument exactness each regressed by one case. Raw strict public and
      heldout exact sequence stayed `0/8`, so naive structural token-ID
      weighting is a probe, not the next main recipe.
    - argument-span weighting probe:
      `models/qwen3.5-9b-fastdllm-init/modeling.py` now also supports an
      opt-in per-label argument-span loss mask controlled by
      `FASTDLLM_ARGUMENT_SPAN_LOSS_WEIGHT`,
      `FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS`, and
      `FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS`; the launcher derives the marker
      IDs with `scripts/fastdllm_argument_span_token_ids.py`
    - argument-span one-step gate:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw2_b896_step1_gate`
      used weight `2.0`, saw pre-MDM valid labels `[27]`, argument-span labels
      `15/27`, post-MDM valid labels `[8, 19]`, shifted weighted labels
      `15/27`, and produced train loss `9.805089950561523`
    - argument-span 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw2_b896_step300`,
      block size `896`, max steps `300`, LR `3e-5`, train loss
      `3.9026308631896973`, runtime `646.444s`, throughput `0.464` steps/s
    - argument-span comparable eval at `max_new_tokens=96`:
      public one-call strict sequence/args `2/8` / `2/8`, constrained
      sequence/args `6/8` / `3/8`; Qwen3.6 teacher train strict sequence/args
      `2/12` / `2/12`, constrained sequence/args `8/12` / `3/12`; Qwen3.6
      teacher heldout strict sequence/args `0/8` / `0/8`, constrained
      sequence/args `6/8` / `1/8`
    - argument-span interpretation:
      first current 9B diffusion QLoRA run to move public raw strict exact
      arguments above zero and to `2/8`; public constrained exact arguments
      improve to `3/8`, but constrained sequence coverage drops versus the
      label-aware/structural probes. This is the better trainer-side direction,
      but needs a weight/data-mix sweep and generation-time constraints.
    - combined structural plus argument-span probe:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_structw2_argspanw2_b896_step300`
      used both structural-token and argument-span weights at `2.0`, with
      max-combined per-token weighting. First batch coverage was pre-MDM
      `[33]`, argument-span labels `19/33`, structural labels `9/33`, and
      max-combined weighted labels `24/33`.
    - combined 300-step/eval result:
      train loss `4.07765515645345`, runtime `646.5271s`, throughput `0.464`
      steps/s. Comparable eval at `max_new_tokens=96`: public one-call strict
      sequence/args `2/8` / `0/8`, constrained sequence/args `4/8` / `1/8`;
      Qwen3.6 teacher train strict sequence/args `1/12` / `0/12`, constrained
      sequence/args `9/12` / `3/12`; Qwen3.6 teacher heldout strict
      sequence/args `1/8` / `0/8`, constrained sequence/args `5/8` / `1/8`.
    - combined interpretation:
      negative result for this objective. It preserves some public strict
      sequence but loses the argument-span-only public exact-argument gain
      (`2/8 -> 0/8`) and drops public constrained sequence/args. Do not repeat
      structural `2.0` plus argument-span `2.0` as-is.
    - argument-span weight-1.5 sweep:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw1p5_b896_step300`
      used argument-span weight `1.5` with structural weighting disabled. First
      batch coverage was pre-MDM `[33]`, argument-span labels `19/33`, and
      post-MDM `[5, 28]`. Train loss was `3.0483706188201904`, runtime
      `646.3971s`, throughput `0.464` steps/s.
    - argument-span weight-1.5 comparable eval:
      public one-call strict sequence/args `2/8` / `1/8`, constrained
      sequence/args `7/8` / `3/8`; Qwen3.6 teacher train strict sequence/args
      `2/12` / `1/12`, constrained sequence/args `10/12` / `4/12`; Qwen3.6
      teacher heldout strict sequence/args `1/8` / `1/8`, constrained
      sequence/args `6/8` / `1/8`.
    - argument-span weight-1.5 interpretation:
      best balanced 300-step checkpoint so far under constrained decoding:
      public constrained args tie weight `2.0` at `3/8`, public constrained
      sequence recovers to `7/8`, and teacher-train constrained sequence/args
      improve to `10/12` / `4/12`. Weight `2.0` still has the better public raw
      exact-argument count (`2/8` vs `1/8`), so keep both as comparison points.
    - argument-span weight-3.0 sweep:
      `runs/fastdllm_qwen35_9b_toolcall_labelaware_argspanw3_b896_step300`
      completed the planned high-pressure short probe. No-grouping one-step
      gate passed with pre-MDM `[33]`, argument-span labels `21/33`, and train
      loss `12.57717227935791`; the 300-step comparable run had first-batch
      argument-span labels `19/33`, train loss `5.558430992762248`, runtime
      `646.4547s`, throughput `0.464` steps/s.
    - argument-span weight-3.0 comparable eval:
      public one-call strict sequence/args `1/8` / `0/8`, constrained
      sequence/args `7/8` / `1/8`; Qwen3.6 teacher train strict sequence/args
      `0/12` / `0/12`, constrained sequence/args `10/12` / `1/12`; Qwen3.6
      teacher heldout strict sequence/args `1/8` / `0/8`, constrained
      sequence/args `5/8` / `1/8`.
    - argument-span weight-3.0 interpretation:
      negative result. It preserves some constrained sequence selection but
      regresses exact argument recovery versus both `1.5` and `2.0`. Stop the
      raw argument-span-weight sweep here; next work should move to data mix,
      generation-time constraints, and possibly teacher-KL/value-copy objectives.
    - model-repair eval pass:
      `scripts/eval_fastdllm_toolcall_cases.py` now supports
      `--model-repair-pass` and `--model-repair-max-new-tokens`. The evaluator
      first generates the normal diffusion draft, then asks the same checkpoint
      to rewrite that draft into valid Qwen `<tool_call>` JSON using the
      original request and tool schema. Scores are tracked as `model_repair_*`
      beside raw strict, deterministic repair, and deterministic constrained
      projection metrics.
    - model-repair curriculum:
      `scripts/build_toolcall_model_repair_curriculum.py` builds
      `data/qwen35_9b_toolcall_model_repair_curriculum` from label-aware
      originals plus train-slice raw drafts from five previous 300-step
      checkpoints. The dataset has `227` rows: `147` label-aware originals and
      `80` accepted model-repair rows. Accepted rendered length min/p50/p90/max
      is `239 / 591 / 840 / 890`, kept assistant labels min/p50/p90/max is
      `24 / 41 / 78 / 315`, with `0` accepted zero-label or partial-label rows.
    - model-repair 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300`
      used the repair curriculum with argument-span weight `1.5`, block size
      `896`, max steps `300`, LR `3e-5`. First batch coverage was pre-MDM
      `[29]`, argument-span labels `17/29`, post-MDM `[1, 28]`; train loss was
      `3.2297114753723144`, runtime `646.4761s`, throughput `0.464` steps/s.
    - model-repair comparable eval at `max_new_tokens=96`:
      public one-call raw strict sequence/args `3/8` / `2/8`, constrained
      sequence/args `7/8` / `4/8`, learned model-repair sequence/args
      `4/8` / `2/8`; Qwen3.6 teacher train raw strict sequence/args
      `2/12` / `2/12`, constrained `10/12` / `5/12`, learned model-repair
      `5/12` / `2/12`; Qwen3.6 teacher heldout raw strict sequence/args
      `1/8` / `0/8`, constrained `6/8` / `1/8`, learned model-repair
      `3/8` / `1/8`.
    - model-repair 160-token decode check:
      `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300_eval96_modelrepair160`
      kept the first diffusion draft at `max_new_tokens=96` and only changed
      `--model-repair-max-new-tokens` from `96` to `160`. Valid JSON improved
      slightly, but exact results did not: public learned repair sequence/args
      fell to `2/8` / `1/8`, teacher train was `5/12` / `1/12`, and heldout
      fell to `2/8` / `1/8`. Long schedule cases still hit the 160-token cap,
      so simply decoding the repair pass longer is not the next scaling lever.
    - single-call constrained projection:
      `scripts/eval_fastdllm_toolcall_cases.py` and
      `scripts/rescore_fastdllm_toolcall_outputs.py` now support
      `--constrained-max-calls`. Rescoring the current best model-repair
      checkpoint with `--constrained-max-calls 1` raises public one-call
      constrained sequence/args to `8/8` / `5/8` from `7/8` / `4/8`, keeps
      Qwen3.6 teacher train at `10/12` / `5/12`, and raises heldout to `7/8` /
      `2/8` from `6/8` / `1/8`. This fixes the garage-door case where the raw
      model emitted the correct `open_garage_door` call plus an extra
      `close_garage_door` call. It is still an inference projection, not full
      token-level denoising constraints.
    - argument-diff diagnostic:
      `scripts/diagnose_toolcall_argument_errors.py` compares selected row
      prefixes such as `constrained` against gold arguments by schema path.
      On the current best checkpoint with `--constrained-max-calls 1`, public
      one-call has `3/8` exact-tool-sequence rows with wrong arguments, made up
      of `5` scalar value mismatches and `1` missing required field. Qwen3.6
      teacher train has `5/12` exact-tool-sequence/wrong-argument rows with
      `5` missing required fields, `3` scalar mismatches, and `2` missing tool
      calls. Heldout has `5/8` exact-tool-sequence/wrong-argument rows with
      `9` missing required fields, `4` scalar mismatches, and `1` missing tool
      call.
      Interpretation: after extra-call suppression, the gap is value-copy and
      required-field completion, not mainly JSON syntax. The next curriculum
      should use train-slice hard argument-completion rows only.
    - clean-repair curriculum probe:
      `scripts/build_toolcall_model_repair_curriculum.py` now has optional
      controlled-corruption repair rows via `--clean-repair-cap`,
      `--clean-repair-repeat`, `--clean-repair-sources`,
      `--clean-repair-variants`, and `--clean-repair-onecall-only`.
      The clean dataset
      `data/qwen35_9b_toolcall_model_repair_clean_curriculum` has `294` rows:
      `147` label-aware originals, `80` accepted raw model-repair rows, and
      `67` accepted clean-repair rows. It passed the no-group one-step gate
      with pre-MDM labels `[43]`, argument-span labels `27/43`, post-MDM
      labels `[18, 25]`, and loss `6.055093288421631`.
    - clean-repair 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_modelrepair_clean_argspanw1p5_b896_step300`
      used the clean dataset with argument-span weight `1.5`, block size `896`,
      max steps `300`, LR `3e-5`. First batch coverage was pre-MDM `[61]`,
      argument-span labels `48/61`, post-MDM `[25, 36]`; train loss was
      `2.613187549908956`, runtime `647.0471s`, throughput `0.464` steps/s.
    - clean-repair comparable eval at `max_new_tokens=96`:
      public one-call raw strict sequence/args fell to `1/8` / `0/8`,
      constrained fell to `5/8` / `1/8`, learned model-repair was `3/8` /
      `1/8`; Qwen3.6 teacher train raw strict was `1/12` / `0/12`,
      constrained `9/12` / `1/12`, learned model-repair `2/12` / `2/12`;
      Qwen3.6 teacher heldout raw strict was `1/8` / `0/8`, constrained
      `6/8` / `1/8`, learned model-repair `2/8` / `0/8`.
    - model-repair interpretation:
      strongest current 300-step public raw strict result and first useful
      learned repair signal. It still trails deterministic constrained
      projection on exact arguments and regresses heldout raw exact arguments,
      and the 160-token repair check does not improve exactness, so the next
      step is to scale/clean repair data and keep constrained decoding as the
      fallback rather than treating this as final.
      The cap-80 clean-repair mix is also a negative result: it trains stably
      but over-weights repair syntax enough to regress the main generator and
      constrained metrics. If clean repair rows are retried, use a smaller cap,
      harder variants, or a separate repair adapter.
    - hard argument-completion probe:
      `scripts/build_toolcall_model_repair_curriculum.py` now has optional
      hard argument rows via `--hard-argument-cases`,
      `--hard-argument-outputs`, `--hard-argument-prefix`,
      `--hard-argument-cap`, and `--hard-argument-repeat`. The dataset
      `data/qwen35_9b_toolcall_hard_argument_curriculum` has `239` rows:
      `147` label-aware originals, `80` accepted raw model-repair rows, `12`
      hard argument-completion rows, and `96` format rows. The one-step gate
      passed with pre-MDM labels `[27]`, argument-span labels `15/27`,
      post-MDM labels `[11, 16]`, and loss `6.3488993644714355`.
    - hard argument 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_hardarg_argspanw1p5_b896_step300`
      used argument-span weight `1.5`, block size `896`, max steps `300`, LR
      `3e-5`. First batch coverage was pre-MDM `[28]`, argument-span labels
      `16/28`, post-MDM `[11, 17]`; train loss was `2.5478537392616274`,
      runtime `646.9774s`, throughput `0.464` steps/s.
    - hard argument comparable eval with `--constrained-max-calls 1`:
      public one-call raw strict sequence/args fell to `0/8` / `0/8`,
      constrained was `7/8` / `1/8`, learned model-repair was `1/8` / `1/8`;
      Qwen3.6 teacher train raw strict was `1/12` / `0/12`, constrained
      `11/12` / `1/12`, learned model-repair `2/12` / `2/12`; Qwen3.6
      teacher heldout raw strict was `0/8` / `0/8`, constrained `7/8` /
      `1/8`, learned model-repair `0/8` / `0/8`.
    - hard argument arg-diff:
      under constrained max-1 projection, exact-tool-sequence rows with wrong
      arguments rose to public `6/8`, teacher train `10/12`, and heldout
      `6/8`; the main errors are still missing required fields and scalar
      value mismatches. This regresses versus the prior best diagnostic
      (`3/8`, `5/12`, `5/8`), so the current hard-argument mix widens the
      argument gap.
    - hard argument interpretation:
      negative result for the main generator. It improves teacher-train
      constrained tool sequence selection but does not improve exact argument
      recovery and regresses public raw exactness. Keep hard argument rows as a
      diagnostic or separate repair-adapter candidate; do not heavily mix this
      version into the next main generator run.
    - Qwen3.5/Qwen3.6 architecture check:
      Qwen3.5-9B and Qwen3.6-27B are Gated-DeltaNet hybrid-attention models,
      not Qwen2.5-style dense full-attention transformers. Online
      primary-source configs confirm Qwen3.5-9B has 32 layers and Qwen3.6-27B
      has 64 layers, both repeating
      `linear_attention,linear_attention,linear_attention,full_attention`;
      Qwen2.5-7B is `Qwen2ForCausalLM` / `model_type: qwen2` with no
      `layer_types` linear-attention layout. The local Qwen3.5 bridge uses
      `gdn_mode: option_a_causal_gdn_v0` and includes GDN projections
      `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` in
      the LoRA target list. Future speed work needs GDN recurrent-state cache
      support, not only standard KV-cache logic.
      Rechecked on 2026-06-27: keep Qwen2.5-1.5B as a sampler/objective lab,
      but do not promote Qwen2.5 implementation assumptions into the real
      Qwen3.5/3.6 target loop. Standalone note:
      `qwen35_gdn_vs_qwen25_research.md`.
    - same-recipe 600-step probe:
      `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step600`
      used the current best model-repair corpus, argument-span weight `1.5`,
      block size `896`, max steps `600`, LR `3e-5`, and the GDN-aware LoRA
      target list. Train loss was `1.8730474283297858`, runtime `1293.9047s`,
      throughput `0.464` steps/s.
    - same-recipe 600-step eval with `--constrained-max-calls 1`:
      public one-call raw strict sequence/args was `0/8` / `0/8`,
      constrained `7/8` / `3/8`, learned model-repair `2/8` / `0/8`;
      Qwen3.6 teacher train raw strict was `0/12` / `0/12`, constrained
      `12/12` / `5/12`, learned model-repair `1/12` / `0/12`; Qwen3.6
      teacher heldout raw strict was `0/8` / `0/8`, constrained `7/8` /
      `0/8`, learned model-repair `2/8` / `0/8`.
    - same-recipe 600-step interpretation:
      negative scaling point. It improves teacher-train constrained sequence
      selection but regresses public and heldout exact arguments and learned
      repair. Add checkpoint sweep/early stopping around `250-350` steps before
      longer runs.
    - checkpoint-275 sweep:
      `scripts/eval_fastdllm_toolcall_cases.py` now supports
      `--tokenizer-path`, which lets checkpoint-only PEFT adapter folders use
      tokenizer files from the parent run.
      `scripts/run_fastdllm_toolcall_checkpoint_sweep.sh` wraps the standard
      public/train/heldout eval suite for retained checkpoints and writes a
      `checkpoint_sweep_summary.tsv`. Evaluating
      `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
      with tokenizer path
      `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300`
      gives public raw strict sequence/args `3/8` / `2/8`, constrained
      `8/8` / `5/8`, learned model-repair `4/8` / `2/8`; Qwen3.6 teacher
      train raw strict `2/12` / `2/12`, constrained `10/12` / `5/12`,
      learned model-repair `4/12` / `3/12`; Qwen3.6 teacher heldout raw
      strict `1/8` / `0/8`, constrained `8/8` / `3/8`, learned model-repair
      `3/8` / `1/8`.
    - checkpoint-275 interpretation:
      active best for the current one-call objective. It ties final step-300 on
      public and teacher-train constrained exact arguments and improves heldout
      constrained sequence/args from `7/8` / `2/8` to `8/8` / `3/8`. The next
      run should evaluate `250/275/300/325` before promoting longer training.
    - checkpoint-275 agentic eval:
      public Hermes multi-call with `max_new_tokens=384` and unconstrained call
      count reaches raw strict sequence/args `1/12` / `0/12`, deterministic
      repaired `7/12` / `0/12`, constrained `7/12` / `1/12`, learned
      model-repair `1/12` / `1/12`; Qwen3.6 teacher is `11/12` / `10/12`.
      Synthetic text-compatible tool-result with `--constrained-max-calls 1`
      reaches raw strict `5/10` / `3/10`, constrained `10/10` / `8/10`.
      Synthetic OpenAI-style `role=tool` tool-result reaches raw strict
      `6/10` / `6/10`, constrained `10/10` / `9/10`, learned model-repair
      `5/10` / `5/10`.
    - agentic interpretation:
      tool-result next-action behavior is now a positive signal under
      constrained decoding, but public multi-call is still far below teacher,
      mostly because chained call arguments are wrong. The next data target is
      multi-call continuation and chained argument-copy rows, not more one-call
      wrapper repair.
    - multi-call continuation curriculum:
      `scripts/build_toolcall_multicall_curriculum.py` builds a 467-row mix on
      top of the active model-repair curriculum: 227 base rows plus 240 accepted
      multi-call rows from 56 public-train multi-call records. Row types include
      full multi-call, exact ordered plan rendering, and continuation after
      already-completed calls. The 896-token label-retention audit has zero
      selected rows with zero or partial retained labels.
    - multi-call continuation 300-step run:
      `runs/fastdllm_qwen35_9b_toolcall_multicall_argspanw1p5_b896_step300`
      used block size `896`, argument-span weight `1.5`, dataset grouping
      disabled, and GDN-aware LoRA targets. One-step gate train loss was
      `9.725969314575195`; the 300-step train loss was
      `2.3689375511805215`, runtime `2574.0471s`, throughput `0.117` steps/s,
      with retained checkpoints `275` and `300`.
    - multi-call continuation eval:
      checkpoint-275 on public Hermes multi-call with `max_new_tokens=384`,
      unconstrained call count, deterministic projection, and model-repair pass
      reaches raw strict `0/12` / `0/12`, deterministic repaired `3/12` /
      `0/12`, constrained `4/12` / `1/12`, learned model-repair `3/12` /
      `1/12`. Final checkpoint-300 without the learned repair pass reaches raw
      strict `0/12` / `0/12`, deterministic repaired `5/12` / `0/12`, and
      constrained `5/12` / `0/12`.
    - multi-call continuation interpretation:
      negative result for the main generator. It does not improve exact
      arguments and regresses constrained tool-sequence recovery versus the
      active best multi-call result (`7/12` / `1/12`). Argument diffs are still
      dominated by scalar value mismatches (`19-21`) plus a few missing required
      fields and missing tool calls. Do not scale this first full-chain
      multi-call continuation mix as-is; use smaller argument-repair/extraction
      windows or a separate repair adapter next.
    - separate multi-call repair-adapter probe:
      `scripts/build_toolcall_multicall_repair_curriculum.py` produced a
      434-row repair-only dataset: 120 base repair rows plus 314 accepted
      public multi-call corruption/repair rows, all with full 896-token label
      retention. The 100-step repair adapter
      `runs/fastdllm_qwen35_9b_toolcall_multicall_repair_argspanw1p5_b896_step100`
      trained with loss `3.8560502338409424`, runtime `859.6796s`, throughput
      `0.116` steps/s. Evaluating it as a second-stage repair model over the
      active-best public multi-call drafts gives raw-draft repair-constrained
      `5/12` exact sequence and `3/12` exact arguments; constrained-draft
      repair-constrained gives `5/12` sequence and `2/12` arguments. This is a
      partial argument gain over the active constrained baseline (`7/12` /
      `1/12`) but not promotable because it loses too much tool-sequence
      accuracy. Argument diffs remain scalar-value dominated.
    - fixed-sequence repair-adapter probe:
      `scripts/build_toolcall_sequence_repair_curriculum.py` produced a
      274-row fixed-sequence repair dataset from 56 public train multi-call
      records. The 100-step adapter
      `runs/fastdllm_qwen35_9b_toolcall_sequence_repair_argspanw1p5_b896_step100`
      trained with loss `3.77995418548584`, runtime `855.6122s`, throughput
      `0.117` steps/s. Evaluated on active-best constrained multi-call drafts
      with `--repair-prompt-mode preserve_sequence`, the input draft was
      `7/12` exact sequence and `1/12` exact arguments, while learned repair
      plus constrained projection fell to `3/12` sequence and `1/12` arguments.
      This is a negative result: even fixed-sequence repair prompts/training
      damage tool order.
    - sequence-preserving deterministic projection:
      moved the preserve-order repair idea out of a learned second-stage model
      and into deterministic constrained projection. `scripts/rescore_fastdllm_toolcall_outputs.py`
      now supports `--text-field` and `--sequence-preserving-constrained`; the
      projection preserves repeated function names/order/count from the chosen
      draft and fills arguments per call. After tightening the string extractor
      so parsed string values win over broad request spans, active-best public
      multi-call improves from `7/12` exact sequence and `1/12` exact arguments
      to `7/12` exact sequence and `4/12` exact arguments when run over
      `constrained_assistant`. Running the same projection over raw
      `assistant` gives `6/12` sequence and `3/12` arguments, so the constrained
      draft remains the right starting point. Remaining diffs are `14` scalar
      value mismatches, `3` missing required fields, and `2` missing tool calls.
      A follow-up contextual projection over the same constrained draft adds
      call-local request-evidence fills for quoted IDs, date/time fields, and
      explicit missing required scalar values. It reaches `7/12` sequence and
      `7/12` arguments, with zero exact-sequence rows left wrong on arguments.
      Running the same contextual projection suite over public one-call,
      teacher one-call, and both tool-result slices is neutral, so this is a
      public multi-call scalar-grounding fix rather than a broad metric
      inflator. Result note:
      `qwen35_9b_contextual_projection_suite_result.md`.
      Promote this as the current public multi-call reporting path and the
      blueprint for generation-time constrained `<tool_call>` decoding.
    - multi-call scalar extraction curriculum:
      `scripts/build_toolcall_multicall_scalar_curriculum.py` builds one-call
      argument-repair windows from public multi-call training records. It keeps
      only the relevant single tool schema, includes a short request excerpt,
      corrupts one scalar argument, and targets one corrected `<tool_call>`
      block. The first full build produced `1184` accepted rows from `56`
      public multi-call records with zero rejected rows, zero label-loss rows,
      length min/p50/p90/max `340/551/667/883`, and labels min/p50/p90/max
      `25/45/95/194` at block size `896`. A cgroup-protected one-step QLoRA
      gate on the local 5090 saved an adapter and reached train loss
      `7.224119186401367`. Do not promote that one-step adapter; use this data
      as the next staged/lower-weight scalar-copy candidate before reattempting
      full-chain multi-call training.
    - multi-call scalar repair adapter:
      a 100-step adapter trained on the scalar curriculum
      (`runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step100`)
      reached train loss `5.111620244979858` in `215.4411s`. The new
      per-call evaluator
      `scripts/eval_fastdllm_toolcall_scalar_repair_outputs.py` preserves the
      draft tool sequence, runs one scalar repair generation per parsed call,
      applies a conservative accept policy for missing/noisy/repeated values,
      and then scores the recomposed chain. On public multi-call, input
      constrained drafts are `7/12` exact sequence and `1/12` exact arguments;
      scalar repair alone reaches `7/12` / `3/12`; scalar repair plus
      sequence-preserving constrained projection reaches `7/12` / `5/12`.
      This is a positive two-stage result over the prior best deterministic
      projection (`7/12` / `4/12`), but still not a promoted first-pass
      generator. A same-curriculum 300-step extension trained cleanly to loss
      `3.1057442967096964`, but checkpoint-275 and checkpoint-300 both matched
      the 100-step top line at `7/12` sequence and `5/12` arguments. More
      steps on this exact scalar curriculum are not the next useful lever.
      A deterministic contextual scalar projection prototype then rescored the
      same scalar-repair outputs by extracting call-local quoted IDs and
      date/time values from the request. It closed the exact-sequence
      wrong-scalar gap on the 12-row public slice, reaching `7/12` sequence and
      `7/12` arguments with zero exact-sequence rows left wrong on arguments.
      After adding the explicit missing-required scalar fill, direct contextual
      projection over the constrained draft ties that same `7/12` / `7/12`
      result without running the scalar adapter. This is not a model-only score;
      it is the strongest evidence so far that generation-time constrained
      scalar decoding/per-field extraction is the right next implementation
      direction.
    - model-repair scalar-mix main-generator test:
      `scripts/build_toolcall_modelrepair_scalar_mix.py` built a 355-row mix
      with 227 model-repair rows plus 128 balanced scalar rows. The one-step
      gate and 300-step QLoRA run both trained cleanly, but checkpoint-275
      regressed the target public multi-call gate: sequence-preserving
      constrained projection fell to `4/12` sequence and `2/12` arguments
      versus the active `7/12` / `4/12` path and the scalar-repair two-stage
      `7/12` / `5/12` path. Do not promote direct scalar mixing as the next
      main-generator recipe; keep scalar extraction as a repair/decoding signal
      unless a much lower-ratio replay-heavy run passes an early multi-call
      sequence gate.
    - multi-call gap curriculum:
      `scripts/build_toolcall_multicall_gap_curriculum.py` builds smaller
      public-train rows for the remaining failure classes after contextual
      scalar projection: missing-call recovery from a draft chain and exact
      complex array/object extraction from request evidence. The first gated
      build produced `181` accepted rows from `56` public train multi-call
      records: `137` missing-call rows and `44` complex-extraction rows. At
      block size `896`, accepted rendered length min/p50/p90/max is
      `405/728/842/894`, kept assistant labels are `25/45/98/194`, and no
      accepted row has zero or partial labels after truncation. Treat this as a
      staged repair/extraction curriculum, not a promoted main-generator mix,
      until a one-step gate and short adapter show it does not regress the
      active `7/12` public multi-call sequence path.
    - multi-call gap adapter probe:
      `scripts/build_toolcall_multicall_gap_eval_cases.py` builds a 38-row
      held-out public multi-call gap eval: `31` missing-call rows and `7`
      complex-extraction rows. The launcher now exposes `LORA_MODEL_PATH`, so
      gap runs can continue from the active checkpoint-275 adapter. A 50-step
      continuation from checkpoint-275 trained cleanly to loss `2.5502`, but
      regressed the held-out gap eval versus checkpoint-275: raw exact
      sequence/arguments changed from `13/38` / `9/38` to `11/38` / `6/38`,
      and constrained exact arguments changed from `26/38` to `23/38`.
      Complex constrained arguments improved slightly (`2/7 -> 3/7`), but
      missing-call constrained arguments regressed (`24/31 -> 20/31`). Do not
      promote this adapter; split future gap probes by kind.
    - complex-only gap probe:
      The complex-extraction lane was split into
      `data/qwen35_9b_toolcall_multicall_complex_extract_curriculum` with `44`
      accepted rows and zero label-loss rows, plus a 7-row held-out complex eval
      at `data/toolcall_eval/public_multicall_complex_extract_eval.jsonl`.
      A 25-step continuation from checkpoint-275 trained cleanly to loss
      `3.2027`. On the 7-row complex eval, constrained exact arguments improved
      from `2/7` to `3/7` and constrained required-present improved from `3/7`
      to `5/7`, but raw exact arguments regressed from `1/7` to `0/7`. Treat
      this as evidence for constrained complex extraction, not as a promoted
      model update.
  - summary metrics for:
    - schema pass rate
    - correct tool rate
    - repeated-call rate
    - extra/missing-call rate
    - argument exactness
    - unresolved mask examples
    - tokens/s
- Remaining for the first full loop:
  - keep the one-step nonzero-label gate before each training run.
  - implement proper KV-cache-aware sampling for the Qwen3.5 bridge, or keep
    full-context sampling as the slow correctness path until then.
  - turn the schema-repair diagnostic into constrained `<tool_call>` decoding
    so punctuation/key corruption does not dominate otherwise-correct tool
    choices.
  - use label-aware packing for every future public/Qwen3.6 teacher-distilled
    dataset.
  - do not scale the current explicit argument-copy mix as-is; it improved some
    sequence recovery but did not move public argument exactness.
  - keep the structural-token loss hook for sweeps, but do not scale the naive
    token-ID weighting as-is; next trainer-side work should target complete
    argument spans, value-copy behavior, and possibly teacher KL.
  - keep argument-span weighting as the active trainer-side candidate. Use
    weight `1.5` as the balanced longer-run default and weight `2.0` as the
    raw-public-argument comparison point. Avoid weight `3.0`, which regressed
    exact arguments, and avoid structural `2.0` plus argument-span `2.0`, which
    regressed public exact arguments.
  - if combining structural and argument weighting again, use weaker structural
    pressure such as `1.25-1.5` with argument-span `2.0`, or move structural
    enforcement to decoding rather than the training loss.
  - move constrained decoding from post-hoc rescoring toward generation-time
    constraints for `<tool_call>`, `name`, `arguments`, schema keys, and scalar
    delimiters. Keep the new model-repair pass in the eval suite as the first
    model-in-loop approximation while a stricter generation-time constrained
    sampler is developed.
  - for one-call slices, include `--constrained-max-calls 1` in constrained
    scoring because it matches the task contract and exposes argument quality
    after extra-call suppression.
  - do not spend the next run simply increasing
    `--model-repair-max-new-tokens`; the 160-token pass improves syntax a bit
    but does not improve exact tool/argument metrics.
  - do not scale the cap-80 clean-repair curriculum mix as the next main
    generator recipe; it regresses public raw/constrained exactness.
  - do not scale the current hard argument-completion mix as the next main
    generator recipe; it regresses public raw exactness and does not recover
    constrained exact arguments.
  - do not scale the same model-repair corpus to 600 steps as the next default;
    it regresses public and heldout exact arguments. Select checkpoints around
    `250-350` steps before attempting longer runs.
  - use checkpoint-275 from the model-repair + argument-span-1.5 run as the
    active comparison point: public constrained `8/8` / `5/8`, teacher-train
    constrained `10/12` / `5/12`, heldout constrained `8/8` / `3/8`.
  - use `scripts/run_fastdllm_toolcall_checkpoint_sweep.sh` after future runs
    so checkpoint promotion is based on comparable public/train/heldout evals.
  - include public multi-call and both tool-result slices in checkpoint
    promotion. Current checkpoint-275 is strong on constrained tool-result
    (`10/10` sequence on both variants) but weak on public multi-call exact
    arguments (`1/12`).
  - do not scale the first multi-call continuation curriculum as-is. It
    trained stably but regressed public multi-call constrained sequence to
    `4/12-5/12` and did not improve exact arguments.
  - do not promote the first separate multi-call repair adapter. It improves
    exact arguments to `2/12-3/12` after repair/constrained projection, but
    drops exact sequence to `5/12` from the active constrained `7/12`.
  - do not promote the fixed-sequence repair adapter. It drops active
    constrained multi-call sequence from `7/12` to `3/12` after repair plus
    constrained projection and does not improve exact arguments.
  - for the next multi-call attempt, use smaller argument-repair or exact
    scalar-extraction windows before asking for full-chain generation; consider
    keeping that as a separate repair adapter until it stops hurting the
    first-pass generator.
  - move multi-call sequence preservation into deterministic/generation-time
    constraints before another learned repair pass. The learned repair model
    should not be allowed to add, drop, or reorder tool calls if the draft
    already has the right sequence.
  - use the new multi-call gap curriculum first as a separate repair or
    per-field extraction lane for missing calls and complex payloads; avoid
    heavy main-generator mixing until it passes an early public multi-call
    sequence gate.
  - do not promote the first 50-step gap continuation from checkpoint-275. It
    improves syntax and one complex-extraction constrained row but regresses
    missing-call recovery and overall exact arguments.
  - do not promote the first 25-step complex-only continuation either. It helps
    constrained complex extraction by one row, but still regresses raw exact
    arguments and needs decoding/accept-policy work before more training.
  - promote the complex-context constrained decoder change instead of the
    complex-only adapter. The decoder now reconstructs conservative array/object
    payloads from request evidence, improving the 7-row held-out complex lane
    from `2/7-3/7` to `7/7` constrained exact arguments while keeping
    cross-slice results neutral-to-positive.
  - promote the guarded sequence-planner projection as the current
    missing-call diagnostic. It uses request list/table structure and tool
    schema evidence to improve active public multi-call from `7/12` sequence
    and `7/12` arguments to `11/12` sequence and `10/12` arguments after
    segment-local scalar extraction, while staying neutral on one-call and
    tool-result slices. Treat it as a constrained-decoding/planner blueprint,
    not as model-only learning.
  - add the train-only sequence-planner distillation rows as a small replay
    component, not a standalone full run. The new builder selects public-train
    multi-call rows where deterministic request/schema planning matches the
    gold tool order: `27/56` rows match sequence, `2/56` match exact arguments,
    and `13` sequence-selected gold-target rows survive the strict block-size
    `896` label-retention gate. A one-step QLoRA gate from active
    checkpoint-275 completed on the local 5090 with train loss
    `1.42819082736969`; result note:
    `qwen35_9b_sequence_planner_distill_curriculum_result.md`.
  - compact-schema sequence-planner recovery was tested after the low-ratio
    replay mix regressed. Compact schemas recover more fully labeled rows:
    `18` at block size `896` and `21` at block size `1024` for request-only
    prompts, or `17` and `20` with the original instruction prompt. Both
    1024-token one-step gates fit on the local RTX 5090 under the cgroup cap.
    Loss was `4.3695` for compact/request-only and `2.6395` for
    compact/instruction. Use compact/instruction only as an auxiliary recovery
    path, and do not launch a longer compact planner run without a full-schema
    eval gate.
  - compact/instruction passed that first full-schema gate at one step. The
    checkpoint-1 sweep matches active constrained/projected top lines on public
    one-call (`8/8`, `5/8`), public multi-call contextual projection (`7/12`,
    `7/12`), guarded planner projection (`11/12`, `10/12`), synthetic
    tool-result (`10/10`, `8/10`), and OpenAI-style tool-result (`10/10`,
    `9/10`). Teacher-heldout constrained exact arguments improve from the
    active documented `3/8` to `4/8`, but raw public multi-call remains weak
    at `1/12` sequence and `0/12` arguments, so this is not promoted as a new
    checkpoint. It only clears compact/instruction as a safe short-continuation
    candidate.
  - do not scale compact/instruction planner-only replay as the next main
    branch. A 25-step continuation from active checkpoint-275 trained cleanly
    on the 20 compact/instruction rows at block size `1024`, but failed the
    one-call promotion gate before the full sweep completed: public one-call
    constrained exact arguments regressed from active `5/8` to `4/8`,
    teacher-train constrained arguments regressed from `5/12` to `4/12`, and
    teacher-heldout constrained arguments fell to `2/8`. The sweep was stopped
    after those one-call failures to save 5090 time. Result note:
    `qwen35_9b_sequence_planner_distill_curriculum_result.md`.
  - do not promote the first low-ratio model-repair plus sequence-planner mix.
    `scripts/build_toolcall_modelrepair_sequence_planner_mix.py` built a
    240-row mix with 227 base model-repair rows and 13 sequence-planner rows.
    The 100-step checkpoint-100 continuation from active checkpoint-275 trained
    cleanly, but regressed public multi-call contextual projection from active
    `7/12` sequence and `7/12` arguments to `5/12` and `4/12`; guarded
    sequence-planner projection regressed from active `11/12` and `10/12` to
    `7/12` and `5/12`. OpenAI tool-result constrained exact arguments also
    fell from active `9/10` to `5/10`. Result note:
    `qwen35_9b_modelrepair_sequence_planner_mix_result.md`.
  - target stronger strict public exact arguments while preserving the
    synthetic exact-argument gains from the format-only run.
  - use the argument-diff diagnostic to build hard argument-completion rows
    from train-slice failures only, but stage or lower-weight them, or keep
    them in a separate repair adapter, until they stop hurting the first-pass
    generator.
  - longer Qwen3.5-9B diffusion/QLoRA run against offline Qwen3.6 labels and
    the shared tool-call slices after public one-call exact sequence moves
    above zero.
  - first 9B diffusion summary with schema pass rate, correct tool rate,
    repeated-call rate, stop-boundary failures, and tokens/s.
  - actual diffusion target geometry is now controllable separately from the
    data `BLOCK_SIZE`: `TRAIN_BD_SIZE` and `TRAIN_BD_SIZE_CHOICES` passed
    compile plus fixed/dynamic one-step gates. Use
    `qwen35_blockdiffusion_target_ablation_result.md` as the target-objective
    ablation record before interpreting any block-size sweep.
  - first matched target-geometry signal: fixed `bd_size=16` checkpoint-5
    improved heldout policy-target constrained exact sequence from the active
    `5/12` line to `6/12`; dynamic `8,16,32` stayed at `5/12`. Raw valid and
    exact arguments remain `0/12`, so the next branch should add
    tool-sensitive boundaries and repair/teacher objectives rather than merely
    extending the same block-size run.
  - first arg/value span-pressure branch from fixed `bd_size=16` checkpoint-5
    produced the first raw exact sequence/argument hit on the heldout
    policy-target slice (`1/12`), but regressed constrained exact sequence from
    `6/12` to `4/12` and increased extra/repeated calls. Treat this as evidence
    for split sequence/order preservation plus argument repair, not as a
    promoted generator checkpoint.
  - lower-pressure arg/value span masking from the same fixed `bd_size=16`
    anchor trained cleanly, but did not solve the tradeoff. Checkpoint-5
    reached `5/12` constrained exact sequence and `0/12` exact arguments;
    checkpoint-10 reached `1/12` raw exact sequence but still `0/12` exact
    arguments and only `3/12` constrained exact sequence. This makes uniform
    span masking a poor next scaling target unless paired with an explicit
    skeleton/route objective or generation-time grammar/value constraints.
  - simple LoRA delta blending is not a good split-route mechanism for this
    branch. Blends `plain_bd16_ckpt5 + 0.10/0.25 * (argvalue_ckpt5 -
    plain_bd16_ckpt5)` each produced `1/12` raw valid JSON but collapsed
    constrained exact sequence to `3/12`. Use a separate repair/infill adapter,
    constrained remask loop, or skeleton-then-infill target instead of direct
    weight interpolation.
  - scalar repair sidecar transfer onto the plain fixed `bd_size=16`
    checkpoint-5 constrained drafts completed cleanly. It preserved the
    `6/12` heldout policy-target constrained exact sequence and improved
    schema/required-argument validity (`7/12` to `9/12` schema-valid,
    `8/12` to `10/12` required-args-present), but exact arguments stayed
    `0/12` and extra/missing/repeated call totals were unchanged. This keeps
    scalar repair useful as a constrained decoding/repair diagnostic, but not
    as the heldout argument-grounding solution. Next branch should make the
    value/evidence decision part of generation-time infill or a learned
    skeleton-then-infill target.
  - tool-call JSON completability diagnostic is now implemented:
    `scripts/diagnose_toolcall_json_completability.py`, with results in
    `qwen35_toolcall_json_completability_diagnostic_result.md`. On the
    heldout policy-target slice, raw fixed `bd_size=16`, dynamic `8,16,32`,
    and low-pressure arg/value generations all have unrecoverable JSON-prefix
    errors on `12/12` rows. Fixed `bd_size=16` raw has only `5/30` complete
    JSON segments and `25/30` invalid segments; dynamic has `5/38` complete
    and `31/38` invalid; low-pressure arg/value has `10/45` complete and
    `31/45` invalid. Projection/repair makes JSON complete, but exact
    arguments remain `0/12`, and the scalar repair sidecar stays at `6/12`
    sequence and `0/12` arguments. Treat this as evidence that the next
    experiment must be generation-time grammar-completable tool-call commits
    plus skeleton/value split, not another uniform broad span-masking run.
  - first generation-time JSON-prefix guard smoke is complete:
    `qwen35_toolcall_json_prefix_guard_smoke_result.md`. The opt-in sampler
    flag `--guard-tool-json-prefix` now keeps scheduled JSON/tool intervals
    left-to-right and checks that the active `<tool_call>` body remains a
    completable JSON prefix before commit. On a one-row public multi-call
    tool-tag-only comparison, the unguarded run produced malformed raw JSON
    and `0/1` exact sequence, while the guarded run produced `3/3` complete
    JSON segments, raw valid JSON `1/1`, and raw exact sequence `1/1`. Exact
    arguments still stayed `0/1` because the timestamps missed `Z`. The
    unforced smoke also showed the boundary condition: if the model starts with
    prose/thinking and never enters `<tool_call>`, JSON-prefix checking has no
    active body to constrain. Next sampler work should pair this guard with
    literal tool-call mode/sentinel protection and then add schema-aware
    key/value masks plus value-candidate infill.
  - tool-call mode/sentinel guard smoke is complete:
    `qwen35_toolcall_mode_guard_smoke_result.md`. The new opt-in sampler flag
    `--guard-tool-call-mode` hard-fills only scheduled `tool_tag` sentinel
    tokens and records separate mode-force counters. On the same one-row public
    multi-call smoke, JSON-prefix-only generation started with prose and had
    zero raw tool-call segments; mode + JSON-prefix protection forced `12`
    sentinel tokens, produced `3/3` complete raw JSON tool-call segments, raw
    valid JSON `1/1`, and raw exact sequence `1/1`. Exact arguments remain
    `0/1` due timestamp suffix/value grounding, so the next step is
    schema-aware key/value masks plus value-candidate/ranker infill, then the
    12-case public and heldout scheduled scorecards.
  - value/name candidate guard scorecard is complete:
    `qwen35_tool_value_name_guard_scorecard_result.md`. New named sampler
    flags `--guard-tool-value-candidates` and `--guard-tool-name-candidates`
    wrap the existing whole-candidate sequence machinery with separate
    counters. On public multi-call 12, mode + JSON-prefix + value candidates
    reaches raw valid JSON `12/12`, raw exact sequence `11/12`, raw exact
    arguments `11/12`, schema/required `12/12`, with `343` value-candidate
    tokens forced. The only miss is a voice-command route/name row. Adding the
    tool-name guard fixes the tool-name set (`12/12`) and constrained sequence
    (`12/12`), but raw valid JSON drops to `11/12` because the closing
    `</tool_call>` sentinel can still be forced while the active JSON string is
    incomplete-but-completable. Next sampler bug: force opening sentinels, but
    force closing sentinels only when the active tool-call JSON body is
    complete.
  - close-tag completeness under `--guard-tool-call-mode` is now implemented
    and scored in `qwen35_tool_value_name_guard_scorecard_result.md`. On public
    multi-call 12 with mode + JSON-prefix + name + value + close protection,
    raw valid JSON improves to `12/12`, raw exact tool-name set to `12/12`,
    raw exact tool sequence to `12/12`, and raw exact arguments to `11/12`.
    The diagnostic reports `31/31` raw complete JSON segments and zero invalid
    segments. The close guard fires once, exactly on the earlier truncation
    row. The remaining miss is no longer structural: gold expects
    `location: ""` for the third voice-command call while raw fills
    `location: "home"`. Treat the next work as evidence-grounded value
    learning/infill plus heldout validation, not another close-boundary fix.
  - heldout policy-target close-guard scorecard is complete:
    `qwen35_heldout_policy_close_guard_scorecard_result.md`. The lean named
    guard stack transfers to the 12-row heldout policy target at raw valid JSON
    `11/12`, exact sequence `11/12`, and exact arguments `11/12`. The miss is
    `heldout_seed_multicall_0004`, a long nested `create_campaign` array where
    JSON keys/structure drift after `83` rejected prefix-guard commits and `83`
    unsafe fallbacks. Target fallback does not help. Adding only
    `--force-schedule-token-kinds json_key,json_structure` while keeping named
    mode/name/value guards reaches raw valid JSON `12/12`, exact sequence
    `12/12`, exact arguments `12/12`, and `29/29` complete raw JSON segments.
    This is a protected structural ceiling, not raw model promotion. The next
    model-side target should learn skeleton/key/structure stability and
    skeleton-conditioned value infill rather than relying on full schedule
    forcing.
  - skeleton-conditioned value infill artifacts are now built:
    `qwen35_skeleton_value_infill_artifacts_result.md`. New builder:
    `scripts/build_skeleton_value_infill_artifacts.py`. Diagnostic heldout
    artifacts live under
    `data/skeleton_value_infill/heldout_policy_diagnostic/` with `123` value
    slots and `promotion_allowed=false`. Trainable artifacts live under
    `data/skeleton_value_infill/public_train_no_public_smoke/` with `45`
    clean filtered records, `331` usable value slots, `711` candidate rows,
    `4667` boundary labels, `331` value-infill train instances, and
    `promotion_allowed=true`. A source overlap audit checks `85` filtered
    train records against `37` public/heldout eval records and finds `0` exact
    or user overlaps. The older
    `data/toolcall_eval/public_train_multicall_gold_cases.jsonl` is not safe
    for this purpose: direct audit finds `11/12` public multi-call overlaps.
  - skeleton-conditioned value-infill training gate is complete:
    `qwen35_skeleton_value_infill_training_gate_result.md`. A one-file staging
    dataset at
    `data/qwen35_9b_skeleton_value_infill_no_public_smoke_curriculum/` avoids
    LMFlow globbing non-training JSON artifacts. The one-step checkpoint-275
    continuation gate saved an adapter with train loss `3.3805`. The 75-step
    QLoRA sweep completed on the local RTX 5090 without OOM, saved
    checkpoint adapters at steps `25`, `50`, and `75`, and ended with train
    loss `2.6479`. This is trainability evidence only; promotion now requires
    checkpoint evaluation on public and heldout tool-call gates without
    structural regressions.
  - skeleton-conditioned value-infill checkpoint eval is complete:
    `qwen35_skeleton_value_infill_checkpoint_eval_result.md`. Checkpoints
    `25`, `50`, and `75` all tie active checkpoint-275 on public multi-call
    closeguard at raw valid JSON `12/12`, exact sequence `12/12`, exact
    arguments `11/12`, and heldout lean closeguard at raw valid JSON `11/12`,
    exact sequence `11/12`, exact arguments `11/12`. The public miss remains
    the voice-command empty-location value; the heldout miss remains
    `heldout_seed_multicall_0004` nested skeleton/key corruption with `83`
    unsafe prefix fallbacks. Checkpoint-25 shows a small one-call raw/model-
    repair improvement, but the adapter line is not promotion-worthy.
  - schedule-state selector curriculum gate is complete:
    `qwen35_schedule_state_selector_curriculum_gate_result.md`. New builder:
    `scripts/build_schedule_state_selector_curriculum.py`. It converts the
    clean no-public-smoke skeleton value slots into `539` selector/policy
    instances that emit a compact JSON decision with `candidate_index`,
    block-size, denoise-step, value-candidate, JSON-prefix, and close-complete
    protection labels. The artifact lives at
    `data/qwen35_9b_schedule_state_selector_no_public_smoke_curriculum/`,
    has `0` rejected rows under the `1024`-token left-truncation audit, and a
    one-step QLoRA continuation from active checkpoint-275 trains/saves on the
    5090 with loss `4.8791`. This replaces the standalone value-span objective
    as the next sweep candidate.
  - schedule-state selector free-generation sweep is complete and not
    promotable. The `75`-step QLoRA continuation trains without OOM on the RTX
    5090 and saves checkpoints `25/50/75`, but a fixed `16`-example ambiguous
    selector eval gives `0/16` valid JSON and `0/16` exact decisions for active
    checkpoint-275 and all selector checkpoints. Loose regex index hits are
    `3/16`, `1/16`, `3/16`, and `2/16` respectively. Next selector work should
    preserve the schedule-state representation but use constrained JSON
    prefix-forcing or masked/pairwise scoring, not free assistant JSON
    generation.
  - constrained schedule-state selector ranking is now implemented:
    `scripts/eval_fastdllm_schedule_state_selector_ranking.py`. In
    `index_only` mode it force-prefixes `{"candidate_index":` and scores only
    candidate indices, then the sampler can inject the fixed JSON protection
    policy. Active checkpoint-275 reaches `59/64` index accuracy and `63/64`
    target top-2 on the first ambiguous selector slice; selector checkpoints
    `25/50/75` tie exactly and are not promotion-worthy. On all `349`
    ambiguous rows, active checkpoint-275 reaches `312/349` (`89.40%`) top-1
    and `334/349` (`95.70%`) target top-2 with `0` runtime errors. This
    promotes the constrained scorer/injector pattern, not the selector-SFT
    adapter.
  - schedule-state selector ranking is now wired into sampler schedules:
    `scripts/inject_schedule_state_selector_ranking_choices.py`. Candidate
    order matches between selector prompts and sampler schedules for all `539`
    curriculum instances. The rank-1/rank-2 injected no-public-smoke schedules
    restrict `163` argument schedule items across `30` records. A four-case
    generation smoke with the protected valueguard/mode/json-prefix stack gives
    rank-1 raw valid JSON `4/4`, exact sequence `4/4`, exact arguments `3/4`;
    rank-2 keeps valid JSON/sequence `4/4` but drops exact arguments to `2/4`.
    Treat rank-1 as the default injected control path; top-2 needs a separate
    repair/rerank policy before promotion.

Then use the failure cases to define the next training corpus and objective.

Closeout target:

- full target definitions live in `qwen36_diffusion_closeout_metrics.md`
- minimum full SWE-bench Verified closeout:
  `DIFF_TRAIN_Q36 >= max(DIFF_INIT_Q36 + 15pp, 0.70 * AR_Q36, 30%)`
- project-success target:
  `DIFF_TRAIN_Q36 >= max(DIFF_INIT_Q36 + 25pp, 0.80 * AR_Q36, 45%)`
- speed target:
  `DIFF_TRAIN_Q36` should resolve at least 1.3x as many Verified instances per
  hour as `AR_Q36` under the same harness.

## Open Questions

- Should the first real agentic training use only LoRA, or LoRA plus selected
  structural-token embedding/head updates?
- Is AR-teacher KL enough, or do we need sequence-level constrained decoding loss?
- Should block size stay small for tool-call spans and grow only for natural text?
- Can the sampler force structural tokens more safely without ruining speed?
- Is the released Fast-dLLM 1.5B good enough on tool calls to serve as a local
  diffusion baseline, or do we need to compare with DiffusionGemma directly?
- Can SGLang Qwen3.6 FP8 fit comfortably enough on the 5090 after the NVFP4/MTP
  path, or should NVFP4 remain the default local teacher profile?
