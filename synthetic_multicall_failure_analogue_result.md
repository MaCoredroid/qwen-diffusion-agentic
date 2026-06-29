# Synthetic Multi-Call Failure Analogue Result

Date: 2026-06-28

## Purpose

Create non-eval analogues for the two remaining public multi-call failure
families before doing more training:

- voice-command camera routing: the user asks for a spoken command that mentions
  cameras, but the correct tool is still `activate_voice_command`, not the
  direct `activate_security_cameras` API.
- security installation-code scoping: lock, motion-detector, and alarm actions
  each carry a separate code; the motion-detector call must not copy the smart
  lock code.

These cases are synthetic analogues, not copied public rows. A literal sanity
check found none of the public-row values `YRD256`, `91MHZPIR`, `73829SL`,
`ALRM328SEC`, `Activate security cameras in away mode`, `Activate kitchen
lights`, `Honeywell SiXPIR`, or `Yale Assure` in the synthetic JSONL.

## Artifacts

- builder:
  `scripts/build_synthetic_multicall_failure_analogues.py`
- eval cases:
  `data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl`
- eval summary:
  `data/toolcall_eval/synthetic_multicall_failure_analogues.summary.json`
- pure planner audit:
  `runs/synthetic_multicall_failure_analogues/sequence_planner_from_empty.summary.json`
- bad-draft conservative audit:
  `runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_conservative.summary.json`
- bad-draft sequence-override audit:
  `runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_sequence_override.summary.json`
- bad-draft safe sequence-mismatch audit:
  `runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.summary.json`

The synthetic set has `8` records and `24` gold tool calls:

- `4` voice-command camera analogues
- `4` security installation-code analogues

Each row also includes `bad_draft_assistant`:

- voice rows use the right first two calls but incorrectly choose
  `activate_security_cameras` for the camera command.
- security rows use the right tool sequence but incorrectly copy the lock
  `installation_code` into `configure_motion_detectors`.

## Commands

```bash
.venv-fastdllm/bin/python scripts/build_synthetic_multicall_failure_analogues.py

.venv-fastdllm/bin/python scripts/rescore_toolcall_sequence_planner_projection.py \
  --cases-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --input-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --out-jsonl runs/synthetic_multicall_failure_analogues/sequence_planner_from_empty.jsonl \
  --text-field contextual_projection_assistant \
  --min-input-calls-for-plan 0

.venv-fastdllm/bin/python scripts/rescore_toolcall_sequence_planner_projection.py \
  --cases-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --input-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --out-jsonl runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_conservative.jsonl \
  --text-field bad_draft_assistant \
  --min-input-calls-for-plan 2

.venv-fastdllm/bin/python scripts/rescore_toolcall_sequence_planner_projection.py \
  --cases-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --input-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --out-jsonl runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_sequence_override.jsonl \
  --text-field bad_draft_assistant \
  --min-input-calls-for-plan 2 \
  --use-plan-on-sequence-mismatch

.venv-fastdllm/bin/python scripts/rescore_toolcall_sequence_planner_projection.py \
  --cases-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --input-jsonl data/toolcall_eval/synthetic_multicall_failure_analogues.jsonl \
  --out-jsonl runs/synthetic_multicall_failure_analogues/sequence_planner_bad_draft_safe_seqmismatch.jsonl \
  --text-field bad_draft_assistant \
  --min-input-calls-for-plan 2 \
  --use-safe-plan-on-sequence-mismatch
```

## Results

Pure planner from empty drafts:

- exact tool sequence: `8/8`
- exact arguments: `8/8`
- schema valid: `8/8`
- required args present: `8/8`

Bad-draft conservative projection:

- input exact tool sequence: `4/8`
- input exact arguments: `0/8`
- planned exact tool sequence: `4/8`
- planned exact arguments: `4/8`

The conservative planner repairs the four security-code rows because the draft
tool sequence already matches the plan, so segment-local argument extraction can
replace the copied detector code. It does not repair the four voice-command
camera rows because the draft has the same number of calls but a different tool
sequence, and the existing guard refuses to override that mismatch.

Bad-draft sequence-override projection:

- planned exact tool sequence: `8/8`
- planned exact arguments: `8/8`
- schema valid: `8/8`
- required args present: `8/8`

Bad-draft safe sequence-mismatch projection:

- planned exact tool sequence: `8/8`
- planned exact arguments: `8/8`
- schema valid: `8/8`
- required args present: `8/8`

The repeated-call counter reports `4` rows because the correct voice-command
gold sequence intentionally calls `activate_voice_command` twice. That counter
is not an error for this analogue family; exact sequence and exact arguments
are the decisive metrics here.

## Change Made

`scripts/rescore_toolcall_sequence_planner_projection.py` now has an opt-in
flag:

```bash
--use-plan-on-sequence-mismatch
```

It allows the request-derived plan to replace a same-length draft sequence when
the planned tool names disagree with the draft tool names. The default remains
conservative, so existing scorecards do not change unless this flag is passed.

Follow-up change:

- `--use-safe-plan-on-sequence-mismatch` only applies same-length sequence
  replacement when every planned segment clears score and margin thresholds
  (`14.0` score and `2.0` margin by default).
- a targeted camera voice-command conflict resolver detects prompts where
  earlier prose says to execute a quoted camera command "by saying" it, while a
  later argument list resembles a direct camera status API. It then chooses
  `activate_voice_command` and fills `command`, `device_type`, and `location`
  from the spoken-command evidence.
- anchored code extraction now returns the nearest code after an anchor in
  anchored mode. This fixes detector-code scoping where a motion-detector model
  anchor is followed by the detector installation code and then a later alarm
  code.

## Interpretation

The analogue set shows that the rule-based planner has enough request evidence
to recover these two failure families without model training. The current
integration guard is the limiting factor for the voice-command camera family,
not planner understanding.

## Public Multi-Call Ablation

The opt-in sequence-mismatch override was also run on the public multi-call
smoke as a safety ablation only:

```bash
.venv-fastdllm/bin/python scripts/rescore_toolcall_sequence_planner_projection.py \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_preserve_complex_contextual_v4.jsonl \
  --out-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v4_seqmismatch.jsonl \
  --text-field contextual_projection_assistant \
  --min-input-calls-for-plan 2 \
  --use-plan-on-sequence-mismatch
```

Result:

- existing guarded planner v3: `11/12` exact sequence, `10/12` exact arguments
- global sequence-mismatch override: `9/12` exact sequence, `8/12` exact
  arguments

This means the override is not safe as a global default. It regresses rows where
the request-derived planner has weak margins and a same-length draft was already
right, such as camera live/recorded-feed routing and quantum syntax/examples.
It also does not fix the public voice-command camera row: the public prompt has
earlier prose saying to execute a camera command "by saying" a phrase, but the
later argument list says `status/mode`; the current planner over-weights that
later list and chooses `activate_security_cameras`.

Implication:

- keep `--use-plan-on-sequence-mismatch` as an ablation/debug flag only.
- do not use it in protected scores without an additional confidence or
  family-specific guard.
- the next planner fix should cross-reference earlier prose and quoted spoken
  commands when a later argument list conflicts with a voice-command task.

That fix is now implemented and verified:

```bash
.venv-fastdllm/bin/python scripts/rescore_toolcall_sequence_planner_projection.py \
  --cases-jsonl data/toolcall_eval/public_multicall_hermes_smoke.jsonl \
  --input-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_preserve_complex_contextual_v4.jsonl \
  --out-jsonl runs/qwen35_9b_split_route_sidecar_eval/live_smoke_public_multicall_planner/public_multi_call_planner/public_multicall_12_sequence_planner_segmentargs_v5_voice_safe.jsonl \
  --text-field contextual_projection_assistant \
  --min-input-calls-for-plan 2
```

Targeted voice resolver plus anchored-code fix:

- public multi-call diagnostic planned exact sequence: `12/12`
- public multi-call diagnostic planned exact arguments: `12/12`
- schema valid: `12/12`
- required args present: `12/12`

This is a protected planner/projection result, not model-side promotion
evidence. The public slice is also known to overlap older public-train-derived
artifacts, so use it as a diagnostic regression check only.

Next useful step from here:

1. Turn the synthetic analogue rows and the protected planner decisions into
   train-only distillation labels.
2. Keep the public multi-call v5 result as a protected regression gate, not as
   checkpoint-promotion evidence.
3. Move the targeted voice and anchored-code rules toward generation-time
   constrained decoding so raw diffusion outputs need less post-hoc projection.
