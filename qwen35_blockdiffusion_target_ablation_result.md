# Qwen3.5 Block-Diffusion Target Ablation

Date: 2026-06-28

## Why This Matters

Earlier runs changed `BLOCK_SIZE`, which is the max token window/chunk length
used by the data pipeline. That is not the same as the diffusion block target.
The actual block-diffusion objective uses `config.bd_size`, which controls:

- how `input_ids` are reshaped into denoising blocks
- the block-diffusion attention mask
- GDN chunking/state handling inside diffusion mode

So the next useful experiments need to vary `bd_size`, not just the context
window.

## Code Change

Added training-time controls:

- `TRAIN_BD_SIZE=16`
  - exports `FASTDLLM_TRAIN_BD_SIZE`
  - sets model/config/decoder-layer `bd_size`
  - sets LMFlow data padding multiple
- `TRAIN_BD_SIZE_CHOICES=8,16,32`
  - exports `FASTDLLM_TRAIN_BD_SIZE_CHOICES`
  - pads batches to the LCM of choices
  - randomly selects a valid `bd_size` per training forward
  - updates model/config/layer `bd_size` before MDM reshape and attention mask

Touched paths:

- `fast-dllm/v2/train_scripts/finetune.py`
- `models/qwen3.5-9b-fastdllm-init/modeling.py`
- `scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh`

Compile gate:

```bash
.venv-fastdllm/bin/python -m py_compile \
  fast-dllm/v2/train_scripts/finetune.py \
  models/qwen3.5-9b-fastdllm-init/modeling.py
```

Result: passed.

## Smoke Gates

Both runs used:

- local RTX 5090 only
- cgroup scope: `MemoryMax=28G`, `MemorySwapMax=4G`
- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- corpus:
  `data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum`
- `MAX_STEPS=1`
- `MAX_TRAIN_SAMPLES=32`
- `BLOCK_SIZE=1536`
- `TRUNCATION_SIDE=left`
- `LEARNING_RATE=1e-6`
- `GRAD_ACCUM=4`
- `LORA_R=8`, `LORA_ALPHA=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`
- `VALUE_SPAN_LABEL_ONLY=1`

### Fixed `bd_size=16`

Run:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step1_gate
```

Log:

```text
logs/fastdllm_qwen35_sequence_value_bd16_step1_gate.log
```

Debug evidence:

```text
[fastdllm-qwen35-debug] post_mdm ... bd_size=16 ...
```

Result:

- checkpoint written
- train loss: `0.4349699020385742`
- train runtime: `16.2155s`
- samples/sec: `0.247`

### Dynamic `bd_size` Choices `8,16,32`

Run:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bdchoices8_16_32_from_ckpt275_step1_gate
```

Log:

```text
logs/fastdllm_qwen35_sequence_value_bdchoices8_16_32_step1_gate.log
```

Debug evidence from one optimizer step:

```text
bd_size=32
bd_size=16
bd_size=8
bd_size=16
```

Result:

- checkpoint written
- train loss: `0.6011598706245422`
- train runtime: `16.2501s`
- samples/sec: `0.246`

These are wiring gates, not quality conclusions. Some early debug batches had
zero valid labels after `VALUE_SPAN_LABEL_ONLY=1`; later forwards in the same
step did contain valid value-span labels.

## Fundamental Levers To Explore

1. Block geometry

   Fixed `bd_size` controls how much text is denoised jointly. Smaller blocks
   reduce structural smear and may help tool calls; larger blocks increase
   parallelism and longer-range joint planning. Dynamic choices may train
   robustness across both regimes.

2. Tool-sensitive boundaries

   Tool-call spans, JSON delimiters, function names, parameter names, and
   quoted values should probably not be split casually across diffusion blocks.
   The next version should choose block boundaries from the tokenized tool-call
   structure, not just fixed token offsets.

3. Mask/noise schedule

   Current training uses per-block random `t` with linear mask probability.
   Agentic spans likely need non-uniform schedules: always/usually mask
   arguments, lower noise on structural delimiters, higher noise on copied
   values, and harder schedules for failed eval cases.

4. Training target

   Current loss is masked-token CE with complementary masking and optional span
   weights. Other targets worth testing are AR-teacher KL on masked positions,
   sequence-level exact tool-call loss/proxy reward, edit-delta prediction,
   skeleton-then-infill, and confidence-calibrated denoising targets.

   Prioritized target variants:

   - masked CE plus complementary AR-preservation loss: current baseline
   - top-k teacher KL only on sensitive spans: tool name, JSON key, argument
     scalar, call boundary, stop marker
   - repair-denoising: corrupt model-produced malformed tool calls and train
     toward the repaired teacher/tool trace
   - skeleton-then-infill: first generate tool sequence/schema skeleton, then
     denoise values under schema/request constraints
   - later verifier/RL objective: use tool-call validity, exact sequence,
     grounded values, and code/unit-test rewards after offline logging is solid

5. GDN state handling

   `bd_size` interacts with Qwen3.5/3.6 Gated DeltaNet behavior. We should keep
   testing causal GDN, noisy block isolation, clean-state injection, and
   dual-pass modes because the state path may determine whether diffusion
   preserves AR behavior at agentic boundaries.

6. Structured decoding and repair

   Protected tool calling can be grammar/constrained decoding, deterministic
   repair, or a learned sidecar/adapter. The most likely near-term recipe is
   hybrid: diffusion proposes content, a constrained decoder/repair layer keeps
   `<tool_call>` syntax valid, and learned adapters improve route/value choice.

   Treat tool calls as typed sections, not prose:

   - `tool_tag`: literal/prefix constrained
   - `tool_name`: enum constrained
   - `json_key`: schema-key enum constrained
   - `json_structure`: grammar constrained
   - `argument_value`: request-evidence and schema constrained
   - `inter_call_boundary` and `stop`: tiny, high-confidence, no-prose spans

   The near-term sampler should restrict tokens for enums/literals, run a
   JSON/schema incompletable-prefix check, and remask only offending positions
   instead of regenerating the whole answer.

7. Test-time compute

   Diffusion only wins if extra denoising passes buy quality or speed. Sweeps
   need to track denoising steps, remask policy, confidence thresholds, and
   retry/repair cost, not just checkpoint accuracy.

   Use TTC locally:

   - early-commit easy spans by confidence
   - spend extra denoising steps only on tool names, JSON structure, values, and
     stop boundaries
   - run N stochastic rollouts with the same prompt/cache and rank by schema
     validity, exact tool sequence, request grounding, and code tests
   - compare enhanced adapter logits versus reference/base logits on sensitive
     spans before accepting changes

8. Scaling law discipline

   One-step gates only prove the path runs. A failed tiny sample does not rule
   out a target. Promote/reject target variants only after matched 10/25/100-step
   curves on heldout tool-call slices plus at least one coding/tool eval.

## Next Sweep

Stage A: matched target geometry sweep from checkpoint-275 on the same
sequence/value/retention curriculum:

| Variant | Steps | Purpose |
|---|---:|---|
| fixed `bd_size=8` | 10 | smallest structural/tool-call block |
| fixed `bd_size=16` | 10 | smoke-passed middle-small block |
| fixed `bd_size=32` | 10 | current baseline target |
| fixed `bd_size=64` | 10 | larger joint-planning block |
| dynamic `8,16,32` | 10 | robustness across small blocks |
| dynamic `16,32,64` | 10 | robustness across mid/large blocks |

Evaluate each checkpoint on:

- heldout policy-target forced-prefix eval
- public one-call constrained projection
- public multi-call contextual/sequence-planner projection
- synthetic tool-result and OpenAI-style tool-result slices

Stage B: implement token/tool-sensitive block planning:

- keep each `<tool_call>...</tool_call>` inside one or few protected blocks
- avoid splitting function name and parameter-name spans
- optionally isolate quoted argument values into smaller high-noise blocks
- compare against fixed/dynamic token-offset blocks

Stage C: target/objective variants:

- `VALUE_SPAN_LABEL_ONLY=0` versus `1`
- argument/value force-mask probabilities
- structural-token loss weights
- AR-teacher KL on masked spans when logits are available
- skeleton-then-infill curriculum for tool calls
- constrained decode/repair in eval as a first-class metric, not an afterthought

Promotion condition for the next trained branch:

- no public one-call constrained regression versus checkpoint-275
- improve heldout policy-target constrained exact arguments above the current
  `0/12` branch results
- improve raw valid tool-call rate above the current `0-1/12` range
- preserve or improve synthetic tool-result exact arguments

## First Matched Sweep Started

Two 10-step continuations from checkpoint-275 were run on the same
sequence/value/retention curriculum.

Shared settings:

- corpus:
  `data/qwen35_9b_sequence_value_retention_mix_nooverlap_curriculum`
- start adapter:
  `runs/fastdllm_qwen35_9b_toolcall_modelrepair_argspanw1p5_b896_step300/checkpoint-275/adapter_model`
- `MAX_STEPS=10`
- `SAVE_STEPS=5`
- `MAX_TRAIN_SAMPLES=387`
- `BLOCK_SIZE=1536`
- `TRUNCATION_SIDE=left`
- `LEARNING_RATE=1e-6`
- `GRAD_ACCUM=4`
- `LORA_R=8`, `LORA_ALPHA=16`
- target modules:
  `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b`
- `VALUE_SPAN_LABEL_ONLY=1`
- local RTX 5090 under cgroup memory cap

### Fixed `bd_size=16`, 10 steps

Run:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step10
```

Log:

```text
logs/fastdllm_qwen35_sequence_value_bd16_step10.log
```

Training result:

- checkpoints: `5`, `10`
- train loss: `0.7233704566955567`
- runtime: `156.2219s`
- samples/sec: `0.256`

### Dynamic `bd_size=8,16,32`, 10 steps

Run:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bdchoices8_16_32_from_ckpt275_step10
```

Log:

```text
logs/fastdllm_qwen35_sequence_value_bdchoices8_16_32_step10.log
```

Training result:

- checkpoints: `5`, `10`
- train loss: `0.6896651029586792`
- runtime: `156.5624s`
- samples/sec: `0.255`

### Dynamic checkpoint-5 heldout eval

Adapter:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bdchoices8_16_32_from_ckpt275_step10/checkpoint-5/adapter_model
```

Output:

```text
runs/target_geometry_eval/bdchoices8_16_32_checkpoint5_policy_targets_forcedprefix.summary.json
```

Heldout policy-target forced-prefix settings:

- max new tokens: `900`
- eval block size: `32`
- eval small block size: `8`
- full-context sampling
- forced `<tool_call>\n` prefix
- constrained tool decoding
- sequence-preserving constrained projection
- constrained max calls: `3`

Result:

| metric | score |
|---|---:|
| raw valid JSON | `0/12` |
| raw exact sequence | `0/12` |
| raw exact arguments | `0/12` |
| constrained valid JSON | `12/12` |
| constrained exact name set | `7/12` |
| constrained exact sequence | `5/12` |
| constrained exact arguments | `0/12` |
| records with extra calls | `1/12` |
| records with missing calls | `12/12` |
| records with repeated calls | `1/12` |

Runtime:

- elapsed: `941.715s`
- generated tokens: `5478`
- tokens/sec: `5.817`

Interpretation:

- dynamic small-block training is not an immediate heldout policy-target win
  at checkpoint-5
- it preserves the old checkpoint-275 constrained exact sequence line
  (`5/12`) and improves constrained name set over the old baseline (`7/12`
  versus `6/12`)
- it does not reproduce the old sequence/value checkpoint-5 raw-valid bump
  (`0/12` here versus prior `1/12`)
- exact arguments remain unmoved at `0/12`, so block geometry alone is not
  enough; the next target needs tool-sensitive boundaries, repair-denoising, or
  teacher-KL/value objectives

### Fixed `bd_size=16` checkpoint-5 heldout eval

Adapter:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step10/checkpoint-5/adapter_model
```

Output:

```text
runs/target_geometry_eval/bd16_checkpoint5_policy_targets_forcedprefix.summary.json
```

Result:

| metric | score |
|---|---:|
| raw valid JSON | `0/12` |
| raw exact sequence | `0/12` |
| raw exact arguments | `0/12` |
| constrained valid JSON | `12/12` |
| constrained exact name set | `7/12` |
| constrained exact sequence | `6/12` |
| constrained exact arguments | `0/12` |
| records with extra calls | `1/12` |
| records with missing calls | `12/12` |
| records with repeated calls | `1/12` |

Runtime:

- elapsed: `679.889s`
- generated tokens: `4161`
- tokens/sec: `6.120`

Row-level constrained sequence passes:

```text
heldout_seed_multicall_0004
heldout_seed_multicall_0005
heldout_seed_multicall_0009
heldout_seed_multicall_0010
heldout_seed_multicall_0011
heldout_seed_multicall_0012
```

Interpretation:

- fixed small blocks are the first target-geometry branch to beat the active
  checkpoint-275 heldout constrained sequence line: `6/12` versus `5/12`
- dynamic `8,16,32` was neutral on constrained sequence (`5/12`) and slower on
  this eval, so dynamic choices are not automatically better at this scale
- raw valid/exact and constrained exact arguments are still flat at `0/12`; the
  sequence win is meaningful but not sufficient
- next best experiment is fixed `bd_size=16` plus tool-sensitive block
  boundaries and argument/value repair-denoising, not a blind longer
  continuation of the same objective

## Argument/Value Span Objective Branch

To test whether the fixed `bd_size=16` sequence gain can be pushed toward
argument correctness, a 10-step continuation was run from the best
`bd_size=16` checkpoint-5.

Start adapter:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step10/checkpoint-5/adapter_model
```

Run:

```text
runs/fastdllm_qwen35_9b_bd16_ckpt5_argvalue_mask_w_step10
```

Log:

```text
logs/fastdllm_qwen35_bd16_ckpt5_argvalue_mask_w_step10.log
```

Training settings versus the plain `bd_size=16` branch:

- `TRAIN_BD_SIZE=16`
- `VALUE_SPAN_LABEL_ONLY=0`
- `ARGUMENT_SPAN_LOSS_WEIGHT=1.5`
- `VALUE_SPAN_LOSS_WEIGHT=2.0`
- `ARGUMENT_SPAN_MASK_PROB=0.25`
- `VALUE_SPAN_MASK_PROB=0.5`

Debug evidence that the target changed:

```text
argument_span=51 value_span=34 forced_argument_mask=13 forced_value_mask=16
argument_span=81 value_span=67 forced_argument_mask=24 forced_value_mask=33
argument_span=166 value_span=132 forced_argument_mask=42 forced_value_mask=73
```

Training result:

- checkpoints: `5`, `10`
- train loss: `3.6441720962524413`
- runtime: `156.4173s`
- samples/sec: `0.256`

### Span-objective checkpoint-5 heldout eval

Adapter:

```text
runs/fastdllm_qwen35_9b_bd16_ckpt5_argvalue_mask_w_step10/checkpoint-5/adapter_model
```

Output:

```text
runs/target_geometry_eval/bd16_argvalue_mask_w_checkpoint5_policy_targets_forcedprefix.summary.json
```

Result:

| metric | plain `bd_size=16` ckpt5 | arg/value span ckpt5 |
|---|---:|---:|
| raw valid JSON | `0/12` | `0/12` |
| raw exact name set | `0/12` | `3/12` |
| raw exact sequence | `0/12` | `1/12` |
| raw exact arguments | `0/12` | `1/12` |
| raw all schema valid | `2/12` | `4/12` |
| constrained valid JSON | `12/12` | `12/12` |
| constrained exact name set | `7/12` | `6/12` |
| constrained exact sequence | `6/12` | `4/12` |
| constrained exact arguments | `0/12` | `0/12` |
| records with extra calls | `1/12` | `3/12` |
| records with missing calls | `12/12` | `9/12` |
| records with repeated calls | `1/12` | `3/12` |
| total extra calls | `1` | `10` |
| total missing calls | `26` | `19` |
| total repeated calls | `1` | `10` |

Runtime:

- elapsed: `944.283s`
- generated tokens: `5545`
- tokens/sec: `5.872`

Row-level raw win:

```text
heldout_seed_multicall_0010: raw exact sequence=True, raw exact arguments=True,
strict valid_tool_json=False
```

Interpretation:

- Argument/value pressure is the first branch here to produce any raw exact
  sequence/argument signal (`1/12`), so the objective direction is not dead.
- It also increases raw exact name-set and schema-valid counts, suggesting the
  model is paying more attention to the tool-call content rather than only the
  sequence projection.
- It destabilizes call count/order: constrained exact sequence drops from
  `6/12` to `4/12`, extra calls rise from `1` to `10`, and repeated calls rise
  from `1` to `10`.
- Do not promote this adapter. The next target should separate route/order from
  argument repair instead of forcing all argument spans inside the main
  generator with high weights.
- Most likely next experiments:
  - lower-pressure version: `ARGUMENT_SPAN_MASK_PROB=0.1`,
    `VALUE_SPAN_MASK_PROB=0.25`, keep `VALUE_SPAN_LABEL_ONLY=0`
  - two-adapter/blend route: preserve plain `bd_size=16` generator for
    sequence/order, add a smaller argument-repair adapter or sidecar
  - skeleton-then-infill: train tool sequence/schema skeleton first, then value
    infill on protected argument spans
  - eval-time validator/remask: remask only malformed/extra repeated call spans
    instead of pushing all spans in one training objective

## Lower-Pressure Argument/Value Branch

The lower-pressure follow-up tested whether the high-pressure raw argument
signal could be recovered without the route/order collapse.

Start adapter:

```text
runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step10/checkpoint-5/adapter_model
```

Run:

```text
runs/fastdllm_qwen35_9b_bd16_ckpt5_argvalue_lowpressure_step10
```

Log:

```text
logs/fastdllm_qwen35_bd16_ckpt5_argvalue_lowpressure_step10.log
```

Training settings versus the high-pressure branch:

- `TRAIN_BD_SIZE=16`
- `VALUE_SPAN_LABEL_ONLY=0`
- `ARGUMENT_SPAN_LOSS_WEIGHT=1.25`
- `VALUE_SPAN_LOSS_WEIGHT=1.5`
- `ARGUMENT_SPAN_MASK_PROB=0.10`
- `VALUE_SPAN_MASK_PROB=0.25`

Training result:

- checkpoints: `5`, `10`
- train loss: `2.7437017440795897`
- runtime: `156.4734s`
- samples/sec: `0.256`

Eval outputs:

```text
runs/target_geometry_eval/bd16_argvalue_lowpressure_checkpoint5_policy_targets_forcedprefix.summary.json
runs/target_geometry_eval/bd16_argvalue_lowpressure_checkpoint10_policy_targets_forcedprefix.summary.json
```

Heldout policy-target forced-prefix results:

| metric | plain `bd_size=16` ckpt5 | high-pressure ckpt5 | low-pressure ckpt5 | low-pressure ckpt10 |
|---|---:|---:|---:|---:|
| raw valid JSON | `0/12` | `0/12` | `0/12` | `0/12` |
| raw exact name set | `0/12` | `3/12` | `1/12` | `1/12` |
| raw exact sequence | `0/12` | `1/12` | `0/12` | `1/12` |
| raw exact arguments | `0/12` | `1/12` | `0/12` | `0/12` |
| raw all schema valid | `2/12` | `4/12` | `3/12` | `4/12` |
| constrained valid JSON | `12/12` | `12/12` | `12/12` | `12/12` |
| constrained exact name set | `7/12` | `6/12` | `6/12` | `7/12` |
| constrained exact sequence | `6/12` | `4/12` | `5/12` | `3/12` |
| constrained exact arguments | `0/12` | `0/12` | `0/12` | `0/12` |
| constrained all schema valid | not recorded in table | not recorded in table | `8/12` | `8/12` |
| constrained required args present | not recorded in table | not recorded in table | `9/12` | `8/12` |
| total extra calls | `1` | `10` | `3` | `2` |
| total missing calls | `26` | `19` | `25` | `21` |
| total repeated calls | `1` | `10` | `3` | `1` |

Runtime:

- checkpoint-5 eval: `839.277s`, `5318` generated tokens, `6.336` tokens/sec
- checkpoint-10 eval: `1140.704s`, `6905` generated tokens, `6.053` tokens/sec

Interpretation:

- Lower pressure reduces the high-pressure branch's extra/repeated-call damage,
  but it also loses the only raw exact-argument hit.
- Checkpoint-5 is close to the old checkpoint-275 constrained sequence line
  (`5/12`) but still below the plain fixed `bd_size=16` anchor (`6/12`).
- Checkpoint-10 gets `1/12` raw exact sequence, but constrained exact sequence
  falls to `3/12` and exact arguments remain `0/12`.
- This is not a promoted generator branch. Both high and low arg/value pressure
  confirm that route/order and argument grounding should be separated, rather
  than jointly optimized by uniform span masking in the main adapter.

## LoRA Delta Blend Attempt

Because the plain fixed `bd_size=16` adapter preserves sequence/order better and
the arg/value span branch introduces the first raw argument hit, two LoRA delta
blends were tested:

```text
blend = plain_bd16_ckpt5 + alpha * (argvalue_ckpt5 - plain_bd16_ckpt5)
```

Source adapters:

- plain sequence/order anchor:
  `runs/fastdllm_qwen35_9b_sequence_value_retention_mix_bd16_from_ckpt275_step10/checkpoint-5/adapter_model`
- arg/value branch:
  `runs/fastdllm_qwen35_9b_bd16_ckpt5_argvalue_mask_w_step10/checkpoint-5/adapter_model`

Blend adapters:

```text
runs/lora_blends/qwen35_bd16_ckpt5_plus_0p1_argvalue_delta/adapter_model
runs/lora_blends/qwen35_bd16_ckpt5_plus_0p25_argvalue_delta/adapter_model
runs/lora_blends/qwen35_bd16_ckpt5_plus_0p5_argvalue_delta/adapter_model
```

Only `0.10` and `0.25` were evaluated; `0.50` was created but not evaluated
after both lower-alpha blends showed the same sequence collapse.

Heldout policy-target forced-prefix results:

| metric | plain bd16 | arg/value branch | blend 0.10 | blend 0.25 |
|---|---:|---:|---:|---:|
| raw valid JSON | `0/12` | `0/12` | `1/12` | `1/12` |
| raw exact name set | `0/12` | `3/12` | `1/12` | `0/12` |
| raw exact sequence | `0/12` | `1/12` | `0/12` | `0/12` |
| raw exact arguments | `0/12` | `1/12` | `0/12` | `0/12` |
| constrained exact name set | `7/12` | `6/12` | `7/12` | `7/12` |
| constrained exact sequence | `6/12` | `4/12` | `3/12` | `3/12` |
| constrained exact arguments | `0/12` | `0/12` | `0/12` | `0/12` |
| total extra calls | `1` | `10` | `0` | `1` |
| total missing calls | `26` | `19` | `26` | `25` |
| total repeated calls | `1` | `10` | `0` | `1` |

Blend eval outputs:

```text
runs/target_geometry_eval/bd16_argvalue_blend0p1_checkpoint5_policy_targets_forcedprefix.summary.json
runs/target_geometry_eval/bd16_argvalue_blend0p25_checkpoint5_policy_targets_forcedprefix.summary.json
```

Interpretation:

- Simple LoRA interpolation is not a good split-route mechanism here.
- The blends inherit a small syntax-validity benefit (`1/12` raw valid JSON),
  but lose the plain bd16 sequence/order gain (`6/12` constrained sequence down
  to `3/12`).
- The arg/value behavior is not linearly separable from route/order behavior in
  these LoRA deltas.
- Next direction should be a separate repair/infill path or sampler-time
  validator/remask, not weight blending.

## Scalar Repair Sidecar Transfer Test

The next split-route test kept the plain fixed `bd_size=16` checkpoint-5 as the
main generator and applied the existing public multi-call scalar repair adapter
to its constrained drafts.

Generator draft:

```text
runs/target_geometry_eval/bd16_checkpoint5_policy_targets_forcedprefix.jsonl
```

Scalar repair adapter:

```text
runs/fastdllm_qwen35_9b_toolcall_multicall_scalar_argspanw1p5_b896_step100/checkpoint-100/adapter_model
```

Output:

```text
runs/target_geometry_eval/bd16_checkpoint5_scalar_repair_sidecar_policy_targets.summary.json
```

Evaluator:

```text
scripts/eval_fastdllm_toolcall_scalar_repair_outputs.py
```

The evaluator now sets the generation-default arguments expected by the shared
Fast-DLLM generation helper, so this path can reuse the same repair generation
code as the other tool-call evals.

Result:

| metric | draft | scalar repair | scalar repair + constrained projection |
|---|---:|---:|---:|
| valid tool JSON | `12/12` | `12/12` | `12/12` |
| exact tool name set | `7/12` | `7/12` | `7/12` |
| exact tool sequence | `6/12` | `6/12` | `6/12` |
| exact arguments | `0/12` | `0/12` | `0/12` |
| all schema valid | `7/12` | `9/12` | `9/12` |
| all required args present | `8/12` | `10/12` | `10/12` |
| total extra calls | `5` | `5` | `5` |
| total missing calls | `6` | `6` | `6` |
| total repeated calls | `6` | `6` | `6` |

Runtime:

- elapsed: `399.137s`
- generated tokens: `3271`
- tokens/sec: `8.195`
- scalar generation calls: `28`
- scalar repaired calls accepted: `2`

Interpretation:

- The scalar sidecar transfers as a syntax/schema repair aid: schema-valid
  records improve from `7/12` to `9/12`, and required-argument presence improves
  from `8/12` to `10/12`.
- It does not fix the heldout policy-target argument grounding problem:
  exact arguments stay at `0/12`.
- It also does not change route/order: exact sequence stays at `6/12`, with the
  same extra/missing/repeated call totals.
- This confirms the current split of responsibilities. Keep the plain
  `bd_size=16` generator as the route/order anchor, use scalar repair as a
  constrained decoding/repair diagnostic, but do not expect the public scalar
  sidecar alone to solve heldout policy arguments.
- The next model-side branch should be skeleton-then-infill or tool-sensitive
  remasking with value/evidence targets, not direct LoRA blending and not more
  of the same scalar sidecar training.
