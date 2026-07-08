# TEACHER SWAP — epoch 2: stock-9B-AR → Qwen3.6-27B NVFP4 + native MTP

Durable operational record (DATAGEN_STATUS.txt is auto-overwritten by `ledger.py state`
each cycle, so the narrative lives here + in `../../swe_endgoal_plan.md`).

## What changed
Datagen teacher swapped per user directive. All three acceptance gates PASSED
(`gate_27b_20260708T181604Z/gate_verdict.json`: format schema-equiv 0 mismatches;
arg-grounding 63/64 = 98.4% source-verbatim, 0 malformed; live 4/4 resolved; MTP A/B
1.59× @ 0.93 accept). GPU was idle + host RAM clean at swap time (one server at a time).

## Frozen/certified server config (bootprobe_27b/FROZEN_CONFIG.json)
- checkpoint: `nvidia/Qwen3.6-27B-NVFP4` (modelopt MIXED_PRECISION, mtp_num_hidden_layers=1)
- quant `modelopt_fp4` (NVFP4 mixed; fp8 linear-attn projs); spec `qwen3_5_mtp` n=1
- **MAX_NUM_SEQS=2** (2×32k always resident, zero preemption; do NOT copy the 9B C=4)
- **KV_OFFLOAD_GB=0** (frozen OFF: offload drops expandable_segments + shrinks the fp8 KV
  pool 83012→77550; net-negative when 2 seqs fit. Validated opt-in lever KV_OFFLOAD_GB=4|8,
  hard-capped ≤8G per HOST-RAM directive)
- KV_CACHE_DTYPE auto → **fp8_e4m3** (ckpt kv_cache_quant_algo=FP8)
- **ATTENTION_BACKEND=TRITON_ATTN** (decode); head-256 fp8 prefill auto-routes to FlashInfer
- APC: `--enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align
  --mamba-block-size 1024 --mamba-ssm-cache-dtype float32 --gdn-prefill-backend triton`
- tools: `qwen3_xml` tool-parser + `qwen3` reasoning-parser + codex chat template
- gmu dynamic `min(0.85,(total-used-1800)/total)` floor 0.74
- measured: boot ~46s, on-GPU KV 83,012 tok (2.53× @32k), GPU headroom 4343 MiB,
  cage RSS 11.5G, host-avail ~23G, MTP accept 0.935 / mean-accept-len 1.935

## Wiring
- `datagen_gen.sh`: `RUNCAGE_SCRIPT` default → `runcage_27b.sh`; per-teacher `case` sets
  the driver-requested model (`qwen3.6-27b-nvfp4` == launcher `--served-model-name`), the
  keeper teacher label, and the certified primitives (TRITON_ATTN / KV auto / offload 0),
  all passed through the RAM cage to the launcher.
- `datagen_orch.sh`: single knob `RUNCAGE_SCRIPT` → resolves+exports the teacher, sets the
  certified concurrency `C=2`, exports `TEACHER_LABEL` + `KEEPER_GENERATOR`.
- `extract_keepers.py`: keepers stamp `provenance.teacher = qwen3.6-27b-nvfp4-mtp` +
  `provenance.generator` (from env; fallback = historical 9B).

### ONE-LINE ROLLBACK to the stock-9B-AR teacher
```
RUNCAGE_SCRIPT=runcage_ar_probe.sh C=4 setsid bash runs/swe_datagen_s1/datagen_orch.sh \
  >>runs/swe_datagen_s1/logs/orch.log 2>&1 & echo $! > runs/swe_datagen_s1/orch.pid
```

## Epoch governance (amendment, not tampering)
New teacher == new epoch. `ledger.py`:
- `ledger.py epoch <attempts.jsonl> <label>` appends `{"epoch_marker":true,"epoch":...}`.
- The rolling KILL window counts only valid real attempts AFTER the latest marker, so the
  27B is judged on its OWN yield — it does NOT inherit the 9B epoch's spent 0.075 window.
- UNCHANGED: kill bar (0.10/200), lifetime yield, keepers, coverage/eligibility,
  DONE_EXHAUSTED (all span every epoch).
- Epoch-1 kill record preserved as `DATAGEN_KILL_9b_epoch1.txt`.
- Verified: pre-marker `KILL_YIELD_COLLAPSE rolling 0.075` → post-marker `CONTINUE` window 0,
  lifetime unchanged 0.2362, keepers 282, `nextbatch` still draws the fresh pandas/getmoto/dask head.

## Fresh-coverage probe — cancelled by monitor (batch 2)
`probe_freshcov_20260708T153555Z`: BATCH 1 complete (50 attempts, 12 resolved, pooled fresh
yield 0.24 > 0.15 bar); BATCH 2 CANCELLED (teacher switch moots the 9B-continuation question).
Batch-1 artifacts + 12 isolated keepers preserved, NOT promoted. See the probe dir's
`CANCELLED_BY_MONITOR.md`.

## ON-SPEC CONVERGENCE + single relaunch (2026-07-08T~20:10Z) — teacher-stack-converge workflow

The standup's early epoch-2 relaunch was intentionally superseded (GPU-contention stop during
the re-cert); this workflow owns the **single on-spec relaunch on the final stack**. Coordination
guardrail honored: both sibling workflows had already landed (standup: teacher swap 69b39c3 +
ledger epoch support + `DATAGEN_KILL_9b_epoch1.txt` archive + C=2; flywheel: qwen-code 0.19.4
bf7c0cf/1a104c1). GPU free, no orch running, orch.pid stale — clean launch, no boundary swap needed.

### RE-CERT (banked): parser + template
- **Parser = `qwen3_xml` FINALIZED.** RAW template-native completions for 5 tool-call cases
  round-tripped through vLLM `qwen3_xml` AND `qwen3_coder`: both byte-faithful AND byte-identical
  extractions (incl. a 318-byte multiline write and a 99-byte diff). Tie broken to `qwen3_xml` =
  what the live server, production datagen, and all 282 keepers already run (zero-risk, consistent).
- **Template = codex RETAINED.** Adjudicated: the codex and ckpt-native templates instruct the
  *identical* `<tool_call><function=NAME><parameter=key>…` XML (signature (True,True,True) both) and
  the same `enable_thinking` gating — the task premise that native emits a different XML needing a
  parser switch is FALSE. Codex = format-equivalence-by-construction with the 9B keepers.

### The actual on-spec gap that remained: the ENVELOPE + THINKING mode (not the parser/template)
The gate-2 config audit (`gate_27b_20260708T181604Z/CONFIG_DELTAS.md`) found the serving flags
already correct; the defect was the sampling/thinking chimera inherited from the non-thinking 9B AR
teacher (thinking forced OFF by the proxy + a thinking-mode coding sampler 0.6/0.95/20). Fixed to
**Regime T** (Qwen3.6 official thinking-general agentic path == ckpt `generation_config` defaults):

| delta | change | where |
|---|---|---|
| D2 envelope | 0.6 → **1.0 / 0.95 / 20 / min_p 0 / pp 0** (thinking ON) | `datagen_gen.sh` case block (teacher-coupled) + `datagen_orch.sh` `ENVELOPE_JSON` keeper stamp |
| D6 thinking | proxy `enable_thinking` unconditional-False → **env-gated `LUMO_ENABLE_THINKING`** (default-OFF ⇒ 9B rollback stays byte-identical; 27B sets true) | `scripts/qwen_code_sglang_proxy.py` |
| D7 max-tokens | `DEFAULT_PROXY_MAX_TOKENS` 2048 → **8192** (thinking trace + patch headroom; gate-2 ran 8192) | `scripts/run_swe_bench_qwen_code.py` |
| parser | `qwen3_xml` (unchanged, recert-finalized) | `runcage_27b.sh` |
| C | **2** (server `max_num_seqs=2`); orch resolves it, gen passes `$C`; gen's standalone default 4 is never reached via the orch | `datagen_orch.sh` (already), `datagen_gen.sh:MAX_NUM_SEQS=$C` |

All teacher-coupled so the one-line 9B rollback (`RUNCAGE_SCRIPT=runcage_ar_probe.sh C=4`) stays
byte-faithful (0.6/0.95/20, thinking OFF). `bash -n` + `py_compile` clean.

### FIRST-BATCH VERIFICATION — PASS (batch_0001_20260708T200704Z, orch.pid 2830633, detached)
- **Boot on-spec:** `served=qwen3.6-27b-nvfp4 seqs=2 quant=modelopt_fp4 spec=qwen3_5_mtp/1 kv=auto
  kv_offload=0 attn=TRITON_ATTN`, codex template + `qwen3_xml` + reasoning `qwen3`; KV pool **83,012 tok
  (2.53x @32k)**; engine init 10.0 s; `[ready] :9951`.
- **Envelope forwarded (proxy dump `chat_0001.json`):** `temperature 1.0, top_p 0.95, top_k 20,
  min_p 0.0, presence_penalty 0.0, seed 1001234, max_tokens 8192`, `chat_template_kwargs
  {enable_thinking: true}`, 15 tools.
- **Frontier-head draw:** pandas-dev__pandas-47446 / getmoto__moto-4867 / dask__dask-10149.
- **Live thinking-mode rollout:** `Running: 2 reqs`, gen ~50 tok/s, prefix-cache hit ~50%, turns
  advancing (chat_0010), `finish_reason` tool_calls+stop, `completion_tokens` 182–333 (thinking +
  tool call). Pull was ~80 s (49/50 images cached); GPU 29.7–29.9 GB / 97–98 %.

### #98 (promote the 12 isolated 9B probe keepers) — DEFERRED, not left-undone
`extract_keepers.py` **dedups by instance_id and skips** any instance already in `keepers.jsonl`.
The 12 probe keepers are **9B**-generated for pandas/getmoto/dask — the exact frontier HEAD the 27B
is re-covering right now. Promoting them would make the 27B's higher-quality keepers **silently
skipped** for those 11 fresh instances (dask-10212 is the 12th; already in prod attempts). That
would lock in inferior data and defeat the teacher swap. Decision: let the 27B cover them first; the
12 keepers stay preserved in `probe_freshcov_20260708T153555Z/keepers/`. **Fallback recipe** (only
for instances the 27B ultimately fails to resolve): `awk`-filter those iids from the probe keepers,
append to prod `keepers/keepers.jsonl` (dedup), stamp `provenance.teacher=stock-qwen3.5-9b-ar`; do
NOT add post-epoch-marker attempts rows (would misattribute 9B resolves to the 27B rolling window).
