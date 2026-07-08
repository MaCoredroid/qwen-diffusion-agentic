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
