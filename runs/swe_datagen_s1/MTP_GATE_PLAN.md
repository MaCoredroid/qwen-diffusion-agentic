# MTP SPECULATIVE-DECODE GATE — datagen teacher wall-clock lever

**STATUS: STAGED, NOT ACTIVE.** Nothing in this document is live. The stock-AR
teacher (`runcage_ar_probe.sh`) remains the default and is what the running
orchestrator uses. This is a turnkey runbook for a LATER workflow to flip the
datagen teacher to MTP self-speculative-decode **at a cycle boundary**, iff the
frontier endgame probe verdicts CONTINUE (15+ cycles to the 400-keeper floor).

---

## 0. Feasibility verdict — WORKABLE PATH, no hard blocker

Verified on this host (CPU-only inspection, GPU untouched):

| Check | Result |
|---|---|
| MTP head in pinned snapshot @c202236 | **YES** — 15 `mtp.*` tensors in `model.safetensors.index.json`; one full-attention draft decoder layer, ~464 MiB bf16. No separate/draft checkpoint. |
| `config.json` text_config | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` (draft shares base `embed_tokens` + `lm_head`) → `n_predict = 1`. |
| vLLM registry (0.23.0, `.venv-vllm`) | `"Qwen3_5MTP": ("qwen3_5_mtp", "Qwen3_5MTP")` present (registry.py:640); model file `model_executor/models/qwen3_5_mtp.py` present. |
| Method auto-resolve | `model_type qwen3_5` → `qwen3_5_mtp` → normalized `self.method = "mtp"`, `n_predict = mtp_num_hidden_layers = 1` (config/speculative.py:460–473, 550–561). |
| Draft weights source | Auto-loaded from the SAME snapshot: `self.model = self.target_model_config.model` (speculative.py:561). No extra download. |
| CLI flags exist in this vLLM | `--speculative-config` and `--mamba-cache-mode {align,all,none}` both confirmed via `vllm serve --help=all`. |
| Losslessness | Default `rejection_sample_method="standard"` (probabilistic) = **distribution-preserving** ⇒ teacher output is identical in law to plain AR decode. Do NOT set `"synthetic"`. |
| cudagraph | qwen3_5 is NOT special-cased to `enforce_eager` (only deepseek_v32 is, speculative.py:562) ⇒ cudagraph stays on. |
| Composes with GDN hybrid | Qwen3.5 is IsHybrid + HasInnerState and does NOT declare SupportsMambaPrefixCaching ⇒ with prefix-caching ON, vLLM 0.23 auto-resolves `mamba_cache_mode` to `align`. Draft layer is full-attention (adds attention-KV only, no extra mamba/GDN state). We pass `--mamba-cache-mode align` explicitly (belt-and-suspenders). |

**Why quality is unchanged in principle:** speculative decode drafts `n_predict`
tokens with the MTP head, then the base model verifies them in one forward;
standard (probabilistic) rejection sampling accepts/repairs so the emitted token
stream has the *same distribution* as plain AR. It trades GPU compute for
wall-clock; it does not change *what* the teacher emits. The only real risk
surface is vLLM-0.23 MTP × GDN-hybrid × `max_num_seqs=4` correctness/perf on the
RTX 5090 — which is exactly what this empirical gate measures before promoting.

---

## 1. Artifacts staged (this change)

- **`runcage_ar_mtp.sh`** — byte-for-byte copy of `runcage_ar_probe.sh` (same
  snapshot, same vLLM, same cage/env conventions) + exactly two additions:
  - `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": <NUM_SPEC_TOKENS>}'`
  - `--mamba-cache-mode align` (explicit; already the auto-resolved value)
  - env-overridable `NUM_SPEC_TOKENS` (default `1` = `n_predict`).
- **`datagen_gen.sh`** — minimal, safe edit (invoked fresh per cycle):
  - new selector `RUNCAGE_SCRIPT="${RUNCAGE_SCRIPT:-runcage_ar_probe.sh}"` (default = stock AR).
  - server boot now execs `$HERE/$RUNCAGE_SCRIPT` and threads `NUM_SPEC_TOKENS` through.
  - boot log line prints `runcage=$RUNCAGE_SCRIPT` for observability.
  - **Default behavior is byte-identical to before** — with no env override the
    stock-AR probe boots exactly as it does today.

Nothing here is wired into the orchestrator. The orchestrator (`datagen_orch.sh`)
calls `datagen_gen.sh` as a plain subprocess (orch.sh:148) that inherits the
orch's environment, so the flip is purely an env var at orch launch.

---

## 2. THE GATE — run at a cycle boundary, standalone (does NOT touch the orch)

Preconditions: run only when the orchestrator is between batches (or paused) so no
other heavy GPU job overlaps — the campaign's "ONE heavy job-class at a time"
invariant must hold. The gate boots its OWN caged MTP server on a scratch port and
tears it down; it does not modify `attempts.jsonl`, `keepers/`, or the ledger.

### 2a. Boot the MTP server (caged, scratch port)
```
cd /home/mark/qwen_diffusion
GATE=runs/swe_datagen_s1/gate_mtp_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$GATE"
# derive gpu_util from measured free exactly as datagen_gen.sh does (desktop-drift safe)
read -r U T < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', *' 'NR==1{print $1,$2}')
GU=$(python3 -c "print(f'{min(0.85,($T-$U-1800)/$T):.2f}')")
systemd-run --user --scope --unit=gate_mtp -p MemoryMax=22G -p MemorySwapMax=4G \
  bash -c "MAX_NUM_SEQS=4 MAX_MODEL_LEN=32768 GPU_UTIL=$GU PORT=9971 NUM_SPEC_TOKENS=1 \
           bash runs/swe_datagen_s1/runcage_ar_mtp.sh" > "$GATE/server.log" 2>&1 &
# wait for readiness (mirror datagen_gen.sh wait_ready): curl :9971/v1/models until 200, boot deadline 600s
```
**GATE-CHECK A (boot):** server reaches `/v1/models` 200 within the boot deadline,
`server.log` shows the spec-decode config accepted (method mtp, n_predict=1) and
NO error/traceback, and (from the log) cudagraph is on (not forced eager) and
`mamba_cache_mode=align`. Fail any → ABORT, do not promote.

### 2b. Correctness through the FULL pipeline — 2 gym + 2 Verified bounded episodes
Drive 4 real bounded episodes through the same driver `datagen_gen.sh` uses
(`scripts/run_swe_bench_qwen_code.py`, `--runtime container`, native `qwen3_xml`,
reference envelope 0.6/0.95/20, `--eval-mode skip` for gen), pointed at the MTP
endpoint `http://127.0.0.1:9971/v1`, then score with the SAME official fork harness
the campaign uses (`datagen_score.sh` path). Pick 2 gym + 2 Verified-adjacent
instance ids that the campaign has ALREADY resolved with the stock teacher (so a
correct MTP teacher is expected to also produce a scorable, resolvable rollout).

**GATE-CHECK B (correctness):**
- all 4 episodes produce a non-empty prediction (predictions flow end-to-end),
- all 4 get a REAL verdict from the official scorer (no `no_prediction`, no infra_invalid),
- zero server-side errors/tracebacks in `server.log` across all 4,
- resolved/plausible rate is consistent with the stock teacher on the same ids
  (spec-decode is lossless, so a large divergence in outcome = red flag, investigate).

### 2c. Throughput A/B — MTP vs plain, same 2 prompts
Using 2 fixed prompts (reuse two of the 2b first-turns, or two canned long-context
SWE first-turns), measure decode tokens/s from vLLM usage metrics (`completion_tokens`
/ server-side generation time) on BOTH servers, one at a time (never co-resident):
1. MTP server (already up from 2a) — record tokens/s per prompt.
2. Stop `gate_mtp.scope`, settle GPU, boot `runcage_ar_probe.sh` on :9971 with the
   same MAX_NUM_SEQS/MAX_MODEL_LEN/GPU_UTIL, replay the same 2 prompts, record tokens/s.
Compute `speedup = mean(tokens/s MTP) / mean(tokens/s plain)`.

**GATE-CHECK C (throughput):** `speedup >= 1.20`.
(Optional: if C is borderline, re-probe MTP with `NUM_SPEC_TOKENS=2` — note this
loops the single MTP layer autoregressively; vLLM warns lower acceptance, so it may
help or hurt. Keep 1 unless 2 clearly wins on the A/B.)

### PROMOTE IFF: GATE-CHECK A ok AND B (all 4 flow + score, zero errors) AND C (speedup >= 1.2x).
Any failure → do NOT promote; leave the stock-AR teacher live; append the numbers
and verdict to `$GATE/verdict.txt`.

---

## 3. PROMOTION (flip to MTP) — one env var, at a cycle boundary

The orchestrator reads its env at launch and passes it to `datagen_gen.sh`. Kill +
restart is resume-safe by design (`attempts.jsonl` is the source of truth; ledger
`nextbatch` = frontier-order minus already-attempted). So promotion is:

1. Stop the orch cleanly at a cycle boundary (between batches):
   `kill "$(cat runs/swe_datagen_s1/orch.pid)"` (or wait for its current batch to finish).
2. Relaunch detached WITH the override (and same knobs it runs with today):
```
cd /home/mark/qwen_diffusion
RUNCAGE_SCRIPT=runcage_ar_mtp.sh NUM_SPEC_TOKENS=1 \
  setsid bash runs/swe_datagen_s1/datagen_orch.sh \
  >runs/swe_datagen_s1/logs/orch.log 2>&1 &
echo $! > runs/swe_datagen_s1/orch.pid
```
3. Confirm the next batch's `logs/gen_server.log` / boot line shows
   `runcage=runcage_ar_mtp.sh` and the spec-decode config, and that the batch scores
   normally (watch the first post-flip batch's yield vs recent rolling — lossless, so
   yield should be statistically unchanged; a drop = investigate/rollback).

## 4. ROLLBACK — instant, one line

Relaunch the orch WITHOUT the override (default = stock AR):
```
cd /home/mark/qwen_diffusion
setsid bash runs/swe_datagen_s1/datagen_orch.sh \
  >runs/swe_datagen_s1/logs/orch.log 2>&1 &
echo $! > runs/swe_datagen_s1/orch.pid
```
Because `RUNCAGE_SCRIPT` defaults to `runcage_ar_probe.sh`, unsetting the env var
fully reverts. No file edits, no re-staging. Rollback is safe mid-campaign at any
cycle boundary.

---

## 5. Risk register / notes
- **max_num_seqs=4 × MTP × GDN:** the one un-de-riskable-on-CPU surface; GATE-CHECK
  A/B exercise it on real concurrency-representative episodes before any promotion.
- **VRAM:** draft head is ~464 MiB extra weights + a small draft attention-KV; the
  gen-time gpu_util derivation already leaves desktop headroom. If boot OOMs, the
  gate ABORTS at CHECK A (no promotion) — lower GPU_UTIL is NOT a valid workaround
  to force it through; investigate first.
- **Do NOT** export `VLLM_USE_V2_MODEL_RUNNER=1` in the datagen env (would change
  the mamba cache path assumptions).
- **Do NOT** set `rejection_sample_method: "synthetic"` — that breaks losslessness.
- `NUM_SPEC_TOKENS > 1` is legal (divisible by n_predict=1) but loops the single MTP
  layer; start at 1.
