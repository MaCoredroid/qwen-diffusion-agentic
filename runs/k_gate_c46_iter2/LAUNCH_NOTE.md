# C46 RE-GATE ITERATION-2 — LAUNCH NOTE (#129)

**Launched 2026-07-13 ~20:34Z, detached (setsid + cage), ~4 h expected.** This is the
iteration-2 re-run of the Tier1-C46 twin@K1 entry gate. It mirrors the iteration-1 gate
(`runs/k_gate_c46/launch.sh` + `launch_ar.sh`, run scripts + frozen shard_plan + official
docker scoring + `build_report.py`/`build_ar_paired_report.py`) EXACTLY, with only the four
pre-registered iteration-2 changes below. Same frozen 48-instance pool
(`runs/k_gate_c46/shard_plan.json`, pool_sha256 `49d8f46dc202bf50…`), same envelope
(temp 0.6 / top_p 0.95 / top_k 20, NO presence_penalty, per-shard base seeds
{1234,101234,201234,301234}, turn cap 75, empty-patch re-drive 1, c=4), same official
swebench-harness scoring, same `>=12/46` entry bar.

## Iteration-2 changes (ONLY these)
- **(a) diffusion arm** = `models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16`, served FLARE
  hybrid_clean K=1 (max_model_len 32768, mask 248077, **gmu 0.74 / max_num_seqs 4**), **WITH
  the CERTIFIED read-clamp proxy shim active** — `runs/k_gate_c46/proxy_readclamp.py`
  (certified 7ae55d4), wired exactly as the cert did: `--proxy-script` +
  `LUMO_PROXY_READCLAMP_LIMIT=100` (injects a bounded `limit` into `read_file` calls that drop
  it; offset/file_path untouched). Runner: `run_arm_twin.sh` (a clone of the certified
  `run_candidate.sh`, ARM=diffusion).
- **(b) AR arm** = `models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16` AR-decoded — the SAME
  fold the iteration-2 KILL-T1 anchor gate served (#127, `2632c13`; the arm-S windowed-pool
  SWE-SFT export). Served via `runs/stage_c_driver/runcage_ar.sh` SNAP-override, stock vLLM
  0.23 AR, **gmu 0.85 / max_num_seqs 4**. Runner: `run_arm_ar.sh`.
- **(c) ctx-overflow truth-telling labels active** (`52ffcc2`): both report builders bucket
  `terminal_cause=="ctx_overflow"` cap-deaths as a distinct `ctx_overflow_deaths` bucket,
  kept OUT of `clean_exit0` / `empty_patches`.
- **(d) output** = `runs/k_gate_c46_iter2/`.

## MEMORY-BUDGET RULE (honored)
The diffusion arm's gmu (0.74) is NEVER the AR arm's (0.85). The GDN align-cache lives
OUTSIDE the KV pool, so the iteration-1 gate's measured gmu/concurrency (twin 0.74/4,
AR 0.85/4) is authoritative and not copied across arms.

## Runner (self-bounded, one server at a time)
`run_gate.sh` (pidfile `gate.pid`; detached via `setsid`; caged servers via
`systemd-run --user --scope`): verify 48 images + GPU-idle preflight → **twin+clamp arm to
completion + teardown + GPU-settle → AR arm + teardown + GPU-settle** → OFFICIAL docker
scoring both (servers DOWN) → gate reports: `build_report.py` (twin resolve@1 vs the
**>=12/46** entry bar + ctx_overflow accounting) and `build_ar_paired_report.py` (**McNemar
twin-vs-AR paired** + locus + ctx_overflow). `[state]` lines (`eps done / 48`, wall) emit to
`logs/run_gate.log` every 60 s per arm. **STOP-file** `runs/k_gate_c46_iter2/STOP` aborts
gracefully (server torn down, exit 9).

Docker is via the docker group (plain `docker`) on this host — no `sudo -A`/SUDO_ASKPASS
(the iteration-1 `sudo -A docker` path is not available here). The scoring dataset is the
committed `runs/k_gate_c46/inputs/swe_verified_c46.json` (frozen at build_inputs time).

## Launch evidence (verified before exit)
- gate pid **816168** alive; twin server **:9952 up** (~45 s boot).
- **proxy smoke PASS** — a real prior divergence `read_file` routed through the enabled clamp
  against the LIVE iter-2 server reassembled a well-formed, `[DONE]`-terminated stream.
- first episodes producing **real turns**: 4 shards fanned out (12 ids each, correct seeds);
  ~147 server chat-completions (200 OK), ~84 turn dumps across shards, **11 clamp injections
  logged** (shim actively firing), first episode completed (`[state] eps=1/48`).

## Pending verdict
Report writes `report.md` / `report.json` (twin `>=12/46` ENTRY-PASS vs INCONCLUSIVE-BY-POWER)
and `AR_PAIRED_READ.md` / `ar_paired_report.json` (McNemar twin-vs-AR). Per #129, a `<12/46`
result is a principled STOP, adjudicated honestly against the AR paired read.
