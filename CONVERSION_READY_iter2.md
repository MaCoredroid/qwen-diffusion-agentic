# CONVERSION_READY (iteration-2) — K-carrying re-conversion inputs (do NOT execute here)

**Status:** iteration-2 SWE-SFT arms (windowed pool, retrain-freely) **both PASS KILL-T1**
(2026-07-13). D1(iter2) recorded in `swe_tuning_campaign_design.md` STATUS (2026-07-13, later). This
file is the iter2-suffixed twin of `CONVERSION_READY.md`: it freezes the iteration-2 inputs the next,
monitor-dispatched step (the K-track re-conversion of the iteration-2 M_swe base) needs. Authority:
`k_raise_campaign_design.md` + the convert-after-RL #29 protocol. **Nothing here is launched by this
turn** (conversion is a separate GPU-tenant step). The design procedure is UNCHANGED from
`CONVERSION_READY.md`; only the SFT base objects (and their provenance = the shape-corrected windowed
pool) advance.

---

## 0. Advancing objects (D1 default, iteration-2)

| role | object | how to build (diffusion-loadable base for re-conversion) |
|---|---|---|
| **PRIMARY** | **M_swe_S (iter2)** = init+RL-v2 + windowed SWE-SFT | merge `runs/swe_sft_arm1_iter2/Aswe_S_step400_seed71101/checkpoint-400` into `models/qwen3.5-9b-fastdllm-mtplus1-merged` |
| **CONTROL TWIN** | **M_swe_T (iter2)** = stock init + windowed SWE-SFT | merge `runs/swe_sft_arm2_iter2/Aswe_T_step400_seed71101/checkpoint-400` into `models/qwen3.5-9b-fastdllm-init` |

Both re-convert (twin@K1 par-power gate decides per-arm). D2 tiebreak = **N=5 AR SWE resolve@1**
(design step 2b), **still not measured** — the one open input before committing K-track spend to a
single arm. The iteration-2 shape fix targets the Tier1-C46 edit-commitment deficit (3/48 in
iteration-1, `k_raise_campaign_design.md` STATUS-2026-07-10) — that is a SWE-resolve measurement, a
separate step, NOT this tool-call anchor gate.

## 1. Merged SFT weights path / adapter+base recipe (the re-conversion base)

The K-track re-conversion needs a **diffusion-loadable** merged base (mask token 248077 / bd_size 32 /
bridge preserved) — **NOT** the AR vllm-bf16 exports (those strip mask/bridge). Use the HF-stack
CPU-exact merge (bit-identical, gate reads maxabs 0.0), exactly as iteration-1:

- Script: `scripts/merge_adapter_into_fastdllm_candidate.py --device cpu`
- **M_swe_S iter2 (primary):** `--init models/qwen3.5-9b-fastdllm-mtplus1-merged` `--adapter runs/swe_sft_arm1_iter2/Aswe_S_step400_seed71101/checkpoint-400` `--out models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged`
- **M_swe_T iter2 (twin):** `--init models/qwen3.5-9b-fastdllm-init` `--adapter runs/swe_sft_arm2_iter2/Aswe_T_step400_seed71101/checkpoint-400` `--out models/qwen3.5-9b-fastdllm-mswe-T-iter2-merged`
- Adapter recipe (both, frozen, identical to iter-1): r16/α32 dropout0.05, W += 2.0·(B@A); 11 targets = q,k,v,o + GDN in_proj_{qkv,z,a,b} + out_proj + **MLP gate_up_proj/down_proj** (`lora_merge_count=184` verified on the AR export of BOTH iteration-2 arms). Merge sanity gate must PASS (mask 248077, bd_size 32, has_weights). ANY failure ⇒ KILL-1.
- **AR-serving exports already built** (for the anchor gate this turn; NOT for re-conversion): `models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16`, `models/qwen3.5-9b-fastdllm-mswe-T-iter2-vllm-bf16`.

## 2. Provenance delta vs iteration-1 (why these are fresh objects)

- **Data shape:** iteration-1 trained on a single **front-truncated** window/episode (assistant-label retention 69.88 %, early-third labels 0.2 %). Iteration-2 trains on **episode-windowing** (Amendment C, §2.3): serve-exact sliding windows tiling each episode, retention **100 %**, mid-episode context management now a training target. Builder `runs/swe_datagen_s1/build_windowed_dataset.py` (seed 71101, block 12288, ctx_overlap 3072, cap 6).
- **Pool:** tranche-2-promoted **383 keepers** → **987 windows** (`data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl`; 0 rows left-truncated, max_seq 12286).
- **Leakage firewall UNCHANGED:** enforced holdout = inner5 ∪ tier0_20 ∪ tier1_100 (113 ids), KILL-D1 hash-asserted; windows drawn from keeper episodes only.

## 3–7. K schedule / entry gate / serving cert / budget / open item

**UNCHANGED from `CONVERSION_READY.md` §§3–7** (the K-track procedure, the Tier1-C46 twin@K1
≥12/46 entry gate, the serving cert plan, budget, and the N=5 AR SWE resolve D2 open item all carry
forward verbatim). The only substitution is the advancing base objects above.

---

## GATE FREEZE — KILL-T1 iteration-2 (2026-07-13)

AR-mode tool-call matched-20 exact_args, paired McNemar vs each arm's own iteration-1 base anchor
(reused, not regenerated), `gold_sha256` mismatch 0/63 both arms:

| arm | base anchor (iter-1) | post-SFT iter2 | McNemar b / c / p | KILL-T1 |
|---|---:|---:|---|:--|
| **S** (M_swe_S iter2) | 49/63 | **49/63** | 0 / 0 / 1.0 | **PASS** — CONVERSION_READY (iter2) |
| **T** (M_swe_T iter2) | 51/63 | **51/63** | 0 / 0 / 1.0 | **PASS** — CONVERSION_READY (iter2) |

Both arms are **CONVERSION_READY (iteration-2)**. Gate JSONs:
`runs/swe_sft_arm1_iter2/anchor_gate/anchor_mcnemar_result.json`,
`runs/swe_sft_arm2_iter2/anchor_gate/anchor_mcnemar_result.json`.
