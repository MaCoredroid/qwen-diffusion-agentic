#!/usr/bin/env python3
"""FLARE hybrid-clean ENGINE best-of-N (GRPO same-prompt) bench.

The GRPO rollout regime: N samples of the SAME prompt (shared prefix -> APC
prefix reuse), temp=0.7, per-sample DISTINCT seeds. Measures BOTH:
  Q1 throughput: samples/sec per (prompt, N) group.
  Q2 signal quality: unique-output frac, unique-argument-set frac, valid frac,
     per-sample exact_args -> pass@1 / pass@N.

Seeds are NESTED (N=4 seeds subset of N=8 subset of N=16) so pass@N is monotone.
Per group: reset APC (cold prompt) -> ONE engine.generate() of N identical-prompt
requests (the sync scheduler co-batches; APC reuses the shared prefix) -> score.

Engine-side AUDIT (proj==0, verify): a monkeypatch on _hybrid_clean_step reads
EVERY live hybrid_clean decoder's cumulative stats each step; the group's
max value_projection_events must stay 0 (no value ever grammar-projected), forced
tokens present (>0), model value tokens present (>0), and forwards==model_chosen.

One heavy process; RAM cage; foreground; incremental JSONL.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
HERE = ROOT / "runs/p2_bestofn_grpo"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "runs/p2_batched_rollout_bench"))  # gpu_sampler
sys.path.insert(0, str(HERE))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
MANIFEST = HERE / "prompts_manifest.json"
OUT = Path(os.environ.get("BENCH_OUT", str(HERE / "engine_groups.jsonl")))

os.environ["VLLM_FLARE_CUDAGRAPH"] = "1"
os.environ.setdefault("VLLM_FLARE_BIDIR_PROBE", "1")

from gpu_sampler import GpuSampler, host_ram_peak_gb, gpu_snapshot  # noqa: E402
import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_flare_northstar_matched import decode_text, trim_scored_assistant  # noqa: E402
import grpo_metrics as GM  # noqa: E402

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
MAXSEQ = int(os.environ.get("BENCH_MAXSEQ", "16"))
NS = [int(x) for x in os.environ.get("BENCH_NS", "4 8 16").split()]
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
GMU = float(os.environ.get("BENCH_GMU", "0.62"))  # b16 fits at 0.62 (throughput bench)

# ---- forward ledger + proj audit (reads ALL live decoders each step) ----
LED = {"fwds": 0, "nreq": []}
AUD = {"max_proj": 0, "any_forced": False, "any_value": False,
       "fwd_ne_model_chosen": 0, "min_value_tokens": None}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _reset_group_ledger():
    LED["fwds"] = 0
    LED["nreq"] = []
    AUD["max_proj"] = 0
    AUD["any_forced"] = False
    AUD["any_value"] = False
    AUD["fwd_ne_model_chosen"] = 0


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    LED["fwds"] += 1
    LED["nreq"].append(int(input_batch.num_reqs))
    ret = _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                     decode_indices_np, decode_slots_np, valid_len_np,
                     is_committing, num_reqs, input_batch)
    decoders = getattr(self, "_hc_decoders", None)
    if decoders:
        for dec in list(decoders.values()):
            s = dec.stats
            if int(s.value_projection_events) > AUD["max_proj"]:
                AUD["max_proj"] = int(s.value_projection_events)
            if int(s.fsm_committed_tokens) > 0:
                AUD["any_forced"] = True
            if int(s.value_tokens) > 0:
                AUD["any_value"] = True
            if int(s.forwards) != int(s.model_chosen_tokens):
                AUD["fwd_ne_model_chosen"] += 1
    return ret


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step


def main():
    manifest = json.loads(MANIFEST.read_text())
    prompts = manifest["prompts"]
    mask_id = int(manifest["mask_id"])
    block_size = int(manifest["block_size"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    print(f"[eng] n_prompts={len(prompts)} exact={manifest['n_exact']} "
          f"miss={manifest['n_miss']} NS={NS} temp={TEMP} gmu={GMU} maxseq={MAXSEQ} "
          f"block={block_size}", flush=True)

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done.add((int(d["N"]), int(d["global_turn"])))
            except Exception:
                pass
    print(f"[eng] resume: {len(done)} groups already done", flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    from vllm import SamplingParams

    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=SEED,
        gpu_memory_utilization=GMU)
    adapter._engine_kwargs.update({
        "max_num_seqs": MAXSEQ, "max_num_batched_tokens": 4096,
        "enable_prefix_caching": True})
    t0 = time.time()
    engine = adapter._build_engine()
    boot_s = time.time() - t0
    _u0, m0 = gpu_snapshot()
    print(f"[eng] booted boot_s={boot_s:.1f} idle_mem={m0}MB", flush=True)

    def seeds_for(prompt_idx, n):
        base = SEED + 1 + prompt_idx * 10000
        return [base + i for i in range(n)]

    def make_sp(rec, seed):
        tools = [{"type": "function", "function": {"name": nm, "parameters": p}}
                 for nm, p in (rec["schemas"] or {}).items()]
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=TEMP, top_p=1.0,
            seed=int(seed),
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])})

    def gen_group(rec, seeds):
        engine.reset_prefix_cache()  # cold prompt; APC reuse happens WITHIN the wave
        prompts_in = [{"prompt_token_ids": list(rec["prompt_ids"])} for _ in seeds]
        sps = [make_sp(rec, s) for s in seeds]
        outs = engine.generate(prompts_in, sps, use_tqdm=False)
        return outs

    # warmup one wave per distinct N (cudagraph capture + first-call excluded)
    warm = prompts[0]
    for n in sorted(set(NS)):
        gen_group(warm, seeds_for(0, n))
    torch.cuda.empty_cache()

    fh = OUT.open("a")
    for n in NS:
        for pidx, rec in enumerate(prompts):
            gt = int(rec["global_turn"])
            if (n, gt) in done:
                continue
            seeds = seeds_for(pidx, n)
            _reset_group_ledger()
            sampler = GpuSampler(interval=0.2)
            sampler.start()
            t_start = time.time()
            outs = gen_group(rec, seeds)
            wall = time.time() - t_start
            sampler.stop()
            gs = sampler.summary()

            samples = []
            fin_counts = {}
            total_tok = 0
            for i, o in enumerate(outs):
                out0 = o.outputs[0]
                ids = [int(x) for x in out0.token_ids]
                total_tok += len(ids)
                fr = str(getattr(out0, "finish_reason", None))
                fin_counts[fr] = fin_counts.get(fr, 0) + 1
                text = trim_scored_assistant(
                    decode_text(tok, torch.tensor(ids, dtype=torch.long)))
                sc = GM.score_sample(text, rec["tools"], rec["gold_block"])
                samples.append({"seed": int(seeds[i]), "n_tok": len(ids),
                                "token_ids": ids, "text": text,
                                "finish_reason": fr, **sc})
            met = GM.group_metrics(samples)
            nreq = LED["nreq"]
            occ = round(float(np.mean(nreq)) / n, 3) if nreq else 0
            audit = {
                "max_value_projection_events": AUD["max_proj"],
                "proj_zero_ok": AUD["max_proj"] == 0,
                "any_forced_tokens": AUD["any_forced"],
                "any_model_value_tokens": AUD["any_value"],
                "forwards_ne_model_chosen": AUD["fwd_ne_model_chosen"],
                "verify_ok": (AUD["max_proj"] == 0 and AUD["any_forced"]
                              and AUD["any_value"] and AUD["fwd_ne_model_chosen"] == 0),
            }
            import hashlib
            psha = hashlib.sha256(
                json.dumps(list(rec["prompt_ids"])).encode()).hexdigest()
            row = {
                "side": "engine", "global_turn": gt,
                "source_family": rec["source_family"],
                "hf_exact_arguments": bool(rec["hf_exact_arguments"]),
                "prompt_len": rec["prompt_len"], "n_ref": rec["n_ref"],
                "wall_s": round(wall, 3),
                "samples_per_sec": round(n / wall, 4),
                "total_gen_tokens": total_tok,
                "tokens_per_sec": round(total_tok / wall, 2),
                "mean_gen_tokens_per_sample": round(total_tok / n, 1),
                "engine_step_calls": LED["fwds"],
                "engine_step_calls_per_sample": round(LED["fwds"] / n, 2),
                "mean_batch_in_forward": round(float(np.mean(nreq)), 3) if nreq else 0,
                "batch_occupancy_efficiency": occ,
                "seeds": seeds,
                "per_sample": [{k: s[k] for k in
                                ("seed", "n_tok", "valid", "exact", "argset",
                                 "finish_reason")} for s in samples],
                "finish_reasons": fin_counts,
                "audit": audit,
                "prompt_sha256_manifest": rec["prompt_sha256"],
                "gpu_mem_used_mb": gpu_snapshot()[1],
                "host_ram_peak_gb": host_ram_peak_gb(),
                "seed_base": SEED,
            }
            row.update(met)
            row.update(gs)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            print(f"[eng] N={n:2d} gt{gt:3d} {rec['source_family'][:12]:12s} "
                  f"hf={int(rec['hf_exact_arguments'])} wall={wall:.2f}s "
                  f"samp/s={row['samples_per_sec']:.2f} "
                  f"uniqOut={met['unique_output_frac']} "
                  f"uniqArg={met['unique_argset_frac']} "
                  f"valid={met['valid_frac']} pass1={met['pass1']} passN={met['passN']} "
                  f"fwd/s={row['engine_step_calls_per_sample']} occ={occ} "
                  f"proj={AUD['max_proj']} vok={int(audit['verify_ok'])} "
                  f"util~{gs['gpu_util_mean_pct']}%", flush=True)
    fh.close()
    print("[eng] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[eng] FATAL:", repr(e), flush=True)
        traceback.print_exc()
        raise
