#!/usr/bin/env python3
"""STOCK-AR (guided) best-of-N (GRPO same-prompt) bench -- the AR baseline.

Mirrors bench_engine_bestofn.py EXACTLY (same manifest, same nested per-sample
seeds, same N in {4,8,16}, same wave method, same grpo_metrics scorer) but on
stock Qwen3.5-9B (c202236) + stock vLLM guided decoding (regex_from_qwen_xml,
the endgame-scoreboard baseline). AR co-batches N identical prefixes with APC on
and runs its FAST path (cudagraph) -- the conservative baseline for the thesis.

One heavy process; RAM cage; foreground; incremental JSONL.
"""
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")

ROOT = Path("/home/mark/qwen_diffusion")
HERE = ROOT / "runs/p2_bestofn_grpo"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "runs/p2_batched_rollout_bench"))  # gpu_sampler
sys.path.insert(0, str(HERE))
MANIFEST = HERE / "prompts_manifest.json"
OUT = Path(os.environ.get("BENCH_OUT", str(HERE / "ar_groups.jsonl")))
MODEL = os.environ.get(
    "AR_MODEL",
    "/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")

from gpu_sampler import GpuSampler, host_ram_peak_gb, gpu_snapshot  # noqa: E402
import grpo_metrics as GM  # noqa: E402

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
MAXSEQ = int(os.environ.get("BENCH_MAXSEQ", "16"))
NS = [int(x) for x in os.environ.get("BENCH_NS", "4 8 16").split()]
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
GMU = float(os.environ.get("AR_GMU", "0.66"))
EAGER = os.environ.get("AR_EAGER", "0") == "1"


# --- guided_tool_call_regex: byte-identical to the scoreboard / throughput bench ---
def regex_literal(text):
    return re.escape(str(text))


def schema_type(schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        return next((i for i in expected if i != "null"), expected[0] if expected else None)
    return expected


def guided_value_regex(schema):
    if not isinstance(schema, dict):
        return "[^<]*"
    ev = schema.get("enum")
    if isinstance(ev, list) and ev:
        ch = []
        for v in ev:
            if isinstance(v, str):
                ch.append(regex_literal(v))
            elif isinstance(v, bool):
                ch.extend([str(v).lower(), str(v)])
            elif v is None:
                ch.append("null")
            else:
                ch.append(regex_literal(json.dumps(v, ensure_ascii=False)))
        return "(?:" + "|".join(dict.fromkeys(ch)) + ")"
    e = schema_type(schema)
    return {
        "integer": "-?[0-9]+",
        "number": "-?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        "boolean": "(?:true|false|True|False)",
        "array": "\\[[^<]*\\]",
        "object": "\\{[^<]*\\}",
    }.get(e, "[^<]*")


def guided_tool_call_regex(tools):
    alts = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        name = str(fn["name"])
        schema = fn.get("parameters") or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        props = props if isinstance(props, dict) else {}
        required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
        params = []
        for pn, ps in props.items():
            body = (f"<parameter={regex_literal(pn)}>\\n"
                    f"{guided_value_regex(ps)}\\n</parameter>\\n")
            params.append(body if pn in required else f"(?:{body})?")
        alts.append("<tool_call>\\n" + f"<function={regex_literal(name)}>\\n"
                    + "".join(params) + "</function>\\n</tool_call>")
    if not alts:
        return ("<tool_call>\\n<function=[^>]+>\\n(?:<parameter=[^>]+>\\n[^<]*\\n"
                "</parameter>\\n)*</function>\\n</tool_call>")
    return "(?:" + "|".join(alts) + ")"


def decode_text(tok, ids):
    return tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def trim(text):
    text = text.strip()
    end = text.find("</tool_call>")
    return text if end < 0 else text[: end + len("</tool_call>")].strip()


def main():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer

    manifest = json.loads(MANIFEST.read_text())
    prompts = manifest["prompts"]
    for r in prompts:
        r["_regex"] = guided_tool_call_regex(r["tools"])
    print(f"[ar] model={Path(MODEL).name} n_prompts={len(prompts)} "
          f"exact={manifest['n_exact']} miss={manifest['n_miss']} NS={NS} "
          f"temp={TEMP} gmu={GMU} eager={EAGER}", flush=True)

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
    print(f"[ar] resume: {len(done)} groups already done", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    kw = dict(model=MODEL, trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=GMU, max_num_seqs=MAXSEQ,
              max_num_batched_tokens=4096, enable_prefix_caching=True, seed=SEED,
              mamba_cache_mode="align", mamba_block_size=1024, enforce_eager=EAGER)
    t0 = time.time()
    engine = LLM(**kw)
    boot_s = time.time() - t0
    print(f"[ar] booted boot_s={boot_s:.1f}", flush=True)

    def seeds_for(prompt_idx, n):
        base = SEED + 1 + prompt_idx * 10000
        return [base + i for i in range(n)]

    def make_sp(rec, seed):
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=TEMP, top_p=1.0,
            seed=int(seed),
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            structured_outputs=StructuredOutputsParams(regex=rec["_regex"]))

    def gen_group(rec, seeds):
        try:
            engine.reset_prefix_cache()
        except Exception:
            pass
        prompts_in = [{"prompt_token_ids": list(rec["prompt_ids"])} for _ in seeds]
        sps = [make_sp(rec, s) for s in seeds]
        return engine.generate(prompts_in, sps, use_tqdm=False)

    warm = prompts[0]
    for n in sorted(set(NS)):
        gen_group(warm, seeds_for(0, n))

    fh = OUT.open("a")
    for n in NS:
        for pidx, rec in enumerate(prompts):
            gt = int(rec["global_turn"])
            if (n, gt) in done:
                continue
            seeds = seeds_for(pidx, n)
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
                text = trim(decode_text(tok, ids))
                sc = GM.score_sample(text, rec["tools"], rec["gold_block"])
                samples.append({"seed": int(seeds[i]), "n_tok": len(ids),
                                "token_ids": ids, "text": text,
                                "finish_reason": fr, **sc})
            met = GM.group_metrics(samples)
            row = {
                "side": "ar", "global_turn": gt,
                "source_family": rec["source_family"],
                "hf_exact_arguments": bool(rec["hf_exact_arguments"]),
                "prompt_len": rec["prompt_len"], "n_ref": rec["n_ref"],
                "wall_s": round(wall, 3),
                "samples_per_sec": round(n / wall, 4),
                "total_gen_tokens": total_tok,
                "tokens_per_sec": round(total_tok / wall, 2),
                "mean_gen_tokens_per_sample": round(total_tok / n, 1),
                "ar_decode_steps_per_sample": round(total_tok / n, 1),
                "seeds": seeds,
                "per_sample": [{k: s[k] for k in
                                ("seed", "n_tok", "valid", "exact", "argset",
                                 "finish_reason")} for s in samples],
                "finish_reasons": fin_counts,
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
            print(f"[ar] N={n:2d} gt{gt:3d} {rec['source_family'][:12]:12s} "
                  f"hf={int(rec['hf_exact_arguments'])} wall={wall:.2f}s "
                  f"samp/s={row['samples_per_sec']:.2f} "
                  f"uniqOut={met['unique_output_frac']} "
                  f"uniqArg={met['unique_argset_frac']} "
                  f"valid={met['valid_frac']} pass1={met['pass1']} passN={met['passN']} "
                  f"gtok/s={row['ar_decode_steps_per_sample']} "
                  f"util~{gs['gpu_util_mean_pct']}%", flush=True)
    fh.close()
    print("[ar] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[ar] FATAL:", repr(e), flush=True)
        traceback.print_exc()
        raise
