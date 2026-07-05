#!/usr/bin/env python3
"""Join the engine + AR-guided throughput sweeps into compare.json (the ratio
curve that matters) + a markdown table fragment. Pure post-processing; no GPU."""
import json
from pathlib import Path

HERE = Path("/home/mark/qwen_diffusion/runs/p2_batched_rollout_bench")
eng = json.loads((HERE / "engine_throughput.json").read_text())
ar = json.loads((HERE / "ar_throughput.json").read_text())


def by_batch(summ, variant="rl_temp0.7"):
    d = {}
    for p in summ["points"]:
        if p.get("error"):
            d[(p["variant"], p["batch"])] = p
            continue
        d[(p["variant"], p["batch"])] = p
    return d


E = by_batch(eng)
A = by_batch(ar)
batches = [1, 2, 4, 8, 16]
rows = []
for B in batches:
    e = E.get(("rl_temp0.7", B))
    a = A.get(("rl_temp0.7", B))
    if not e or e.get("error") or not a or a.get("error"):
        rows.append({"batch": B,
                     "engine_error": (e or {}).get("error"),
                     "ar_error": (a or {}).get("error")})
        continue
    rows.append({
        "batch": B,
        "engine_samples_per_sec": e["samples_per_sec"],
        "ar_samples_per_sec": a["samples_per_sec"],
        "engine_over_ar_samples": round(e["samples_per_sec"] / a["samples_per_sec"], 3),
        "engine_tok_per_sec": e["tokens_per_sec"],
        "ar_tok_per_sec": a["tokens_per_sec"],
        "engine_over_ar_tok": round(e["tokens_per_sec"] / a["tokens_per_sec"], 3),
        "engine_speedup_vs_b1": e.get("speedup_samples_vs_b1"),
        "ar_speedup_vs_b1": a.get("speedup_samples_vs_b1"),
        "engine_per_forward_ms": e["per_forward_ms"],
        "engine_forwards_per_turn": e["mean_forwards_per_turn"],
        "ar_per_decode_step_ms": a["per_decode_step_ms"],
        "ar_gen_tok_per_turn": a["mean_gen_tokens_per_turn"],
        "engine_batch_occupancy": e["mean_batch_in_forward"],
        "engine_batch_occupancy_eff": e["batch_occupancy_efficiency"],
        "engine_gpu_util_mean": e["gpu_util_mean_pct"],
        "ar_gpu_util_mean": a["gpu_util_mean_pct"],
        "engine_gpu_mem_mb": e["gpu_mem_used_mb"],
        "ar_gpu_mem_mb": a["gpu_mem_used_mb"],
        "engine_host_ram_peak_gb": e["host_ram_peak_gb"],
        "ar_host_ram_peak_gb": a["host_ram_peak_gb"],
    })

greedy = E.get(("greedy", 8))
compare = {
    "note": ("48 never-train tool-call turns (prompt_len 467-1299, nref_mean 50.8), "
             "identical wave harness both sides, RTX 5090, RAM cage. "
             "ENGINE=FLARE hybrid-clean converted-9B, PIECEWISE cudagraph+APC, gmu 0.62, temp=0.7 seeded. "
             "AR=stock Qwen3.5-9B snapshot c202236, vLLM 0.23.0, GUIDED regex_from_qwen_xml_tool_schema "
             "(== scoreboard), cudagraph (fast path, NOT the scoreboard's eager server -> conservative for "
             "the engine thesis), gmu 0.66, temp=0.7 seeded. samples/sec == turns/sec == rollouts/sec/GPU."),
    "pool_n": eng["pool_n"], "prompt_len_band": eng["prompt_len_band"],
    "engine_config": eng["config"], "engine_gmu": eng["gpu_memory_utilization"],
    "ar_config": ar["config"], "ar_gmu": ar["gpu_memory_utilization"],
    "seed": eng["seed"],
    "engine_greedy_b8_samples_per_sec": (greedy or {}).get("samples_per_sec"),
    "engine_greedy_b8_tok_per_sec": (greedy or {}).get("tokens_per_sec"),
    "rows": rows,
}
(HERE / "compare.json").write_text(json.dumps(compare, indent=2))

# markdown fragment
lines = []
lines.append("| batch | eng samp/s | AR samp/s | **eng/AR** | eng tok/s | AR tok/s | eng scale | AR scale | eng fwd/turn (ms/fwd) | AR tok/turn (ms/step) | eng occ (eff) | eng util% | AR util% |")
lines.append("|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    if "engine_samples_per_sec" not in r:
        lines.append(f"| {r['batch']} | eng_err={r.get('engine_error')} | ar_err={r.get('ar_error')} | — | | | | | | | | | |")
        continue
    lines.append(
        f"| {r['batch']} | {r['engine_samples_per_sec']:.3f} | {r['ar_samples_per_sec']:.3f} | "
        f"**{r['engine_over_ar_samples']:.2f}x** | {r['engine_tok_per_sec']:.0f} | {r['ar_tok_per_sec']:.0f} | "
        f"{r['engine_speedup_vs_b1']:.2f}x | {r['ar_speedup_vs_b1']:.2f}x | "
        f"{r['engine_forwards_per_turn']:.1f} ({r['engine_per_forward_ms']:.1f}) | "
        f"{r['ar_gen_tok_per_turn']:.1f} ({r['ar_per_decode_step_ms']:.2f}) | "
        f"{r['engine_batch_occupancy']:.1f} ({r['engine_batch_occupancy_eff']:.2f}) | "
        f"{r['engine_gpu_util_mean']:.0f} | {r['ar_gpu_util_mean']:.0f} |")
(HERE / "compare_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print()
print("greedy engine b8:", (greedy or {}).get("samples_per_sec"), "samp/s",
      (greedy or {}).get("tokens_per_sec"), "tok/s")
print("wrote compare.json + compare_table.md")
