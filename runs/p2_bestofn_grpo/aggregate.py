#!/usr/bin/env python3
"""Aggregate the best-of-N GRPO bench: Q1 throughput + Q2 signal-quality tables."""
import json
from pathlib import Path
from collections import defaultdict

HERE = Path("/home/mark/qwen_diffusion/runs/p2_bestofn_grpo")


def load(p):
    rows = []
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def agg(rows, lane=None):
    """lane: None=all, True=hf_exact, False=hf_miss. keyed by N."""
    by = defaultdict(list)
    for r in rows:
        if lane is not None and bool(r["hf_exact_arguments"]) != lane:
            continue
        by[r["N"]].append(r)
    out = {}
    for N in sorted(by):
        g = by[N]
        n_groups = len(g)
        tot_samples = sum(r["N"] for r in g)
        tot_wall = sum(r["wall_s"] for r in g)
        tot_exact = sum(r["n_exact"] for r in g)
        tot_valid = sum(r["n_valid"] for r in g)
        # micro throughput = total samples / total wall
        micro_sps = tot_samples / tot_wall if tot_wall else 0
        mean_sps = sum(r["samples_per_sec"] for r in g) / n_groups
        out[N] = {
            "n_groups": n_groups,
            "tot_samples": tot_samples,
            "micro_samples_per_sec": round(micro_sps, 3),
            "mean_group_samples_per_sec": round(mean_sps, 3),
            "mean_unique_output_frac": round(sum(r["unique_output_frac"] for r in g) / n_groups, 4),
            "mean_unique_argset_frac": round(sum(r["unique_argset_frac"] for r in g) / n_groups, 4),
            "mean_unique_valid_argset_frac": round(sum(r["unique_valid_argset_frac"] for r in g) / n_groups, 4),
            "mean_valid_frac": round(tot_valid / tot_samples, 4),
            "pass1_micro": round(tot_exact / tot_samples, 4),
            "passN_group_frac": round(sum(r["passN"] for r in g) / n_groups, 4),
            "mean_unique_output_count": round(sum(r["unique_output_count"] for r in g) / n_groups, 3),
            "mean_gen_tok_per_sample": round(sum(r["mean_gen_tokens_per_sample"] for r in g) / n_groups, 1),
        }
    return out


def main():
    eng = load(HERE / "engine_groups.jsonl")
    ar = load(HERE / "ar_groups.jsonl")
    result = {"engine": {}, "ar": {}}
    for name, rows in (("engine", eng), ("ar", ar)):
        result[name] = {
            "all": agg(rows, None),
            "hf_exact": agg(rows, True),
            "hf_miss": agg(rows, False),
        }
    # engine audit rollup
    aud_ok = all(r["audit"]["proj_zero_ok"] for r in eng)
    verify_ok = all(r["audit"]["verify_ok"] for r in eng)
    max_proj = max(r["audit"]["max_value_projection_events"] for r in eng)
    result["engine_audit"] = {
        "n_groups": len(eng),
        "all_proj_zero": aud_ok,
        "all_verify_ok": verify_ok,
        "max_value_projection_events_over_all_groups": max_proj,
        "all_valid_frac_1p0": all(r["valid_frac"] == 1.0 for r in eng),
    }
    # prompt integrity: manifest sha present on every row
    result["prompt_integrity"] = {
        "engine_all_have_sha": all("prompt_sha256_manifest" in r for r in eng),
        "ar_all_have_sha": all("prompt_sha256_manifest" in r for r in ar),
    }
    (HERE / "aggregate.json").write_text(json.dumps(result, indent=2))

    # ---- print Q1 + Q2 tables ----
    def q1_row(N):
        e = result["engine"]["all"][N]
        a = result["ar"]["all"][N]
        ratio = e["micro_samples_per_sec"] / a["micro_samples_per_sec"]
        return (f"| {N:>2} | {e['micro_samples_per_sec']:>6.2f} | "
                f"{a['micro_samples_per_sec']:>6.2f} | **{ratio:.2f}x** | "
                f"{e['mean_group_samples_per_sec']:.2f} | {a['mean_group_samples_per_sec']:.2f} |")

    print("\n### Q1 THROUGHPUT (samples/sec == rollouts/sec/GPU, same-prompt group)\n")
    print("| N | eng micro s/s | AR micro s/s | eng/AR | eng mean-grp | AR mean-grp |")
    print("|---:|---:|---:|:---:|---:|---:|")
    for N in sorted(result["engine"]["all"]):
        print(q1_row(N))

    def q2_block(title, lane_key):
        print(f"\n### Q2 {title}\n")
        print("| N | side | uniqOut | uniqArg | uniqValidArg | valid | pass@1 | pass@N(grp) |")
        print("|---:|:---|---:|---:|---:|---:|---:|---:|")
        for N in sorted(result["engine"][lane_key]):
            for side in ("engine", "ar"):
                d = result[side][lane_key][N]
                print(f"| {N} | {side} | {d['mean_unique_output_frac']:.3f} | "
                      f"{d['mean_unique_argset_frac']:.3f} | {d['mean_unique_valid_argset_frac']:.3f} | "
                      f"{d['mean_valid_frac']:.3f} | {d['pass1_micro']:.3f} | {d['passN_group_frac']:.3f} |")

    q2_block("SIGNAL QUALITY — ALL 16 prompts", "all")
    q2_block("SIGNAL QUALITY — HF-EXACT lane (8 prompts)", "hf_exact")
    q2_block("SIGNAL QUALITY — HF-MISS lane (8 prompts)", "hf_miss")
    print("\n### ENGINE AUDIT\n", json.dumps(result["engine_audit"], indent=2))


if __name__ == "__main__":
    main()
