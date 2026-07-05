#!/usr/bin/env python3
"""Deterministic prompt selection for the best-of-N (GRPO same-prompt) bench.

Picks a balanced, reproducible set of never-train tool-call turns from
runs/p2_engine_nevertrain/nevertrain_ref.json, STRATIFIED across the 4 source
families AND across HF-exact / HF-miss (the pass@1 proxy), so pass@1 fails on a
real fraction (the requirement: "prompts where pass@1 fails sometimes").

The manifest is consumed IDENTICALLY by bench_engine_bestofn.py and
bench_ar_bestofn.py so both sides run byte-identical prompts + per-sample seeds.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
OUT = ROOT / "runs/p2_bestofn_grpo/prompts_manifest.json"

PMIN, PMAX = 400, 1300
# (family, lane) -> how many to take. lane in {exact, miss}. 8 exact + 8 miss = 16.
QUOTA = {
    ("BFCL-AST", "exact"): 3,
    ("API-Bank-Lv1", "exact"): 3,
    ("API-Bank-Lv2", "exact"): 2,
    ("BFCL-multi_turn", "miss"): 3,
    ("API-Bank-Lv1", "miss"): 3,
    ("API-Bank-Lv2", "miss"): 2,
}

KEEP = ["global_turn", "episode", "turn", "episode_id", "prompt_len",
        "prompt_sha256", "n_ref", "prompt_ids", "ref_new_ids", "gold_block",
        "tools", "schemas", "stop_token_ids", "grammar_topk", "mask_id",
        "block_size", "source_family", "hf_exact_arguments", "hf_valid_tool_call",
        "hf_denoise_forwards_total"]


def main():
    recs = json.loads(REF.read_text())
    band = [r for r in recs if PMIN <= r["prompt_len"] <= PMAX]
    cells = defaultdict(list)
    for r in band:
        lane = "exact" if r["hf_exact_arguments"] else "miss"
        cells[(r["source_family"], lane)].append(r)
    # deterministic: within each cell sort by (n_ref, global_turn) and take the
    # smallest-n_ref members (bounds N=16 wall) while staying reproducible.
    picked = []
    for key, k in QUOTA.items():
        pool = sorted(cells.get(key, []), key=lambda r: (r["n_ref"], r["global_turn"]))
        take = pool[:k]
        assert len(take) == k, f"cell {key} has only {len(pool)} < {k}"
        picked.extend(take)
    picked.sort(key=lambda r: (0 if r["hf_exact_arguments"] else 1, r["global_turn"]))
    manifest = {
        "ref": str(REF),
        "prompt_len_band": [PMIN, PMAX],
        "n_prompts": len(picked),
        "n_exact": sum(1 for r in picked if r["hf_exact_arguments"]),
        "n_miss": sum(1 for r in picked if not r["hf_exact_arguments"]),
        "block_size": int(picked[0]["block_size"]),
        "mask_id": int(picked[0]["mask_id"]),
        "prompts": [{k: r[k] for k in KEEP} for r in picked],
    }
    OUT.write_text(json.dumps(manifest))
    print(f"selected n={manifest['n_prompts']} exact={manifest['n_exact']} "
          f"miss={manifest['n_miss']}")
    for r in picked:
        print(f"  gt{r['global_turn']:3d} {r['source_family']:18s} "
              f"exact={int(r['hf_exact_arguments'])} n_ref={r['n_ref']:3d} "
              f"plen={r['prompt_len']}")


if __name__ == "__main__":
    main()
