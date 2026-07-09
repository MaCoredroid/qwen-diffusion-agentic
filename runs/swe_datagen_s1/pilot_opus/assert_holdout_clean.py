#!/usr/bin/env python3
"""Assert the 10 Opus-pilot instance ids are HOLDOUT-CLEAN.

Reconstructs the 113-id eval holdout byte-identically to expand_frontier.py /
leakage_audit.py, hash-asserts it against .eval_holdout_sha256, then asserts none
of the pilot ids are in it. Systemic gate: exits nonzero on any mismatch.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/mark/qwen_diffusion")
HERE = REPO_ROOT / "runs/swe_datagen_s1"
MANIFEST = REPO_ROOT / "data/swe_sft_pool/pool_manifest.json"
PIN = HERE / ".eval_holdout_sha256"
RING_SRC = {
    "tier0_20": REPO_ROOT / "runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json",
    "tier1_100": Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/swe-bench-tier1-verified-instances-20260520.json"),
}
PILOT = HERE / "pilot_opus/batch/shard_0.json"


def _ids(path):
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        return set(d.get("instance_ids", []))
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return set(d)
        return {r.get("instance_id") for r in d if isinstance(r, dict) and r.get("instance_id")}
    return set()


def _sha(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


man = json.loads(MANIFEST.read_text())
inner5 = set(man["held_out_rings"]["inner5"]["ids"])
holdout = inner5 | _ids(RING_SRC["tier0_20"]) | _ids(RING_SRC["tier1_100"])
sha = _sha(holdout)
pinned = PIN.read_text().strip()
if sha != pinned:
    raise SystemExit(f"HOLDOUT HASH MISMATCH: reconstructed {sha} != pinned {pinned}")

pilot_ids = json.loads(PILOT.read_text())["instance_ids"]
overlap = sorted(set(pilot_ids) & holdout)
result = {
    "holdout_n_distinct": len(holdout),
    "holdout_sha256": sha,
    "holdout_sha256_pinned": pinned,
    "sha_match": sha == pinned,
    "pilot_ids": pilot_ids,
    "pilot_n": len(pilot_ids),
    "overlap_with_holdout": overlap,
    "clean": len(overlap) == 0,
}
print(json.dumps(result, indent=1))
if overlap:
    raise SystemExit(f"LEAKAGE: {len(overlap)} pilot ids in the 113-id holdout: {overlap}")
print("HOLDOUT-CLEAN OK", file=sys.stderr)
