#!/usr/bin/env python3
"""Build the FROZEN 50-instance W2 evaluation pool (stock-AR vs diffusion).

W2 directive: 50 instances STRATIFIED from the pre-registered Tier1-100
SWE-bench_Verified subset, with a leakage firewall that makes the pool
DISJOINT from the 5 Tier0 gate/ladder instances the frozen config was tuned on.

Frozen-pool discipline: the instance list + hashes are committed BEFORE any
episode runs, so the pool is immutable and the pre-registered analysis is honest.

Conventions reused verbatim from
  /home/mark/shared/lumoFlyWheel/scripts/build_swe_bench_subset.py :: _stratify
  (proportional floor + largest-remainder + min-1-per-repo + seeded within-repo
   random.sample), operating on the leakage-filtered Tier1 candidate set.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
TIER1 = Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/"
             "swe-bench-tier1-verified-instances-20260520.json")
TIER0 = ROOT / "runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json"
OUT_DIR = ROOT / "data/swe_w2_n50_pool"

TARGET_N = 50
SEED = 0  # canonical project subset seed (build_swe_bench_subset.py default)

# The 5 Tier0 instances the frozen config was tuned on: N=5 loop-halt PASS-GATE
# (runs/stage_c_n5v3_gate) AND the v3 ENVELOPE LADDER (runs/stage_c_n5v3).
# These are the HARD leakage firewall — the W2 pool MUST be disjoint from them.
GATE_LADDER_5 = [
    "django__django-11119",
    "django__django-12754",
    "django__django-13741",
    "pytest-dev__pytest-8399",
    "sympy__sympy-13757",
]


def _docker_arch() -> str:
    import platform
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def _image_for(instance_id: str) -> str:
    # swebench substitutes `__` with `_1776_` (Docker Hub forbids `__`).
    slug = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.{_docker_arch()}.{slug}:latest"


def _stratify(records: list[dict], target_n: int, seed: int) -> list[dict]:
    """Verbatim port of build_swe_bench_subset.py::_stratify."""
    if target_n >= len(records):
        return sorted(records, key=lambda r: r["instance_id"])

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_repo[r["repo"]].append(r)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda r: r["instance_id"])

    total = len(records)
    shares = {repo: len(rs) / total for repo, rs in by_repo.items()}
    raw_alloc = {repo: target_n * share for repo, share in shares.items()}
    floor_alloc = {repo: int(v) for repo, v in raw_alloc.items()}
    remainder = target_n - sum(floor_alloc.values())

    ranked = sorted(
        by_repo.keys(),
        key=lambda repo: (
            -(raw_alloc[repo] - floor_alloc[repo]),
            -len(by_repo[repo]),
            repo,
        ),
    )
    for repo in ranked:
        if remainder <= 0:
            break
        if floor_alloc[repo] < len(by_repo[repo]):
            floor_alloc[repo] += 1
            remainder -= 1

    if target_n >= len(by_repo):
        for repo in by_repo:
            if floor_alloc[repo] == 0:
                donor = max(
                    floor_alloc.keys(),
                    key=lambda r: floor_alloc[r] - target_n * shares[r],
                )
                if donor != repo and floor_alloc[donor] > 1:
                    floor_alloc[donor] -= 1
                    floor_alloc[repo] += 1

    rng = random.Random(seed)
    chosen: list[dict] = []
    for repo in sorted(by_repo.keys()):
        n = floor_alloc.get(repo, 0)
        if n <= 0:
            continue
        chosen.extend(rng.sample(by_repo[repo], n))
    chosen.sort(key=lambda r: r["instance_id"])
    return chosen


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> None:
    os.environ.setdefault("HF_HOME", "/home/mark/.cache/huggingface")
    from datasets import load_dataset

    tier1 = json.loads(TIER1.read_text())
    tier0 = json.loads(TIER0.read_text())
    tier1_ids = list(tier1["instance_ids"])
    tier0_ids = set(tier0["instance_ids"])
    assert len(tier1_ids) == 100, len(tier1_ids)

    # ---- LEAKAGE FIREWALL (hard requirement: disjoint from the gate/ladder 5)
    leak5 = set(GATE_LADDER_5)
    leaked_in_tier1 = sorted(leak5 & set(tier1_ids))
    candidates_ids = [i for i in tier1_ids if i not in leak5]

    # ---- Pull repo + base_commit + version + difficulty from the Verified pin
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    meta = {ex["instance_id"]: ex for ex in ds}
    records = [{
        "instance_id": i,
        "repo": meta[i]["repo"],
        "base_commit": meta[i]["base_commit"],
        "version": str(meta[i].get("version", "")),
        "difficulty": meta[i].get("difficulty", ""),
    } for i in candidates_ids]

    chosen = _stratify(records, TARGET_N, SEED)
    assert len(chosen) == TARGET_N, len(chosen)

    chosen_ids = [r["instance_id"] for r in chosen]
    chosen_set = set(chosen_ids)

    # ---- LEAKAGE ASSERTS (fail-closed) --------------------------------------
    asserts = {}
    inter_gate = sorted(chosen_set & leak5)
    asserts["disjoint_from_gate_ladder_5"] = {
        "gate_ladder_5": sorted(leak5),
        "gate_ladder_5_in_tier1_candidates_removed": leaked_in_tier1,
        "intersection_with_pool": inter_gate,
        "PASS": inter_gate == [],
    }
    # Secondary (informational, NOT a firewall): overlap with the broader
    # Tier0-20 held-out ring. These 5 were reserved but NEVER tuned on (config
    # tuning touched only the gate/ladder 5); reported as a covariate.
    tier0_nongate = sorted((chosen_set & tier0_ids) - leak5)
    asserts["tier0_20_holdout_ring_overlap_informational"] = {
        "note": "reserved-ring instances that were NOT used for config tuning; "
                "no training on any Verified instance (SWE-Gym SFT pool is "
                "repo-disjoint). Reported as covariate, not a leakage failure.",
        "overlap": tier0_nongate,
        "n": len(tier0_nongate),
    }
    # Structural sanity
    asserts["all_pool_in_tier1_100"] = {
        "PASS": chosen_set.issubset(set(tier1_ids)),
    }
    asserts["no_duplicates"] = {"PASS": len(chosen_ids) == len(chosen_set)}

    hard_pass = (asserts["disjoint_from_gate_ladder_5"]["PASS"]
                 and asserts["all_pool_in_tier1_100"]["PASS"]
                 and asserts["no_duplicates"]["PASS"])

    # ---- Composition
    by_repo = defaultdict(int)
    for r in chosen:
        by_repo[r["repo"]] += 1
    by_repo = dict(sorted(by_repo.items()))

    tier1_by_repo = tier1.get("by_repo", {})

    # ---- Per-instance frozen rows (id + base_commit + image + row hash)
    instances = []
    for r in chosen:
        row_hash = _sha(f"{r['instance_id']}@{r['base_commit']}")
        instances.append({
            "instance_id": r["instance_id"],
            "repo": r["repo"],
            "base_commit": r["base_commit"],
            "version": r["version"],
            "difficulty": r["difficulty"],
            "image": _image_for(r["instance_id"]),
            "row_sha256": row_hash,
        })

    # ---- Pool fingerprint: sha256 over the canonical id@commit list
    canon = "\n".join(f"{r['instance_id']}@{r['base_commit']}"
                      for r in sorted(chosen, key=lambda x: x["instance_id"]))
    pool_sha256 = _sha(canon)
    ids_only_sha256 = _sha("\n".join(sorted(chosen_ids)))

    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = None

    manifest = {
        "artifact": "data/swe_w2_n50_pool/pool_manifest.json",
        "purpose": "FROZEN W2 evaluation pool (N=50) — stock-AR vs diffusion, "
                   "paired resolve@1 McNemar. Immutable; committed before any episode.",
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_head_at_build": git_head,
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "n_total": len(ds),
        },
        "sampling": {
            "target_n": TARGET_N,
            "seed": SEED,
            "method": "proportional-stratified-by-repo, largest-remainder, "
                      "min-1-per-repo, seeded within-repo random.sample "
                      "(verbatim build_swe_bench_subset.py::_stratify)",
            "source_subset": "Tier1-100 (seed=0) pre-registered",
            "source_subset_file": str(TIER1),
            "candidate_pool_after_leakage_removal": len(candidates_ids),
        },
        "leakage": asserts,
        "leakage_firewall_PASS": hard_pass,
        "composition_by_repo": by_repo,
        "tier1_100_by_repo": tier1_by_repo,
        "pool_sha256": pool_sha256,
        "pool_ids_only_sha256": ids_only_sha256,
        "instance_ids": sorted(chosen_ids),
        "instances": sorted(instances, key=lambda x: x["instance_id"]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pool_manifest.json").write_text(json.dumps(manifest, indent=1))
    # Plain id list for the orchestrator + a docker pull list.
    (OUT_DIR / "instance_ids.txt").write_text(
        "\n".join(sorted(chosen_ids)) + "\n")
    (OUT_DIR / "images.txt").write_text(
        "\n".join(i["image"] for i in manifest["instances"]) + "\n")

    print(json.dumps({
        "n": len(chosen_ids),
        "leakage_firewall_PASS": hard_pass,
        "pool_sha256": pool_sha256,
        "composition_by_repo": by_repo,
        "leaked_removed_from_candidates": leaked_in_tier1,
        "gate_ladder_5_intersection_with_pool": inter_gate,
        "tier0_20_holdout_overlap_informational": tier0_nongate,
    }, indent=1))


if __name__ == "__main__":
    main()
