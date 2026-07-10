#!/usr/bin/env python3
"""Build the FROZEN Tier1-C46 evaluation pool for the K-raise campaign step 5.

k_raise_campaign_design.md §7 (LEAKAGE firewall):
  Tier1-C46 = Tier1-100 \\ (w2_n50_ids ∪ gate_ladder_5)  -- the FRESH slice
  never used in any prior tuning or eval decision. w2_n50 drew 50 of the 98
  leakage-cleared Tier1 candidates for the AR-vs-diffusion horse race; its
  complement was never drawn, never scored, never used to make a decision.

KILL-D1 hash asserts (build-time; re-asserted at eval-launch):
  Tier1_C46 ∩ train_ids = ∅ · ∩ w2_n50_ids = ∅ · ∩ gate_ladder_5 = ∅ ·
  Tier1_C46 ⊂ tier1_100. Any nonzero intersection ⇒ do not eval.

Reuses build_swe_w2_n50_pool.py conventions verbatim (image slug, row hashes,
pool fingerprint) so the manifest is directly comparable to the w2 pool.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
TIER1 = Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/"
             "swe-bench-tier1-verified-instances-20260520.json")
W2_IDS = ROOT / "data/swe_w2_n50_pool/instance_ids.txt"
SFT_POOL = ROOT / "data/swe_sft_pool/pool_manifest.json"
OUT_DIR = ROOT / "data/swe_kraise_c46_pool"

# The gate/ladder 5 (verbatim build_swe_w2_n50_pool.py).
GATE_LADDER_5 = [
    "django__django-11119",
    "django__django-12754",
    "django__django-13741",
    "pytest-dev__pytest-8399",
    "sympy__sympy-13757",
]


def _docker_arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64"


def _image_for(instance_id: str) -> str:
    slug = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.{_docker_arch()}.{slug}:latest"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> None:
    os.environ.setdefault("HF_HOME", "/home/mark/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    tier1 = json.loads(TIER1.read_text())
    tier1_ids = list(tier1["instance_ids"])
    assert len(tier1_ids) == 100, len(tier1_ids)
    tier1_set = set(tier1_ids)

    w2_ids = [l.strip() for l in W2_IDS.read_text().splitlines() if l.strip()]
    w2_set = set(w2_ids)
    assert len(w2_set) == 50, len(w2_set)
    assert w2_set.issubset(tier1_set), "w2 not subset of tier1"

    gl5 = set(GATE_LADDER_5)

    # Train ids (KILL-D1 firewall): the SWE-SFT pool instances.
    sft = json.loads(SFT_POOL.read_text())
    train_ids = sorted({r["instance_id"] for r in sft.get("instances", [])})
    train_set = set(train_ids)

    # ---- Tier1-C46 = Tier1-100 \ (w2 ∪ gate_ladder_5) -----------------------
    c46_ids = sorted(tier1_set - w2_set - gl5)
    c46_set = set(c46_ids)

    # ---- KILL-D1 hash asserts (fail-closed) ---------------------------------
    asserts = {}
    int_train = sorted(c46_set & train_set)
    int_w2 = sorted(c46_set & w2_set)
    int_gl5 = sorted(c46_set & gl5)
    subset_ok = c46_set.issubset(tier1_set)
    asserts["c46_cap_train_ids_empty"] = {"intersection": int_train, "PASS": int_train == []}
    asserts["c46_cap_w2_n50_empty"] = {"intersection": int_w2, "PASS": int_w2 == []}
    asserts["c46_cap_gate_ladder_5_empty"] = {"intersection": int_gl5, "PASS": int_gl5 == []}
    asserts["c46_subset_of_tier1_100"] = {"PASS": subset_ok}
    asserts["no_duplicates"] = {"PASS": len(c46_ids) == len(c46_set)}
    kill_d1_pass = all(a["PASS"] for a in asserts.values())

    # ---- Verified metadata --------------------------------------------------
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    meta = {ex["instance_id"]: ex for ex in ds}

    instances = []
    by_repo = defaultdict(int)
    for i in c46_ids:
        ex = meta[i]
        row_hash = _sha(f"{i}@{ex['base_commit']}")
        by_repo[ex["repo"]] += 1
        instances.append({
            "instance_id": i,
            "repo": ex["repo"],
            "base_commit": ex["base_commit"],
            "version": str(ex.get("version", "")),
            "difficulty": ex.get("difficulty", ""),
            "image": _image_for(i),
            "source_ring": "Tier1-100 (seed=0) complement of w2_n50 -- FRESH, "
                           "never used in any prior tuning/eval decision",
            "row_sha256": row_hash,
        })
    by_repo = dict(sorted(by_repo.items()))

    canon = "\n".join(f"{r['instance_id']}@{r['base_commit']}"
                      for r in sorted(instances, key=lambda x: x["instance_id"]))
    pool_sha256 = _sha(canon)
    ids_only_sha256 = _sha("\n".join(c46_ids))

    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = None

    manifest = {
        "artifact": "data/swe_kraise_c46_pool/pool_manifest.json",
        "purpose": "FROZEN Tier1-C46 evaluation pool for the K-raise campaign "
                   "step 5 (twin@K1 entry gate, resolve@1). Tier1-100 minus the "
                   "w2_n50 50 and the gate_ladder_5 -- the FRESH slice never used "
                   "in any prior tuning or eval decision. Immutable; committed "
                   "before any episode.",
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_head_at_build": git_head,
        "dataset": {"name": "princeton-nlp/SWE-bench_Verified", "split": "test",
                    "n_total": len(ds)},
        "construction": {
            "formula": "Tier1-100 \\ (w2_n50_ids ∪ gate_ladder_5)",
            "tier1_100_source": str(TIER1),
            "w2_n50_source": str(W2_IDS),
            "gate_ladder_5": sorted(gl5),
            "gate_ladder_5_in_tier1": sorted(gl5 & tier1_set),
            "n_tier1": 100, "n_w2": 50,
            "n_gate_ladder_in_tier1": len(gl5 & tier1_set),
            "n_c46": len(c46_ids),
        },
        "kill_d1_check": {
            "train_ids_source": str(SFT_POOL),
            "train_ids_n": len(train_ids),
            "asserts": asserts,
            "KILL_D1_PASS": kill_d1_pass,
        },
        "composition_by_repo": by_repo,
        "pool_sha256": pool_sha256,
        "pool_ids_only_sha256": ids_only_sha256,
        "instance_ids": c46_ids,
        "instances": instances,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pool_manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT_DIR / "instance_ids.txt").write_text("\n".join(c46_ids) + "\n")
    (OUT_DIR / "images.txt").write_text(
        "\n".join(i["image"] for i in instances) + "\n")

    print(json.dumps({
        "n_c46": len(c46_ids),
        "KILL_D1_PASS": kill_d1_pass,
        "asserts": {k: v["PASS"] for k, v in asserts.items()},
        "pool_sha256": pool_sha256,
        "composition_by_repo": by_repo,
    }, indent=1))
    if not kill_d1_pass:
        raise SystemExit("KILL-D1: nonzero leakage intersection -- DO NOT EVAL")


if __name__ == "__main__":
    main()
