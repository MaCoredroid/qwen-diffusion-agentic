#!/usr/bin/env python3
"""Materialize one BATCH's local dataset + subset + shard subsets (CPU, offline).

DUAL-SOURCE (belt-lever, 2026-07-07): the frontier now mixes two provenances —
  * SWE-Gym instances (families python/pydantic/conan-io/iterative/pandas/...),
    metadata from the offline-cached `SWE-Gym/SWE-Gym` train split, prebuilt
    images from xingyaoww, scored by the SWE-Bench-Fork harness; and
  * SWE-bench_Verified-adjacent instances (families django/sympy/astropy/...),
    metadata from the offline-cached `princeton-nlp/SWE-bench_Verified` test
    split, OFFICIAL `swebench/sweb.eval.x86_64.<slug_1776>` images, scored by the
    OFFICIAL swebench harness (the W2-proven path — matches the image provenance).

Source is determined by DATASET MEMBERSHIP (the two repo sets are disjoint), so an
id is unambiguously one or the other. We emit both a combined dataset (for the
driver, which reads one dataset.json across all shards) and SOURCE-PARTITIONED
datasets + a sources map so `datagen_pull.sh` pulls the right image and
`datagen_score.sh` routes each id to the matching harness:

  <batchdir>/dataset.json          list[record] ALL ids (driver reads this via the
                                   shard subset's dataset_name; core fork/driver
                                   schema + environment_setup_commit).
  <batchdir>/dataset_gym.json      SWE-Gym records only (fork scorer input).
  <batchdir>/dataset_verified.json Verified records only (official scorer input).
  <batchdir>/sources.json          {instance_id: "swe_gym"|"swe_verified"}.
  <batchdir>/subset.json           {dataset_name:<abs dataset.json>, split:"train",
                                    instance_ids:[...]}  (whole-batch id list)
  <batchdir>/shard_<k>.json        C disjoint round-robin shard subsets (same
                                   dataset_name), each driven with --only.

BACKWARD-COMPAT: a pure-SWE-Gym batch produces dataset_gym.json == dataset.json
(content), an empty dataset_verified.json, and an all-"swe_gym" sources map — so
every downstream that keys off sources.json degrades to the original behavior. If
`princeton-nlp/SWE-bench_Verified` is not cacheable, Verified lookups just become
"missing" (recorded, never fatal for a gym batch).

Round-robin over the batch order keeps each shard repo-balanced (the frontier
order is already stratified). Per-shard base seed = BASE + k*100000 so the proxy's
per-request seed counter (base+index) never collides across shards.

usage: build_batch_dataset.py <batchdir> <ids_csv_or_@file> <concurrency> <base_seed>
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# The exact record fields the fork/official harness + driver read. Verified rows
# carry `environment_setup_commit`; SWE-Gym rows usually don't. We build the record
# via .get so the combined schema is a superset, but `environment_setup_commit` is
# SPECIAL: the fork harness keys off KEY PRESENCE, not truthiness (get_environment_yml:
# `instance["environment_setup_commit"] if "environment_setup_commit" in instance else
# instance["base_commit"]`). `.get(k)` turns a SWE-Gym row's true ABSENCE into a
# present `None`, which defeats that fallback -> os.path.join(url, repo, None, path)
# raises TypeError and aborts the ENTIRE fork run before any container (every gym id
# then records no_prediction). So below we DROP `environment_setup_commit` when it is
# None, restoring true absence -> the harness base_commit fallback fires. Verified rows
# keep it (populated, non-None); the official harness is unaffected.
FIELDS = ["instance_id", "repo", "base_commit", "version", "problem_statement",
          "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text",
          "created_at", "environment_setup_commit"]

GYM = "swe_gym"
VERIFIED = "swe_verified"


def _read_ids(arg: str) -> list[str]:
    if arg.startswith("@"):
        return [l.strip() for l in Path(arg[1:]).read_text().splitlines() if l.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def _load_source(name: str, split: str) -> dict[str, dict]:
    """Best-effort offline load of one HF dataset -> {instance_id: record}."""
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split=split)
        return {ex["instance_id"]: dict(ex) for ex in ds}
    except Exception as e:  # noqa: BLE001
        print(f"[build] source {name}[{split}] unavailable: {type(e).__name__}: {e}",
              file=sys.stderr)
        return {}


def main() -> int:
    batchdir = Path(sys.argv[1]); batchdir.mkdir(parents=True, exist_ok=True)
    ids = _read_ids(sys.argv[2])
    C = int(sys.argv[3]); BASE = int(sys.argv[4])

    gym_rec = _load_source("SWE-Gym/SWE-Gym", "train")
    ver_rec = _load_source("princeton-nlp/SWE-bench_Verified", "test")

    records, present, missing = [], [], []
    sources: dict[str, str] = {}
    for iid in ids:
        if iid in gym_rec:
            ex, src = gym_rec[iid], GYM
        elif iid in ver_rec:
            ex, src = ver_rec[iid], VERIFIED
        else:
            missing.append(iid); continue
        rec = {k: ex.get(k) for k in FIELDS}
        # A present-`None` environment_setup_commit crashes the fork harness (it
        # tests key PRESENCE, not value); drop it so true absence -> base_commit
        # fallback. Verified rows carry a real value and are untouched.
        if rec.get("environment_setup_commit") is None:
            rec.pop("environment_setup_commit", None)
        records.append(rec)
        present.append(iid)
        sources[iid] = src

    dataset_path = (batchdir / "dataset.json").resolve()
    dataset_path.write_text(json.dumps(records, indent=1))

    # source-partitioned datasets for the dual scorer
    gym_records = [r for r in records if sources[r["instance_id"]] == GYM]
    ver_records = [r for r in records if sources[r["instance_id"]] == VERIFIED]
    (batchdir / "dataset_gym.json").write_text(json.dumps(gym_records, indent=1))
    (batchdir / "dataset_verified.json").write_text(json.dumps(ver_records, indent=1))
    (batchdir / "sources.json").write_text(json.dumps(sources, indent=1))

    subset = {"dataset_name": str(dataset_path), "split": "train",
              "instance_ids": present}
    (batchdir / "subset.json").write_text(json.dumps(subset, indent=1))

    shards = []
    for k in range(C):
        members = [present[i] for i in range(len(present)) if i % C == k]
        sh = {"dataset_name": str(dataset_path), "split": "train",
              "instance_ids": members, "shard_id": k,
              "base_seed": BASE + k * 100000, "proxy_port": 30030 + k}
        (batchdir / f"shard_{k}.json").write_text(json.dumps(sh, indent=1))
        shards.append(len(members))

    meta = {"n_requested": len(ids), "n_present": len(present),
            "n_missing": len(missing), "missing": missing,
            "n_gym": len(gym_records), "n_verified": len(ver_records),
            "concurrency": C, "base_seed": BASE, "shard_sizes": shards,
            "dataset_json": str(dataset_path)}
    (batchdir / "batch_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0 if present else 3


if __name__ == "__main__":
    raise SystemExit(main())
