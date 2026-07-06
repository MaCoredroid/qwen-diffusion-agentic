#!/usr/bin/env python3
"""Materialize one BATCH's local dataset + subset + shard subsets (CPU, offline).

Given a batch of SWE-Gym instance_ids, filter the offline-cached SWE-Gym train
split to just those rows and write the exact artifacts the reused toolchain
consumes:

  <batchdir>/dataset.json        list[record] with the fork/driver schema
                                 (instance_id, repo, base_commit, version,
                                  problem_statement, patch, test_patch,
                                  FAIL_TO_PASS, PASS_TO_PASS, hints_text,
                                  created_at) — the same shape as the probe's
                                  swegym_probe20_dataset.json, consumed BOTH by
                                  run_swe_bench_qwen_code._load_dataset (local
                                  .json branch) AND the fork scorer.
  <batchdir>/subset.json         {dataset_name: <abs dataset.json>, split:"train",
                                  instance_ids:[...]}  (whole-batch id list)
  <batchdir>/shard_<k>.json      C disjoint round-robin shard subsets (same
                                  dataset_name), each driven with --only.

Round-robin over the batch order keeps each shard repo-balanced (the frontier
order is already stratified). Per-shard base seed = BASE + k*100000 so the
proxy's per-request seed counter (base+index) never collides across shards
(the gen_shard_plan.py convention).

usage: build_batch_dataset.py <batchdir> <ids_csv_or_@file> <concurrency> <base_seed>
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# The exact record fields the fork harness + driver read (probe dataset schema).
FIELDS = ["instance_id", "repo", "base_commit", "version", "problem_statement",
          "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text",
          "created_at"]


def _read_ids(arg: str) -> list[str]:
    if arg.startswith("@"):
        return [l.strip() for l in Path(arg[1:]).read_text().splitlines() if l.strip()]
    return [x.strip() for x in arg.split(",") if x.strip()]


def main() -> int:
    batchdir = Path(sys.argv[1]); batchdir.mkdir(parents=True, exist_ok=True)
    ids = _read_ids(sys.argv[2])
    C = int(sys.argv[3]); BASE = int(sys.argv[4])

    from datasets import load_dataset
    ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
    rec = {ex["instance_id"]: dict(ex) for ex in ds}

    records, present, missing = [], [], []
    for iid in ids:
        ex = rec.get(iid)
        if ex is None:
            missing.append(iid); continue
        records.append({k: ex.get(k) for k in FIELDS})
        present.append(iid)

    dataset_path = (batchdir / "dataset.json").resolve()
    dataset_path.write_text(json.dumps(records, indent=1))

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
            "concurrency": C, "base_seed": BASE, "shard_sizes": shards,
            "dataset_json": str(dataset_path)}
    (batchdir / "batch_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0 if present else 3


if __name__ == "__main__":
    raise SystemExit(main())
