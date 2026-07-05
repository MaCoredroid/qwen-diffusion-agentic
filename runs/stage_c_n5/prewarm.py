#!/usr/bin/env python3
"""Pre-warm + verify: load the 5 chosen Tier0 instances from the cached
SWE-bench_Verified dataset, print repo/base_commit/statement size, and ensure the
shared repo cache has each repo + base_commit fetched (blob:none clone)."""
import json, os, subprocess, sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset

SUBSET = Path("/home/mark/qwen_diffusion/runs/stage_c_n5/subset_n5.json")
CACHE = Path("/home/mark/qwen_diffusion/runs/stage_c_driver/repo_cache")
payload = json.loads(SUBSET.read_text())
want = set(payload["instance_ids"])
ds = load_dataset(payload["dataset_name"], split="test")
recs = {ex["instance_id"]: ex for ex in ds if ex["instance_id"] in want}
missing = want - set(recs)
if missing:
    print("MISSING FROM DATASET:", missing); sys.exit(1)

def clone(repo):
    safe = repo.replace("/", "__")
    p = CACHE / safe
    if not p.is_dir():
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none",
                        f"https://github.com/{repo}.git", str(p)], check=True)
    return p

for iid in payload["instance_ids"]:
    r = recs[iid]
    p = clone(r["repo"])
    have = subprocess.run(["git", "-C", str(p), "cat-file", "-e", r["base_commit"]]).returncode == 0
    if not have:
        subprocess.run(["git", "-C", str(p), "fetch", "origin", r["base_commit"]], check=False)
        have = subprocess.run(["git", "-C", str(p), "cat-file", "-e", r["base_commit"]]).returncode == 0
    print(f"{iid:32s} repo={r['repo']:26s} base={r['base_commit'][:10]} "
          f"stmt={len(r.get('problem_statement') or '')}B gold={len(r.get('patch') or '')}B "
          f"cache_ok={have}")
print("PREWARM_DONE")
