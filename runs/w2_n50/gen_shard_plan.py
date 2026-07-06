#!/usr/bin/env python3
"""Deterministic, ARM-INDEPENDENT shard plan for the W2 N=50 concurrency fan-out.

Splits the 50 frozen-pool instances across C shards (round-robin over the frozen
pool order => balanced repo mix per shard => balanced wall time). BOTH arms use
the SAME instance->shard assignment and the SAME per-shard base seed, so instance
X is attempted under matched sampling conditions on both arms (paired-McNemar
validity; seeds only add reproducibility). Per-shard base seeds are spaced 100000
apart so the proxy's per-request seed counter (base+index) never collides across
shards.

usage: gen_shard_plan.py <subset_n50.json> <concurrency> <base_seed> <out.json>
"""
import json, sys, hashlib
subset = json.load(open(sys.argv[1]))
C = int(sys.argv[2]); BASE = int(sys.argv[3]); outp = sys.argv[4]
ids = list(subset["instance_ids"])
assert len(ids) == 50, len(ids)
shards = []
for k in range(C):
    members = [ids[i] for i in range(len(ids)) if i % C == k]
    shards.append({
        "shard_id": k,
        "base_seed": BASE + k * 100000,
        "ar_proxy_port": 31000 + k,
        "diff_proxy_port": 32000 + k,
        "instance_ids": members,
        "n": len(members),
    })
plan = {
    "concurrency": C,
    "base_seed": BASE,
    "pool_sha256": subset.get("pool_sha256"),
    "assignment": "round-robin over frozen pool order (i %% C)",
    "shards": shards,
    "total_instances": sum(s["n"] for s in shards),
}
# integrity: every instance assigned exactly once
allm = [i for s in shards for i in s["instance_ids"]]
assert sorted(allm) == sorted(ids), "assignment lost/duplicated instances"
plan["assignment_sha256"] = hashlib.sha256(
    json.dumps([s["instance_ids"] for s in shards], sort_keys=True).encode()).hexdigest()
json.dump(plan, open(outp, "w"), indent=1)
print(f"wrote {outp}: C={C} base_seed={BASE} shards={[s['n'] for s in shards]} total={plan['total_instances']}")
print("assignment_sha256:", plan["assignment_sha256"][:16])
