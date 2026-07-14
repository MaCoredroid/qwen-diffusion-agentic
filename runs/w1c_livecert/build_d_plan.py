#!/usr/bin/env python3
"""Build the (d) 6-episode C46 A/B shard plan: the first 6 django ctx_overflow
ids, each preserving its banked-run base_seed so the gate-ON arm is seed-matched
to the banked gate-OFF arm. Groups ids by base_seed into shards; assigns fresh
proxy ports. Emits runs/w1c_livecert/d_plan.json in run_arm_twin's schema.
"""
import json

ROOT = "/home/mark/qwen_diffusion"
TARGET = ["django__django-11163", "django__django-11211", "django__django-13195",
          "django__django-14170", "django__django-14631", "django__django-14725"]

plan = json.load(open(f"{ROOT}/runs/k_gate_c46/shard_plan.json"))
# map id -> banked base_seed
id_seed = {}
for sh in plan["shards"]:
    for iid in sh["instance_ids"]:
        id_seed[iid] = sh["base_seed"]

by_seed = {}
for iid in TARGET:
    seed = id_seed.get(iid, 1234)
    by_seed.setdefault(seed, []).append(iid)

shards = []
port = 33200
for sid, (seed, ids) in enumerate(sorted(by_seed.items())):
    shards.append({"shard_id": sid, "base_seed": seed, "diff_proxy_port": port,
                   "instance_ids": ids, "n": len(ids)})
    port += 1

out = {"concurrency": len(shards), "base_seed": 1234,
       "pool_sha256": "w1c_d_ctxoverflow6", "total_instances": len(TARGET),
       "shards": shards}
open(f"{ROOT}/runs/w1c_livecert/d_plan.json", "w").write(json.dumps(out, indent=1))
print(f"wrote d_plan.json: {len(shards)} shards, {len(TARGET)} ids")
for sh in shards:
    print(f"  shard {sh['shard_id']} seed={sh['base_seed']} port={sh['diff_proxy_port']} ids={sh['instance_ids']}")
