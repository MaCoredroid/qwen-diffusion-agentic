#!/usr/bin/env python3
"""Merge per-shard predictions.jsonl into one per-arm predictions.jsonl.

usage: merge_predictions.py <arm_dir> <n_shards> <out.jsonl>
Reads <arm_dir>/shard_<k>/verified/predictions.jsonl for k in [0,n_shards).
Dedups by instance_id (last wins), warns on missing/dupes, writes merged file.
"""
import json, sys, pathlib
arm_dir = pathlib.Path(sys.argv[1]); nsh = int(sys.argv[2]); outp = pathlib.Path(sys.argv[3])
by_id = {}
seen_files = 0
for k in range(nsh):
    pf = arm_dir / f"shard_{k}" / "verified" / "predictions.jsonl"
    if not pf.is_file():
        print(f"[merge] WARNING missing shard predictions: {pf}", file=sys.stderr); continue
    seen_files += 1
    for line in pf.read_text().splitlines():
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        iid = rec["instance_id"]
        if iid in by_id:
            print(f"[merge] WARNING duplicate instance across shards: {iid}", file=sys.stderr)
        by_id[iid] = line
outp.write_text("\n".join(by_id[i] for i in sorted(by_id)) + ("\n" if by_id else ""))
print(f"[merge] shards_seen={seen_files}/{nsh} unique_instances={len(by_id)} -> {outp}", file=sys.stderr)
