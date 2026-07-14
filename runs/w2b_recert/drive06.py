#!/usr/bin/env python3
"""W-2b live recert driver: post a corpus at TEMP 0.6 with a FIXED SEED.

Usage: drive06.py <corpus.jsonl> <gold.json> <tag>

Injects temperature=0.6 + seed=SEED into every row (so gate-OFF and gate-ON with
the same seed decode the identical seeded trajectory), extracts the write_file
content, and writes out_<tag>.jsonl. Bit-identity vs the gate-OFF twin and the
w1 counters are adjudicated by compare06.py off the server logs.
"""
import json
import os
import sys
import time
import urllib.request

PORT = int(os.environ.get("PORT", "9952"))
SEED = int(os.environ.get("RECERT_SEED", "20260714"))
TEMP = float(os.environ.get("RECERT_TEMP", "0.6"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))

CORPUS = sys.argv[1]
GOLD = sys.argv[2]
TAG = sys.argv[3]


def post(row):
    body = {k: v for k, v in row.items() if not k.startswith("_") and k != "meta"}
    body["temperature"] = TEMP
    body["seed"] = SEED
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    args = tcs[0]["function"]["arguments"] if tcs else ""
    content = None
    try:
        content = json.loads(args).get("content")
    except Exception:
        pass
    return {"raw_args": args, "content": content,
            "usage": resp.get("usage", {}), "latency_s": round(dt, 3)}


def main():
    corpus = [json.loads(x) for x in open(CORPUS)]
    gold = json.load(open(GOLD))
    out = []
    for row in corpus:
        idx, rep = str(row["_idx"]), int(row["_rep"])
        r = post(row)
        rec = {"idx": idx, "rep": rep, **r}
        out.append(rec)
        ok = r["content"] == gold.get(idx)
        print(f"[{TAG}] idx={idx} rep={rep} exact={ok} lat={r['latency_s']}s "
              f"ctok={r['usage'].get('completion_tokens')}", flush=True)
    with open(f"{HERE}/out_{TAG}.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[{TAG}] wrote {len(out)} rows")


if __name__ == "__main__":
    main()
