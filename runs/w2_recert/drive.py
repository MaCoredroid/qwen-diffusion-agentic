#!/usr/bin/env python3
"""W-2 live recert driver: 6 snippets x 5 reps against a booted FLARE server.

Posts the fixed corpus at temp 0, extracts the write_file tool-call content, and
scores per-idx (a) EXACT vs gold and (b) BIT-REPRODUCIBLE (all 5 reps' raw
tool-call arguments identical). Writes out_<tag>.jsonl + det_<tag>.json."""
import json
import os
import sys
import time
import urllib.request

PORT = 9952
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
W1D = "/home/mark/qwen_diffusion/runs/w1d_recert"
TAG = sys.argv[1] if len(sys.argv) > 1 else "on"


def post(row):
    body = {k: v for k, v in row.items() if not k.startswith("_") and k != "meta"}
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
    return {
        "raw_args": args, "content": content,
        "usage": resp.get("usage", {}), "latency_s": round(dt, 3),
    }


def main():
    corpus = [json.loads(x) for x in open(f"{W1D}/corpus.jsonl")]
    gold = json.load(open(f"{W1D}/gold.json"))
    out = []
    by_idx = {}
    for row in corpus:
        idx, rep = str(row["_idx"]), int(row["_rep"])
        r = post(row)
        rec = {"idx": idx, "rep": rep, **r}
        out.append(rec)
        by_idx.setdefault(idx, []).append(rec)
        ok = r["content"] == gold.get(idx)
        print(f"idx={idx} rep={rep} exact={ok} lat={r['latency_s']}s "
              f"ctok={r['usage'].get('completion_tokens')}", flush=True)
    with open(f"{HERE}/out_{TAG}.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    det = []
    total_exact = 0
    total_repro = 0
    for idx, recs in sorted(by_idx.items(), key=lambda x: int(x[0])):
        distinct = len(set(r["raw_args"] for r in recs))
        exact = sum(1 for r in recs if r["content"] == gold.get(idx))
        repro = distinct == 1
        total_exact += exact
        total_repro += int(repro)
        det.append({"idx": int(idx), "n": len(recs), "distinct": distinct,
                    "exact": exact, "bit_reproducible": repro})
    summary = {"tag": TAG, "per_idx": det,
               "exact_total": f"{total_exact}/{sum(len(v) for v in by_idx.values())}",
               "bit_reproducible_total": f"{total_repro}/{len(by_idx)}"}
    json.dump(summary, open(f"{HERE}/det_{TAG}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
