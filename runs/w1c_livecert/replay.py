#!/usr/bin/env python3
"""Drive a corpus of chat-completion request bodies through the live engine.

Reads argv[1] (jsonl, one request body per line), forces non-stream, POSTs each
to the FLARE server, and records the full assistant message (content + tool
calls), the chatcmpl id (to correlate with the EngineCore per-request counters
in the server log), usage, and latency. Writes argv[2] (jsonl).

Deterministic temp-0 regime is expected to already be set in the bodies. Used
for (b) throughput and (c) A6, on BOTH the gate-OFF and gate-ON boots.
"""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:9952"
CORPUS = sys.argv[1]
OUT = sys.argv[2]


def main():
    for _ in range(120):
        try:
            if requests.get(f"{BASE}/health", timeout=5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)
    bodies = [json.loads(l) for l in open(CORPUS) if l.strip()]
    with open(OUT, "w") as w:
        for i, b in enumerate(bodies):
            meta = {k: b[k] for k in list(b.keys()) if k.startswith("_")}
            body = {k: v for k, v in b.items() if not k.startswith("_")}
            body["stream"] = False
            body.setdefault("temperature", 0.0)
            t0 = time.time()
            try:
                r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
                dt = time.time() - t0
                r.raise_for_status()
                d = r.json()
                msg = d["choices"][0]["message"]
                tcs = msg.get("tool_calls") or []
                row = {
                    "idx": i, "meta": meta,
                    "cmpl_id": d.get("id"),
                    "content": msg.get("content"),
                    "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                                    "arguments": (tc.get("function") or {}).get("arguments")}
                                   for tc in tcs],
                    "finish_reason": d["choices"][0].get("finish_reason"),
                    "usage": d.get("usage"),
                    "latency_s": round(dt, 3),
                }
            except Exception as e:
                row = {"idx": i, "meta": meta, "error": repr(e),
                       "latency_s": round(time.time() - t0, 3)}
            w.write(json.dumps(row) + "\n")
            w.flush()
            cls = meta.get("_class") or meta.get("_src", "")
            ntc = len(row.get("tool_calls") or [])
            print(f"[{i}] {cls} id={row.get('cmpl_id')} tcs={ntc} "
                  f"fr={row.get('finish_reason')} lat={row.get('latency_s')}s "
                  f"usage={row.get('usage')}")


if __name__ == "__main__":
    main()
