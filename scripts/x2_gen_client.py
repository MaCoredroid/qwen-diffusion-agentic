#!/usr/bin/env python3
"""X.2 AR-self-distillation — generation client.

Reads gen_prefixes.jsonl (each = the token prefix ENDING at a <parameter=limit|offset>
marker, i.e. the read-phase state), POSTs each to the AR teacher's /v1/completions
endpoint at temperature 0 (greedy = deterministic, the 12/48 policy's conditional), and
records the emitted value continuation. The AR teacher is the SAME weights as the student
(mswe-S-iter2 vllm-bf16 export, careful_live_grammar = native AR path) — so the target is
the model's own conditional, not a foreign teacher (SECTION X.2).

Token-id prompt (exact, no detok/retok of the prefix). Resumable: skips req_ids already
present in the output file.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
import urllib.error


def post(url, payload, timeout=240):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP{e.code}:{e.read().decode()[:200]}"
    except Exception as e:
        return None, f"ERR:{type(e).__name__}:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:9953/v1/completions")
    ap.add_argument("--model", default="qwen3.5-9b-flare-hybrid-clean")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    reqs = [json.loads(l) for l in open(args.prefixes)]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["req_id"])
            except Exception:
                pass
    todo = [r for r in reqs if r["req_id"] not in done]
    print(f"[x2-gen] total={len(reqs)} done={len(done)} todo={len(todo)}", flush=True)

    # smoke: one request first, fail loud if the endpoint/prompt-shape is wrong
    if todo:
        s = todo[0]
        resp, err = post(args.url, {"model": args.model, "prompt": s["prefix_ids"],
                                    "temperature": 0.0, "max_tokens": args.max_tokens,
                                    "stop": ["</function>"]})
        if err:
            print(f"[x2-gen] SMOKE FAILED (token-id prompt): {err}", flush=True)
            sys.exit(4)
        txt = resp["choices"][0].get("text", "")
        print(f"[x2-gen] SMOKE ok: slot={s['slot_key']} text={txt!r}", flush=True)

    lock_out = open(args.out, "a")
    n_ok = [0]; n_err = [0]; t0 = time.time()

    def work(s):
        resp, err = post(args.url, {"model": args.model, "prompt": s["prefix_ids"],
                                    "temperature": 0.0, "max_tokens": args.max_tokens,
                                    "stop": ["</function>"]})
        if err:
            return {"req_id": s["req_id"], "slot_key": s["slot_key"],
                    "completion_text": "", "error": err}
        ch = resp["choices"][0]
        return {"req_id": s["req_id"], "slot_key": s["slot_key"],
                "completion_text": ch.get("text", ""),
                "finish_reason": ch.get("finish_reason")}

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s for s in todo}
        for i, fut in enumerate(cf.as_completed(futs)):
            rec = fut.result()
            lock_out.write(json.dumps(rec) + "\n"); lock_out.flush()
            if rec.get("error"):
                n_err[0] += 1
            else:
                n_ok[0] += 1
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"[x2-gen] {i+1}/{len(todo)} ok={n_ok[0]} err={n_err[0]} "
                      f"{(i+1)/dt:.1f} req/s", flush=True)
    lock_out.close()
    print(f"[x2-gen] DONE ok={n_ok[0]} err={n_err[0]} elapsed={time.time()-t0:.0f}s", flush=True)
    if n_ok[0] == 0:
        sys.exit(5)


if __name__ == "__main__":
    main()
