#!/usr/bin/env python3
"""K1 committal REPRO: replay divergence-point prompts against the live twin
(FLARE hybrid_clean) and quantify read-argument grounding at the frozen envelope.

The hybrid_clean diffusion serving path does NOT expose per-token logprobs
(500 'list index out of range'), so we measure the operational decode
distribution by RESAMPLING at the exact frozen envelope (temp 0.6 / top_p 0.95 /
top_k 20). The sampling frequency IS the served confidence.

Same script serves both arms: pass --tag to label diffusion vs ar-control runs
(the AR control boots the SAME export on DECODE_POLICY=careful_live_grammar).
"""
import argparse, concurrent.futures as cf, json, sys, time, urllib.request, urllib.error

DIFF_ROOT = "/home/mark/qwen_diffusion/runs/k_gate_c46/diffusion"

# (instance, dump relative to DIFF_ROOT, what the run actually generated here)
PROMPTS = [
    ("matplotlib-25122", "dumps_shard_1/chat_0174.json", "offset=423 no-limit (paired w/ AR limit=50)"),
    ("sympy-13647",      "dumps_shard_1/chat_0273.json", "whole common.py (2320 ln)"),
    ("django-12273",     "dumps_shard_2/chat_0020.json", "whole base.py (1913 ln)"),
    ("matplotlib-20859", "dumps_shard_0/chat_0133.json", "whole figure.py (3228 ln)"),
    ("django-16256",     "dumps_shard_1/chat_0147.json", "whole file (10-consecutive-read instance)"),
]

URL = "http://127.0.0.1:9952/v1/chat/completions"
MODEL = "qwen3.5-9b-flare-hybrid-clean"


def post(payload, timeout=180):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP{e.code}:{e.read().decode()[:200]}"
    except Exception as e:
        return None, f"ERR:{type(e).__name__}:{e}"


def classify(resp):
    """Return dict describing the generated call."""
    ch = resp["choices"][0]
    msg = ch["message"]
    tcs = msg.get("tool_calls") or []
    ptoks = resp.get("usage", {}).get("prompt_tokens")
    ctoks = resp.get("usage", {}).get("completion_tokens")
    fr = ch.get("finish_reason")
    if not tcs:
        # no tool call -> text / quit
        txt = (msg.get("content") or "").strip()
        return {"cat": "NO_TOOL", "tool": None, "prompt_tokens": ptoks,
                "completion_tokens": ctoks, "finish_reason": fr, "text_head": txt[:80]}
    fn = tcs[0]["function"]
    name = fn["name"]
    try:
        args = json.loads(fn["arguments"])
    except Exception:
        args = {}
    rec = {"tool": name, "prompt_tokens": ptoks, "completion_tokens": ctoks,
           "finish_reason": fr, "args_keys": sorted(args.keys())}
    if name == "read_file":
        has_off = "offset" in args
        has_lim = "limit" in args
        rec["offset"] = args.get("offset")
        rec["limit"] = args.get("limit")
        if has_lim:
            rec["cat"] = "READ_BOUNDED"      # has limit -> windowed (grounded)
        elif has_off:
            rec["cat"] = "READ_OFFSET_NOLIMIT"  # offset..EOF (schema-invalid; unbounded)
        else:
            rec["cat"] = "READ_WHOLE"        # whole file to cap (unbounded)
    else:
        rec["cat"] = "OTHER_TOOL"
    return rec


def run_prompt(inst, dump, note, n, base_seed, max_tokens, temp, top_p, top_k, workers):
    d = json.load(open(f"{DIFF_ROOT}/{dump}"))
    base = {
        "model": MODEL,
        "messages": d["messages"],
        "tools": d["tools"],
        "chat_template_kwargs": d.get("chat_template_kwargs", {"enable_thinking": False}),
        "stream": False,
        "max_tokens": max_tokens,
    }
    results = []
    # 1 greedy modal decode
    greedy = dict(base, temperature=0.0)
    resp, err = post(greedy)
    greedy_rec = classify(resp) if resp else {"cat": "ERROR", "err": err}
    # N sampled at frozen envelope
    def one(i):
        p = dict(base, temperature=temp, top_p=top_p, top_k=top_k, seed=base_seed + i)
        resp, err = post(p)
        if resp is None:
            return {"cat": "ERROR", "err": err}
        return classify(resp)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, range(n)))
    return {"instance": inst, "dump": dump, "note": note,
            "greedy": greedy_rec, "samples": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diffusion")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"tag": args.tag, "envelope": {"temp": args.temp, "top_p": args.top_p, "top_k": args.top_k,
            "n": args.n, "max_tokens": args.max_tokens}, "prompts": []}
    for inst, dump, note in PROMPTS:
        t0 = time.time()
        r = run_prompt(inst, dump, note, args.n, 700000, args.max_tokens,
                       args.temp, args.top_p, args.top_k, args.workers)
        dt = time.time() - t0
        # per-prompt summary
        from collections import Counter
        cats = Counter(s["cat"] for s in r["samples"])
        reads = [s for s in r["samples"] if s.get("tool") == "read_file"]
        n_read = len(reads)
        n_lim = sum(1 for s in reads if s["cat"] == "READ_BOUNDED")
        pt = [s["prompt_tokens"] for s in r["samples"] if s.get("prompt_tokens")]
        r["summary"] = {
            "seconds": round(dt, 1),
            "cats": dict(cats),
            "greedy_cat": r["greedy"]["cat"],
            "greedy_detail": {k: r["greedy"].get(k) for k in ("tool", "offset", "limit", "args_keys")},
            "n_read_file": n_read,
            "n_read_with_limit": n_lim,
            "p_limit_given_read": round(n_lim / n_read, 3) if n_read else None,
            "prompt_tokens_med": sorted(pt)[len(pt)//2] if pt else None,
        }
        out["prompts"].append(r)
        s = r["summary"]
        print(f"[{args.tag}] {inst:18s} {dt:5.1f}s  greedy={s['greedy_cat']:18s} "
              f"cats={s['cats']}  P(limit|read)={s['p_limit_given_read']} "
              f"ptok~{s['prompt_tokens_med']}", flush=True)
    # pooled
    from collections import Counter
    allsamp = [s for p in out["prompts"] for s in p["samples"]]
    reads = [s for s in allsamp if s.get("tool") == "read_file"]
    n_read = len(reads)
    n_lim = sum(1 for s in reads if s["cat"] == "READ_BOUNDED")
    out["pooled"] = {
        "n_samples": len(allsamp),
        "cats": dict(Counter(s["cat"] for s in allsamp)),
        "n_read_file": n_read,
        "n_read_with_limit": n_lim,
        "p_limit_given_read": round(n_lim / n_read, 3) if n_read else None,
        "p_read_unbounded": round(sum(1 for s in reads if s["cat"] != "READ_BOUNDED") / n_read, 3) if n_read else None,
    }
    print(f"[{args.tag}] POOLED: {out['pooled']}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[{args.tag}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
