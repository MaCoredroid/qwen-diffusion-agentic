#!/usr/bin/env python3
"""KILL-T1 SPOT (X.1): 3 matched clean single-turn tool-call turns, exact_args on the X.1 twin.

Purpose (dispatch STEP 2): a conversion that breaks tool-calls is dead regardless of grounding.
Serves the X.1 twin (whatever export the launcher booted at :9952), replays 3 matched anchor
turns from the native scaleup eval (each has gold_tool_calls with exact arguments), and reports
exact_arguments = (emitted first tool call name == gold name) AND (emitted args == gold args).

Deterministic (temperature 0 / greedy) — a KILL-T1 anchor is a modal-decode identity check.
"""
import argparse, json, urllib.request, urllib.error

URL = "http://127.0.0.1:9952/v1/chat/completions"
MODEL = "qwen3.5-9b-flare-hybrid-clean"
NATIVE = "/home/mark/qwen_diffusion/data/toolcall_eval_native/flare_scaleup_native_58.jsonl"


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


def norm(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=[
        "heldout_seed_run1clean_0000",
        "heldout_seed_run1clean_0002",
        "heldout_seed_run1clean_0003",
    ])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = {json.loads(l)["id"]: json.loads(l) for l in open(NATIVE)}
    results = []
    n_exact = n_validname = n_validcall = 0
    for eid in args.ids:
        r = rows[eid]
        gold = r["gold_tool_calls"][0]
        gname = gold["name"]
        gargs = gold["arguments"]
        payload = {
            "model": MODEL,
            "messages": r["prompt_messages"],
            "tools": r["tools"],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        }
        resp, err = post(payload)
        rec = {"id": eid, "gold_name": gname}
        if resp is None:
            rec.update({"error": err, "valid_tool_call": False, "valid_name": False, "exact_arguments": False})
            results.append(rec)
            print(f"[killt1] {eid:32s} ERROR {err}", flush=True)
            continue
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        valid_call = bool(tcs)
        ename = tcs[0]["function"]["name"] if tcs else None
        try:
            eargs = json.loads(tcs[0]["function"]["arguments"]) if tcs else None
        except Exception:
            eargs = None
        valid_name = (ename == gname)
        exact = bool(valid_call and valid_name and eargs is not None and norm(eargs) == norm(gargs))
        n_validcall += valid_call
        n_validname += valid_name
        n_exact += exact
        rec.update({"valid_tool_call": valid_call, "emitted_name": ename,
                    "valid_name": valid_name, "exact_arguments": exact,
                    "emitted_args": eargs})
        results.append(rec)
        print(f"[killt1] {eid:32s} valid_call={valid_call} name_match={valid_name} exact_args={exact}", flush=True)

    out = {"n": len(args.ids), "n_valid_tool_call": n_validcall, "n_valid_name": n_validname,
           "n_exact_arguments": n_exact, "results": results}
    print(f"[killt1] SPOT exact_args={n_exact}/{len(args.ids)} valid_call={n_validcall}/{len(args.ids)} "
          f"name_match={n_validname}/{len(args.ids)}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[killt1] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
