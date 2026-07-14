#!/usr/bin/env python3
"""W-1c analysis: (a) FA byte-diff, (c) A6 byte-diff, (b) throughput from logs.

Usage:
  compare.py fa   <gateoff.jsonl> <gateon.jsonl>
  compare.py a6   <gateoff.jsonl> <gateon.jsonl> <gateon_server.log>
  compare.py thru <gateoff.jsonl> <gateon.jsonl> <gateoff_server.log> <gateon_server.log>
Prints a compact JSON summary (bounded).
"""
import json
import re
import sys

DONE = re.compile(
    r"req=(chatcmpl-\S+) done: model_forwards=(\d+) forced_token_count=(\d+) "
    r"value_tokens=(\d+) projected_value_tokens_exact=(\d+) generated_tokens=(\d+) "
    r"stop_reason=(\S+)(?:.*?w1\[on=(\w+) spans=(\d+) toks=(\d+) vfwd=(\d+) rej=(\d+)\])?"
)


def parse_log(path):
    """Return ordered list of per-request dicts (w1 fields are cumulative)."""
    out = []
    for ln in open(path, errors="replace"):
        m = DONE.search(ln)
        if not m:
            continue
        cid, fwd, forced, val, proj, gen, stop = m.group(1), *map(int, m.group(2, 3, 4, 5, 6)), m.group(7)
        d = {"cmpl_id": cid, "model_forwards": fwd, "forced": forced, "value_tokens": val,
             "proj_exact": proj, "generated_tokens": gen, "stop": stop}
        if m.group(8):
            d.update(w1_on=m.group(8) == "True", w1_spans=int(m.group(9)),
                     w1_toks=int(m.group(10)), w1_vfwd=int(m.group(11)), w1_rej=int(m.group(12)))
        out.append(d)
    return out


def log_by_id(path):
    lst = [d for d in parse_log(path) if "_warmup" not in d["cmpl_id"]]
    # de-cumulate w1 fields into per-request deltas
    prev = {"w1_spans": 0, "w1_toks": 0, "w1_vfwd": 0, "w1_rej": 0}
    for d in lst:
        if "w1_spans" in d:
            for k in ("w1_spans", "w1_toks", "w1_vfwd", "w1_rej"):
                d[k + "_delta"] = d[k] - prev[k]
                prev[k] = d[k]
    # the server req id has an extra "-<suffix>" the client id lacks; key by the
    # client-visible prefix (everything up to the last dash).
    return {d["cmpl_id"].rsplit("-", 1)[0]: d for d in lst}, lst


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def fa_cmp(off_p, on_p):
    off, on = load(off_p), load(on_p)
    byidx_off = {r["idx"]: r for r in off}
    rows = []
    for r in on:
        o = byidx_off.get(r["idx"], {})
        same_content = (o.get("content_text") == r.get("content_text"))
        same_args = (o.get("args_raw") == r.get("args_raw"))
        byte_ident = same_content and same_args
        rows.append({"idx": r["idx"], "cls": r.get("cls"), "deploy": r.get("deploy"),
                     "byte_identical_on_vs_off": byte_ident,
                     "on_emit_eq_canonical": r.get("emit_eq_canonical"),
                     "on_emit_eq_distractor": r.get("emit_eq_distractor"),
                     "off_emit_eq_canonical": o.get("emit_eq_canonical")})
    # a deploy-class FALSE ACCEPT = gate-ON diverges from gate-OFF on a deploy case
    # (the fast path changed a value-corruption/near-dup outcome) OR gate-ON emits
    # the perturbed distractor.
    deploy = [x for x in rows if x["deploy"]]
    dep_fa = [x for x in deploy if (not x["byte_identical_on_vs_off"]) or x["on_emit_eq_distractor"]]
    allcls = {}
    for x in rows:
        allcls.setdefault(x["cls"], {"n": 0, "byte_div": 0, "emit_distractor": 0})
        allcls[x["cls"]]["n"] += 1
        allcls[x["cls"]]["byte_div"] += 0 if x["byte_identical_on_vs_off"] else 1
        allcls[x["cls"]]["emit_distractor"] += 1 if x["on_emit_eq_distractor"] else 1 if False else 0
        if x["on_emit_eq_distractor"]:
            allcls[x["cls"]]["emit_distractor"] += 1
    summary = {
        "n_cases": len(rows),
        "deploy_cases": len(deploy),
        "deploy_false_accepts": len(dep_fa),
        "deploy_false_accept_ids": [x["idx"] for x in dep_fa],
        "byte_identical_all": sum(1 for x in rows if x["byte_identical_on_vs_off"]),
        "per_class": allcls,
        "on_emit_eq_canonical": sum(1 for x in rows if x["on_emit_eq_canonical"]),
        "BAR_deploy_full_span_false_accepts": 0,
        "PASS": len(dep_fa) == 0,
    }
    print(json.dumps(summary, indent=1))
    return summary


def a6_cmp(off_p, on_p, on_log):
    off, on = load(off_p), load(on_p)
    byidx_off = {r["idx"]: r for r in off}
    lg, _ = log_by_id(on_log)
    rows = []
    for r in on:
        o = byidx_off.get(r["idx"], {})
        same = (o.get("content") == r.get("content")) and \
               (json.dumps(o.get("tool_calls")) == json.dumps(r.get("tool_calls")))
        lgd = lg.get(r.get("cmpl_id"), {})
        fired = lgd.get("w1_spans_delta", 0) > 0
        # exact-args validity: every tool_call's arguments parse as JSON
        args_valid = True
        for tc in (r.get("tool_calls") or []):
            try:
                json.loads(tc.get("arguments") or "{}")
            except Exception:
                args_valid = False
        rows.append({"idx": r["idx"], "src": (r.get("meta") or {}).get("_src"),
                     "byte_identical": same, "fired": fired,
                     "spans": lgd.get("w1_spans_delta", 0), "args_valid": args_valid,
                     "gen": lgd.get("generated_tokens"), "fwd": lgd.get("model_forwards")})
    unfired = [x for x in rows if not x["fired"]]
    fired = [x for x in rows if x["fired"]]
    summary = {
        "n": len(rows),
        "fired": len(fired), "unfired": len(unfired),
        "unfired_byte_identical": sum(1 for x in unfired if x["byte_identical"]),
        "fired_byte_identical": sum(1 for x in fired if x["byte_identical"]),
        "fired_args_valid": sum(1 for x in fired if x["args_valid"]),
        "all_byte_identical": sum(1 for x in rows if x["byte_identical"]),
        "rows": rows,
        "PASS_unfired_byte_identical": all(x["byte_identical"] for x in unfired),
        "PASS_fired_args_valid": all(x["args_valid"] for x in fired),
    }
    print(json.dumps(summary, indent=1))
    return summary


def thru_cmp(off_p, on_p, off_log, on_log):
    off, on = load(off_p), load(on_p)
    lg_off, list_off = log_by_id(off_log)
    lg_on, list_on = log_by_id(on_log)

    def agg(rows, lg, label):
        buckets = {}
        for r in rows:
            cls = (r.get("meta") or {}).get("_class", "?")
            lgd = lg.get(r.get("cmpl_id"), {})
            if not lgd:
                continue
            b = buckets.setdefault(cls, {"n": 0, "gen": 0, "fwd": 0, "spans": 0, "rej": 0,
                                         "vfwd": 0, "lat": 0.0, "proj_nonzero": 0})
            b["n"] += 1
            b["gen"] += lgd.get("generated_tokens", 0)
            b["fwd"] += lgd.get("model_forwards", 0)
            b["spans"] += lgd.get("w1_spans_delta", 0)
            b["rej"] += lgd.get("w1_rej_delta", 0)
            b["vfwd"] += lgd.get("w1_vfwd_delta", 0)
            b["lat"] += r.get("latency_s", 0.0) or 0.0
            b["proj_nonzero"] += 1 if lgd.get("proj_exact", 0) else 0
        for cls, b in buckets.items():
            b["tok_per_fwd"] = round(b["gen"] / b["fwd"], 3) if b["fwd"] else None
            b["ms_per_committed_tok"] = round(b["lat"] * 1000 / b["gen"], 2) if b["gen"] else None
            b["reject_tax_share"] = round(b["rej"] / (b["spans"] + b["rej"]), 3) if (b["spans"] + b["rej"]) else None
        # overall
        tg = sum(b["gen"] for b in buckets.values()); tf = sum(b["fwd"] for b in buckets.values())
        tl = sum(b["lat"] for b in buckets.values())
        overall = {"tok_per_fwd": round(tg / tf, 3) if tf else None,
                   "ms_per_committed_tok": round(tl * 1000 / tg, 2) if tg else None,
                   "gen": tg, "fwd": tf}
        return {"label": label, "buckets": buckets, "overall_blended": overall}

    off_a = agg(off, lg_off, "gate_off_K1")
    on_a = agg(on, lg_on, "gate_on_W1")
    # per-turn byte-identity (gate-ON vs gate-OFF) + firing, split by class
    off_by_idx = {r["idx"]: r for r in off}
    byte = {}
    for r in on:
        cls = (r.get("meta") or {}).get("_class", "?")
        o = off_by_idx.get(r["idx"], {})
        ident = (o.get("content") == r.get("content")) and \
                (json.dumps(o.get("tool_calls")) == json.dumps(r.get("tool_calls")))
        fired = lg_on.get(r.get("cmpl_id"), {}).get("w1_spans_delta", 0) > 0
        b = byte.setdefault(cls, {"n": 0, "byte_identical": 0, "fired": 0})
        b["n"] += 1
        b["byte_identical"] += 1 if ident else 0
        b["fired"] += 1 if fired else 0
    summary = {
        "gate_off": off_a, "gate_on": on_a,
        "byte_and_firing_by_class": byte,
        "cpu_cert_ref": {"copy_tok_per_fwd": 14.41, "blended_speedup": 1.863,
                         "w0_baseline_recompute": 19.62},
        "k1_gate_off_baseline_tok_per_fwd_note": "gate_off buckets above ARE the matched K=1 baseline",
    }
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fa":
        fa_cmp(sys.argv[2], sys.argv[3])
    elif mode == "a6":
        a6_cmp(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "thru":
        thru_cmp(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
