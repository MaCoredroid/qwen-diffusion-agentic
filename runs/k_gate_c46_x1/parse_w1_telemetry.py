#!/usr/bin/env python3
"""C46-new-envelope: parse the twin server logs into w1 draft-verify telemetry.

The FLARE engine emits ONE audit line per finished request (per agent turn):

  FLARE hybrid_clean req=<id> done: model_forwards=<N> forced_token_count=<F>
    value_tokens=<V> projected_value_tokens_exact=<P> generated_tokens=<G>
    stop_reason=<S> w1[on=<bool> spans=<s> toks=<t> vfwd=<vf> rej=<r> arej=<a>]

model_forwards / generated_tokens are PER-REQUEST (from the decoder stats). The
w1[...] block is CUMULATIVE over the whole server lifetime (the counters live on
the shared sampler), so the FINAL (max) w1 snapshot is the run total, and the
per-request stream is kept for distribution. The banked gate-OFF twin log predates
the w1 telemetry, so its lines carry model_forwards/generated_tokens but no w1[...]
block -> gate-OFF tok/fwd is still directly comparable (both are the same
decoder-stat ratio).

Computes live blended tok/fwd = sum(generated_tokens)/sum(model_forwards) for the
gate-ON arm and the banked gate-OFF arm, plus wall/episode, and asserts arej==0.

usage: parse_w1_telemetry.py <gate_on_server.log> <gate_off_server.log> \
                             <gate_on_arm_timing.json> <gate_off_wall_seconds> \
                             <n_episodes> <out.json>
"""
import json
import re
import sys

DONE = re.compile(
    r"hybrid_clean req=(?P<req>\S+) done: "
    r"model_forwards=(?P<mf>\d+) "
    r"forced_token_count=(?P<fc>\d+) "
    r"value_tokens=(?P<vt>\d+) "
    r"projected_value_tokens_exact=(?P<pv>\d+) "
    r"generated_tokens=(?P<gt>\d+) "
    r"stop_reason=(?P<sr>\S+)"
    r"(?: w1\[on=(?P<on>\w+) spans=(?P<sp>\d+) toks=(?P<tk>\d+) "
    r"vfwd=(?P<vf>\d+) rej=(?P<rj>\d+) arej=(?P<ar>\d+)\])?"
)


def parse_log(path):
    """Return (records, aggregate) for one server log; aggregate is None if empty."""
    recs = []
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return [], None
    for m in DONE.finditer(txt):
        d = m.groupdict()
        rec = {
            "req": d["req"],
            "model_forwards": int(d["mf"]),
            "forced_token_count": int(d["fc"]),
            "value_tokens": int(d["vt"]),
            "projected_value_tokens_exact": int(d["pv"]),
            "generated_tokens": int(d["gt"]),
            "stop_reason": d["sr"],
        }
        if d["on"] is not None:
            rec["w1"] = {
                "on": d["on"] == "True",
                "spans": int(d["sp"]),
                "toks": int(d["tk"]),
                "vfwd": int(d["vf"]),
                "rej": int(d["rj"]),
                "arej": int(d["ar"]),
            }
        recs.append(rec)
    if not recs:
        return [], None
    tmf = sum(r["model_forwards"] for r in recs)
    tgt = sum(r["generated_tokens"] for r in recs)
    tvt = sum(r["value_tokens"] for r in recs)
    tpv = sum(r["projected_value_tokens_exact"] for r in recs)
    w1recs = [r["w1"] for r in recs if "w1" in r]
    gate_on = bool(w1recs) and all(w["on"] for w in w1recs)
    # w1[...] is cumulative on the shared sampler -> the run total is the max snapshot.
    final_w1 = None
    if w1recs:
        final_w1 = {
            "spans": max(w["spans"] for w in w1recs),
            "toks": max(w["toks"] for w in w1recs),
            "vfwd": max(w["vfwd"] for w in w1recs),
            "rej": max(w["rej"] for w in w1recs),
            "arej": max(w["arej"] for w in w1recs),
        }
    agg = {
        "requests_logged": len(recs),
        "total_model_forwards": tmf,
        "total_generated_tokens": tgt,
        "total_value_tokens": tvt,
        "total_projected_value_tokens_exact": tpv,
        "live_tok_per_fwd": round(tgt / tmf, 4) if tmf else None,
        "gate_on": gate_on,
        "w1_final_cumulative": final_w1,
        "arej_total": (final_w1["arej"] if final_w1 else None),
    }
    return recs, agg


def main():
    (on_log, off_log, on_timing_p, off_wall_s, n_eps_s, outp) = sys.argv[1:7]
    n_eps = int(n_eps_s)
    off_wall = int(off_wall_s)

    on_recs, on_agg = parse_log(on_log)
    off_recs, off_agg = parse_log(off_log)

    try:
        on_wall = int(json.load(open(on_timing_p)).get("wall_seconds") or 0)
    except OSError:
        on_wall = 0

    def wall_per_ep(w):
        return round(w / n_eps, 1) if n_eps else None

    speedups = {}
    if on_agg and off_agg and off_agg["live_tok_per_fwd"]:
        speedups["tok_per_fwd_gain_on_vs_off"] = round(
            on_agg["live_tok_per_fwd"] / off_agg["live_tok_per_fwd"], 3
        )
    if on_wall and off_wall:
        speedups["wall_per_ep_ratio_off_over_on"] = round(off_wall / on_wall, 3)

    arej_total = on_agg["arej_total"] if on_agg else None
    out = {
        "read": "C46-new-envelope w1 draft-verify telemetry (gate-ON vs banked gate-OFF)",
        "n_episodes": n_eps,
        "gate_on_arm": {
            "server_log": on_log,
            "aggregate": on_agg,
            "wall_seconds": on_wall,
            "wall_per_episode_s": wall_per_ep(on_wall),
        },
        "gate_off_arm_banked": {
            "server_log": off_log,
            "aggregate": off_agg,
            "wall_seconds": off_wall,
            "wall_per_episode_s": wall_per_ep(off_wall),
        },
        "speedups": speedups,
        "arej_total": arej_total,
        " AREJ_CLEAN": (arej_total == 0),
        "per_request_gate_on": on_recs,
    }
    json.dump(out, open(outp, "w"), indent=2)
    # console summary
    if on_agg:
        print(f"gate-ON: reqs={on_agg['requests_logged']} "
              f"fwd={on_agg['total_model_forwards']} tok={on_agg['total_generated_tokens']} "
              f"tok/fwd={on_agg['live_tok_per_fwd']} "
              f"w1={on_agg['w1_final_cumulative']} arej={arej_total} "
              f"wall={on_wall}s ({wall_per_ep(on_wall)}s/ep)")
    else:
        print("gate-ON: NO done lines parsed yet")
    if off_agg:
        print(f"gate-OFF(banked): reqs={off_agg['requests_logged']} "
              f"tok/fwd={off_agg['live_tok_per_fwd']} wall={off_wall}s "
              f"({wall_per_ep(off_wall)}s/ep)")
    print(f"speedups={speedups}  AREJ_CLEAN={arej_total == 0}")


if __name__ == "__main__":
    main()
