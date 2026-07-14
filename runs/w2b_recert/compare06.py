#!/usr/bin/env python3
"""W-2b recert adjudicator: bit-identity gate-ON==gate-OFF @ temp 0.6, +tok/fwd.

Reads out_recert_{off,on}.jsonl + out_fa_{off,on}.jsonl and the two server logs.
Emits results06.json with:
  * bit_identity: per idx/rep, gate-ON raw_args == gate-OFF raw_args (the
    decisive bar -- identical by construction since the seeded trajectory is the
    same); reported as N_identical / N_total for recert and FA separately.
  * reproducibility: within each arm, the 5 reps of a recert idx are identical.
  * w1 counters live: final cumulative spans/toks/vfwd/rej/arej from server_on.log
    (spans>0 == the dormancy blocker is fixed; arej MUST be 0).
  * fa: 12/12 correct-resolution == 12/12 gate-ON==gate-OFF, false_accepts=0.
  * tok/fwd: blended generated/model_forwards per arm + forward speedup.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
W1D = "/home/mark/qwen_diffusion/runs/w1d_recert"

REQ_RE = re.compile(
    r"model_forwards=(\d+).*?generated_tokens=(\d+).*?"
    r"w1\[on=(\w+) spans=(\d+) toks=(\d+) vfwd=(\d+) rej=(\d+) arej=(\d+)\]"
)
REQ_RE_NOW1 = re.compile(r"req=(?!_warmup)[^ ]+ done: model_forwards=(\d+).*?generated_tokens=(\d+)")
WARM = re.compile(r"req=_warmup")


def load(tag):
    p = f"{HERE}/out_{tag}.jsonl"
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p)]


def key(rec):
    return (rec["idx"], rec["rep"])


def bit_identity(off, on):
    offm = {key(r): r["raw_args"] for r in off}
    onm = {key(r): r["raw_args"] for r in on}
    keys = sorted(set(offm) & set(onm))
    identical = sum(1 for k in keys if offm[k] == onm[k])
    mismatches = [k for k in keys if offm[k] != onm[k]]
    return identical, len(keys), mismatches


def reproducibility(recs):
    by_idx = {}
    for r in recs:
        by_idx.setdefault(r["idx"], []).append(r["raw_args"])
    repro = sum(1 for v in by_idx.values() if len(set(v)) == 1)
    return repro, len(by_idx)


def parse_log(path):
    """Return (final cumulative w1 dict or None, sum_forwards, sum_gen) over
    non-warmup requests."""
    if not os.path.exists(path):
        return None, 0, 0
    last_w1 = None
    fwd = gen = 0
    for line in open(path, errors="ignore"):
        if "FLARE hybrid_clean req=" not in line or "done:" not in line:
            continue
        if WARM.search(line):
            continue
        m = REQ_RE.search(line)
        if m:
            f, g = int(m.group(1)), int(m.group(2))
            fwd += f
            gen += g
            last_w1 = {
                "on": m.group(3), "spans": int(m.group(4)), "toks": int(m.group(5)),
                "vfwd": int(m.group(6)), "rej": int(m.group(7)), "arej": int(m.group(8)),
            }
        else:
            m2 = REQ_RE_NOW1.search(line)
            if m2:
                fwd += int(m2.group(1))
                gen += int(m2.group(2))
    return last_w1, fwd, gen


def exact_count(recs, goldpath):
    gold = json.load(open(goldpath))
    return sum(1 for r in recs if r.get("content") == gold.get(r["idx"])), len(recs)


def main():
    r_off, r_on = load("recert_off"), load("recert_on")
    f_off, f_on = load("fa_off"), load("fa_on")

    res = {"temp": 0.6, "seed": int(os.environ.get("RECERT_SEED", "20260714"))}

    # DECISIVE BAR: gate-ON == gate-OFF bit-identical.
    ri, rn, rmm = bit_identity(r_off, r_on)
    fi, fn, fmm = bit_identity(f_off, f_on)
    res["bit_identity"] = {
        "recert": f"{ri}/{rn}", "recert_mismatches": rmm,
        "fa": f"{fi}/{fn}", "fa_mismatches": fmm,
        "VERDICT": "BIT-IDENTICAL" if (ri == rn and fi == fn and rn > 0)
        else "DIVERGENT",
    }
    # Reproducibility within each arm.
    res["reproducibility"] = {
        "recert_off": "%d/%d" % reproducibility(r_off),
        "recert_on": "%d/%d" % reproducibility(r_on),
    }
    # Exactness vs gold (informational; content is greedy value -> temp-independent).
    try:
        res["exact_vs_gold"] = {
            "recert_off": "%d/%d" % exact_count(r_off, f"{W1D}/gold.json"),
            "recert_on": "%d/%d" % exact_count(r_on, f"{W1D}/gold.json"),
            "fa_off": "%d/%d" % exact_count(f_off, f"{HERE}/fa_gold.json"),
            "fa_on": "%d/%d" % exact_count(f_on, f"{HERE}/fa_gold.json"),
        }
    except Exception as e:
        res["exact_vs_gold"] = f"err: {e}"

    # FA correct-resolution == gate-ON==gate-OFF; false_accepts = FA divergences.
    res["fa_battery"] = {
        "correct_resolution": f"{fi}/{fn}",
        "false_accepts": fn - fi,
    }

    # w1 counters live + tok/fwd from server logs.
    log_on = f"{HERE}/server_on.log"
    log_off = f"{HERE}/server_off.log"
    w1_on, fwd_on, gen_on = parse_log(log_on)
    _w1_off, fwd_off, gen_off = parse_log(log_off)
    res["w1_counters_live"] = w1_on
    res["w1_dormancy_fixed"] = bool(w1_on and w1_on["spans"] > 0)
    res["arej_zero"] = bool(w1_on and w1_on["arej"] == 0)
    tokfwd_on = round(gen_on / fwd_on, 3) if fwd_on else None
    tokfwd_off = round(gen_off / fwd_off, 3) if fwd_off else None
    res["tok_per_fwd"] = {
        "gate_off": tokfwd_off, "gate_on": tokfwd_on,
        "forwards_off": fwd_off, "forwards_on": fwd_on,
        "gen_off": gen_off, "gen_on": gen_on,
        "forward_speedup_off_over_on": round(fwd_off / fwd_on, 3)
        if (fwd_on and fwd_off) else None,
    }

    json.dump(res, open(f"{HERE}/results06.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
