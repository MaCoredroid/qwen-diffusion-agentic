#!/usr/bin/env python3
"""Build the ENVELOPE-CORRECTED 4-arm N=5 v3 report from aligned-runtime artifacts.

Joins, per (arm, instance):
  * OFFICIAL swebench verdict (resolved) from runs/stage_c_n5v3/scoring/*.json
  * agent metrics from <arm>/verified/per_task/<iid>/runner_metadata.json + trace
  * exact turns+tokens from the proxy usage.jsonl (robust to qwen exit mode:
    turn-limit 53 / loop-detect 1 / budget 55 write NOTHING to CLI stdout).
  * FLARE engine counters for BOTH diffusion arms (diffusion, diffstock) from
    logs/<arm>_server.log.
Prints the 4x5 table + per-arm rollups + the GREEDY(v2)->ENVELOPE(v3) loop-halt /
gave-up / resolve delta table; writes report.json.
"""
import json, re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/home/mark/qwen_diffusion/runs/stage_c_n5v3")
ARMS = ["ar", "mergedar", "diffusion", "diffstock"]
DIFFUSION_ARMS = {"diffusion", "diffstock"}
ARM_LABEL = {"ar": "stock-AR", "mergedar": "merged-AR",
             "diffusion": "diffusion", "diffstock": "diffstock"}
IDS = ["django__django-11119", "django__django-12754", "django__django-13741",
       "pytest-dev__pytest-8399", "sympy__sympy-13757"]
SHORT = {i: i.split("__")[-1] for i in IDS}
EXIT_MEANING = {0: "ok", 53: "turn-limit", 1: "loop-halt", 55: "budget", -1: "wall-timeout/kill"}

# GREEDY (v2) baseline rollups -- source: runs/stage_c_n5v2/report_table.txt +
# runs/stage_c_n5v2/diffstock_report_table.txt (the confounded greedy ladder).
GREEDY_BASELINE = {
    "ar":        {"resolve": "4/5", "loops": 0, "exits": {"ok": 3, "turn-limit": 2}},
    "mergedar":  {"resolve": "2/5", "loops": 1, "exits": {"ok": 3, "turn-limit": 1, "loop-halt": 1}},
    "diffusion": {"resolve": "1/5", "loops": 2, "exits": {"loop-halt": 2, "turn-limit": 3}},
    "diffstock": {"resolve": "0/5", "loops": 2, "exits": {"turn-limit": 3, "loop-halt": 2}},
}


def _epoch(iso):
    if not iso:
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def load_official():
    out = {}
    for arm in ARMS:
        rid = f"n5v3_{arm}"
        files = list(ROOT.glob(f"scoring/*.{rid}.json"))
        rec = {"resolved": set(), "completed": set(), "error": set(), "empty": set(), "report_file": None}
        if files:
            d = json.loads(files[0].read_text())
            rec["resolved"] = set(d.get("resolved_ids", []))
            rec["completed"] = set(d.get("completed_ids", []))
            rec["error"] = set(d.get("error_ids", []))
            rec["empty"] = set(d.get("empty_patch_ids", []))
            rec["report_file"] = str(files[0])
        out[arm] = rec
    return out


def load_usage(arm):
    recs = []
    uf = ROOT / f"dumps_{arm}" / "usage.jsonl"
    if uf.is_file():
        for ln in uf.read_text().splitlines():
            ln = ln.strip()
            if ln:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    return recs


FLARE_RE = re.compile(
    r"FLARE \w+ req=(\S+) done: model_forwards=(\d+) forced_token_count=(\d+) "
    r"value_tokens=(\d+) projected_value_tokens_exact=(\d+) generated_tokens=(\d+) stop_reason=(\S+)")
TS_RE = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")


def load_flare(server_log, year=2026):
    rows = []
    if not Path(server_log).is_file():
        return rows
    for ln in Path(server_log).read_text(errors="replace").splitlines():
        m = FLARE_RE.search(ln)
        if not m:
            continue
        t = TS_RE.search(ln)
        ts = None
        if t:
            mo, da, hh, mm, ss = map(int, t.groups())
            try:
                ts = datetime(year, mo, da, hh, mm, ss, tzinfo=timezone.utc).timestamp()
            except Exception:
                ts = None
        rows.append({"req": m.group(1), "model_forwards": int(m.group(2)),
                     "forced": int(m.group(3)), "value_tokens": int(m.group(4)),
                     "proj_exact": int(m.group(5)), "generated": int(m.group(6)),
                     "stop_reason": m.group(7), "ts": ts})
    return rows


def loop_flag(task_dir):
    for name in ("qwen_stderr.log", "qwen_trace.json"):
        p = task_dir / name
        if p.is_file():
            txt = p.read_text(errors="replace")
            if "Loop detection" in txt or "loop detected" in txt.lower():
                m = re.search(r"Loop detection halted the run \(([^)]*)\)", txt)
                return True, (m.group(1) if m else "loop_detected")
    return False, None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def build():
    official = load_official()
    report = {"envelope": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed_base": 1234,
                           "empty_patch_retries": 1},
              "arms": {}, "instances": IDS, "greedy_baseline": GREEDY_BASELINE}
    for arm in ARMS:
        usage = load_usage(arm)
        flare = load_flare(ROOT / "logs" / f"{arm}_server.log") if arm in DIFFUSION_ARMS else []
        per_task = ROOT / arm / "verified" / "per_task"
        arm_rec = {"label": ARM_LABEL[arm], "instances": {}, "official": {
            k: sorted(v) if isinstance(v, set) else v for k, v in official[arm].items()}}
        for iid in IDS:
            td = per_task / iid
            meta = {}
            mp = td / "runner_metadata.json"
            if mp.is_file():
                meta = json.loads(mp.read_text())
            q = meta.get("qwen") or {}
            s_ep, e_ep = _epoch(meta.get("started_at")), _epoch(meta.get("ended_at"))
            win = [u for u in usage if u.get("ts") is not None and s_ep is not None
                   and e_ep is not None and s_ep <= u["ts"] <= e_ep]
            in_tok = sum((u.get("usage") or {}).get("prompt_tokens", 0) for u in win if u.get("usage"))
            out_tok = sum((u.get("usage") or {}).get("completion_tokens", 0) for u in win if u.get("usage"))
            tot_tok = sum((u.get("usage") or {}).get("total_tokens", 0) for u in win if u.get("usage"))
            n_req = len(win)
            n_length = sum(1 for u in win if u.get("finish_reason") == "length")
            turns = q.get("num_turns")
            turns_src = "qwen"
            if turns is None:
                turns = n_req
                turns_src = "proxy_reqs"
            qusage = q.get("usage") or {}
            if qusage.get("total_tokens"):
                in_tok2, out_tok2, tot_tok2 = (qusage.get("input_tokens", 0),
                                               qusage.get("output_tokens", 0), qusage["total_tokens"])
                tok_src = "qwen"
            else:
                in_tok2, out_tok2, tot_tok2 = in_tok, out_tok, tot_tok
                tok_src = "proxy"
            looped, loop_reason = loop_flag(td)
            exit_code = q.get("exit_code")
            rec = {
                "resolved": iid in official[arm]["resolved"],
                "official_completed": iid in official[arm]["completed"],
                "official_error": iid in official[arm]["error"],
                "official_empty": iid in official[arm]["empty"],
                "patch_bytes": meta.get("patch_bytes"),
                "edit": bool(meta.get("patch_bytes")),
                "turns": turns, "turns_src": turns_src,
                "in_tok": in_tok2, "out_tok": out_tok2, "tot_tok": tot_tok2, "tok_src": tok_src,
                "wall_s": q.get("elapsed_s"),
                "exit_code": exit_code, "exit_meaning": EXIT_MEANING.get(exit_code, str(exit_code)),
                "subtype": q.get("subtype"), "timed_out": q.get("timed_out"),
                "n_length_finish": n_length, "n_proxy_reqs": n_req,
                "loop_detected": looped, "loop_reason": loop_reason,
                "empty_patch_retry": meta.get("empty_patch_retry"),
                "status": meta.get("status"),
            }
            if arm in DIFFUSION_ARMS:
                fw = [f for f in flare if f.get("ts") is not None and s_ep is not None
                      and e_ep is not None and s_ep <= f["ts"] <= e_ep]
                if fw:
                    tf = sum(f["model_forwards"] for f in fw)
                    tg = sum(f["generated"] for f in fw)
                    rec["engine"] = {
                        "n_reqs": len(fw), "model_forwards": tf, "generated_tokens": tg,
                        "value_tokens": sum(f["value_tokens"] for f in fw),
                        "forced_tokens": sum(f["forced"] for f in fw),
                        "fwd_per_tok": round(tf / tg, 3) if tg else None,
                        "stop_reasons": _count([f["stop_reason"] for f in fw]),
                    }
            arm_rec["instances"][iid] = rec
        insts = arm_rec["instances"]
        n = len(IDS)
        arm_rec["rollup"] = {
            "resolve_at_1": round(sum(1 for i in IDS if insts[i]["resolved"]) / n, 3),
            "n_resolved": sum(1 for i in IDS if insts[i]["resolved"]),
            "edit_rate": round(sum(1 for i in IDS if insts[i]["edit"]) / n, 3),
            "n_edit": sum(1 for i in IDS if insts[i]["edit"]),
            "turns_mean": _mean([insts[i]["turns"] for i in IDS]),
            "wall_mean_s": _mean([insts[i]["wall_s"] for i in IDS]),
            "tot_tok_mean": _mean([insts[i]["tot_tok"] for i in IDS]),
            "exit_dist": _count([insts[i]["exit_meaning"] for i in IDS]),
            "n_loop_detected": sum(1 for i in IDS if insts[i]["loop_detected"]),
            "n_gave_up": sum(1 for i in IDS if insts[i].get("subtype") == "agent_gave_up"),
        }
        if arm in DIFFUSION_ARMS:
            engs = [insts[i].get("engine") for i in IDS if insts[i].get("engine")]
            if engs:
                tf = sum(e["model_forwards"] for e in engs)
                tg = sum(e["generated_tokens"] for e in engs)
                arm_rec["rollup"]["engine_total"] = {
                    "model_forwards": tf, "generated_tokens": tg,
                    "fwd_per_tok": round(tf / tg, 3) if tg else None}
        report["arms"][arm] = arm_rec
    return report


def fmt(report):
    L = []
    L.append("=" * 96)
    L.append("ENVELOPE-CORRECTED 4-ARM N=5 (v3)  |  temp0.6/top_p0.95/top_k20/seed1234  |  OFFICIAL docker scoring")
    L.append("=" * 96)
    hdr = f"{'instance':<20}" + "".join(f"{ARM_LABEL[a]:>15}" for a in ARMS)
    for metric, label in [("resolved", "RESOLVE@1 (official)"), ("cell", "turns | tok | wall | exit")]:
        L.append("")
        L.append(label)
        L.append(hdr)
        for iid in IDS:
            row = f"{SHORT[iid]:<20}"
            for a in ARMS:
                r = report["arms"][a]["instances"][iid]
                if metric == "resolved":
                    mark = "RESOLVED" if r["resolved"] else ("edit" if r["edit"] else "no-edit")
                    if r["official_error"]:
                        mark = "ERROR"
                    row += f"{mark:>15}"
                else:
                    cell = f"{r['turns']}|{r['tot_tok']}|{_i(r['wall_s'])}s|{r['exit_meaning']}"
                    row += f"{cell:>15}"
            L.append(row)
    L.append("")
    L.append("PER-ARM ROLLUP (v3 ENVELOPE)")
    for a in ARMS:
        ru = report["arms"][a]["rollup"]
        L.append(f"  {ARM_LABEL[a]:<11} resolve@1={ru['resolve_at_1']} ({ru['n_resolved']}/5)  "
                 f"edit={ru['edit_rate']} ({ru['n_edit']}/5)  turns~{ru['turns_mean']}  "
                 f"wall~{ru['wall_mean_s']}s  tok~{ru['tot_tok_mean']}  exits={ru['exit_dist']}  "
                 f"loops={ru['n_loop_detected']}  gave_up={ru['n_gave_up']}")
        if "engine_total" in ru:
            et = ru["engine_total"]
            L.append(f"              engine: model_forwards={et['model_forwards']} "
                     f"generated_tokens={et['generated_tokens']} fwd/tok={et['fwd_per_tok']}")
    L.append("")
    L.append("GREEDY (v2) -> ENVELOPE (v3)   [the degenerate-regime test]")
    L.append(f"  {'arm':<11}{'resolve@1':>18}{'loop-halts':>16}{'turn-limit':>14}")
    for a in ARMS:
        ru = report["arms"][a]["rollup"]
        gb = GREEDY_BASELINE[a]
        env_res = f"{ru['n_resolved']}/5"
        env_loops = ru["exit_dist"].get("loop-halt", 0)
        env_tl = ru["exit_dist"].get("turn-limit", 0)
        g_loops = gb["exits"].get("loop-halt", 0)
        g_tl = gb["exits"].get("turn-limit", 0)
        L.append(f"  {ARM_LABEL[a]:<11}"
                 f"{gb['resolve']+' -> '+env_res:>18}"
                 f"{str(g_loops)+' -> '+str(env_loops):>16}"
                 f"{str(g_tl)+' -> '+str(env_tl):>14}")
    return "\n".join(L)


def _i(x):
    return int(round(x)) if x is not None else "?"


if __name__ == "__main__":
    rep = build()
    (ROOT / "report.json").write_text(json.dumps(rep, indent=2))
    print(fmt(rep))
    print("\nwrote", ROOT / "report.json")
