#!/usr/bin/env python3
"""KEEPER extraction — resolved SWE-Gym episodes -> per-episode SFT trajectory JSONL.

Rejection-sampling on the ground-truth reward: KEEP ONLY instances the official
docker harness marked `resolved` (patch applied ∧ all FAIL_TO_PASS + PASS_TO_PASS
green). For each such episode this reconstructs the native `qwen3_xml` trajectory
and appends ONE row to keepers/keepers.jsonl. Idempotent: an instance already in
keepers.jsonl is skipped (safe to re-run per batch or after a resume).

Trajectory source of truth (given the reused proxy captures REQUEST bodies, not
streamed responses — see qwen_code_sglang_proxy.py):
  * per_task/<iid>/                unambiguously keyed by instance_id:
      - runner_metadata.json        repo/base_commit/image/started_at/elapsed/turns/
                                    tool_by_name/usage/container
      - patch.diff                  the ANSWER (git diff vs base_commit)
      - prompt.md                   the initial task prompt (AGENTS.md problem)
      - qwen_trace.json             the qwen-code structured result (terminal turn)
  * dumps_<shard>/chat_NNNN.json    the proxy request dumps. A shard runs its
                                    instances STRICTLY sequentially through ONE
                                    proxy, so an instance's dumps are the ones
                                    whose mtime ∈ [started_at, started_at+elapsed].
                                    The RICHEST (max len(messages)) dump in that
                                    window carries the full native conversation:
                                    system + user + [assistant, tool]* for every
                                    completed turn (empty-patch re-drives leave
                                    several segments; the richest is the winning
                                    drive). usage.jsonl rows in-window give
                                    per-turn tokens + finish_reason + ts.

Keeper record schema (one JSON object per line):
  instance_id, repo, base_commit, image, source, split
  verify   : {resolved, fail_to_pass_n, pass_to_pass_n, run_id, scored_by,
              harness_report_path}          # ground-truth reward metadata
  messages : [...]                          # per-turn native messages/tool-calls
  tools    : [...]                          # tool schemas presented to the model
  sft      : {assistant_turn_idxs, context_turn_idxs, note}   # loss mask plan
  final_patch : "<patch.diff>"              # the answer
  prompt   : {prompt_md, system_len, first_user_len}
  trajectory_meta : {num_turns, tool_by_name, usage, elapsed_s, sampling}
  provenance : {batch_id, shard, gen_dir, dumps_dir, richest_dump,
                n_dumps_in_window, generator, envelope, extracted_at, fidelity}

FIDELITY CAVEAT (recorded per row): the final assistant turn's streamed text is
NOT in a request dump; every assistant TOOL-CALL turn (the load-bearing grounding
targets, §1.1) IS present. For a fully-lossless terminal turn, enable proxy
response-body capture (one-line follow-up) and re-extract — training is cheap
([[retrain-freely-rule]]). `prompt_ids` are materialized at SFT-build time by
applying the native chat_template to `messages` ([[native-function-format-rule]]);
storing messages is strictly more general and re-tokenizable.

usage: extract_keepers.py <batchdir> <batch_id> <gen_root> <score_report_json>
                          <keepers_dir> [envelope_json]
"""
from __future__ import annotations
import json, sys, time, datetime as dt
from pathlib import Path


def _iso_to_epoch(s: str) -> float | None:
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc).timestamp()
    except Exception:
        try:
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None


def _resolved_ids(report_path: Path) -> tuple[list[str], dict]:
    """Read the swebench harness run report; return (resolved_ids, full_report).
    Falls back to scanning resolved_instances if resolved_ids absent."""
    if not report_path or not report_path.exists():
        return [], {}
    rep = json.loads(report_path.read_text())
    ids = rep.get("resolved_ids")
    if ids is None:
        # older schema: derive from per-key maps if present
        ids = [k for k, v in (rep.get("resolved") or {}).items() if v]
    return list(ids or []), rep


def _find_per_task(gen_root: Path, iid: str) -> tuple[Path | None, str | None]:
    """Locate <gen_root>/<shard>/verified/per_task/<iid>/ and return (dir, shard)."""
    for pt in gen_root.glob(f"*/verified/per_task/{iid}"):
        shard = pt.parents[2].name  # <shard>/verified/per_task/<iid>
        return pt, shard
    # tolerate a flat/other dataset-tag layout
    for pt in gen_root.glob(f"*/*/per_task/{iid}"):
        return pt, pt.parents[2].name
    return None, None


def _dumps_dir_for(gen_root: Path, shard: str) -> Path | None:
    cand = gen_root / f"dumps_{shard}"
    return cand if cand.is_dir() else None


def _richest_dump_in_window(dumps_dir: Path, t0: float | None, t1: float | None):
    """Return (dump_path, payload, n_in_window). Prefer dumps whose mtime is in
    [t0,t1]; among those pick max len(messages). If the window yields nothing
    (clock skew / missing metadata) fall back to the global richest dump."""
    best = None; best_n = -1; n_win = 0
    fallback = None; fb_n = -1
    for f in sorted(dumps_dir.glob("chat_*.json")):
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        m = len(payload.get("messages", []))
        if m > fb_n:
            fallback, fb_n = (f, payload), m
        mt = f.stat().st_mtime
        in_win = (t0 is None or mt >= t0 - 5) and (t1 is None or mt <= t1 + 5)
        if in_win:
            n_win += 1
            if m > best_n:
                best, best_n = (f, payload), m
    if best is not None:
        return best[0], best[1], n_win
    if fallback is not None:
        return fallback[0], fallback[1], 0
    return None, None, 0


def _usage_rows_in_window(dumps_dir: Path, t0, t1) -> list[dict]:
    p = dumps_dir / "usage.jsonl"
    if not p.exists():
        return []
    rows = []
    for l in p.read_text().splitlines():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        ts = r.get("ts")
        if ts is None or ((t0 is None or ts >= t0 - 5) and (t1 is None or ts <= t1 + 5)):
            rows.append(r)
    return rows


def main() -> int:
    batchdir = Path(sys.argv[1]); batch_id = sys.argv[2]
    gen_root = Path(sys.argv[3]); report_path = Path(sys.argv[4])
    keepers_dir = Path(sys.argv[5]); keepers_dir.mkdir(parents=True, exist_ok=True)
    envelope = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed_base": 1234}
    raw_dir = keepers_dir / "raw"; raw_dir.mkdir(exist_ok=True)
    kfile = keepers_dir / "keepers.jsonl"

    # dataset records for verify metadata (FAIL_TO_PASS / PASS_TO_PASS counts)
    dsmap = {}
    dspath = batchdir / "dataset.json"
    if dspath.exists():
        for r in json.loads(dspath.read_text()):
            dsmap[r["instance_id"]] = r

    # per-id provenance (belt-lever): swe_gym vs swe_verified. Absent -> all gym.
    src_map = {}
    spath = batchdir / "sources.json"
    if spath.exists():
        try:
            src_map = json.loads(spath.read_text())
        except Exception:
            src_map = {}
    SRC_LABEL = {"swe_gym": "SWE-Gym", "swe_verified": "SWE-bench_Verified"}

    already = set()
    if kfile.exists():
        for l in kfile.read_text().splitlines():
            l = l.strip()
            if l:
                try:
                    already.add(json.loads(l)["instance_id"])
                except Exception:
                    pass

    resolved, report = _resolved_ids(report_path)
    run_id = report.get("run_id") or batch_id
    added, skipped, problems = [], [], []

    for iid in resolved:
        if iid in already:
            skipped.append(iid); continue
        pt, shard = _find_per_task(gen_root, iid)
        if pt is None:
            problems.append({"iid": iid, "why": "no_per_task_dir"}); continue
        meta = {}
        mp = pt / "runner_metadata.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        patch = ""
        pf = pt / "patch.diff"
        if pf.exists():
            patch = pf.read_text()
        prompt_md = ""
        pmd = pt / "prompt.md"
        if pmd.exists():
            prompt_md = pmd.read_text()

        t0 = _iso_to_epoch(meta.get("started_at", ""))
        elapsed = (meta.get("qwen") or {}).get("elapsed_s")
        t1 = (t0 + float(elapsed) + 30) if (t0 and elapsed) else None

        messages, tools, dump_name, n_win, usage_rows = [], [], None, 0, []
        dumps_dir = _dumps_dir_for(gen_root, shard) if shard else None
        if dumps_dir:
            dpath, payload, n_win = _richest_dump_in_window(dumps_dir, t0, t1)
            if payload:
                messages = payload.get("messages", [])
                tools = payload.get("tools", [])
                dump_name = dpath.name
            usage_rows = _usage_rows_in_window(dumps_dir, t0, t1)

        # loss-mask plan (§1.1): assistant turns are SFT targets, others context.
        asst = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        ctx = [i for i, m in enumerate(messages) if m.get("role") != "assistant"]
        sys_len = next((len(m.get("content") or "") for m in messages
                        if m.get("role") == "system"), 0)
        fu = next((m for m in messages if m.get("role") == "user"), {})
        fu_c = fu.get("content")
        fu_len = len(fu_c if isinstance(fu_c, str) else json.dumps(fu_c or ""))

        dsr = dsmap.get(iid, {})
        ftp = dsr.get("FAIL_TO_PASS")
        ptp = dsr.get("PASS_TO_PASS")
        ftp_n = len(ftp) if isinstance(ftp, list) else (len(json.loads(ftp)) if isinstance(ftp, str) and ftp.strip().startswith("[") else None)
        ptp_n = len(ptp) if isinstance(ptp, list) else (len(json.loads(ptp)) if isinstance(ptp, str) and ptp.strip().startswith("[") else None)

        fidelity = "high(all_toolcall_turns)" if messages else "LOW(no_dump_matched)"
        if messages and n_win == 0:
            fidelity = "medium(window_miss_used_global_richest)"

        rec = {
            "instance_id": iid,
            "repo": meta.get("repo") or dsr.get("repo"),
            "base_commit": meta.get("base_commit") or dsr.get("base_commit"),
            "image": meta.get("image"),
            "source": SRC_LABEL.get(src_map.get(iid, "swe_gym"), "SWE-Gym"),
            "split": "train",
            "verify": {
                "resolved": True,
                "fail_to_pass_n": ftp_n, "pass_to_pass_n": ptp_n,
                "run_id": run_id, "scored_by": "swebench.harness.run_evaluation(fork,official)",
                "harness_report_path": str(report_path),
            },
            "messages": messages,
            "tools": tools,
            "sft": {
                "assistant_turn_idxs": asst,
                "context_turn_idxs": ctx,
                "note": "assistant turns = loss targets; user/tool/system = "
                        "loss-masked context (§1.1). prompt_ids materialized at "
                        "SFT-build via native chat_template.",
            },
            "final_patch": patch,
            "prompt": {"prompt_md": prompt_md, "system_len": sys_len,
                       "first_user_len": fu_len},
            "trajectory_meta": {
                "num_turns": (meta.get("qwen") or {}).get("num_turns"),
                "tool_by_name": (meta.get("qwen") or {}).get("tool_by_name"),
                "usage": (meta.get("qwen") or {}).get("usage"),
                "elapsed_s": elapsed,
                "sampling": envelope,
                "per_turn_usage_rows": len(usage_rows),
            },
            "provenance": {
                "batch_id": batch_id, "shard": shard,
                "gen_dir": str(pt), "dumps_dir": str(dumps_dir) if dumps_dir else None,
                "richest_dump": dump_name, "n_dumps_in_window": n_win,
                "generator": "stock-Qwen3.5-9B-AR (qwen_code, native qwen3_xml)",
                "envelope": envelope,
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fidelity": fidelity,
            },
        }
        with kfile.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        # verbatim provenance bundle for audit
        (raw_dir / f"{iid}.json").write_text(json.dumps(
            {"messages": messages, "tools": tools, "usage_rows": usage_rows,
             "patch": patch, "runner_metadata": meta}, indent=1))
        added.append(iid)

    total = len(already) + len(added)
    summary = {"batch_id": batch_id, "resolved_in_report": len(resolved),
               "kept_new": len(added), "already_had": len(skipped),
               "problems": problems, "keepers_total": total,
               "keepers_jsonl": str(kfile)}
    (batchdir / "keepers_extract_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
