#!/usr/bin/env python3
"""Ledger + stop-condition brain for the image-cycling data-gen orchestrator.

Centralizes the three pieces the shell loop must get exactly right — RESUME,
the ROLLING-YIELD KILL, and the STOP conditions — in one testable module.

subcommands:

  nextbatch <frontier.json> <attempts.jsonl> <batch_size>
      Print the next <=batch_size instance_ids: frontier order MINUS every id
      already in attempts.jsonl (resume by construction — a resumed run never
      re-attempts a done/skipped instance). One id per line. Empty => exhausted.

  record <batchdir> <batch_id> <attempts.jsonl>
      Classify every id in <batchdir>/subset.json from the batch score report +
      pull.jsonl and APPEND one attempts row each (idempotent: ids already in
      attempts.jsonl are not re-added). Verdicts:
        resolved | unresolved | empty_patch | error   (episode ran + scored)
        no_prediction                                  (episode ran, nothing scoreable)
        env_unavailable                                (image pull failed — never ran)

  state <attempts.jsonl> <keepers.jsonl> <frontier.json> <status_file>
        --target N --floor N --kill-yield F --kill-window N
      Compute counters + the STOP verdict, WRITE the human status line to
      <status_file>, and print a JSON blob the shell parses. Verdict is one of:
        DONE_TARGET | DONE_EXHAUSTED | KILL_YIELD_COLLAPSE | CONTINUE
      Rolling yield = resolved / real-attempts over the last <kill-window> REAL
      attempts (env_unavailable excluded — a missing image is a skip, not a
      generator failure). The kill only fires once the window is full.
"""
from __future__ import annotations
import json, sys, time, argparse
from pathlib import Path

REAL_VERDICTS = {"resolved", "unresolved", "empty_patch", "error", "no_prediction"}


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for l in p.read_text().splitlines():
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def _attempted_ids(attempts: Path) -> set[str]:
    return {r["instance_id"] for r in _read_jsonl(attempts) if "instance_id" in r}


def cmd_nextbatch(frontier: Path, attempts: Path, batch_size: int) -> int:
    order = json.loads(frontier.read_text())["order"]
    done = _attempted_ids(attempts)
    picked = [i for i in order if i not in done][:batch_size]
    print("\n".join(picked))
    return 0


def _pull_failed_ids(batchdir: Path) -> set[str]:
    p = batchdir / "pull.jsonl"
    failed = set()
    for r in _read_jsonl(p):
        if r.get("status") in ("pull_failed",) or r.get("ok") is False:
            if r.get("instance_id"):
                failed.add(r["instance_id"])
    return failed


def _load_report(batchdir: Path) -> dict:
    reps = sorted((batchdir / "score").glob("*.json"))
    for r in reps:
        if r.name == "timing.json":
            continue
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if "resolved_ids" in d or "resolved_instances" in d:
            return d
    return {}


def cmd_record(batchdir: Path, batch_id: str, attempts: Path) -> int:
    ids = json.loads((batchdir / "subset.json").read_text())["instance_ids"]
    rep = _load_report(batchdir)
    resolved = set(rep.get("resolved_ids", []))
    unresolved = set(rep.get("unresolved_ids", []))
    empty = set(rep.get("empty_patch_ids", []))
    error = set(rep.get("error_ids", []))
    pull_failed = _pull_failed_ids(batchdir)
    have = _attempted_ids(attempts)

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    added = {"resolved": 0, "unresolved": 0, "empty_patch": 0, "error": 0,
             "no_prediction": 0, "env_unavailable": 0}
    with attempts.open("a") as f:
        for iid in ids:
            if iid in have:
                continue
            if iid in pull_failed:
                v = "env_unavailable"
            elif iid in resolved:
                v = "resolved"
            elif iid in empty:
                v = "empty_patch"
            elif iid in error:
                v = "error"
            elif iid in unresolved:
                v = "unresolved"
            else:
                v = "no_prediction"
            f.write(json.dumps({"instance_id": iid, "batch_id": batch_id,
                                "verdict": v, "ts": ts}) + "\n")
            added[v] += 1
    print(json.dumps({"batch_id": batch_id, "recorded": sum(added.values()),
                      "by_verdict": added}))
    return 0


def cmd_state(a: argparse.Namespace) -> int:
    attempts = _read_jsonl(Path(a.attempts))
    keepers = _read_jsonl(Path(a.keepers))
    order = json.loads(Path(a.frontier).read_text())["order"]

    n_keepers = len(keepers)
    attempted_ids = {r["instance_id"] for r in attempts if "instance_id" in r}
    remaining = [i for i in order if i not in attempted_ids]

    real = [r for r in attempts if r.get("verdict") in REAL_VERDICTS]
    n_real = len(real)
    window = real[-a.kill_window:]
    win_resolved = sum(1 for r in window if r["verdict"] == "resolved")
    rolling_yield = (win_resolved / len(window)) if window else None
    lifetime_resolved = sum(1 for r in real if r["verdict"] == "resolved")
    lifetime_yield = (lifetime_resolved / n_real) if n_real else None

    if n_keepers >= a.target:
        verdict = "DONE_TARGET"
    elif not remaining:
        verdict = "DONE_EXHAUSTED"
    elif len(window) >= a.kill_window and rolling_yield is not None and rolling_yield < a.kill_yield:
        verdict = "KILL_YIELD_COLLAPSE"
    else:
        verdict = "CONTINUE"

    state = {
        "verdict": verdict,
        "keepers": n_keepers,
        "target": a.target,
        "floor": a.floor,
        "floor_met": n_keepers >= a.floor,
        "attempts_real": n_real,
        "attempts_total": len(attempts),
        "remaining_frontier": len(remaining),
        "lifetime_yield": round(lifetime_yield, 4) if lifetime_yield is not None else None,
        "rolling_window": len(window),
        "rolling_yield": round(rolling_yield, 4) if rolling_yield is not None else None,
        "kill_yield_bar": a.kill_yield,
        "kill_window": a.kill_window,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    line = (f"{verdict} keepers={n_keepers}/{a.target} (floor {a.floor}"
            f"{'✓' if state['floor_met'] else '·'}) attempts={n_real} "
            f"lifetime_yield={state['lifetime_yield']} "
            f"rolling_yield={state['rolling_yield']}(w={len(window)}) "
            f"remaining={len(remaining)} {state['ts']}")
    Path(a.status_file).write_text(line + "\n")
    print(json.dumps(state))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nextbatch"); p.add_argument("frontier"); p.add_argument("attempts"); p.add_argument("batch_size", type=int)
    p = sub.add_parser("record"); p.add_argument("batchdir"); p.add_argument("batch_id"); p.add_argument("attempts")
    p = sub.add_parser("state")
    p.add_argument("attempts"); p.add_argument("keepers"); p.add_argument("frontier"); p.add_argument("status_file")
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--floor", type=int, default=400)
    p.add_argument("--kill-yield", type=float, default=0.10)
    p.add_argument("--kill-window", type=int, default=200)

    a = ap.parse_args()
    if a.cmd == "nextbatch":
        return cmd_nextbatch(Path(a.frontier), Path(a.attempts), a.batch_size)
    if a.cmd == "record":
        return cmd_record(Path(a.batchdir), a.batch_id, Path(a.attempts))
    if a.cmd == "state":
        return cmd_state(a)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
