#!/usr/bin/env python3
"""Ledger + stop-condition brain for the image-cycling data-gen orchestrator.

Centralizes the four pieces the shell loop must get exactly right — RESUME,
BEST-OF-K selection, the ROLLING-YIELD KILL, and the STOP conditions — in one
testable module.

INTERVENTION (cycle-2): the frontier now carries a `best_of_k` block (see
restratify_frontier.py). When present it turns on:
  * per-family best-of-k: an instance is re-attempted (distinct per-cycle seed —
    wired in datagen_gen.sh) up to k_resolvable times on a RESOLVABLE family and
    k_default (1) times otherwise, stopping early the moment it resolves
    (keep-any-resolve). Each attempt is one attempts.jsonl row (counts toward
    the rolling-yield denominator — bar unchanged at kill-yield/kill-window).
  * exploit/explore batch composition: ~ (1-explore_rate) of each batch from the
    resolvable "exploit" pool (best-of-k, coverage-before-depth), ~explore_rate
    from the FRESH zero-yield "explore" pool so the yield map keeps GROWING.
  * a GROWING resolvable set: resolvable = seed_resolvable_families UNION every
    family with >=1 'resolved' row in attempts.jsonl (a zero family that starts
    resolving auto-earns best-of-k).
With NO best_of_k block the module falls back to the original best-of-1 behavior
(frontier order MINUS attempted; per-instance idempotent record), so a partial
install is always safe.

subcommands:

  nextbatch <frontier.json> <attempts.jsonl> <batch_size>
      Print the next <=batch_size instance_ids. One id per line. Empty => exhausted.
      best_of_k ON: exploit(resolvable, best-of-k, by (attempt_count,index)) +
      explore(fresh zero-family, by index) split by explore_rate, backfilling from
      the other pool when one runs short. best_of_k OFF: frontier order MINUS
      every id already attempted.

  record <batchdir> <batch_id> <attempts.jsonl>
      Classify every id in <batchdir>/subset.json from the batch score report +
      pull.jsonl and APPEND one attempts row each. Idempotent per (instance_id,
      batch_id) — the SAME instance may be recorded once per batch (best-of-k),
      and re-running a batch's record never double-writes it. Verdicts:
        resolved | unresolved | empty_patch | error   (episode ran + scored)
        no_prediction                                  (episode ran, nothing scoreable)
        env_unavailable                                (image pull failed — never ran)

  state <attempts.jsonl> <keepers.jsonl> <frontier.json> <status_file>
        --target N --floor N --kill-yield F --kill-window N
      Compute counters + the STOP verdict, WRITE the human status line to
      <status_file>, and print a JSON blob the shell parses. Verdict is one of:
        DONE_TARGET | DONE_EXHAUSTED | KILL_YIELD_COLLAPSE | CONTINUE
      Rolling yield = resolved / real-attempts over the last <kill-window> REAL
      attempts (env_unavailable AND infra_invalid excluded — a missing image is a
      skip, and an infra_invalid batch [gen never booted a server] is our bug, not
      a teacher signal; the kill must judge the teacher). The kill only fires once
      the window is full.
      DONE_EXHAUSTED tracks the SAME eligibility nextbatch uses (best-of-k aware).
"""
from __future__ import annotations
import json, sys, time, argparse
from collections import Counter
from pathlib import Path

REAL_VERDICTS = {"resolved", "unresolved", "empty_patch", "error", "no_prediction"}


def _valid(rows: list[dict]) -> list[dict]:
    """Drop INFRA-INVALID rows. An attempt is infra_invalid when the batch never
    got a real shot at the teacher (e.g. gen preflight-timed-out / server never
    booted -> the whole batch is no_prediction). Such rows are NOT evidence about
    the teacher, so they must be invisible to yield, the kill window, coverage
    (attempt_count) and exhaustion — the instance is re-drawable, and the kill
    judges the teacher, not our infra bug. Written by `record --infra-invalid`."""
    return [r for r in rows if not r.get("infra_invalid")]


def _fam(iid: str) -> str:
    return iid.split("__")[0]


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
    return {r["instance_id"] for r in _valid(_read_jsonl(attempts)) if "instance_id" in r}


# ---------------------------------------------------------------------------
# best-of-k eligibility (shared by nextbatch + state so DONE_EXHAUSTED and the
# batch draw can NEVER disagree)
# ---------------------------------------------------------------------------
def _attempt_stats(attempt_rows: list[dict]):
    """Per-instance: real-attempt count, resolved?, env_unavailable? (terminal).
    INFRA-INVALID rows are dropped first (never a real attempt)."""
    real = Counter()
    resolved: set[str] = set()
    env_unavail: set[str] = set()
    for r in _valid(attempt_rows):
        iid = r.get("instance_id")
        v = r.get("verdict")
        if not iid:
            continue
        if v in REAL_VERDICTS:
            real[iid] += 1
        if v == "resolved":
            resolved.add(iid)
        if v == "env_unavailable":
            env_unavail.add(iid)
    return real, resolved, env_unavail


def _resolvable_families(cfg: dict, resolved_ids: set[str]) -> set[str]:
    """seed_resolvable_families UNION families with >=1 resolved attempt row.
    The map GROWS: a zero-family that starts resolving auto-earns best-of-k."""
    fams = set(cfg.get("seed_resolvable_families", []))
    fams |= {_fam(iid) for iid in resolved_ids}
    return fams


def eligible_pools(order: list[str], attempt_rows: list[dict], cfg: dict):
    """Return (exploit_ids, explore_ids) — each already priority-sorted.

    exploit = RESOLVABLE-family ids not yet resolved/skipped with attempt_count <
              k_resolvable, sorted by (attempt_count, frontier_index) so every
              fresh instance is tried once before any 2nd attempt (coverage first).
    explore = zero-family ids not yet resolved/skipped with attempt_count <
              k_default, sorted by frontier_index.
    """
    real, resolved, env_unavail = _attempt_stats(attempt_rows)
    resolvable = _resolvable_families(cfg, resolved)
    k_res = int(cfg.get("k_resolvable", 3))
    k_def = int(cfg.get("k_default", 1))
    exploit, explore = [], []
    for idx, iid in enumerate(order):
        if iid in resolved or iid in env_unavail:
            continue
        f = _fam(iid)
        if f in resolvable:
            if real[iid] < k_res:
                exploit.append((real[iid], idx, iid))
        else:
            if real[iid] < k_def:
                explore.append((real[iid], idx, iid))
    exploit.sort()
    explore.sort()
    return [x[2] for x in exploit], [x[2] for x in explore]


def cmd_nextbatch(frontier: Path, attempts: Path, batch_size: int) -> int:
    front = json.loads(frontier.read_text())
    order = front["order"]
    cfg = front.get("best_of_k")
    if not cfg or not cfg.get("enabled", True):
        # backward-compatible best-of-1: frontier order MINUS attempted
        done = _attempted_ids(attempts)
        picked = [i for i in order if i not in done][:batch_size]
        print("\n".join(picked))
        return 0

    exploit, explore = eligible_pools(order, _read_jsonl(attempts), cfg)
    rate = float(cfg.get("explore_rate", 0.13))
    n_explore = min(len(explore), round(batch_size * rate))
    n_exploit = min(len(exploit), batch_size - n_explore)
    picked = exploit[:n_exploit] + explore[:n_explore]
    if len(picked) < batch_size:  # one pool short -> backfill from the other
        extra = exploit[n_exploit:] + explore[n_explore:]
        picked += extra[: batch_size - len(picked)]
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


def cmd_record(batchdir: Path, batch_id: str, attempts: Path,
               infra_invalid: str | None = None) -> int:
    ids = json.loads((batchdir / "subset.json").read_text())["instance_ids"]
    rep = _load_report(batchdir)
    resolved = set(rep.get("resolved_ids", []))
    unresolved = set(rep.get("unresolved_ids", []))
    empty = set(rep.get("empty_patch_ids", []))
    error = set(rep.get("error_ids", []))
    pull_failed = _pull_failed_ids(batchdir)
    # Idempotent per (instance_id, batch_id): the same instance may appear across
    # batches (best-of-k), but a given batch's row is written at most once.
    have = {(r.get("instance_id"), r.get("batch_id")) for r in _read_jsonl(attempts)}

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    added = {"resolved": 0, "unresolved": 0, "empty_patch": 0, "error": 0,
             "no_prediction": 0, "env_unavailable": 0}
    with attempts.open("a") as f:
        for iid in ids:
            if (iid, batch_id) in have:
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
            row = {"instance_id": iid, "batch_id": batch_id, "verdict": v, "ts": ts}
            # INFRA-INVALID: the batch never got a real shot at the teacher (gen
            # failed -> server never booted). Flag every row so ledger yield / the
            # kill window / coverage all ignore it and the id stays re-drawable.
            if infra_invalid:
                row["infra_invalid"] = True
                row["infra_reason"] = infra_invalid
            f.write(json.dumps(row) + "\n")
            added[v] += 1
    print(json.dumps({"batch_id": batch_id, "recorded": sum(added.values()),
                      "infra_invalid": bool(infra_invalid),
                      "by_verdict": added}))
    return 0


def _trailing_infra_batches(attempt_rows: list[dict]) -> tuple[int, list[str]]:
    """Count how many of the MOST RECENT recorded batches are ENTIRELY infra_invalid.

    Batches are ordered by first appearance in attempts.jsonl (the orchestrator
    appends one batch's rows contiguously via `record`). A batch counts as
    infra_invalid when it has rows and EVERY row carries the infra_invalid flag
    (record --infra-invalid flags the whole batch uniformly). We walk BACKWARDS
    from the newest batch and stop at the first non-infra batch, so the result is
    the length of the trailing run of consecutive infra-invalid batches — a single
    recovered batch (e.g. the gmu hot-fix finally boots a server) resets it to 0.
    Feeds the HALT_INFRA belt in cmd_state: 'the last 2+ recorded batches are BOTH
    infra_invalid' == trailing >= 2."""
    order: list[str] = []
    rows_by_batch: dict[str, list[dict]] = {}
    for r in attempt_rows:
        bid = r.get("batch_id")
        if not bid:
            continue
        if bid not in rows_by_batch:
            rows_by_batch[bid] = []
            order.append(bid)
        rows_by_batch[bid].append(r)
    trailing = 0
    trailing_ids: list[str] = []
    for bid in reversed(order):
        rows = rows_by_batch[bid]
        if rows and all(r.get("infra_invalid") for r in rows):
            trailing += 1
            trailing_ids.append(bid)
        else:
            break
    return trailing, list(reversed(trailing_ids))


# Fire the infra-halt belt when this many most-recent batches are ALL infra_invalid.
INFRA_HALT_MIN_BATCHES = 2


def cmd_state(a: argparse.Namespace) -> int:
    attempt_rows = _read_jsonl(Path(a.attempts))
    keepers = _read_jsonl(Path(a.keepers))
    front = json.loads(Path(a.frontier).read_text())
    order = front["order"]
    cfg = front.get("best_of_k")

    n_keepers = len(keepers)

    if cfg and cfg.get("enabled", True):
        exploit, explore = eligible_pools(order, attempt_rows, cfg)
        remaining = len(exploit) + len(explore)
        remaining_detail = {"exploit_eligible": len(exploit),
                            "explore_eligible": len(explore)}
    else:
        attempted_ids = {r["instance_id"] for r in attempt_rows if "instance_id" in r}
        remaining = len([i for i in order if i not in attempted_ids])
        remaining_detail = None

    n_infra_invalid = len(attempt_rows) - len(_valid(attempt_rows))
    # yield / kill window / lifetime are computed over VALID real attempts only:
    # infra_invalid rows are our bug, not teacher evidence, so the kill judges the
    # teacher, not the infra failure.
    real = [r for r in _valid(attempt_rows) if r.get("verdict") in REAL_VERDICTS]
    n_real = len(real)
    window = real[-a.kill_window:]
    win_resolved = sum(1 for r in window if r["verdict"] == "resolved")
    rolling_yield = (win_resolved / len(window)) if window else None
    lifetime_resolved = sum(1 for r in real if r["verdict"] == "resolved")
    lifetime_yield = (lifetime_resolved / n_real) if n_real else None

    # HALT_INFRA belt (task-2): how many of the most-recent batches were ALL
    # infra_invalid (gen never booted a server)? >=2 means the run is spinning on a
    # persistent infra fault, not teaching — cycles 4-5 each burned BATCH_SIZE (50)
    # attempts this way because the verdict stayed CONTINUE through infra_invalid.
    infra_trailing, infra_trailing_ids = _trailing_infra_batches(attempt_rows)
    infra_halt = infra_trailing >= INFRA_HALT_MIN_BATCHES

    halt_reason = None
    if n_keepers >= a.target:
        verdict = "DONE_TARGET"
    elif infra_halt:
        # The RUNNING orchestrator honors ONLY {DONE_TARGET, DONE_EXHAUSTED,
        # KILL_YIELD_COLLAPSE} in its `case "$VERDICT"` (datagen_orch.sh ~L96-102);
        # any novel string (e.g. a bare "HALT_INFRA") falls through and CONTINUEs,
        # so it would NOT stop the loop. We therefore emit the honored KILL string —
        # whose arm writes DATAGEN_KILL.txt and `break`s — and carry the TRUE cause
        # in halt_reason + the infra_* fields + the human status line, so the KILL
        # flag and STATUS both read HALT_INFRA rather than a teacher-yield collapse.
        verdict = "KILL_YIELD_COLLAPSE"
        halt_reason = "HALT_INFRA"
    elif remaining <= 0:
        verdict = "DONE_EXHAUSTED"
    elif len(window) >= a.kill_window and rolling_yield is not None and rolling_yield < a.kill_yield:
        verdict = "KILL_YIELD_COLLAPSE"
    else:
        verdict = "CONTINUE"

    state = {
        "verdict": verdict,
        "halt_reason": halt_reason,
        "infra_trailing_batches": infra_trailing,
        "infra_trailing_batch_ids": infra_trailing_ids,
        "keepers": n_keepers,
        "target": a.target,
        "floor": a.floor,
        "floor_met": n_keepers >= a.floor,
        "attempts_real": n_real,
        "attempts_total": len(attempt_rows),
        "attempts_infra_invalid": n_infra_invalid,
        "remaining_frontier": remaining,
        "remaining_detail": remaining_detail,
        "best_of_k": bool(cfg and cfg.get("enabled", True)),
        "lifetime_yield": round(lifetime_yield, 4) if lifetime_yield is not None else None,
        "rolling_window": len(window),
        "rolling_yield": round(rolling_yield, 4) if rolling_yield is not None else None,
        "kill_yield_bar": a.kill_yield,
        "kill_window": a.kill_window,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    label = halt_reason or verdict  # human line leads with the TRUE cause when halting
    line = (f"{label} keepers={n_keepers}/{a.target} (floor {a.floor}"
            f"{'✓' if state['floor_met'] else '·'}) attempts={n_real} "
            f"lifetime_yield={state['lifetime_yield']} "
            f"rolling_yield={state['rolling_yield']}(w={len(window)}) "
            f"remaining={remaining}{'(bok)' if state['best_of_k'] else ''}"
            f"{f' INFRA_TRAILING={infra_trailing} via={verdict}' if halt_reason else ''}"
            f" {state['ts']}")
    Path(a.status_file).write_text(line + "\n")
    print(json.dumps(state))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nextbatch"); p.add_argument("frontier"); p.add_argument("attempts"); p.add_argument("batch_size", type=int)
    p = sub.add_parser("record"); p.add_argument("batchdir"); p.add_argument("batch_id"); p.add_argument("attempts")
    p.add_argument("--infra-invalid", default=None,
                   help="reason string; flags every recorded row infra_invalid "
                        "(excluded from yield/kill/coverage; id stays re-drawable)")
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
        return cmd_record(Path(a.batchdir), a.batch_id, Path(a.attempts),
                          infra_invalid=a.infra_invalid)
    if a.cmd == "state":
        return cmd_state(a)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
