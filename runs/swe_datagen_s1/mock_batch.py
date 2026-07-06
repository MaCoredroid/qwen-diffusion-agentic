#!/usr/bin/env python3
"""DRY-RUN stubs for the batch lifecycle — NO GPU, NO docker. Synthesizes the
exact artifact SHAPES that the real pull/gen/score steps produce, so the REAL
keeper-extraction + ledger + resume + stop-condition logic runs unchanged over
them. This exercises the orchestrator's control flow end-to-end on one cycle.

Faithfulness choices that make this a real test (not a rubber stamp):
  * Each episode gets a DISJOINT past time-window (started_at + elapsed) and its
    proxy dumps are os.utime'd into that window, so extract_keepers' mtime-window
    dump->instance association is genuinely exercised (windows do not overlap).
  * `resolved` is a deterministic ~25% of patch-producing instances (hash of
    instance_id) — matching the measured envelope yield 0.25 — and gen/score
    agree on which ids resolve. Non-resolved + empty-patch cases are present too.
  * dumps carry a real system+user+[assistant,tool]* message array so the loss-
    mask plan (assistant_turn_idxs) is computed on real structure.

subcommands:
  gen   <batchdir> <gen_root> <C>
  score <batchdir> <gen_root>
"""
from __future__ import annotations
import json, os, sys, time, hashlib
from pathlib import Path

MODEL = "datagen-stockAR-env"
ENVELOPE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1234}
_BASE = time.time() - 3600           # episodes "ran" in the last hour
_SLOT = 200                          # seconds between episode start slots (disjoint)
_ELAPSED = 80


def _h(iid: str) -> int:
    return int(hashlib.sha256(iid.encode()).hexdigest(), 16)


def _produces_patch(iid: str) -> bool:
    return (_h(iid) % 100) >= 5           # ~95% patch-produced (probe: 0.95)


def _resolves(iid: str) -> bool:
    return _produces_patch(iid) and (_h(iid) % 100) < 25   # ~25% yield


def _iso(ep: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ep))


def _messages(iid: str, repo: str, turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "You are Qwen Code, a CLI agent. " * 40},
            {"role": "user", "content": [{"type": "text",
             "text": f"Fix the bug in {repo} (instance {iid}). Run the tests."}]}]
    for t in range(turns):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{t}", "type": "function",
                       "function": {"name": ["read_file", "grep_search", "edit",
                                             "run_shell_command"][t % 4],
                                    "arguments": json.dumps({"path": f"src/mod_{t}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{t}",
                     "content": f"<result of turn {t}>"})
    return msgs


def _tools() -> list[dict]:
    return [{"type": "function", "function": {"name": n, "parameters": {}}}
            for n in ("read_file", "edit", "run_shell_command", "grep_search")]


def cmd_gen(batchdir: Path, gen_root: Path, C: int) -> None:
    subset = json.loads((batchdir / "subset.json").read_text())
    ids = subset["instance_ids"]
    ds = {r["instance_id"]: r for r in json.loads((batchdir / "dataset.json").read_text())}
    gen_root.mkdir(parents=True, exist_ok=True)

    for k in range(C):
        shard_ids = [ids[i] for i in range(len(ids)) if i % C == k]
        pt_root = gen_root / f"shard_{k}" / "verified" / "per_task"
        pt_root.mkdir(parents=True, exist_ok=True)
        dumps = gen_root / f"dumps_shard_{k}"; dumps.mkdir(parents=True, exist_ok=True)
        preds = gen_root / f"shard_{k}" / "verified" / "predictions.jsonl"
        counter = 0
        usage_rows = []
        pred_lines = []
        for slot, iid in enumerate(shard_ids):
            rec = ds.get(iid, {})
            repo = rec.get("repo", "unknown/unknown")
            base_commit = rec.get("base_commit", "0" * 40)
            slug_1776 = iid.replace("__", "_1776_")
            image = f"swebench/sweb.eval.x86_64.{slug_1776}:latest"
            t0 = _BASE + slot * _SLOT
            turns = 3 + (_h(iid) % 5)
            has_patch = _produces_patch(iid)
            patch = ("" if not has_patch else
                     f"diff --git a/src/mod.py b/src/mod.py\n"
                     f"--- a/src/mod.py\n+++ b/src/mod.py\n"
                     f"@@ -1,1 +1,1 @@\n-old  # {iid}\n+new  # {iid}\n")

            d = pt_root / iid; d.mkdir(parents=True, exist_ok=True)
            (d / "patch.diff").write_text(patch)
            (d / "prompt.md").write_text(f"# Task {iid}\nFix bug in {repo}.\n")
            (d / "qwen_trace.json").write_text(json.dumps(
                [{"type": "result", "subtype": "success", "num_turns": turns}]))
            (d / "runner_metadata.json").write_text(json.dumps({
                "instance_id": iid, "repo": repo, "base_commit": base_commit,
                "image": image, "agent": "qwen_code", "eval_mode": "skip",
                "runtime": "container", "container": f"swe_ep_s{k}_{iid.replace('__','_')}",
                "started_at": _iso(t0),
                "qwen": {"elapsed_s": float(_ELAPSED), "exit_code": 0,
                         "num_turns": turns, "tool_calls": turns,
                         "tool_by_name": {"read_file": {"count": turns}},
                         "usage": {"input_tokens": 50000 + turns * 1000,
                                   "output_tokens": turns * 200,
                                   "total_tokens": 50000 + turns * 1200}},
            }, indent=1))

            # proxy dumps in this episode's disjoint window (utime into the past)
            msgs_full = _messages(iid, repo, turns)
            for t in range(turns):
                counter += 1
                payload = {"model": MODEL, "messages": msgs_full[: 2 + 2 * (t + 1)],
                           "tools": _tools(), **ENVELOPE}
                fp = dumps / f"chat_{counter:04d}.json"
                fp.write_text(json.dumps(payload, indent=2))
                mt = t0 + 5 + t * 3
                os.utime(fp, (mt, mt))
                usage_rows.append({"idx": counter, "ts": round(mt, 3),
                                   "usage": {"total_tokens": 51000 + t * 1200},
                                   "finish_reason": "tool_calls" if t < turns - 1 else "stop"})

            pred_lines.append(json.dumps({"instance_id": iid,
                                          "model_name_or_path": MODEL,
                                          "model_patch": patch}))
        preds.write_text("\n".join(pred_lines) + ("\n" if pred_lines else ""))
        (dumps / "usage.jsonl").write_text(
            "\n".join(json.dumps(r) for r in usage_rows) + ("\n" if usage_rows else ""))
    print(json.dumps({"mock_gen": True, "n": len(ids), "C": C, "gen_root": str(gen_root)}))


def cmd_score(batchdir: Path, gen_root: Path) -> None:
    subset = json.loads((batchdir / "subset.json").read_text())
    ids = subset["instance_ids"]
    run_id = batchdir.name
    submitted = [i for i in ids if _produces_patch(i)]
    resolved = [i for i in ids if _resolves(i)]
    empty = [i for i in ids if not _produces_patch(i)]
    report = {
        "run_id": run_id,
        "total_instances": len(ids),
        "submitted_instances": len(submitted),
        "completed_instances": len(submitted),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(submitted) - len(resolved),
        "empty_patch_instances": len(empty),
        "error_instances": 0,
        "resolved_ids": resolved,
        "unresolved_ids": [i for i in submitted if i not in set(resolved)],
        "empty_patch_ids": empty,
        "error_ids": [],
        "schema_version": 2,
        "MOCK": True,
    }
    score = batchdir / "score"; score.mkdir(parents=True, exist_ok=True)
    out = score / f"{MODEL}.{run_id}.json"
    out.write_text(json.dumps(report, indent=1))
    (score / "timing.json").write_text(json.dumps({"score_wall_s": 0, "report": str(out), "MOCK": True}))
    print(json.dumps({"mock_score": True, "resolved": len(resolved),
                      "submitted": len(submitted), "empty": len(empty), "report": str(out)}))


def main() -> int:
    sub = sys.argv[1]
    if sub == "gen":
        cmd_gen(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
    elif sub == "score":
        cmd_score(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(f"unknown subcommand {sub}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
