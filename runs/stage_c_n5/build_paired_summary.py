#!/usr/bin/env python3
"""Consolidate the N=5 paired run into paired_summary.json (data artifact)."""
import json, subprocess
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion/runs/stage_c_n5")
IIDS = ["django__django-11119","django__django-12754","django__django-13741",
        "pytest-dev__pytest-8399","sympy__sympy-13757"]

EXIT_MEANING = {0:"clean_success", 1:"loop_detector_halt(consecutive_identical_tool_calls)",
                53:"turn_limit(FatalTurnLimitedError, max_session_turns=50)",
                55:"wall_budget(qwen --max-wall-time)", -1:"harness_wall_timeout"}

def per_instance(arm, iid):
    p = ROOT/arm/"verified"/"per_task"/iid/"runner_metadata.json"
    m = json.loads(p.read_text())
    q = m.get("qwen") or {}; u = q.get("usage") or {}; er = m.get("eval_report") or {}
    turns = q.get("num_turns")
    wall = q.get("elapsed_s")
    tools = q.get("tool_by_name") or {}
    def tc(name):
        t=tools.get(name);
        return (t.get("count") if isinstance(t,dict) else t) if t else 0
    return {
      "made_edit": (m.get("patch_bytes",0) or 0) > 0,
      "patch_bytes": m.get("patch_bytes"),
      "mock_verdict": er.get("verdict"), "mock_failure_mode": er.get("failure_mode"),
      "turns": turns, "wall_s": wall,
      "per_turn_s": round(wall/turns,2) if (wall and turns) else None,
      "cli_exit": q.get("exit_code"), "cli_exit_meaning": EXIT_MEANING.get(q.get("exit_code"),"?"),
      "subtype": q.get("subtype"),
      "edit_calls": tc("edit"), "read_calls": tc("read_file"), "shell_calls": tc("run_shell_command"),
      "gen_tokens_out": u.get("output_tokens"),
      "usage_total_tokens_cumulative": u.get("total_tokens"),
    }

out = {"note": "N=5 paired Stage-C SWE run. VERDICTS ARE MOCK (gold-line-subset stand-in), "
       "NOT docker resolve@1 (docker+swebench absent on 5090; alienware x86 offload unreachable "
       "this session). predictions.jsonl emitted per arm for later offload scoring.",
       "context_ceiling": "max_model_len=32768, proxy max_tokens=2048 -> usable input ~30720; "
       "once conversation exceeds it every turn 400s. Hit BOTH arms (AR 3x400, diffusion 5x400).",
       "instances": {}}
for iid in IIDS:
    out["instances"][iid] = {"ar": per_instance("ar",iid), "diffusion": per_instance("diffusion",iid)}

# engine counters (diffusion)
cc = subprocess.run(["/home/mark/qwen_diffusion/.venv/bin/python",
                     "/home/mark/qwen_diffusion/runs/stage_a_smoke/read_counters.py",
                     str(ROOT/"logs"/"diffusion_server.log")], capture_output=True, text=True)
try: out["diffusion_engine_counters"] = json.loads(cc.stdout)
except Exception: out["diffusion_engine_counters"] = {"raw": cc.stdout[-500:]}

out["arm_rollup"] = {
  "ar": {"made_edit": sum(out["instances"][i]["ar"]["made_edit"] for i in IIDS),
         "mock_resolved": sum(out["instances"][i]["ar"]["mock_verdict"]=="resolved" for i in IIDS),
         "clean_exit0": sum(out["instances"][i]["ar"]["cli_exit"]==0 for i in IIDS)},
  "diffusion": {"made_edit": sum(out["instances"][i]["diffusion"]["made_edit"] for i in IIDS),
         "mock_resolved": sum(out["instances"][i]["diffusion"]["mock_verdict"]=="resolved" for i in IIDS),
         "clean_exit0": sum(out["instances"][i]["diffusion"]["cli_exit"]==0 for i in IIDS)},
}
(ROOT/"paired_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["arm_rollup"], indent=2))
print("WROTE", ROOT/"paired_summary.json")
