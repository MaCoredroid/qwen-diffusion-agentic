#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion/runs/stage_c_n5")
IIDS = ["django__django-11119","django__django-12754","django__django-13741",
        "pytest-dev__pytest-8399","sympy__sympy-13757"]

def row(arm, iid):
    p = ROOT/arm/"verified"/"per_task"/iid/"runner_metadata.json"
    if not p.is_file(): return {"iid":iid,"arm":arm,"status":"MISSING"}
    m = json.loads(p.read_text())
    q = m.get("qwen") or {}
    u = q.get("usage") or {}
    er = m.get("eval_report") or {}
    # token fields vary by qwen version; try common keys
    def tok(*ks):
        for k in ks:
            if isinstance(u.get(k),(int,float)): return u[k]
        return None
    return {
      "iid":iid,"arm":arm,
      "verdict":er.get("verdict"),"failure_mode":er.get("failure_mode"),
      "patch_bytes":m.get("patch_bytes"),
      "wall_s":q.get("elapsed_s"),"api_ms":q.get("duration_api_ms"),
      "exit":q.get("exit_code"),"timed_out":q.get("timed_out"),
      "subtype":q.get("subtype"),"turns":q.get("num_turns"),
      "tool_calls":q.get("tool_calls"),"tools":q.get("tool_by_name"),
      "in_tok":tok("input_tokens","prompt_tokens","inputTokens"),
      "out_tok":tok("output_tokens","completion_tokens","outputTokens"),
      "tot_tok":tok("total_tokens","totalTokens"),
      "usage_keys": list(u.keys()) if u else None,
      "empty_retry": m.get("empty_patch_retry"),
      "result_tail": (q.get("result_tail") or "")[-240:],
    }

print("="*100)
for iid in IIDS:
    for arm in ("ar","diffusion"):
        r = row(arm,iid)
        print(json.dumps(r))
