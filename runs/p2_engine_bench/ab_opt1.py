#!/usr/bin/env python3
"""A/B: dump ENGINE greedy token ids for a set of global_turns, one boot, in order.
Run once on OPT-1 (58cfe2c) and once on pre-OPT-1 (6b81154 hybrid_clean.py); if the
engine ids are token-identical across the two, OPT-1 is behavior-preserving (a pure
speedup) and any divergence-from-HF is a pre-OPT-1 forward property, not OPT-1.
Env: AB_TURNS (comma global_turns), AB_OUT (json), AB_CAP (hard cap, default 94)."""
import json, os, sys, time
from pathlib import Path
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import torch  # noqa
ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = ROOT / "runs/p2_engine_bench/matched20_ref.json"
import parity_audit_flare_engine as H  # noqa


def main():
    records = {r["global_turn"]: r for r in json.loads(REF.read_text())}
    turns = [int(x) for x in os.environ["AB_TURNS"].replace(",", " ").split()]
    cap = int(os.environ.get("AB_CAP", "94"))
    out = Path(os.environ["AB_OUT"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(int(records[turns[0]]["mask_id"]))
    adapter = H.build_engine_adapter("vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
                                     canvas_length=32, decode_mode="hybrid_clean", seed=20260701)
    engine = adapter._build_engine()
    from vllm import SamplingParams
    res = {}
    for gt in turns:
        rec = records[gt]
        tools = [{"type": "function", "function": {"name": n, "parameters": p}} for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(max_tokens=min(rec["n_ref"] + 16, cap), temperature=0.0, top_p=1.0, seed=20260701,
                            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
                            extra_args={"decode_policy": "hybrid_clean", "tools": tools, "grammar_topk": int(rec["grammar_topk"])})
        t0 = time.time()
        req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        ids = [int(x) for x in req.outputs[0].token_ids]
        fd_hf = next((i for i in range(min(len(ids), rec["n_ref"])) if ids[i] != rec["ref_new_ids"][i]), None)
        res[str(gt)] = {"ids": ids, "n": len(ids), "wall": round(time.time() - t0, 2), "first_div_vs_hf": fd_hf}
        print(f"[ab] gt{gt} n={len(ids)} first_div_vs_hf={fd_hf} wall={res[str(gt)]['wall']}", flush=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print("[ab] DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
