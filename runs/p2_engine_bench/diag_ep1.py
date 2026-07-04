#!/usr/bin/env python3
"""Diagnose WHERE the per-turn wall goes on a long turn (default ep1/t0, 110 tok):
grammar-FSM host cost (OPT-5) vs model forwards (OPT-3/4). Times the grammar hot
path (legal_candidates / _keeps_prefix / native_tool_candidate_token_ids / text /
truly_forced_token) and reports forwards + per-step committed-length curve.
Env: DIAG_GT (global_turn index, default 4 = ep1/t0), DIAG_CAP (extra tokens, default 16).
"""
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
OUT = Path(os.environ.get("DIAG_OUT", str(ROOT / "runs/p2_engine_bench/diag_ep1.json")))

import parity_audit_flare_engine as H  # noqa
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa
import vllm.v1.sample.hybrid_clean as HC  # noqa

T = {}  # name -> [cum_seconds, calls]
def _mk(name, fn):
    T[name] = [0.0, 0]
    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name][0] += time.perf_counter() - t0
            T[name][1] += 1
    return wrap

# wrap grammar hot path
HC.HybridCleanGrammar.legal_candidates = _mk("legal_candidates", HC.HybridCleanGrammar.legal_candidates)
HC.HybridCleanGrammar._keeps_prefix = _mk("_keeps_prefix", HC.HybridCleanGrammar._keeps_prefix)
HC.HybridCleanGrammar.text = _mk("grammar.text", HC.HybridCleanGrammar.text)
HC.HybridCleanGrammar.truly_forced_token = _mk("truly_forced_token", HC.HybridCleanGrammar.truly_forced_token)
HC.native_tool_candidate_token_ids = _mk("native_tool_candidate_token_ids", HC.native_tool_candidate_token_ids)
HC.HybridCleanDecodePolicy.decode_model_token = _mk("decode_model_token", HC.HybridCleanDecodePolicy.decode_model_token)
HC.HybridCleanDecodePolicy.bulk_commit_forced = _mk("bulk_commit_forced", HC.HybridCleanDecodePolicy.bulk_commit_forced)

TRACE = Path(os.environ.get("DIAG_TRACE", str(ROOT / "runs/p2_engine_bench/diag_ep1_steps.jsonl")))
_tf = TRACE.open("w")
STEP = {"n": 0, "last": None, "curve": []}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step
def _grammar_cum():
    return sum(T[k][0] for k in ("legal_candidates", "_keeps_prefix", "grammar.text",
               "truly_forced_token", "native_tool_candidate_token_ids"))
def _pstep(self, shifted, block_logits, decode_slots, decode_idx,
           decode_indices_np, decode_slots_np, valid_len_np,
           is_committing, num_reqs, input_batch):
    now = time.perf_counter()
    if STEP["last"] is not None:
        slot0 = int(decode_slots_np[0]); dec = self._hc_decoders.get(slot0)
        clen = len(dec.committed) if dec is not None else -1
        dt = round(now - STEP["last"], 4)
        row = [STEP["n"], clen, dt, bool(is_committing[0].item()), round(_grammar_cum(), 3)]
        STEP["curve"].append(row)
        _tf.write(json.dumps(row) + "\n"); _tf.flush()
    STEP["last"] = now
    STEP["n"] += 1
    return _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                      decode_indices_np, decode_slots_np, valid_len_np,
                      is_committing, num_reqs, input_batch)
QF.Qwen3_5FlareSampler._hybrid_clean_step = _pstep


def main():
    records = json.loads(REF.read_text())
    gt = int(os.environ.get("DIAG_GT", "4"))
    cap_extra = int(os.environ.get("DIAG_CAP", "16"))
    rec = next(r for r in records if r["global_turn"] == gt)
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(int(rec["mask_id"]))
    print(f"[diag] gt={gt} ep{rec['episode']}/t{rec['turn']} n_ref={rec['n_ref']} plen={rec['prompt_len']}", flush=True)
    adapter = H.build_engine_adapter("vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
                                     canvas_length=int(rec["block_size"]), decode_mode="hybrid_clean", seed=20260701)
    engine = adapter._build_engine()
    from vllm import SamplingParams
    tools = [{"type": "function", "function": {"name": n, "parameters": p}} for n, p in (rec["schemas"] or {}).items()]
    sp = SamplingParams(max_tokens=rec["n_ref"] + cap_extra, temperature=0.0, top_p=1.0, seed=20260701,
                        stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
                        extra_args={"decode_policy": "hybrid_clean", "tools": tools, "grammar_topk": int(rec["grammar_topk"])})
    t0 = time.time()
    req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
    wall = time.time() - t0
    ids = [int(x) for x in req.outputs[0].token_ids]
    first_div = next((i for i in range(min(len(ids), rec["n_ref"])) if ids[i] != rec["ref_new_ids"][i]), None)
    timers = {k: {"cum_s": round(v[0], 3), "calls": v[1]} for k, v in sorted(T.items(), key=lambda x: -x[1][1][0])}
    out = {
        "gt": gt, "episode": rec["episode"], "turn": rec["turn"], "n_ref": rec["n_ref"],
        "n_gen": len(ids), "wall_s": round(wall, 2), "finish": getattr(req.outputs[0], "finish_reason", None),
        "first_divergence": first_div, "byte_parity": first_div is None and len(ids) == rec["n_ref"],
        "engine_steps": STEP["n"], "timers": timers,
        "curve_head": STEP["curve"][:8], "curve_tail": STEP["curve"][-12:],
        "grammar_total_s": round(sum(T[k][0] for k in ("legal_candidates", "_keeps_prefix", "grammar.text",
                                    "truly_forced_token", "native_tool_candidate_token_ids")), 2),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("[diag] RESULT " + json.dumps({k: v for k, v in out.items() if k not in ("curve_head", "curve_tail", "timers")}), flush=True)
    print("[diag] TIMERS " + json.dumps(timers), flush=True)
    print("[diag] curve_tail " + json.dumps(STEP["curve"][-12:]), flush=True)


if __name__ == "__main__":
    main()
