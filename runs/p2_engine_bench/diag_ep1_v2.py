#!/usr/bin/env python3
"""Pinpoint the ep1/t0 stall: log every step ENTRY (is_committing + committed len)
and any single FSM call > SLOW seconds, incrementally, so a kill still shows the
exact culprit. Also wrap the engine model forward proxy timing per step."""
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
TRACE = Path(os.environ.get("DIAG2_TRACE", str(ROOT / "runs/p2_engine_bench/diag_ep1_v2.log")))
SLOW = float(os.environ.get("DIAG2_SLOW", "0.3"))

import parity_audit_flare_engine as H  # noqa
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa
import vllm.v1.sample.hybrid_clean as HC  # noqa

_tf = TRACE.open("w")
def log(msg):
    _tf.write(f"{time.strftime('%H:%M:%S')} {msg}\n"); _tf.flush()

def _slowwrap(name, fn, ctx=None):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        dt = time.perf_counter() - t0
        if dt > SLOW:
            extra = ""
            try:
                if ctx == "committed" and len(a) >= 2:
                    extra = f" committed_len={len(a[1])}"
                if name in ("legal_candidates", "native_tool_candidate_token_ids"):
                    extra += f" n_ret={len(r) if hasattr(r,'__len__') else '?'}"
            except Exception:
                pass
            log(f"SLOWCALL {name} dt={dt:.2f}s{extra}")
        return r
    return wrap

HC.HybridCleanGrammar.legal_candidates = _slowwrap("legal_candidates", HC.HybridCleanGrammar.legal_candidates, "committed")
HC.HybridCleanGrammar._keeps_prefix = _slowwrap("_keeps_prefix", HC.HybridCleanGrammar._keeps_prefix, "committed")
HC.HybridCleanGrammar.text = _slowwrap("grammar.text", HC.HybridCleanGrammar.text)
HC.HybridCleanGrammar.truly_forced_token = _slowwrap("truly_forced_token", HC.HybridCleanGrammar.truly_forced_token, "committed")
HC.HybridCleanGrammar.legal_top_token = _slowwrap("legal_top_token", HC.HybridCleanGrammar.legal_top_token, "committed")
HC.HybridCleanGrammar.can_stop = _slowwrap("can_stop", HC.HybridCleanGrammar.can_stop)
HC.HybridCleanGrammar.inside_value = _slowwrap("inside_value", HC.HybridCleanGrammar.inside_value)
HC.HybridCleanGrammar.completable = _slowwrap("completable", HC.HybridCleanGrammar.completable)
HC.HybridCleanGrammar.active = _slowwrap("active", HC.HybridCleanGrammar.active)
HC.native_tool_candidate_token_ids = _slowwrap("native_tool_candidate_token_ids", HC.native_tool_candidate_token_ids)
HC.HybridCleanDecodePolicy.decode_model_token = _slowwrap("decode_model_token", HC.HybridCleanDecodePolicy.decode_model_token, "committed")
HC.HybridCleanDecodePolicy.bulk_commit_forced = _slowwrap("bulk_commit_forced", HC.HybridCleanDecodePolicy.bulk_commit_forced, "committed")

_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step
_N = {"i": 0}
def _pstep(self, shifted, block_logits, decode_slots, decode_idx,
           decode_indices_np, decode_slots_np, valid_len_np,
           is_committing, num_reqs, input_batch):
    slot0 = int(decode_slots_np[0]); dec = self._hc_decoders.get(slot0)
    clen = len(dec.committed) if dec is not None else -1
    committing = bool(is_committing[0].item())
    vlen = int(valid_len_np[0])
    log(f"STEP_ENTER i={_N['i']} committed={clen} committing={committing} vlen={vlen}")
    _N["i"] += 1
    t0 = time.perf_counter()
    r = _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                   decode_indices_np, decode_slots_np, valid_len_np,
                   is_committing, num_reqs, input_batch)
    dt = time.perf_counter() - t0
    if dt > SLOW:
        log(f"STEP_SLOW i={_N['i']-1} committed={clen} committing={committing} dt={dt:.2f}s")
    return r
QF.Qwen3_5FlareSampler._hybrid_clean_step = _pstep


def main():
    records = json.loads(REF.read_text())
    gt = int(os.environ.get("DIAG_GT", "4"))
    rec = next(r for r in records if r["global_turn"] == gt)
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(int(rec["mask_id"]))
    log(f"BOOT gt={gt} ep{rec['episode']}/t{rec['turn']} n_ref={rec['n_ref']} plen={rec['prompt_len']} SLOW={SLOW}")
    adapter = H.build_engine_adapter("vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
                                     canvas_length=int(rec["block_size"]), decode_mode="hybrid_clean", seed=20260701)
    engine = adapter._build_engine()
    log("BOOTED")
    from vllm import SamplingParams
    tools = [{"type": "function", "function": {"name": n, "parameters": p}} for n, p in (rec["schemas"] or {}).items()]
    sp = SamplingParams(max_tokens=rec["n_ref"] + 16, temperature=0.0, top_p=1.0, seed=20260701,
                        stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
                        extra_args={"decode_policy": "hybrid_clean", "tools": tools, "grammar_topk": int(rec["grammar_topk"])})
    log("GEN_START")
    req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
    ids = [int(x) for x in req.outputs[0].token_ids]
    log(f"GEN_DONE n_gen={len(ids)} finish={getattr(req.outputs[0],'finish_reason',None)}")


if __name__ == "__main__":
    main()
