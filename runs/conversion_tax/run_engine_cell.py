#!/usr/bin/env python3
"""Conversion-tax ENGINE cell runner (hybrid_clean diffusion), any capability class.

Byte-identical decode machinery to runs/l1_baseline_b1/run_engine_hardened.py
(pin 0b44dcc, VLLM_FLARE_BIDIR_PROBE=1 + CUDAGRAPH=1, free-text tools=[], grammar
inert), plus the per-turn SIGALRM watchdog + exit-on-first-hang so the reboot loop
sweeps everything on fresh engines. Class-agnostic: captures raw gen_text +
audit counters + verify; scoring is deferred to aggregate.py. Resumable.

Env: CEN_REF, CEN_OUT, CEN_MAXTOK, CEN_START, CEN_END, TURN_TIMEOUT, BENCH_SEED.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import torch  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = Path(os.environ["CEN_REF"])
OUT = Path(os.environ["CEN_OUT"])
TURN_TIMEOUT = int(os.environ.get("TURN_TIMEOUT", "40"))

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

_LAST = {"stats": None, "denoise_forwards": 0}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    if not bool(is_committing[0].item()):
        _LAST["denoise_forwards"] += 1
    ret = _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                     decode_indices_np, decode_slots_np, valid_len_np,
                     is_committing, num_reqs, input_batch)
    dec = self._hc_decoders.get(int(decode_slots_np[0]))
    if dec is not None:
        s = dec.stats
        _LAST["stats"] = {"forwards": int(s.forwards),
                          "fsm_committed_tokens": int(s.fsm_committed_tokens),
                          "value_tokens": int(s.value_tokens),
                          "structural_model_tokens": int(s.structural_model_tokens),
                          "value_projection_events": int(s.value_projection_events),
                          "model_chosen_tokens": int(s.model_chosen_tokens),
                          "generated_tokens": int(s.generated_tokens)}
    return ret


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step


class TurnTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise TurnTimeout()


signal.signal(signal.SIGALRM, _alarm)


def verify(stats):
    if not stats:
        return {"ok": False}
    chk = {"value_projection_events_is_0": stats["value_projection_events"] == 0,
           "forwards_eq_model_chosen": stats["forwards"] == stats["model_chosen_tokens"],
           "generated_eq_fsm_plus_model": stats["generated_tokens"] == (
               stats["fsm_committed_tokens"] + stats["model_chosen_tokens"])}
    chk["ok"] = all(chk.values())
    return chk


def main():
    recs = json.loads(REF.read_text())
    start = int(os.environ.get("CEN_START", "0"))
    end = int(os.environ.get("CEN_END", "10000"))
    maxtok = int(os.environ.get("CEN_MAXTOK", "384"))
    seed = int(os.environ.get("BENCH_SEED", "20260701"))
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(int(recs[0]["mask_id"]))

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                try:
                    done.add(int(json.loads(line)["idx"]))
                except Exception:
                    pass
    todo = [r for r in recs if start <= r["idx"] <= end and r["idx"] not in done]
    print(f"[eng] todo={len(todo)} done={len(done)} timeout={TURN_TIMEOUT}s maxtok={maxtok}", flush=True)
    if not todo:
        print("[eng] nothing to do", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    t_boot = time.time()
    adapter = H.build_engine_adapter("vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
                                     canvas_length=int(recs[0]["block_size"]),
                                     decode_mode="hybrid_clean", seed=seed)
    engine = adapter._build_engine()
    print(f"[eng] booted boot_s={round(time.time()-t_boot,1)}", flush=True)

    from vllm import SamplingParams

    fh = OUT.open("a")
    for rec in todo:
        sp = SamplingParams(max_tokens=maxtok, temperature=0.0, top_p=1.0, seed=seed,
                            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
                            extra_args={"decode_policy": "hybrid_clean", "tools": [],
                                        "grammar_topk": int(rec["grammar_topk"])})
        _LAST["stats"] = None
        _LAST["denoise_forwards"] = 0
        t0 = time.time()
        hung = False
        err = None
        ids = []
        fin = None
        signal.alarm(TURN_TIMEOUT)
        try:
            req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
            o = req.outputs[0]
            ids = [int(x) for x in o.token_ids]
            fin = getattr(o, "finish_reason", None)
        except TurnTimeout:
            hung = True
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        finally:
            signal.alarm(0)
        wall = round(time.time() - t0, 3)
        if hung or err:
            turn = {"idx": rec["idx"], "prompt_len": rec["prompt_len"], "hang": hung,
                    "error": err, "wall_s": wall, "denoise_forwards": _LAST["denoise_forwards"],
                    "partial_counters": _LAST["stats"], "gen_text": ""}
            fh.write(json.dumps(turn) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(f"[eng] idx{rec['idx']:2d} {'HANG' if hung else 'ERR'} wall={wall} {err or ''}", flush=True)
            print("[eng] HANG_EXIT (fresh reboot needed)", flush=True)
            fh.close()
            return
        text = tok.decode(ids, skip_special_tokens=True)
        fwd = _LAST["denoise_forwards"]
        stats = _LAST["stats"]
        turn = {"idx": rec["idx"], "prompt_len": rec["prompt_len"], "n_gen": len(ids),
                "maxtok": maxtok, "finish_reason": fin, "wall_s": wall, "denoise_forwards": fwd,
                "per_forward_ms": round(1000.0 * wall / fwd, 2) if fwd else None,
                "gen_text": text, "counters": stats, "verify": verify(stats)}
        fh.write(json.dumps(turn) + "\n"); fh.flush(); os.fsync(fh.fileno())
        c = stats or {}
        print(f"[eng] idx{rec['idx']:2d} n={len(ids)} fin={fin} fwd={fwd} "
              f"struct={c.get('structural_model_tokens')} proj={c.get('value_projection_events')} "
              f"wall={wall}", flush=True)
    fh.close()
    print("[eng] DONE range", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[eng] FATAL:", repr(e), flush=True)
        traceback.print_exc()
        raise
