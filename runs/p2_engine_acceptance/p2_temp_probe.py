#!/usr/bin/env python3
"""Disambiguate the temp>0 seed-diversity + projection behavior (one boot).

For ep0/t0: sweep temperature; at each T run seedA and seedB and report whether
the two seeds diverge, plus value_projection_events. Also run greedy ep0 3x
back-to-back to characterize the 42-vs-43 length wobble (batch/cache float
non-associativity vs a real nondeterminism).
"""
import json
import os
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
REF = ROOT / "runs/p2_engine_acceptance/gap5a_ref.json"
OUT = Path(os.environ.get("TP_OUT", str(ROOT / "runs/p2_engine_acceptance/p2_temp_probe.json")))

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402

_LAST = {"stats": None}
_real = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched(self, shifted, block_logits, decode_slots, decode_idx,
             decode_indices_np, decode_slots_np, valid_len_np,
             is_committing, num_reqs, input_batch):
    ret = _real(self, shifted, block_logits, decode_slots, decode_idx,
                decode_indices_np, decode_slots_np, valid_len_np,
                is_committing, num_reqs, input_batch)
    slot0 = int(decode_slots_np[0])
    dec = self._hc_decoders.get(slot0)
    if dec is not None:
        s = dec.stats
        _LAST["stats"] = {"proj": int(s.value_projection_events),
                          "forced": int(s.fsm_committed_tokens),
                          "value": int(s.value_tokens),
                          "forwards": int(s.forwards),
                          "model_chosen": int(s.model_chosen_tokens),
                          "generated": int(s.generated_tokens)}
    return ret


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched
RESULT = {"sweep": [], "greedy_repeat": []}


def main():
    records = json.loads(REF.read_text())
    rec = records[0]
    mask_id = int(rec["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    bs = int(rec["block_size"])
    adapter = H.build_engine_adapter("vllm", vllm_workspace=VLLM_WS,
                                     model_path=str(MODEL), canvas_length=bs,
                                     decode_mode="hybrid_clean", seed=20260701)
    engine = adapter._build_engine()
    print("[tp] booted", flush=True)
    from vllm import SamplingParams
    tools = [{"type": "function", "function": {"name": n, "parameters": p}}
             for n, p in (rec["schemas"] or {}).items()]
    maxtok = len(rec["ref_new_ids"]) + 8

    def gen(T, seed):
        sp = SamplingParams(max_tokens=maxtok, temperature=T, top_p=1.0, seed=seed,
                            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
                            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                                        "grammar_topk": int(rec["grammar_topk"])})
        _LAST["stats"] = None
        o = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp).outputs[0]
        return [int(x) for x in o.token_ids], getattr(o, "finish_reason", None), _LAST["stats"]

    for T in [0.0, 0.7, 1.0, 1.5, 2.0]:
        idsA, fA, stA = gen(T, 20260701)
        idsB, fB, stB = gen(T, 987654321)
        row = {"T": T, "nA": len(idsA), "nB": len(idsB),
               "identical": idsA == idsB,
               "finishA": fA, "finishB": fB,
               "projA": (stA or {}).get("proj"), "projB": (stB or {}).get("proj"),
               "first_diff": next((i for i in range(min(len(idsA), len(idsB)))
                                   if idsA[i] != idsB[i]), None),
               "headA": idsA[:14], "headB": idsB[:14]}
        RESULT["sweep"].append(row)
        print(f"[tp] T={T} identical={row['identical']} nA={row['nA']} nB={row['nB']} "
              f"first_diff={row['first_diff']} projA={row['projA']} projB={row['projB']}", flush=True)
        OUT.write_text(json.dumps(RESULT, indent=2) + "\n")

    # greedy repeat (same session) 3x
    for k in range(3):
        ids, f, st = gen(0.0, 20260701)
        RESULT["greedy_repeat"].append({"k": k, "n": len(ids), "finish": f,
                                        "proj": (st or {}).get("proj"), "tail": ids[-4:]})
        print(f"[tp] greedy_repeat k={k} n={len(ids)} finish={f} proj={(st or {}).get('proj')}", flush=True)
        OUT.write_text(json.dumps(RESULT, indent=2) + "\n")
    print("[tp] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        RESULT["error"] = repr(e)
        OUT.write_text(json.dumps(RESULT, indent=2) + "\n")
        traceback.print_exc()
        raise
