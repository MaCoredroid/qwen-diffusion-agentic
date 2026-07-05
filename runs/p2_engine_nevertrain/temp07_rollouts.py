#!/usr/bin/env python3
"""3 seeded temp=0.7 rollouts on NEVER-TRAIN (OOD) prompts -- the RL-contract
sanity on out-of-distribution BFCL/API-Bank inputs. Same FINAL post-fix engine
(v3b config: pin 95d8b47, VLLM_FLARE_BIDIR_PROBE=1 + PIECEWISE cudagraph).

For each selected never-train prompt we do TWO seeded passes (same seed) and store
the FULL engine token ids, so we can certify:
  - bounded  : finish_reason==stop and n_gen <= maxtok (no runaway)
  - valid    : score_tool_calls -> valid_tool_call
  - proj0    : value_projection_events == 0
  - verify   : hybrid-clean invariants ok
  - byte-repro: pass-a token ids == pass-b token ids (seeded determinism at T=0.7)

One heavy process; RAM cage.
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
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
OUT = ROOT / "runs/p2_engine_nevertrain/nevertrain_temp07.jsonl"

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402
from eval_flare_northstar_matched import decode_text, trim_scored_assistant  # noqa: E402

_LAST = {"stats": None, "denoise_forwards": 0}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    committing = bool(is_committing[0].item())
    if not committing:
        _LAST["denoise_forwards"] += 1
    ret = _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                     decode_indices_np, decode_slots_np, valid_len_np,
                     is_committing, num_reqs, input_batch)
    slot0 = int(decode_slots_np[0])
    dec = self._hc_decoders.get(slot0)
    if dec is not None:
        s = dec.stats
        _LAST["stats"] = {
            "forwards": int(s.forwards),
            "fsm_committed_tokens": int(s.fsm_committed_tokens),
            "value_tokens": int(s.value_tokens),
            "structural_model_tokens": int(s.structural_model_tokens),
            "value_projection_events": int(s.value_projection_events),
            "model_chosen_tokens": int(s.model_chosen_tokens),
            "generated_tokens": int(s.generated_tokens),
        }
    return ret


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step

SELECT = [int(x) for x in os.environ.get("SELECT", "0 147 159 172").split()]
TEMP = float(os.environ.get("BENCH_TEMP", "0.7"))
SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))


def verify(stats):
    if not stats:
        return {"ok": False}
    chk = {
        "value_projection_events_is_0": stats["value_projection_events"] == 0,
        "forwards_eq_model_chosen": stats["forwards"] == stats["model_chosen_tokens"],
        "generated_eq_fsm_plus_model": stats["generated_tokens"] == (
            stats["fsm_committed_tokens"] + stats["model_chosen_tokens"]),
        "forced_gt_0": stats["fsm_committed_tokens"] > 0,
    }
    chk["ok"] = all(chk.values())
    return chk


def main():
    records = {r["global_turn"]: r for r in json.loads(REF.read_text())}
    mask_id = int(records[SELECT[0]]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=int(records[SELECT[0]]["block_size"]), decode_mode="hybrid_clean", seed=SEED,
    )
    engine = adapter._build_engine()
    print("[t07] booted", flush=True)
    from vllm import SamplingParams

    def gen(rec, maxtok, seed):
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(
            max_tokens=maxtok, temperature=TEMP, top_p=1.0, seed=seed,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])},
        )
        _LAST["stats"] = None
        _LAST["denoise_forwards"] = 0
        t0 = time.time()
        req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        wall = time.time() - t0
        o = req.outputs[0]
        ids = [int(x) for x in o.token_ids]
        return ids, getattr(o, "finish_reason", None), round(wall, 3), _LAST["denoise_forwards"], _LAST["stats"]

    fh = OUT.open("w")
    for gt in SELECT:
        rec = records[gt]
        maxtok = rec["n_ref"] + MARGIN
        ids_a, fin_a, wall_a, fwd_a, st_a = gen(rec, maxtok, SEED)
        ids_b, fin_b, wall_b, fwd_b, st_b = gen(rec, maxtok, SEED)
        txt_a = trim_scored_assistant(decode_text(tok, torch.tensor(ids_a, dtype=torch.long)))
        sc_a = score_tool_calls(txt_a, rec["tools"], rec["gold_block"])
        row = {
            "global_turn": gt, "episode": rec["episode"], "turn": rec["turn"],
            "source_family": rec["source_family"], "temp": TEMP, "seed": SEED,
            "maxtok": maxtok, "n_ref": rec["n_ref"],
            "n_gen_a": len(ids_a), "n_gen_b": len(ids_b),
            "finish_a": fin_a, "finish_b": fin_b,
            "bounded": (fin_a == "stop" and len(ids_a) <= maxtok),
            "byte_reproducible_a_eq_b": ids_a == ids_b,
            "valid_tool_call": bool(sc_a.get("valid_tool_call")),
            "exact_arguments": bool(sc_a.get("exact_arguments")),
            "wall_a": wall_a, "wall_b": wall_b,
            "denoise_forwards_a": fwd_a, "denoise_forwards_b": fwd_b,
            "counters_a": st_a, "verify_a": verify(st_a),
            "proj0": (st_a or {}).get("value_projection_events") == 0,
        }
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"[t07] gt{gt} [{rec['source_family']}] bounded={row['bounded']} "
              f"repro={row['byte_reproducible_a_eq_b']} valid={row['valid_tool_call']} "
              f"proj0={row['proj0']} verify={row['verify_a']['ok']} n={len(ids_a)}/{maxtok} "
              f"fin={fin_a} fwd={fwd_a} wall={wall_a}", flush=True)
    fh.close()
    print("[t07] DONE", flush=True)


if __name__ == "__main__":
    main()
