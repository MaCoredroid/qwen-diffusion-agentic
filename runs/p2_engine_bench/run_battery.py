#!/usr/bin/env python3
"""P2 ENGINE BENCH — the FULL matched-20 battery (20 episodes / 63 turns) on the
OPTIMIZED engine (vLLM pin 58cfe2c: GAP-5A windowed-probe fix + OPT-1 GPU-native
sampling), greedy, hybrid_clean decode, real export.

Reads runs/p2_engine_bench/matched20_ref.json (63 records; prompts reconstructed +
byte-verified vs the HF matched-20 eval, prompt_sha256/prompt_tokens all match).
For each turn the engine generates greedily on the exact HF prompt and we record:
  * byte-parity vs the HF row's generated_token_ids (first_divergence, full match)
  * TRUE denoise forwards, wall_s, hybrid_clean audit counters + verify_invariants
  * engine exact_arguments (independently scored via score_tool_calls)

Writes each turn incrementally to $BENCH_OUT (JSONL append) so a wall-clock timeout
preserves finished turns. RESUMES by skipping global_turns already present.
Env: EP_START/EP_END (inclusive episode range, default 0..19),
     BENCH_MARGIN (max_tokens = n_ref + margin, default 16),
     BENCH_TEMP (default 0.0), BENCH_SEED (default 20260701),
     BENCH_OUT (default runs/p2_engine_bench/matched20_turns.jsonl).
One heavy process; run inside the RAM cage.
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
REF = ROOT / "runs/p2_engine_bench/matched20_ref.json"
OUT = Path(os.environ.get("BENCH_OUT", str(ROOT / "runs/p2_engine_bench/matched20_turns.jsonl")))

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402
from eval_flare_northstar_matched import decode_text, trim_scored_assistant  # noqa: E402

# ---- per-turn capture of the LAST decoder-stats snapshot (before free) ----
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


def main():
    records = json.loads(REF.read_text())
    ep_start = int(os.environ.get("EP_START", "0"))
    ep_end = int(os.environ.get("EP_END", "19"))
    margin = int(os.environ.get("BENCH_MARGIN", "16"))
    hard_cap = int(os.environ.get("BENCH_HARD_CAP", "0"))  # 0 => no hard cap
    temp = float(os.environ.get("BENCH_TEMP", "0.0"))
    seed = int(os.environ.get("BENCH_SEED", "20260701"))
    mask_id = int(records[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(records[0]["block_size"])

    # resume: skip global_turns already recorded
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["global_turn"]))
            except Exception:
                pass
    skip = set(int(x) for x in os.environ.get("BENCH_SKIP", "").replace(",", " ").split() if x.strip())
    only = set(int(x) for x in os.environ.get("BENCH_ONLY", "").replace(",", " ").split() if x.strip())
    todo = [r for r in records if ep_start <= r["episode"] <= ep_end
            and r["global_turn"] not in done and r["global_turn"] not in skip
            and (not only or r["global_turn"] in only)]
    print(f"[bench] range ep{ep_start}..{ep_end} todo={len(todo)} already_done={len(done)} "
          f"margin={margin} temp={temp} seed={seed}", flush=True)
    if not todo:
        print("[bench] nothing to do", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)

    t_boot = time.time()
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=seed,
    )
    engine = adapter._build_engine()
    boot_s = round(time.time() - t_boot, 1)
    runner = engine.llm_engine.model_executor.driver_worker.model_runner
    ms = getattr(runner, "model_state", None) or getattr(runner, "_model_state", None)
    print(f"[bench] booted boot_s={boot_s} decode_mode={getattr(ms,'decode_mode',None)} "
          f"patched={QF.Qwen3_5FlareSampler._hybrid_clean_step is _patched_step}", flush=True)

    from vllm import SamplingParams

    def gen(rec, maxtok, temperature, sd):
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(
            max_tokens=maxtok, temperature=temperature, top_p=1.0, seed=sd,
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

    fh = OUT.open("a")
    for rec in todo:
        n_ref = rec["n_ref"]
        maxtok = n_ref + margin
        if hard_cap > 0:
            maxtok = min(maxtok, hard_cap)
        ids, finish, wall, fwd, stats = gen(rec, maxtok, temp, seed)
        n_cmp = min(len(ids), n_ref)
        first_div = next((i for i in range(n_cmp) if ids[i] != rec["ref_new_ids"][i]), None)
        byte_parity_full = (first_div is None and len(ids) == n_ref)
        # independent engine scoring
        new_ids_t = torch.tensor(ids, dtype=torch.long)
        assistant_text = trim_scored_assistant(decode_text(tok, new_ids_t))
        sc = score_tool_calls(assistant_text, rec["tools"], rec["gold_block"])
        eng_exact = bool(sc.get("exact_arguments"))
        eng_valid = bool(sc.get("valid_tool_call"))
        turn = {
            "global_turn": rec["global_turn"], "episode": rec["episode"], "turn": rec["turn"],
            "episode_id": rec["episode_id"], "prompt_len": rec["prompt_len"],
            "n_ref": n_ref, "n_gen": len(ids), "maxtok": maxtok,
            "finish_reason": finish, "wall_s": wall, "denoise_forwards": fwd,
            "hard_capped": bool(hard_cap > 0 and maxtok == hard_cap and n_ref + margin > hard_cap),
            "first_divergence": first_div,
            "byte_parity_full": byte_parity_full,
            "byte_parity_over_min": first_div is None,
            "eng_exact_arguments": eng_exact, "eng_valid_tool_call": eng_valid,
            "hf_exact_arguments": rec["hf_exact_arguments"],
            "hf_valid_tool_call": rec["hf_valid_tool_call"],
            "hf_denoise_forwards_total": rec["hf_denoise_forwards_total"],
            "hf_turn_wall_seconds": rec["hf_turn_wall_seconds"],
            "exact_matches_hf": (eng_exact == rec["hf_exact_arguments"]),
            "counters": stats, "verify": verify(stats),
        }
        fh.write(json.dumps(turn) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        pflag = "" if byte_parity_full else " <<PARITY_BREAK"
        print(f"[bench] gt{rec['global_turn']:2d} ep{rec['episode']}/t{rec['turn']} "
              f"parity={byte_parity_full} first_div={first_div} n={len(ids)}/{n_ref} "
              f"fin={finish} fwd={fwd} eng_exact={int(eng_exact)}(hf={int(rec['hf_exact_arguments'])}) "
              f"proj={stats['value_projection_events'] if stats else '?'} wall={wall}{pflag}", flush=True)
    fh.close()
    print("[bench] DONE range", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[bench] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
