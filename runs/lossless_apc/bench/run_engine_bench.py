#!/usr/bin/env python3
"""LOSSLESS-APC END-GOAL BENCH — ENGINE arm (diffusion FLARE, batch=1).

Faithful copy of runs/p2_engine_battery_v3b/run_battery_v3b.py (the trusted,
byte-identical parity harness) with ONE addition: a per-turn PREFILL SPLIT timer.

Why the split is accurate: the first `_hybrid_clean_step` of a turn begins with
`bool(is_committing[0].item())`, and `.item()` on a CUDA tensor forces a device
sync of everything launched before it -- i.e. the whole prefill. We add an explicit
`torch.cuda.synchronize()` + timestamp at the first step of each turn, so
  prefill_s = t(first denoise step, post-sync) - t(gen start)
  decode_s  = wall_s - prefill_s
This isolates prompt-processing (prefill; the ONLY thing APC accelerates) from the
block-diffusion denoise loop (decode), for every turn, in both cache modes.

Cache modes (task a/b): APC stays ENABLED in both (the hybrid model rejects
mamba_block_size without prefix caching), so cold is realized by dropping the
cross-turn cache each turn -- the exact per-turn fresh-context protocol that
produced runs/p2_engine_nevertrain/nevertrain_parity_cert_resetapc.jsonl:
  BENCH_RESET_APC=1 -> engine.reset_prefix_cache() before each turn : cache-COLD
  BENCH_RESET_APC=0 -> keep cache across turns                      : cache-ON

NOTE ON "LOSSLESS": the deployed align-APC reuse (cache-ON here) is byte-LOSSY on a
quality-neutral near-tie class (census 233/247 byte-parity; exact_args APC-invariant).
The Route A lossless publish seam is present in the pin (aedf465) but INERT (gate-1
blocker: capture sinks have zero callers). So cache-ON measured here == deployed lossy
reuse; the lossless variant would carry the SAME prefill savings minus a bounded
per-1024-checkpoint refold (one chunk_gated_delta_rule over <=1024 committed tokens).

Env: EP_START/EP_END, BENCH_MARGIN(16), BENCH_TEMP(0.0), BENCH_SEED(20260701),
     BENCH_APC_OFF, BENCH_OUT, BENCH_REF. Requires VLLM_FLARE_BIDIR_PROBE/CUDAGRAPH.
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
REF = Path(os.environ.get("BENCH_REF", str(ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json")))
OUT = Path(os.environ.get("BENCH_OUT", str(ROOT / "runs/lossless_apc/bench/engine_on.jsonl")))

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402
from eval_flare_northstar_matched import decode_text, trim_scored_assistant  # noqa: E402

# ---- CUDA-graph dispatch counter ----
import vllm.v1.worker.gpu.cudagraph_utils as CG  # noqa: E402
_CG = {"pw": 0}
_real_run_pw = CG.CudaGraphManager.run_pw_graph


def _counted_run_pw(self, model, model_inputs):
    _CG["pw"] += 1
    return _real_run_pw(self, model, model_inputs)


CG.CudaGraphManager.run_pw_graph = _counted_run_pw

# ---- per-turn capture: last decoder stats + FIRST-STEP (prefill boundary) time ----
_LAST = {"stats": None, "denoise_forwards": 0, "first_step_t": None}
_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    committing = bool(is_committing[0].item())
    if _LAST["first_step_t"] is None:
        # First step of this turn: prefill kernels are already resolved (the
        # .item() above synced). Sync explicitly and stamp the prefill boundary.
        torch.cuda.synchronize()
        _LAST["first_step_t"] = time.time()
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
    ep_end = int(os.environ.get("EP_END", "9"))
    margin = int(os.environ.get("BENCH_MARGIN", "16"))
    temp = float(os.environ.get("BENCH_TEMP", "0.0"))
    seed = int(os.environ.get("BENCH_SEED", "20260701"))
    reset_apc = os.environ.get("BENCH_RESET_APC", "0") == "1"
    mask_id = int(records[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)

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
    todo = [r for r in records if ep_start <= r["episode"] <= ep_end
            and r["global_turn"] not in done]
    todo.sort(key=lambda r: r["global_turn"])  # episode order so cache-ON reuse is realistic
    mode = "COLD(reset_apc)" if reset_apc else "ON(apc)"
    print(f"[bench-eng {mode}] ep{ep_start}..{ep_end} todo={len(todo)} done={len(done)} "
          f"seed={seed} BIDIR={os.environ.get('VLLM_FLARE_BIDIR_PROBE')} "
          f"CUDAGRAPH={os.environ.get('VLLM_FLARE_CUDAGRAPH')} OUT={OUT.name}", flush=True)
    if not todo:
        print("[bench-eng] nothing to do", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)

    t_boot = time.time()
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=int(records[0]["block_size"]), decode_mode="hybrid_clean", seed=seed,
    )
    engine = adapter._build_engine()
    boot_s = round(time.time() - t_boot, 1)
    runner = engine.llm_engine.model_executor.driver_worker.model_runner
    ms = getattr(runner, "model_state", None) or getattr(runner, "_model_state", None)
    enforce_eager = engine.llm_engine.vllm_config.model_config.enforce_eager
    cg_mode = getattr(engine.llm_engine.vllm_config.compilation_config, "cudagraph_mode", None)
    apc_live = engine.llm_engine.vllm_config.cache_config.enable_prefix_caching
    print(f"[bench-eng] booted boot_s={boot_s} decode_mode={getattr(ms,'decode_mode',None)} "
          f"enforce_eager={enforce_eager} cudagraph_mode={cg_mode} apc_live={apc_live}", flush=True)

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
        _LAST["first_step_t"] = None
        if reset_apc:
            # cache-COLD: drop cross-turn prefix (KV + mamba align state) so this
            # turn is prefilled fresh. Done OUTSIDE the timed region.
            engine.reset_prefix_cache()
        cg0 = _CG["pw"]
        torch.cuda.synchronize()
        t0 = time.time()
        req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        torch.cuda.synchronize()
        wall = time.time() - t0
        prefill = (_LAST["first_step_t"] - t0) if _LAST["first_step_t"] is not None else None
        o = req.outputs[0]
        ids = [int(x) for x in o.token_ids]
        cg_pw = _CG["pw"] - cg0
        return (ids, getattr(o, "finish_reason", None), round(wall, 4),
                (round(prefill, 4) if prefill is not None else None),
                _LAST["denoise_forwards"], _LAST["stats"], cg_pw)

    fh = OUT.open("a")
    for rec in todo:
        n_ref = rec["n_ref"]
        maxtok = n_ref + margin
        ids, finish, wall, prefill, fwd, stats, cg_pw = gen(rec, maxtok, temp, seed)
        decode_s = round(wall - prefill, 4) if prefill is not None else None
        n_cmp = min(len(ids), n_ref)
        first_div = next((i for i in range(n_cmp) if ids[i] != rec["ref_new_ids"][i]), None)
        byte_parity_full = (first_div is None and len(ids) == n_ref)
        new_ids_t = torch.tensor(ids, dtype=torch.long)
        assistant_text = trim_scored_assistant(decode_text(tok, new_ids_t))
        sc = score_tool_calls(assistant_text, rec["tools"], rec["gold_block"])
        turn = {
            "arm": "engine", "mode": ("cold" if reset_apc else "on"),
            "global_turn": rec["global_turn"], "episode": rec["episode"], "turn": rec["turn"],
            "episode_id": rec["episode_id"], "prompt_len": rec["prompt_len"],
            "n_ref": n_ref, "n_gen": len(ids), "maxtok": maxtok,
            "finish_reason": finish, "wall_s": wall, "prefill_s": prefill,
            "decode_s": decode_s, "denoise_forwards": fwd, "cg_pw_dispatches": cg_pw,
            "reset_apc": reset_apc,
            "prefill_frac": (round(prefill / wall, 4) if (prefill is not None and wall > 0) else None),
            "first_divergence": first_div, "byte_parity_full": byte_parity_full,
            "eng_exact_arguments": bool(sc.get("exact_arguments")),
            "eng_valid_tool_call": bool(sc.get("valid_tool_call")),
            "hf_exact_arguments": rec["hf_exact_arguments"],
            "source_family": rec.get("source_family"),
            "counters": stats,
        }
        fh.write(json.dumps(turn) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"[bench-eng {mode}] gt{rec['global_turn']:3d} ep{rec['episode']}/t{rec['turn']} "
              f"plen={rec['prompt_len']:5d} n={len(ids)}/{n_ref} fwd={fwd} "
              f"wall={wall:.4f} prefill={prefill} dec={decode_s} pf%={turn['prefill_frac']} "
              f"parity={byte_parity_full}", flush=True)
    fh.close()
    print("[bench-eng] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[bench-eng] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
