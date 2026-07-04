#!/usr/bin/env python3
"""P2 FULL ACCEPTANCE (post GAP-5A forward-view fix, vLLM pin 6b81154).

Single engine boot, hybrid_clean decode, real export
``qwen3.5-9b-fastdllm-rlv2-vllm-bf16`` (block/canvas 32, mamba 1024, align+APC).
Compares the ENGINE against the pre-captured HF Fast_dLLM reference in
``gap5a_ref.json`` (the 3 matched-20 parity turns: ep0/t0, ep1/t0, ep2/t0).

Emits, per turn:
  * byte-parity vs ``ref_new_ids`` (token-for-token, first_divergence)
  * finish_reason, wall_s, denoise_forwards
  * the hybrid_clean audit counters captured at the LAST sampler step of the
    request (before it is freed): forwards / fsm_committed / value_tokens /
    value_projection_events (== projected_value_tokens_exact) / model_chosen /
    generated, plus ``verify_invariants()`` (forwards==model_chosen,
    generated==fsm+model_chosen, projection==0).
  * cumulative ModelState delta: read_calls / advance_calls /
    residual_full_context_model_calls.

Then:
  * greedy determinism (ep0, seed A vs seed B -> byte-identical)
  * temp>0 contract (ep0 temp=0.7: seedA x2 identical; seedB differs;
    grammar-valid)
  * 5 temp=0.7 seeded rollouts (RL-rollout sanity: bounded, grammar-valid,
    projection==0)

Writes results incrementally to ``$ACC_OUT`` so a wall-clock timeout still
preserves the finished turns. One heavy process; run inside the RAM cage.
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
OUT = Path(os.environ.get("ACC_OUT", str(ROOT / "runs/p2_engine_acceptance/p2_full_acceptance.json")))

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402

RESULT: dict = {"turns": [], "determinism": None, "temp_contract": None,
                "rollouts": [], "meta": {}}


def _flush():
    OUT.write_text(json.dumps(RESULT, indent=2) + "\n")


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
    seedA = int(os.environ.get("ACC_SEED_A", "20260701"))
    seedB = int(os.environ.get("ACC_SEED_B", "987654321"))
    mask_id = int(records[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(records[0]["block_size"])

    RESULT["meta"] = {
        "pin": "vllm 6b81154 (GAP-5A windowed-probe forward fix)",
        "model": str(MODEL), "mask_id": mask_id, "block_size": block_size,
        "seedA": seedA, "seedB": seedB,
        "windowed_probe": os.environ.get("VLLM_FLARE_WINDOWED_PROBE", "1(default on)"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _flush()

    t_boot = time.time()
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=seedA,
    )
    engine = adapter._build_engine()
    RESULT["meta"]["boot_s"] = round(time.time() - t_boot, 1)
    runner = engine.llm_engine.model_executor.driver_worker.model_runner
    ms = getattr(runner, "model_state", None) or getattr(runner, "_model_state", None)
    print(f"[acc] booted boot_s={RESULT['meta']['boot_s']} model_state={type(ms).__name__ if ms else None} "
          f"decode_mode={getattr(ms,'decode_mode',None)} patched={QF.Qwen3_5FlareSampler._hybrid_clean_step is _patched_step}",
          flush=True)
    _flush()

    from vllm import SamplingParams

    def gen(rec, maxtok, temperature, seed):
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(
            max_tokens=maxtok, temperature=temperature, top_p=1.0, seed=seed,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])},
        )
        _LAST["stats"] = None
        _LAST["denoise_forwards"] = 0
        rc0 = int(getattr(ms, "read_calls", 0)); ac0 = int(getattr(ms, "advance_calls", 0))
        res0 = int(getattr(ms, "residual_full_context_model_calls", 0))
        t0 = time.time()
        req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        wall = time.time() - t0
        o = req.outputs[0]
        ids = [int(x) for x in o.token_ids]
        return {
            "ids": ids, "finish_reason": getattr(o, "finish_reason", None),
            "wall_s": round(wall, 2), "n_gen": len(ids),
            "denoise_forwards": _LAST["denoise_forwards"],
            "stats": _LAST["stats"],
            "ms_delta": {
                "read_calls": int(getattr(ms, "read_calls", 0)) - rc0,
                "advance_calls": int(getattr(ms, "advance_calls", 0)) - ac0,
                "residual_full_context_model_calls": int(getattr(ms, "residual_full_context_model_calls", 0)) - res0,
            },
        }

    def verify(stats, n_gen):
        if not stats:
            return {"ok": False, "why": "no stats captured"}
        chk = {
            "value_projection_events_is_0": stats["value_projection_events"] == 0,
            "forwards_eq_model_chosen": stats["forwards"] == stats["model_chosen_tokens"],
            "generated_eq_fsm_plus_model": stats["generated_tokens"] == (
                stats["fsm_committed_tokens"] + stats["model_chosen_tokens"]),
            "forced_gt_0": stats["fsm_committed_tokens"] > 0,
        }
        chk["ok"] = all(chk.values())
        return chk

    # ep1 (rec index 1) has 110 ref tokens over a 1443-tok prefix; the
    # grammar-FSM cost is O(committed^2) and its TAIL (~tokens 60-110) is
    # pathologically slow (>9 min), NOT the forward -- the committed
    # gap5a_windowed artifact ran it to head-26 (1.1s) for the same reason.
    # Cap ep1 to keep the run bounded; ep0/ep2 run FULL to their stop token.
    EP1_CAP = int(os.environ.get("ACC_EP1_CAP", "32"))

    # ---------- STEP 1: byte-parity + counters on the 3 parity turns ----------
    for ridx, rec in enumerate(records):
        n_ref = len(rec["ref_new_ids"])
        maxtok = n_ref + 8
        if ridx == 1:
            maxtok = min(maxtok, EP1_CAP)
        r = gen(rec, maxtok, 0.0, seedA)
        ref_ids = rec["ref_new_ids"]
        n_cmp = min(len(r["ids"]), n_ref)
        first_div = next((i for i in range(n_cmp) if r["ids"][i] != ref_ids[i]), None)
        turn = {
            "rec": ridx, "episode": rec["episode"], "turn": rec["turn"],
            "prompt_len": rec["prompt_len"], "n_ref": n_ref, "n_gen": r["n_gen"],
            "maxtok": maxtok, "finish_reason": r["finish_reason"], "wall_s": r["wall_s"],
            "denoise_forwards": r["denoise_forwards"],
            "first_divergence": first_div,
            "byte_parity_full": (first_div is None and r["n_gen"] == n_ref),
            "byte_parity_over_min": first_div is None,
            "ref_head": ref_ids[:16], "gen_head": r["ids"][:16],
            "ref_tail": ref_ids[-6:], "gen_tail": r["ids"][-6:],
            "counters": r["stats"], "ms_delta": r["ms_delta"],
            "verify": verify(r["stats"], r["n_gen"]),
        }
        RESULT["turns"].append(turn)
        print(f"[acc] TURN ep{rec['episode']}/t{rec['turn']} parity_full={turn['byte_parity_full']} "
              f"parity_min={turn['byte_parity_over_min']} first_div={first_div} n_gen={r['n_gen']}/{n_ref} "
              f"finish={r['finish_reason']} fwd={r['denoise_forwards']} verify={turn['verify']['ok']} "
              f"proj={r['stats']['value_projection_events'] if r['stats'] else '?'} "
              f"forced={r['stats']['fsm_committed_tokens'] if r['stats'] else '?'} wall={r['wall_s']}", flush=True)
        _flush()

    # ---------- STEP 2a: greedy determinism (ep0 seedA vs seedB) ----------
    rec0 = records[0]
    a = RESULT["turns"][0]["gen_head"]  # not enough; regen full for exactness
    g_a = gen(rec0, len(rec0["ref_new_ids"]) + 8, 0.0, seedA)
    g_b = gen(rec0, len(rec0["ref_new_ids"]) + 8, 0.0, seedB)
    RESULT["determinism"] = {
        "seedA": seedA, "seedB": seedB, "nA": g_a["n_gen"], "nB": g_b["n_gen"],
        "byte_identical": g_a["ids"] == g_b["ids"],
    }
    print(f"[acc] DETERMINISM(greedy) byte_identical={RESULT['determinism']['byte_identical']} "
          f"nA={g_a['n_gen']} nB={g_b['n_gen']}", flush=True)
    _flush()

    # ---------- STEP 2b: temp>0 contract (ep0 temp=0.7) ----------
    T = 0.7
    maxtok0 = len(rec0["ref_new_ids"]) + 8
    a1 = gen(rec0, maxtok0, T, seedA)
    a2 = gen(rec0, maxtok0, T, seedA)
    b1 = gen(rec0, maxtok0, T, seedB)

    def grammar_valid(g):
        st = g["stats"]
        return {
            "finish_reason": g["finish_reason"],
            "n_gen": g["n_gen"],
            "starts_with_scaffold": g["ids"][:1] == rec0["ref_new_ids"][:1],
            "value_projection_events": (st["value_projection_events"] if st else None),
            "forced_gt_0": (st["fsm_committed_tokens"] > 0 if st else None),
            "bounded": g["finish_reason"] in ("stop", "length"),
        }
    RESULT["temp_contract"] = {
        "temperature": T,
        "fixed_seed_2x_identical": a1["ids"] == a2["ids"],
        "two_seeds_differ": a1["ids"] != b1["ids"],
        "seedA_run1": grammar_valid(a1),
        "seedA_run2_ids_head": a2["ids"][:16],
        "seedB_run1": grammar_valid(b1),
        "seedA_ids_head": a1["ids"][:16],
        "seedB_ids_head": b1["ids"][:16],
    }
    print(f"[acc] TEMP>0 fixed_seed_2x_identical={RESULT['temp_contract']['fixed_seed_2x_identical']} "
          f"two_seeds_differ={RESULT['temp_contract']['two_seeds_differ']} "
          f"A_finish={a1['finish_reason']} B_finish={b1['finish_reason']} "
          f"A_proj={a1['stats']['value_projection_events'] if a1['stats'] else '?'}", flush=True)
    _flush()

    # ---------- STEP 3 aux: 5 temp=0.7 seeded rollouts (RL sanity) ----------
    rollout_specs = [
        (0, seedA), (0, seedB), (2, seedA), (2, seedB), (1, seedA),
    ]
    for ridx, seed in rollout_specs:
        rec = records[ridx]
        cap = min(len(rec["ref_new_ids"]) + 8, EP1_CAP if ridx == 1 else 64)
        g = gen(rec, cap, T, seed)
        st = g["stats"]
        RESULT["rollouts"].append({
            "episode": rec["episode"], "seed": seed, "n_gen": g["n_gen"],
            "finish_reason": g["finish_reason"], "wall_s": g["wall_s"],
            "denoise_forwards": g["denoise_forwards"],
            "value_projection_events": (st["value_projection_events"] if st else None),
            "forced": (st["fsm_committed_tokens"] if st else None),
            "grammar_valid": (g["finish_reason"] in ("stop", "length")
                              and st is not None and st["value_projection_events"] == 0),
            "ids_head": g["ids"][:12],
        })
        print(f"[acc] ROLLOUT ep{rec['episode']} seed={seed} n_gen={g['n_gen']} finish={g['finish_reason']} "
              f"proj={st['value_projection_events'] if st else '?'} valid={RESULT['rollouts'][-1]['grammar_valid']}",
              flush=True)
        _flush()

    RESULT["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _flush()
    print("[acc] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # preserve partial results
        import traceback
        RESULT["meta"]["error"] = repr(e)
        RESULT["meta"]["traceback"] = traceback.format_exc()
        _flush()
        print("[acc] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
