#!/usr/bin/env python3
"""A6/A7 OFFLINE reference capture (Stage-A cert).

Boots the FLARE hybrid_clean engine IN-PROCESS (the certified offline ``LLM``
path) at the EXACT shipped-server config (the launcher
``qwen35_9b_flare_hybrid_serve.sh``): eager, gate OFF (no canonical publish),
VLLM_FLARE_BIDIR_PROBE=1, VLLM_QWEN3_5_FLARE_MASK set (the A6 launcher fix),
align-mode APC, mamba_block_size=1024, AND max_num_batched_tokens=1024 +
chunked prefill -- so this offline reference is the true byte-mirror of the
online AsyncLLM server (isolating serve-path fp-residue as the only online var).

Captures, per turn, the RAW generated token ids + detokenized text so the online
client (which can only return text: the FLARE sampler emits no logprobs) can be
compared token/byte-for-byte.

A6 = 10 matched-20 turns, single-turn, APC RESET before each turn (fresh
context, the plan's resetapc parity protocol). A7 = 3 full episodes, warm APC,
turns played in episode order with NO reset (cross-turn prefix reuse).

One heavy process; run inside the RAM cage. Writes offline_a6.jsonl /
offline_a7.jsonl.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
# --- shipped-server engine env (mirror the launcher exactly) ---
os.environ["VLLM_QWEN3_5_FLARE"] = "1"
os.environ["VLLM_QWEN3_5_FLARE_DECODE"] = "hybrid_clean"
os.environ["VLLM_QWEN3_5_FLARE_BLOCK"] = "32"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_FLARE_BIDIR_PROBE"] = "1"
os.environ.pop("VLLM_FLARE_CUDAGRAPH", None)          # eager (== launcher --enforce-eager)
os.environ.pop("VLLM_QWEN3_5_FLARE_CANONICAL_PUBLISH", None)  # gate OFF (shipped)
_CU13 = "/home/mark/qwen_diffusion/.venv-vllm-p2-main/lib/python3.12/site-packages/nvidia/cu13"
os.environ.setdefault("CUDA_HOME", _CU13)
os.environ.setdefault("NVCC_APPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")

import torch  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = json.loads((ROOT / "runs/p2_engine_bench/matched20_ref.json").read_text())
OUTDIR = ROOT / "runs/stage_a_cert"

SEED = 20260701
MARGIN = 16

# Turn selection (shared with the online client via turnsets.json)
TS = json.loads((OUTDIR / "turnsets.json").read_text())
A6_GTS = TS["a6_global_turns"]
A7_EPS = TS["a7_episodes"]

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402
from eval_flare_northstar_matched import decode_text, trim_scored_assistant  # noqa: E402

# per-turn capture of the LAST hybrid_clean decoder stats (before slot release)
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

REC_BY_GT = {r["global_turn"]: r for r in REF}
MASK_ID = int(REF[0]["mask_id"])
os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(MASK_ID)  # the A6 launcher fix, mirrored


def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=int(REF[0]["block_size"]), decode_mode="hybrid_clean",
        seed=SEED, gpu_memory_utilization=0.745,
    )
    # mirror the launcher: chunked prefill at max_num_batched_tokens=1024 (the
    # certified battery used 4096/no-chunk; the SERVER chunks every >1024-tok
    # prompt, so match it here to isolate serve-path from config differences).
    adapter._engine_kwargs.update({
        "max_num_batched_tokens": 1024,
        "enable_chunked_prefill": True,
    })
    t_boot = time.time()
    engine = adapter._build_engine()
    boot_s = round(time.time() - t_boot, 1)
    cfg = engine.llm_engine.vllm_config
    runner = engine.llm_engine.model_executor.driver_worker.model_runner
    ms = getattr(runner, "model_state", None) or getattr(runner, "_model_state", None)
    print(f"[offline] booted boot_s={boot_s} decode_mode={getattr(ms,'decode_mode',None)} "
          f"enforce_eager={cfg.model_config.enforce_eager} "
          f"apc={cfg.cache_config.enable_prefix_caching} "
          f"chunked={cfg.scheduler_config.enable_chunked_prefill} "
          f"max_batched={cfg.scheduler_config.max_num_batched_tokens} mask={MASK_ID}",
          flush=True)

    from vllm import SamplingParams

    def gen(rec):
        maxtok = int(rec["n_ref"]) + MARGIN
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(
            max_tokens=maxtok, temperature=0.0, top_p=1.0, seed=SEED,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])},
        )
        _LAST["stats"] = None
        _LAST["denoise_forwards"] = 0
        t0 = time.time()
        req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        wall = round(time.time() - t0, 3)
        o = req.outputs[0]
        ids = [int(x) for x in o.token_ids]
        return ids, getattr(o, "finish_reason", None), wall, _LAST["denoise_forwards"], _LAST["stats"], maxtok

    def record(rec, ids, finish, wall, fwd, stats, maxtok):
        n_ref = int(rec["n_ref"])
        n_cmp = min(len(ids), n_ref)
        fd_hf = next((i for i in range(n_cmp) if ids[i] != rec["ref_new_ids"][i]), None)
        text_skip = decode_text(tok, torch.tensor(ids, dtype=torch.long))
        text_full = tok.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
        assistant_text = trim_scored_assistant(text_skip)
        sc = score_tool_calls(assistant_text, rec["tools"], rec["gold_block"])
        return {
            "global_turn": rec["global_turn"], "episode": rec["episode"], "turn": rec["turn"],
            "episode_id": rec["episode_id"], "prompt_len": rec["prompt_len"],
            "n_ref": n_ref, "n_gen": len(ids), "maxtok": maxtok, "finish_reason": finish,
            "wall_s": wall, "denoise_forwards": fwd,
            "gen_ids": ids, "gen_text": text_skip, "gen_text_full": text_full,
            "byte_parity_vs_hf": (fd_hf is None and len(ids) == n_ref),
            "first_div_vs_hf": fd_hf,
            "eng_exact_arguments": bool(sc.get("exact_arguments")),
            "eng_valid_tool_call": bool(sc.get("valid_tool_call")),
            "hf_exact_arguments": bool(rec["hf_exact_arguments"]),
            "counters": stats,
        }

    # ---- A6: single-turn, APC RESET before each turn (fresh context) ----
    a6 = (OUTDIR / "offline_a6.jsonl").open("w")
    for gt in A6_GTS:
        rec = REC_BY_GT[gt]
        engine.reset_prefix_cache()
        r = record(rec, *gen(rec))
        r["apc"] = "reset_fresh"
        a6.write(json.dumps(r) + "\n"); a6.flush()
        print(f"[offline-A6] gt{gt:2d} n={r['n_gen']}/{r['n_ref']} fin={r['finish_reason']} "
              f"bp_vs_hf={r['byte_parity_vs_hf']} fd_hf={r['first_div_vs_hf']} "
              f"exact={int(r['eng_exact_arguments'])}(hf={int(r['hf_exact_arguments'])}) "
              f"proj={r['counters']['value_projection_events'] if r['counters'] else '?'}", flush=True)
    a6.close()

    # ---- A7: 3 full episodes, warm APC, episode order, NO reset ----
    engine.reset_prefix_cache()  # clean slate, then warm build-up within episodes
    a7 = (OUTDIR / "offline_a7.jsonl").open("w")
    for ep in A7_EPS:
        gts = sorted(r["global_turn"] for r in REF if r["episode"] == ep)
        for gt in gts:
            rec = REC_BY_GT[gt]
            r = record(rec, *gen(rec))
            r["apc"] = "warm"
            a7.write(json.dumps(r) + "\n"); a7.flush()
            print(f"[offline-A7] ep{ep} gt{gt:2d} t{rec['turn']} n={r['n_gen']}/{r['n_ref']} "
                  f"fin={r['finish_reason']} bp_vs_hf={r['byte_parity_vs_hf']} "
                  f"exact={int(r['eng_exact_arguments'])}(hf={int(r['hf_exact_arguments'])})", flush=True)
    a7.close()
    print("[offline] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[offline] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
