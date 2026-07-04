#!/usr/bin/env python3
"""GAP-5A FORWARD FIX acceptance (variable single-[MASK] schedule width).

Proves the fix that makes each diffusion probe forward exactly
``[committed tail + one MASK]`` (no trailing masks) -- the scheduler now
publishes the sampler's per-slot ``_hc_draft_len`` instead of a fixed 32.

Two processes (two 9B bf16 copies cannot co-reside on a 32 GB card):

  --side reference : HF Fast_dLLM bridge over the vLLM export, sequential
                     single-[MASK] hybrid_clean reference. Captures, on the FIRST
                     model-chosen step (== pos-12, the first logit-dependent
                     choice), the top-k of the probe logit. Saves ref ids + pos12.
  --side engine    : the vLLM FLARE engine (hybrid_clean). Replays the SAME
                     prompt_ids; captures the engine's pos-12 probe top-k from
                     inside the served sampler; then determinism (2x greedy,
                     DIFFERENT seeds, no RNG pinning -> byte-identical) and the
                     1041-tok real turn (5B IMA regression: must decode).
  --side compare   : byte-parity per turn + pos-12 argmax/top-k alignment.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
CHAT = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")

import parity_audit_flare_engine as H  # noqa: E402


def make_args(block_size=32, max_new_tokens=384, temperature=0.0, top_p=1.0, grammar_topk=256):
    return SimpleNamespace(
        input_jsonl=ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl",
        episode_limit=20, min_turns=3, max_turns=6,
        base_model=MODEL, adapter=None, no_merge_adapter=True,
        tokenizer_path=MODEL, chat_template_path=CHAT,
        block_size=block_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, grammar_topk=grammar_topk,
        seed=20260701,
    )


def parse_pairs(spec):
    out = []
    for tok in spec.split(","):
        e, t = tok.split(":")
        out.append((int(e), int(t)))
    return out


def _topk(vec, mask_id, k=8):
    v = vec.detach().float().clone()
    if mask_id is not None and 0 <= int(mask_id) < v.numel():
        v[int(mask_id)] = float("-inf")
    vals, idx = torch.topk(v, k)
    return {"argmax": int(idx[0].item()),
            "top": [[int(i), round(float(x), 4)] for i, x in zip(idx.tolist(), vals.tolist())]}


# ---------------------------------------------------------------------------
def run_reference(pairs, capture_pair, out_path):
    from eval_toolcall_jsonl import tool_schema_by_name
    import flare_hf_cache as FHC

    args_ns = make_args()
    (model, tokenizer, mask_id, stop_token_ids, chat_template, episodes,
     render_matched_prompt) = H.load_real_model_and_data(args_ns)
    model.eval()
    print(f"[reference] loaded mask_id={mask_id}", flush=True)

    # Capture the pos-12 probe: the FIRST shifted_active_logits call whose input
    # ends in the [MASK] sentinel (the model-chosen probe; forced tokens use the
    # grammar only, and _maybe_advance_cache forwards clean blocks -> no mask).
    cap = {"done": False, "vec": None}
    real_sal = FHC.RequestDiffusionState.shifted_active_logits

    def patched_sal(self, model, x_t):
        out = real_sal(self, model, x_t)
        if not cap["done"] and int(x_t[0, -1].item()) == int(mask_id):
            cap["vec"] = out[:, -1, :].detach().float().cpu().clone()[0]
            cap["done"] = True
        return out

    records = []
    for (ep_i, turn_i) in pairs:
        cap["done"] = False
        cap["vec"] = None
        episode = episodes[ep_i]
        ctx_kwargs = dict(block_size=32, max_new_tokens=384, mask_id=int(mask_id),
                          stop_token_ids=stop_token_ids, top_p=1.0, temperature=0.0,
                          grammar_topk=256)
        prompt = H.build_turn_prompt(model, tokenizer, episode, chat_template,
                                     turn_i, ctx_kwargs, render_matched_prompt)
        prompt_ids_t = tokenizer([prompt], return_tensors="pt",
                                 add_special_tokens=False).input_ids.to("cuda")
        prompt_len = int(prompt_ids_t.shape[1])
        schemas = tool_schema_by_name(episode["tools"])
        ctx = H.TurnContext(model=model, tokenizer=tokenizer,
                            prompt_input_ids=prompt_ids_t, schemas=schemas, **ctx_kwargs)
        do_cap = (ep_i, turn_i) == capture_pair
        if do_cap:
            FHC.RequestDiffusionState.shifted_active_logits = patched_sal
        t0 = time.time()
        res = H.ReferenceRunner().run(ctx)
        wall = time.time() - t0
        if do_cap:
            FHC.RequestDiffusionState.shifted_active_logits = real_sal
        new_ids = res.output_ids[prompt_len:]
        rec = {
            "episode": ep_i, "turn": turn_i, "prompt_len": prompt_len,
            "prompt_ids": [int(x) for x in prompt_ids_t.reshape(-1).tolist()],
            "ref_new_ids": [int(x) for x in new_ids],
            "mask_id": int(mask_id), "schemas": schemas,
            "stop_token_ids": sorted(int(x) for x in stop_token_ids),
            "block_size": 32, "grammar_topk": 256,
            "ref_wall_s": round(wall, 2),
            "ref_head_decoded": tokenizer.decode(new_ids[:16]),
        }
        if do_cap and cap["vec"] is not None:
            rec["pos12_ref"] = _topk(cap["vec"], mask_id)
            rec["pos12_ref"]["decoded_token"] = tokenizer.decode([rec["pos12_ref"]["argmax"]])
        records.append(rec)
        print(f"[reference] ep{ep_i} t{turn_i} plen={prompt_len} gen={len(new_ids)} "
              f"wall={wall:.1f}s head={tokenizer.decode(new_ids[:12])!r}", flush=True)
        if do_cap and "pos12_ref" in rec:
            print(f"[reference] POS12 argmax={rec['pos12_ref']['argmax']} "
                  f"({rec['pos12_ref']['decoded_token']!r}) top={rec['pos12_ref']['top'][:4]}",
                  flush=True)
    Path(out_path).write_text(json.dumps(records, indent=2) + "\n")
    print(f"[reference] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
def run_engine(in_path, out_path, capture_pair, ima_len):
    records = json.loads(Path(in_path).read_text())
    mask_id = int(records[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(records[0]["block_size"])

    # Install a capture hook on the served sampler BEFORE the engine boots, so
    # the FIRST denoise (non-committing) probe forward -- the pos-12 probe --
    # top-k is recorded from inside the real serving path.
    from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF
    cap = {"done": False, "vec": None}
    real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step

    def patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                     decode_indices_np, decode_slots_np, valid_len_np,
                     is_committing, num_reqs, input_batch):
        if not cap["done"] and not bool(is_committing[0].item()):
            vlen = int(valid_len_np[0])
            cap["vec"] = shifted[0, max(vlen - 1, 0)].detach().float().cpu().clone()
            cap["done"] = True
            cap["vlen"] = vlen
        return real_step(self, shifted, block_logits, decode_slots, decode_idx,
                         decode_indices_np, decode_slots_np, valid_len_np,
                         is_committing, num_reqs, input_batch)

    QF.Qwen3_5FlareSampler._hybrid_clean_step = patched_step

    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=20260701,
    )
    engine = adapter._build_engine()
    print(f"[engine] booted mask_id={mask_id} block_size={block_size}", flush=True)

    def one_turn(rec, seed, tag, do_capture):
        cap["done"] = False if do_capture else True
        cap["vec"] = None
        from vllm import SamplingParams
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        sp = SamplingParams(
            max_tokens=int(rec.get("max_new_tokens", 384)),
            temperature=0.0, top_p=1.0, seed=int(seed),
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])},
        )
        t0 = time.time()
        req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
        wall = time.time() - t0
        o = req.outputs[0]
        ids = [int(x) for x in o.token_ids]
        stats = adapter._read_engine_stats(engine)
        hc = (stats or {}).get("hybrid_clean") or {}
        out = {"tag": tag, "seed": seed, "n": len(ids), "eng_new_ids": ids,
               "finish": getattr(o, "finish_reason", None), "wall_s": round(wall, 2),
               "projected_value_tokens_exact": hc.get("projected_value_tokens_exact"),
               "model_forwards": hc.get("model_forwards"),
               "forced_token_count": hc.get("forced_token_count")}
        if do_capture and cap["vec"] is not None:
            out["pos12_eng"] = _topk(cap["vec"], mask_id)
            out["pos12_eng"]["vlen"] = cap.get("vlen")
        print(f"[engine] {tag} n={len(ids)} wall={wall:.1f}s "
              f"proj={out['projected_value_tokens_exact']} fwd={out['model_forwards']}", flush=True)
        if "pos12_eng" in out:
            print(f"[engine] POS12 argmax={out['pos12_eng']['argmax']} "
                  f"vlen={out['pos12_eng']['vlen']} top={out['pos12_eng']['top'][:4]}", flush=True)
        return out

    result = {"turns": [], "determinism": None, "ima": None}
    rec0 = records[0]
    cap_rec = next((r for r in records if (r["episode"], r["turn"]) == capture_pair), rec0)

    # Byte-parity turns (capture pos-12 on the capture_pair turn).
    for rec in records:
        do_cap = (rec["episode"], rec["turn"]) == capture_pair
        result["turns"].append(
            {"episode": rec["episode"], "turn": rec["turn"],
             **one_turn(rec, 20260701, f"turn_{rec['episode']}_{rec['turn']}", do_cap)}
        )

    # Determinism: SAME prompt, greedy, DIFFERENT seeds, no RNG pinning.
    dA = one_turn(cap_rec, 20260701, "det_seedA", False)
    dB = one_turn(cap_rec, 987654321, "det_seedB", False)
    result["determinism"] = {
        "seedA": dA["seed"], "seedB": dB["seed"],
        "byte_identical": dA["eng_new_ids"] == dB["eng_new_ids"],
        "nA": dA["n"], "nB": dB["n"],
    }
    print(f"[engine] DETERMINISM byte_identical={result['determinism']['byte_identical']}", flush=True)

    # 5B IMA regression: the real 1041-tok turn must decode (no CUDA IMA).
    real_rec = max(records, key=lambda r: r["prompt_len"])
    try:
        r = one_turn(real_rec, 20260701, f"ima_real_{real_rec['prompt_len']}", False)
        result["ima"] = {"prompt_len": real_rec["prompt_len"], "ok": True,
                         "n": r["n"], "finish": r["finish"],
                         "proj": r["projected_value_tokens_exact"]}
    except Exception as e:  # noqa: BLE001
        result["ima"] = {"prompt_len": real_rec["prompt_len"], "ok": False,
                         "error": repr(e)[:400], "trace": traceback.format_exc()[-1500:]}
    print(f"[engine] IMA ok={result['ima'].get('ok')}", flush=True)

    Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[engine] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
def run_compare(ref_path, eng_path, out_path):
    ref = json.loads(Path(ref_path).read_text())
    eng = json.loads(Path(eng_path).read_text())
    eng_by = {(t["episode"], t["turn"]): t for t in eng["turns"]}
    turns = []
    all_byte = True
    for r in ref:
        key = (r["episode"], r["turn"])
        e = eng_by.get(key)
        if e is None:
            turns.append({**{"episode": key[0], "turn": key[1]}, "engine_missing": True})
            all_byte = False
            continue
        rep = H.token_bytes_identical(r["ref_new_ids"], e["eng_new_ids"], tokenizer=None)
        byte_ok = bool(rep["byte_exact"] and rep["token_exact"])
        all_byte = all_byte and byte_ok
        row = {"episode": key[0], "turn": key[1], "byte_identical": byte_ok,
               "first_divergence": rep.get("first_divergence"),
               "ref_gen": len(r["ref_new_ids"]), "eng_gen": len(e["eng_new_ids"]),
               "ref_head": r["ref_new_ids"][:16], "eng_head": e["eng_new_ids"][:16],
               "engine_projected_value_tokens": e.get("projected_value_tokens_exact")}
        if "pos12_ref" in r and "pos12_eng" in e:
            row["pos12"] = {
                "ref_argmax": r["pos12_ref"]["argmax"],
                "eng_argmax": e["pos12_eng"]["argmax"],
                "argmax_match": r["pos12_ref"]["argmax"] == e["pos12_eng"]["argmax"],
                "ref_top": r["pos12_ref"]["top"][:5],
                "eng_top": e["pos12_eng"]["top"][:5],
                "eng_vlen": e["pos12_eng"].get("vlen"),
                "decoded_token": r["pos12_ref"].get("decoded_token"),
            }
        turns.append(row)
    report = {
        "mode": "gap5a_forward_fix_verify",
        "all_byte_identical": all_byte,
        "determinism": eng.get("determinism"),
        "ima": eng.get("ima"),
        "turns": turns,
    }
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["reference", "engine", "compare"], required=True)
    ap.add_argument("--turns", default="0:0,1:0,2:0")
    ap.add_argument("--capture", default="0:0")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in", dest="in_path", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--eng", type=Path)
    ap.add_argument("--ima-len", type=int, default=1041)
    a = ap.parse_args()
    cap = tuple(int(x) for x in a.capture.split(":"))
    if a.side == "reference":
        run_reference(parse_pairs(a.turns), cap, a.out)
    elif a.side == "engine":
        run_engine(a.in_path, a.out, cap, a.ima_len)
    else:
        run_compare(a.ref, a.eng, a.out)


if __name__ == "__main__":
    main()
