#!/usr/bin/env python3
"""Fast, time-bounded GAP-5A engine probe.

Boots the FLARE engine once, runs a chosen ref turn with a HARD max_tokens cap,
captures the pos-12 probe logit top-k from inside the served sampler, and also
reports per-model-token wall time so we can budget the full byte-parity turns.

Env:
  PROBE_MAXTOK  : hard max_tokens cap (default 16 -> just past first denoise probe)
  PROBE_TURN    : record index in the ref json to run (default 0 = ep0/t0)
  PROBE_SEED    : sampling seed (default 20260701)
"""
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import torch

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = ROOT / "runs/p2_engine_acceptance/gap5a_ref.json"

import parity_audit_flare_engine as H  # noqa: E402


def _topk(vec, mask_id, k=8):
    v = vec.detach().float().clone()
    if mask_id is not None and 0 <= int(mask_id) < v.numel():
        v[int(mask_id)] = float("-inf")
    vals, idx = torch.topk(v, k)
    return {"argmax": int(idx[0].item()),
            "top": [[int(i), round(float(x), 4)] for i, x in zip(idx.tolist(), vals.tolist())]}


def main():
    records = json.loads(REF.read_text())
    maxtok = int(os.environ.get("PROBE_MAXTOK", "16"))
    tidx = int(os.environ.get("PROBE_TURN", "0"))
    seed = int(os.environ.get("PROBE_SEED", "20260701"))
    rec = records[tidx]
    mask_id = int(rec["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(rec["block_size"])

    from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF
    cap = {"done": False, "vec": None, "vlen": None, "count": 0, "t_first": None}
    steps = []  # per-step: (is_committing, valid_len, committed_delta_ids)
    real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step

    def patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                     decode_indices_np, decode_slots_np, valid_len_np,
                     is_committing, num_reqs, input_batch):
        committing = bool(is_committing[0].item())
        vlen = int(valid_len_np[0])
        slot0 = int(decode_slots_np[0])
        dec = self._hc_decoders.get(slot0)
        before = len(dec.committed) if dec is not None else 0
        # denoise probe logit capture (top of the shifted probe position)
        if not committing:
            cap["count"] += 1
            if cap["t_first"] is None:
                cap["t_first"] = time.time()
            if not cap["done"]:
                read_pos = vlen - 1
                if os.environ.get("VLLM_FLARE_WINDOWED_PROBE", "1") != "0" and dec is not None:
                    read_pos = int(self._hc_draft_len.get(slot0, vlen)) - 1
                cap["vec"] = shifted[0, max(read_pos, 0)].detach().float().cpu().clone()
                cap["vlen"] = vlen
                cap["read_pos"] = max(read_pos, 0)
                cap["done"] = True
        # capture block_logits argmax at the last valid position (what a commit
        # boundary / block-parallel read would pick) for diagnosis
        bl_arg = int(block_logits[0, max(vlen - 1, 0)].argmax().item()) if committing else None
        ret = real_step(self, shifted, block_logits, decode_slots, decode_idx,
                        decode_indices_np, decode_slots_np, valid_len_np,
                        is_committing, num_reqs, input_batch)
        after = len(dec.committed) if dec is not None else 0
        delta = dec.committed[before:after] if dec is not None else []
        if len(steps) < 60:
            steps.append({"commit": committing, "vlen": vlen,
                          "delta": [int(x) for x in delta],
                          "bl_argmax": bl_arg})
        return ret

    QF.Qwen3_5FlareSampler._hybrid_clean_step = patched_step

    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=seed,
    )
    if os.environ.get("PROBE_ASYNC") == "0":
        adapter._engine_kwargs["async_scheduling"] = False
        print("[probe] async_scheduling=False", flush=True)
    engine = adapter._build_engine()
    print(f"[probe] booted pid={os.getpid()} mask_id={mask_id} block={block_size} "
          f"plen={rec['prompt_len']} maxtok={maxtok} tidx={tidx}", flush=True)
    # confirm the class method is our patch (in-process worker)
    print(f"[probe] class-patched={QF.Qwen3_5FlareSampler._hybrid_clean_step is patched_step}",
          flush=True)
    try:
        runner = engine.llm_engine.model_executor.driver_worker.model_runner
        ms = getattr(runner, "model_state", None) or getattr(runner, "_model_state", None)
        print(f"[probe] in-process runner={runner is not None} model_state={type(ms).__name__ if ms else None} "
              f"decode_mode={getattr(ms,'decode_mode',None)} sampler={type(getattr(ms,'_flare_sampler',None)).__name__}",
              flush=True)
    except Exception as e:
        print(f"[probe] in-process introspect FAILED: {e!r}", flush=True)

    from vllm import SamplingParams
    tools = [{"type": "function", "function": {"name": n, "parameters": p}}
             for n, p in (rec["schemas"] or {}).items()]
    sp = SamplingParams(
        max_tokens=maxtok, temperature=0.0, top_p=1.0, seed=seed,
        stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
        extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                    "grammar_topk": int(rec["grammar_topk"])},
    )
    t0 = time.time()
    req = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp)
    wall = time.time() - t0
    o = req.outputs[0]
    ids = [int(x) for x in o.token_ids]
    ref_ids = rec["ref_new_ids"]
    n_cmp = min(len(ids), len(ref_ids))
    first_div = next((i for i in range(n_cmp) if ids[i] != ref_ids[i]), None)
    out = {
        "tidx": tidx, "plen": rec["prompt_len"], "maxtok": maxtok,
        "wall_s": round(wall, 2), "n_gen": len(ids),
        "finish_reason": getattr(o, "finish_reason", None),
        "denoise_forwards": cap["count"],
        "gen_ids_head": ids[:24], "ref_ids_head": ref_ids[:24],
        "first_divergence": first_div,
        "prefix_matches_ref": (first_div is None),
        "ref_pos12_argmax": rec.get("pos12_ref", {}).get("argmax"),
    }
    if cap["vec"] is not None:
        pk = _topk(cap["vec"], mask_id)
        pk["vlen"] = cap["vlen"]
        pk["decoded_argmax"] = None
        out["pos12_eng"] = pk
        rp = rec.get("pos12_ref")
        if rp is not None:
            out["pos12_argmax_match"] = (pk["argmax"] == rp["argmax"])
            out["pos12_ref_top"] = rp["top"][:5]
            out["pos12_eng_top"] = pk["top"][:5]
    if cap["t_first"] is not None and cap["count"] > 1:
        out["ms_per_denoise_fwd"] = round(
            1000.0 * (time.time() - cap["t_first"]) / max(cap["count"] - 1, 1), 1)
    if os.environ.get("PROBE_DET") == "1":
        sp2 = SamplingParams(
            max_tokens=maxtok, temperature=0.0, top_p=1.0, seed=987654321,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])},
        )
        req2 = adapter._engine_generate(engine, list(rec["prompt_ids"]), sp2)
        ids2 = [int(x) for x in req2.outputs[0].token_ids]
        out["determinism"] = {
            "seedA": seed, "seedB": 987654321,
            "nA": len(ids), "nB": len(ids2),
            "byte_identical": ids == ids2,
        }
        print("DETERMINISM " + json.dumps(out["determinism"]), flush=True)

    out["steps"] = steps
    print("RESULT " + json.dumps({k: v for k, v in out.items() if k != "steps"}), flush=True)
    # running committed index -> which step/phase emitted each token
    print("STEPS:", flush=True)
    idx = 0
    for si, s in enumerate(steps):
        tags = []
        for t in s["delta"]:
            tags.append(f"[{idx}]={t}")
            idx += 1
        print(f"  step{si:2d} commit={s['commit']} vlen={s['vlen']} "
              f"bl_argmax={s['bl_argmax']} +{len(s['delta'])} {' '.join(tags)}", flush=True)
    outp = os.environ.get("PROBE_OUT")
    if outp:
        Path(outp).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
