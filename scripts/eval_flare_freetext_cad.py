#!/usr/bin/env python3
"""S2 pilot GATE — entropy-gated adaptive-K CAD free-text sampler (spec s2_pilot_design.md sec.3).

The promoted free-text path is the vLLM engine ``Qwen3_5FlareSampler._hybrid_clean_step``
(TRUE sequential single-[MASK] probe, K=1) that produced the 26/30 · 0.862 tok/fwd anchor
(runs/l0l2_final_head_verify/engine_gsm8k_clean_head.jsonl, pin 0b44dcc, seed 20260701). This
script drives that engine with a MONKEYPATCHED step that adds the CAD commit rule behind
``--k-max`` / ``--gamma``:

  * ``--k-max 1`` (any gamma): PURE PASS-THROUGH to the real ``_hybrid_clean_step`` -> the
    single-[MASK] probe. Reproduces the anchor BYTE-EXACT (design sec.3 pinning sanity). This is
    the R1 instrument-validation row and the CTRL-K1 / A_S2-K1 rows.

  * ``--k-max 2``: entropy-gated adaptive commit. Each denoise forward stages ``k`` trailing
    [MASK] probes (k = min(k_max, block_target - tail)); reads the k ``+1``-shifted probe logits;
    per position confidence ``c_i = max softmax``; commits the LEADING CONTIGUOUS RUN with
    ``c_i >= gamma``, clipped to ``[1, k_max]`` (position 0 always commits => "clip min 1"; a single
    sub-gamma position blocks the run => numbers stay K=1). NEVER remask. Native stop-ids honored
    (free-text ``decode_model_token`` honors EOS): a committed stop id ends the run. Values are held
    K=1 by the contiguous-prefix gate, exactly as tool-call values are FSM-forced K=1 on the
    unchanged hybrid_clean path.

Metrics per prompt: committed answer_ids (EOS-trimmed by the engine), denoise_forwards (ACTUAL
model forwards, one per denoise step regardless of how many tokens that step committed),
committed tok/fwd = n_gen / denoise_forwards, the per-forward commit-k histogram, hybrid_clean
counters (value_projection_events etc.) for the sec.6d audit, strict GSM8K correctness, and verify.

Byte-identical engine boot + SIGALRM watchdog + resumable-append machinery to
runs/s2_pilot/run_s2_gen.py (which itself mirrors the pin-0b44dcc hardened census).

Env / args: --model (vLLM dir), --gate-json, --out, --k-max, --gamma, --seed (default 20260701),
--maxtok (384), --limit. Requires VLLM_FLARE_BIDIR_PROBE=1 / VLLM_FLARE_CUDAGRAPH=1
(runs/l1_census/env.sh) for the anchor config.
"""
import argparse
import json
import math
import os
import re
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

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

_REAL_STEP = QF.Qwen3_5FlareSampler._hybrid_clean_step
_REAL_STAGE_DENOISE = QF.Qwen3_5FlareSampler._hc_stage_denoise
_STATE = {
    "k_max": 1,
    "gamma": 1.0,
    "denoise_forwards": 0,
    "commit_k_hist": {},   # k -> count of denoise forwards that committed k tokens
    "stats": None,
    "confs": [],           # per-denoise-forward list of the read confidences (leading position first)
}


def _softmax_max(logit_row: torch.Tensor) -> float:
    """max softmax probability of a single +1-shifted probe logit (fp32)."""
    v = logit_row.detach().float()
    m = torch.max(v)
    ex = torch.exp(v - m)
    return float((torch.max(ex) / torch.sum(ex)).item())


def _patched_stage_denoise(self, slot: int) -> None:
    """Stage ``k = min(k_max, block_target - tail)`` trailing [MASK] probes.

    k_max==1 is byte-identical to the real single-[MASK] staging.
    """
    k_max = _STATE["k_max"]
    if k_max <= 1:
        return _REAL_STAGE_DENOISE(self, slot)
    states = self.diffusion_states
    decoder = self._hc_decoders[slot]
    base = self._hc_block_base[slot]
    tail = decoder.committed[base:]
    tail_len = len(tail)
    block_target = self._hc_block_target(slot, base)
    room = max(1, block_target - tail_len)
    k = max(1, min(k_max, room))
    canvas = states.canvas
    if tail_len:
        canvas[slot, :tail_len] = torch.tensor(
            tail, dtype=canvas.dtype, device=canvas.device
        )
    mask_id = self._hc_mask_id()
    for j in range(k):
        canvas[slot, tail_len + j] = mask_id
    draft_len = tail_len + k
    self.req_states.draft_tokens[slot, :draft_len] = canvas[slot, :draft_len]
    self._hc_draft_len[slot] = draft_len


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch) -> None:
    k_max = _STATE["k_max"]
    gamma = _STATE["gamma"]
    committing_list = is_committing.detach().cpu().tolist()

    if k_max <= 1:
        # PURE PASS-THROUGH: the byte-exact promoted single-[MASK] probe.
        committing = bool(committing_list[0])
        if not committing:
            _STATE["denoise_forwards"] += 1
        ret = _REAL_STEP(self, shifted, block_logits, decode_slots, decode_idx,
                         decode_indices_np, decode_slots_np, valid_len_np,
                         is_committing, num_reqs, input_batch)
        _capture_stats(self, decode_slots_np)
        if not committing:
            _STATE["commit_k_hist"]["1"] = _STATE["commit_k_hist"].get("1", 0) + 1
        return ret

    # k_max >= 2 : adaptive-K contiguous-prefix commit.
    states = self.diffusion_states
    device = shifted.device
    for i, slot in enumerate(decode_slots_np.tolist()):
        di = int(decode_indices_np[i])
        decoder = self._hc_decoders.get(slot)
        if decoder is None:
            self._num_sampled[di] = 0
            continue

        if committing_list[i]:
            # ---- COMMIT branch: verbatim from the real _hybrid_clean_step. -----
            ids = self._hc_pending.get(slot, [])
            n = len(ids)
            if n:
                self._sampled[di, :n] = torch.tensor(
                    ids, dtype=self._sampled.dtype, device=device
                )
            self._num_sampled[di] = n
            states.block_start[slot] = int(states.block_start[slot]) + n
            self._hc_block_base[slot] = self._hc_block_base[slot] + n
            last_pos = max(n - 1, 0)
            states.last_shift_logits[slot] = (
                block_logits[i, last_pos].detach().float()
            )
            states.has_prev_logit[slot] = True
            self._hc_pending[slot] = []
            if not decoder.finished:
                forced_before = decoder.stats.fsm_committed_tokens
                _b = self._hc_block_base[slot]
                decoder.bulk_forced_prefix(
                    block_limit=_b + self._hc_block_target(slot, _b)
                )
                if decoder.stats.fsm_committed_tokens > forced_before:
                    self.hc_zero_forward_rows += 1
            states.is_encoder_phase[slot] = self._hc_set_next_phase(slot)
        else:
            # ---- DENOISE branch: adaptive-K read + contiguous-prefix commit. ---
            _STATE["denoise_forwards"] += 1
            staged = int(self._hc_draft_len.get(slot, int(valid_len_np[i])))
            _b = self._hc_block_base[slot]
            block_target = self._hc_block_target(slot, _b)
            tail_now = len(decoder.committed) - _b
            room = max(1, block_target - tail_now)
            k_avail = max(1, min(k_max, room, staged))
            # Read the k trailing +1-shifted probe logits.
            probes = [shifted[i, staged - k_avail + j] for j in range(k_avail)]
            confs = [_softmax_max(p) for p in probes]
            # Leading contiguous run with c_i >= gamma, clip to [1, k_max].
            run = 0
            for c in confs:
                if c >= gamma:
                    run += 1
                else:
                    break
            k_commit = max(1, min(run if run >= 1 else 1, k_avail))
            # Commit positions 0..k_commit-1 sequentially (each from its own probe
            # logit of THIS single forward). Stop the run the instant a committed
            # token is a native stop id (design sec.3 "native stop-ids only").
            committed_here = 0
            for j in range(k_commit):
                if decoder.finished:
                    break
                if len(decoder.committed) >= block_target + _b:
                    break
                tok = decoder.policy.decode_model_token(
                    decoder.committed, probes[j], decoder.stats
                )
                decoder.committed.append(int(tok))
                committed_here += 1
                # trailing forced run (zero forwards; inert in free-text)
                decoder.bulk_forced_prefix(block_limit=_b + block_target)
                decoder.stats.generated_tokens = len(decoder.committed)
                if _is_stop_token(self, decoder, int(tok)):
                    decoder.finished = True
                    if decoder.stats.stop_reason is None:
                        decoder.stats.stop_reason = "stop_token"
                    break
            key = str(committed_here)
            _STATE["commit_k_hist"][key] = _STATE["commit_k_hist"].get(key, 0) + 1
            _STATE["confs"].append([round(c, 4) for c in confs])
            self._num_sampled[di] = 0
            states.is_encoder_phase[slot] = self._hc_set_next_phase(slot)
    _capture_stats(self, decode_slots_np)


def _is_stop_token(self, decoder, tok: int) -> bool:
    eos = getattr(self.tokenizer, "eos_token_id", None)
    if eos is not None and tok == int(eos):
        return True
    stops = getattr(getattr(decoder, "policy", None), "stop_token_ids", None)
    if stops and tok in {int(s) for s in stops}:
        return True
    return False


def _capture_stats(self, decode_slots_np) -> None:
    slot0 = int(decode_slots_np[0])
    dec = self._hc_decoders.get(slot0)
    if dec is not None:
        s = dec.stats
        _STATE["stats"] = {
            "forwards": int(s.forwards),
            "fsm_committed_tokens": int(s.fsm_committed_tokens),
            "value_tokens": int(s.value_tokens),
            "structural_model_tokens": int(s.structural_model_tokens),
            "value_projection_events": int(s.value_projection_events),
            "model_chosen_tokens": int(s.model_chosen_tokens),
            "generated_tokens": int(s.generated_tokens),
        }


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step
QF.Qwen3_5FlareSampler._hc_stage_denoise = _patched_stage_denoise


class TurnTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise TurnTimeout()


signal.signal(signal.SIGALRM, _alarm)


def strict_answer(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text)
    return m[-1].replace(",", "") if m else None


def verify(stats):
    if not stats:
        return {"ok": False}
    chk = {
        "value_projection_events_is_0": stats["value_projection_events"] == 0,
    }
    chk["ok"] = all(chk.values())
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"))
    ap.add_argument("--gate-json", default=str(ROOT / "runs/l1_census/gsm8k_prompts_clean.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-max", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--maxtok", type=int, default=384)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--turn-timeout", type=int, default=90)
    args = ap.parse_args()

    _STATE["k_max"] = int(args.k_max)
    _STATE["gamma"] = float(args.gamma)

    recs = json.loads(Path(args.gate_json).read_text())[: args.limit]
    mask_id = int(recs[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["idx"]))
            except Exception:
                pass
    todo = [r for r in recs if r["idx"] not in done]
    print(f"[cad] model={Path(args.model).name} k_max={args.k_max} gamma={args.gamma} "
          f"seed={args.seed} todo={len(todo)} done={len(done)}", flush=True)
    if not todo:
        print("[cad] nothing to do", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    t_boot = time.time()
    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(args.model),
        canvas_length=int(recs[0]["block_size"]), decode_mode="hybrid_clean", seed=int(args.seed),
    )
    engine = adapter._build_engine()
    print(f"[cad] booted boot_s={round(time.time()-t_boot,1)} "
          f"patched={QF.Qwen3_5FlareSampler._hybrid_clean_step is _patched_step}", flush=True)

    from vllm import SamplingParams

    fh = out.open("a")
    for rec in todo:
        sp = SamplingParams(
            max_tokens=args.maxtok, temperature=0.0, top_p=1.0, seed=int(args.seed),
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": [],
                        "grammar_topk": int(rec["grammar_topk"])},
        )
        _STATE["stats"] = None
        _STATE["denoise_forwards"] = 0
        _STATE["commit_k_hist"] = {}
        _STATE["confs"] = []
        t0 = time.time()
        hung = False
        err = None
        ids = []
        fin = None
        signal.alarm(args.turn_timeout)
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
        fwd = _STATE["denoise_forwards"]
        stats = _STATE["stats"]
        if hung or err:
            turn = {"idx": rec["idx"], "hang": hung, "error": err, "wall_s": wall,
                    "denoise_forwards": fwd, "correct": False,
                    "k_max": args.k_max, "gamma": args.gamma}
            fh.write(json.dumps(turn) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(f"[cad] idx{rec['idx']:4d} {'HANG' if hung else 'ERR'} {err or ''}", flush=True)
            print("[cad] HANG_EXIT (fresh reboot needed)", flush=True)
            fh.close()
            return
        text = tok.decode(ids, skip_special_tokens=True)
        pred = strict_answer(text)
        gold = strict_answer(rec["gold_answer"])
        turn = {
            "idx": rec["idx"], "k_max": args.k_max, "gamma": args.gamma,
            "n_gen": len(ids), "finish_reason": fin, "wall_s": wall,
            "denoise_forwards": fwd,
            "tok_per_fwd": round(len(ids) / fwd, 4) if fwd else None,
            "commit_k_hist": dict(_STATE["commit_k_hist"]),
            "pred": pred, "gold": gold,
            "correct": (pred is not None and pred == gold),
            "answer_ids": ids, "gen_text": text,
            "counters": stats, "verify": verify(stats),
        }
        fh.write(json.dumps(turn) + "\n"); fh.flush(); os.fsync(fh.fileno())
        c = stats or {}
        print(f"[cad] idx{rec['idx']:4d} n={len(ids)} fin={fin} fwd={fwd} "
              f"tpf={turn['tok_per_fwd']} khist={turn['commit_k_hist']} "
              f"proj={c.get('value_projection_events')} corr={int(turn['correct'])} "
              f"wall={wall}", flush=True)
    fh.close()
    print("[cad] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[cad] FATAL:", repr(e), flush=True)
        traceback.print_exc()
        raise
