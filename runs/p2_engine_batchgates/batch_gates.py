#!/usr/bin/env python3
"""BATCH-CORRECTNESS GATES for the FLARE hybrid-clean engine (RL-rollout regime).

Question: does the engine batch CORRECTLY at bs>1 (the RL-rollout throughput
regime), or is there cross-request state contamination? All prior certification
was bs=1. FLARE forces per-request variable draft widths + per-request GDN
snapshot/restore; a single forward mixes denoising and committing rows across
requests. This harness runs the correctness GATES that must pass BEFORE any
throughput number.

Two configs (BENCH_CONFIG):
  C  (clean control): enforce_eager (CUDAGRAPH off) + APC off. Isolates PURE
     batching correctness (bookkeeping + GDN state + attention). The only
     difference between bs=1 and bs=8 here is the batch dimension itself, so a
     byte divergence = a real batching defect (no cudagraph-bucket / APC-share
     confound). BIDIR_PROBE stays ON (the certified decode semantics).
  P  (production): CUDAGRAPH on + APC on. The real serving regime. Divergences
     here that are ABSENT in C are cudagraph/APC fp-residue (benign near-tie
     flips), which the companion-invariance discriminator confirms.

Gates:
  1 batch-invariance : 8 turns, engine bs=1 (each alone, cold) vs engine bs=8
                       (all concurrent). Every bs=8 output byte-identical to its
                       own bs=1 output. ANY divergence -> auto-run the
                       CONTAMINATION discriminator (is the divergence dependent on
                       WHICH requests are co-batched? yes=contamination/STOP;
                       no=batch-size fp-nondeterminism/benign).
  2 mixed-length     : 8 turns spanning prompt_len 467..2647 concurrently; same
                       gate.
  3 GDN isolation    : 2 concurrent requests, per-forward whole-GDN-cache
                       fingerprint before/after. Pure-denoise batched forward =>
                       cache byte-identical (advance-by-0 across all requests).
                       Mixed forward => only committing requests' state rows
                       change; no denoise request's rows change (no cross-write).
  4 audits           : projected==0, verify_invariants, per-request counters sane
                       under batch (collected from gates 1/2/5).
  5 seeded temp=0.7  : per-request seeds at bs=8 -> each request byte-reproducible
                       across 2 batched runs.

One heavy process; RAM cage. Incremental JSONL per gate.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_V1", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
VLLM_WS = Path("/home/mark/shared/vllm_p2_pr42406")
MODEL = ROOT / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
REF = ROOT / "runs/p2_engine_nevertrain/nevertrain_ref.json"
OUTDIR = ROOT / "runs/p2_engine_batchgates"

CONFIG = os.environ.get("BENCH_CONFIG", "C").strip().upper()
assert CONFIG in ("C", "P"), CONFIG
# C = clean control (enforce_eager, CUDAGRAPH off); P = production (CUDAGRAPH on).
# APC stays ON in BOTH: the certified align/mamba_block_size=1024 config REQUIRES
# enable_prefix_caching (vLLM rejects mamba-block-size without APC). The only
# difference C vs P is the CUDA-graph path -- so C isolates pure batching
# correctness from the cudagraph batch-bucket numeric confound. Cold references
# are established with reset_prefix_cache() between the bs=1 and bs=N passes.
CUDAGRAPH = "1" if CONFIG == "P" else "0"
APC = True
os.environ["VLLM_FLARE_CUDAGRAPH"] = CUDAGRAPH
os.environ.setdefault("VLLM_FLARE_BIDIR_PROBE", "1")

import parity_audit_flare_engine as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import qwen3_5_flare as QF  # noqa: E402
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402
from eval_flare_northstar_matched import decode_text, trim_scored_assistant  # noqa: E402

SEED = int(os.environ.get("BENCH_SEED", "20260701"))
MARGIN = int(os.environ.get("BENCH_MARGIN", "16"))
BAND = int(os.environ.get("VLLM_QWEN3_5_FLARE_READONLY_BAND", "4"))

# ---------------------------------------------------------------------------
# Instrumentation (all via monkeypatch; the FLARE source is unchanged).
# ---------------------------------------------------------------------------
LEDGER = []            # per-forward batch record (batch sizes + phase mix)
PER_SLOT_STATS = {}    # slot -> latest decoder stats (live)
SLOT_TO_REQID = {}     # slot -> vllm req_id (from add_request)
RELEASED_STATS = {}    # req_id -> final decoder stats (captured at completion)
GDN = {"on": False, "fp_before": None, "attr": None, "probe_rows": None, "records": []}


def _stats_of(dec):
    s = dec.stats
    return {
        "forwards": int(s.forwards),
        "fsm_committed_tokens": int(s.fsm_committed_tokens),
        "value_tokens": int(s.value_tokens),
        "structural_model_tokens": int(s.structural_model_tokens),
        "value_projection_events": int(s.value_projection_events),
        "model_chosen_tokens": int(s.model_chosen_tokens),
        "generated_tokens": int(s.generated_tokens),
    }


_real_step = QF.Qwen3_5FlareSampler._hybrid_clean_step


def _patched_step(self, shifted, block_logits, decode_slots, decode_idx,
                  decode_indices_np, decode_slots_np, valid_len_np,
                  is_committing, num_reqs, input_batch):
    committing = is_committing.detach().cpu().tolist()
    slots = [int(s) for s in decode_slots_np.tolist()]
    ndec = len(slots)
    ncommit = int(sum(1 for c in committing if c))
    LEDGER.append({
        "num_reqs": int(num_reqs),
        "num_decode": ndec,
        "num_commit": ncommit,
        "num_denoise": ndec - ncommit,
        "mixed": (ncommit > 0 and (ndec - ncommit) > 0),
        "slots": slots,
    })
    ret = _real_step(self, shifted, block_logits, decode_slots, decode_idx,
                     decode_indices_np, decode_slots_np, valid_len_np,
                     is_committing, num_reqs, input_batch)
    for slot in slots:
        dec = self._hc_decoders.get(slot)
        if dec is not None:
            PER_SLOT_STATS[slot] = _stats_of(dec)
    return ret


QF.Qwen3_5FlareSampler._hybrid_clean_step = _patched_step

_real_add = QF.Qwen3_5FlareModelState.add_request


def _patched_add(self, req_index, new_req_data):
    SLOT_TO_REQID[int(req_index)] = new_req_data.req_id
    return _real_add(self, req_index, new_req_data)


QF.Qwen3_5FlareModelState.add_request = _patched_add

_real_release = QF.Qwen3_5FlareSampler.release_hybrid_clean_slot


def _patched_release(self, slot_idx):
    dec = self._hc_decoders.get(int(slot_idx))
    if dec is not None:
        reqid = SLOT_TO_REQID.get(int(slot_idx))
        if reqid is not None:
            RELEASED_STATS[reqid] = _stats_of(dec)
    return _real_release(self, slot_idx)


QF.Qwen3_5FlareSampler.release_hybrid_clean_slot = _patched_release


# ---- GDN per-forward active-row fingerprint (gate 3; gated by GDN["on"]) ----
# Memory-frugal: fingerprint ONLY the union of active requests' state rows
# (checkpoint slot +/- BAND). A forward only touches rows in its own metadata,
# so "active-rows unchanged" == "whole cache unchanged" for that forward, and
# the few-row fp64 checksum costs ~nothing (the full-cache fp64 copy OOMs at
# gpu_mem_util 0.82).
def _fp_rows(caches, rows_dev):
    """2-moment checksum of the given state slots: [L, len(rows), 2] on CPU."""
    out = []
    for (conv, ssm) in caches:
        c = conv.index_select(0, rows_dev).reshape(rows_dev.shape[0], -1).to(torch.float64)
        s = ssm.index_select(0, rows_dev).reshape(rows_dev.shape[0], -1).to(torch.float64)
        m1 = c.sum(1) + s.sum(1)
        m2 = (c * c).sum(1) + (s * s).sum(1)
        out.append(torch.stack([m1, m2], dim=1))
    return torch.stack(out, 0).cpu()


def _regions_and_rows(self, input_batch, block_tables, kv_cache_config, num_slots):
    num_reqs = input_batch.num_reqs
    slots = input_batch.idx_mapping[:num_reqs]
    is_commit = self.diffusion_states.is_encoder_phase[slots]
    mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
    gid = mamba_group_ids[0]
    full_bt = block_tables[gid][:num_reqs]
    seq_lens = input_batch.seq_lens[:num_reqs]
    state_bt = mamba_get_block_table_tensor(
        full_bt, seq_lens, mamba_spec, self.cache_config.mamba_cache_mode)
    return {
        "slots": [int(x) for x in slots.detach().cpu().numpy().tolist()],
        "is_commit": [bool(x) for x in is_commit.detach().cpu().numpy().tolist()],
        "rows_per_req": state_bt.detach().cpu().numpy().tolist(),
        "num_reqs": int(num_reqs),
    }


_real_prep = QF.Qwen3_5FlareModelState.prepare_attn


def _patched_prep(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                  attn_groups, kv_cache_config, for_capture=False):
    md = _real_prep(self, input_batch, cudagraph_mode, block_tables,
                    slot_mappings, attn_groups, kv_cache_config, for_capture)
    if GDN["on"] and not for_capture and input_batch.num_reqs > 0:
        try:
            caches = self._gdn_caches(kv_cache_config)
            num_slots = int(caches[0][0].shape[0])
            attr = _regions_and_rows(self, input_batch, block_tables,
                                     kv_cache_config, num_slots)
            # union of per-request regions (checkpoint slot +/- BAND)
            probe = set()
            for i in range(attr["num_reqs"]):
                for rr in np.array(attr["rows_per_req"][i]).flatten().tolist():
                    rr = int(rr)
                    if rr < 0:
                        continue
                    probe |= set(range(max(0, rr - BAND), min(num_slots, rr + BAND + 1)))
            probe_rows = sorted(probe)
            rows_dev = torch.tensor(probe_rows, dtype=torch.long,
                                    device=caches[0][0].device)
            GDN["fp_before"] = _fp_rows(caches, rows_dev)
            GDN["probe_rows"] = probe_rows
            GDN["attr"] = attr
        except Exception as e:  # noqa: BLE001
            GDN["fp_before"] = None
            GDN["attr"] = {"error": repr(e)}
    return md


QF.Qwen3_5FlareModelState.prepare_attn = _patched_prep

_real_post = QF.Qwen3_5FlareModelState.postprocess_state


def _patched_post(self, idx_mapping, num_sampled, num_computed_tokens=None):
    ret = _real_post(self, idx_mapping, num_sampled, num_computed_tokens)
    if GDN["on"] and GDN.get("fp_before") is not None and isinstance(GDN.get("attr"), dict) and "error" not in GDN["attr"]:
        try:
            caches = self._gdn_caches()
            probe_rows = GDN["probe_rows"]
            rows_dev = torch.tensor(probe_rows, dtype=torch.long,
                                    device=caches[0][0].device)
            fp_after = _fp_rows(caches, rows_dev)
            fp_before = GDN["fp_before"]
            diff = (fp_before != fp_after).any(dim=2).any(dim=0)
            changed = set(probe_rows[int(i)]
                          for i in torch.nonzero(diff).flatten().tolist())
            attr = GDN["attr"]
            num_slots = int(caches[0][0].shape[0])

            def region(row):
                return set(range(max(0, row - BAND), min(num_slots, row + BAND + 1)))

            denoise_rows, commit_rows = set(), set()
            denoise_anchors, commit_anchors = set(), set()
            com_reqs, den_reqs = [], []
            for i in range(attr["num_reqs"]):
                rrows = [int(x) for x in np.array(attr["rows_per_req"][i]).flatten()
                         if int(x) >= 0]
                reg = set()
                for rr in rrows:
                    reg |= region(rr)
                entry = {"slot": attr["slots"][i], "anchors": rrows, "band_rows": sorted(reg)}
                if attr["is_commit"][i]:
                    commit_rows |= reg
                    commit_anchors |= set(rrows)
                    com_reqs.append(entry)
                else:
                    denoise_rows |= reg
                    denoise_anchors |= set(rrows)
                    den_reqs.append(entry)
            # band-level (loose, has FALSE POSITIVES when anchors are within
            # 2*BAND of each other -- a commit's own footprint lands in the
            # neighbour's band):
            cross_band = sorted(changed & denoise_rows)
            # anchor-level (exact, authoritative): did any DENOISE request's own
            # checkpoint anchor row actually change? This is the true isolation
            # test -- immune to band overlap.
            cross_anchor = sorted(changed & denoise_anchors)
            unattributed = sorted(changed - denoise_rows - commit_rows)
            GDN["records"].append({
                "num_reqs": attr["num_reqs"],
                "num_commit": len(com_reqs),
                "num_denoise": len(den_reqs),
                "changed_slots": sorted(changed),
                "denoise_anchors": sorted(denoise_anchors),
                "commit_anchors": sorted(commit_anchors),
                "cross_write_band_slots": cross_band,
                "cross_write_anchor_slots": cross_anchor,
                "unattributed_changed_slots": unattributed,
                "pure_denoise": len(com_reqs) == 0,
                "pure_denoise_cache_identical": (len(com_reqs) == 0 and len(changed) == 0),
                "isolated": len(cross_anchor) == 0,
                "isolated_band": len(cross_band) == 0,
                "commit_reqs": com_reqs,
                "denoise_reqs": den_reqs,
            })
        except Exception as e:  # noqa: BLE001
            GDN["records"].append({"error": repr(e)})
    GDN["fp_before"] = None
    GDN["attr"] = None
    GDN["probe_rows"] = None
    return ret


QF.Qwen3_5FlareModelState.postprocess_state = _patched_post


# ---------------------------------------------------------------------------
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
    all_recs = list(records.values())
    mask_id = int(all_recs[0]["mask_id"])
    os.environ["VLLM_QWEN3_5_FLARE_MASK"] = str(mask_id)
    block_size = int(all_recs[0]["block_size"])
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)

    from vllm import SamplingParams

    adapter = H.build_engine_adapter(
        "vllm", vllm_workspace=VLLM_WS, model_path=str(MODEL),
        canvas_length=block_size, decode_mode="hybrid_clean", seed=SEED,
    )
    adapter._engine_kwargs.update({
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": APC,
    })
    t_boot = time.time()
    engine = adapter._build_engine()
    boot_s = round(time.time() - t_boot, 1)
    vc = engine.llm_engine.vllm_config
    print(f"[bg] CONFIG={CONFIG} booted boot_s={boot_s} "
          f"enforce_eager={vc.model_config.enforce_eager} "
          f"cudagraph={getattr(vc.compilation_config,'cudagraph_mode',None)} "
          f"apc={vc.cache_config.enable_prefix_caching} "
          f"max_num_seqs={vc.scheduler_config.max_num_seqs} "
          f"bidir={os.environ.get('VLLM_FLARE_BIDIR_PROBE')}", flush=True)

    def make_sp(rec, temp, seed):
        tools = [{"type": "function", "function": {"name": n, "parameters": p}}
                 for n, p in (rec["schemas"] or {}).items()]
        return SamplingParams(
            max_tokens=rec["n_ref"] + MARGIN, temperature=temp, top_p=1.0, seed=seed,
            stop_token_ids=sorted(int(x) for x in rec["stop_token_ids"]),
            extra_args={"decode_policy": "hybrid_clean", "tools": tools,
                        "grammar_topk": int(rec["grammar_topk"])},
        )

    def gen_batch(recs, temp=0.0, seeds=None, reset=True):
        """Submit len(recs) prompts CONCURRENTLY in one generate() call.
        Returns list of (token_ids, finish_reason) aligned to recs order."""
        if reset:
            try:
                engine.reset_prefix_cache()
            except Exception:
                pass
        if seeds is None:
            seeds = [SEED] * len(recs)
        prompts = [{"prompt_token_ids": list(r["prompt_ids"])} for r in recs]
        sps = [make_sp(r, temp, s) for r, s in zip(recs, seeds)]
        LEDGER.clear()
        outs = engine.generate(prompts, sps)
        results = []
        for o in outs:
            out0 = o.outputs[0]
            ids = [int(x) for x in out0.token_ids]
            results.append((ids, getattr(out0, "finish_reason", None), o.request_id))
        return results

    def ledger_summary():
        L = list(LEDGER)
        if not L:
            return {}
        nr = [x["num_reqs"] for x in L]
        nd = [x["num_decode"] for x in L]
        import collections
        return {
            "forwards": len(L),
            "num_reqs_max": max(nr),
            "num_reqs_hist": dict(sorted(collections.Counter(nr).items())),
            "num_decode_max": max(nd),
            "num_decode_hist": dict(sorted(collections.Counter(nd).items())),
            "mixed_phase_forwards": sum(1 for x in L if x["mixed"]),
            "multi_req_forwards": sum(1 for x in L if x["num_reqs"] > 1),
            "multi_decode_forwards": sum(1 for x in L if x["num_decode"] > 1),
        }

    def per_req_audit(results):
        # Read the final live per-slot decoder stats for the slots that were
        # active in the just-completed batched gen (LEDGER holds only that gen's
        # forwards -- gen_batch clears it at start). One slot == one request in
        # the batch; no fragile req_id mapping needed.
        active_slots = sorted(set(s for rec in LEDGER for s in rec["slots"]))
        rows = []
        for slot in active_slots:
            st = PER_SLOT_STATS.get(slot)
            rows.append({
                "slot": slot, "stats": st,
                "verify": verify(st) if st else {"ok": False},
                "proj0": (st or {}).get("value_projection_events") == 0 if st else False,
            })
        return rows

    # =====================================================================
    # GATE 1 + GATE 2: batch-invariance and mixed-length batch-invariance.
    # =====================================================================
    def invariance_gate(name, sel_turns, out_path):
        recs = [records[t] for t in sel_turns]
        print(f"[bg] {name}: turns={sel_turns} "
              f"prompt_lens={[r['prompt_len'] for r in recs]} "
              f"n_ref={[r['n_ref'] for r in recs]}", flush=True)
        # Reference bs=1: each turn alone, cold prefix.
        ref = {}
        for r in recs:
            res = gen_batch([r], temp=0.0, reset=True)
            ref[r["global_turn"]] = res[0][0]
        # Batched bs=N: all concurrent, cold prefix.
        RELEASED_STATS.clear()
        bres = gen_batch(recs, temp=0.0, reset=True)
        led = ledger_summary()
        audit = per_req_audit(bres)
        rows = []
        n_match = 0
        divergent = []
        for (ids, fin, rid), r in zip(bres, recs):
            gt = r["global_turn"]
            ref_ids = ref[gt]
            match = ids == ref_ids
            n_match += int(match)
            fd = None
            if not match:
                m = min(len(ids), len(ref_ids))
                fd = next((i for i in range(m) if ids[i] != ref_ids[i]), m)
                divergent.append((gt, r))
            # score the batched output
            txt = trim_scored_assistant(decode_text(tok, torch.tensor(ids, dtype=torch.long)))
            sc = score_tool_calls(txt, r["tools"], r["gold_block"])
            rows.append({
                "gate": name, "global_turn": gt, "source_family": r["source_family"],
                "prompt_len": r["prompt_len"], "n_ref": r["n_ref"],
                "n_bs1": len(ref_ids), "n_bsN": len(ids),
                "byte_identical_bsN_eq_bs1": match, "first_divergence": fd,
                "bsN_tok_at_fd": (ids[fd] if fd is not None and fd < len(ids) else None),
                "bs1_tok_at_fd": (ref_ids[fd] if fd is not None and fd < len(ref_ids) else None),
                "finish": fin,
                "valid_tool_call": bool(sc.get("valid_tool_call")),
                "exact_arguments": bool(sc.get("exact_arguments")),
            })
        with out_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        summ = {
            "gate": name, "config": CONFIG, "n_turns": len(recs),
            "byte_identical": n_match, "total": len(recs),
            "PASS_byte_invariance": n_match == len(recs),
            "batch_ledger": led, "per_req_audit": audit,
        }
        (OUTDIR / f"{CONFIG}_{name}_summary.json").write_text(json.dumps(summ, indent=2))
        print(f"[bg] {name}: byte_identical={n_match}/{len(recs)} "
              f"ledger_maxreq={led.get('num_reqs_max')} "
              f"multi_decode_fwd={led.get('multi_decode_forwards')} "
              f"mixed_phase_fwd={led.get('mixed_phase_forwards')}", flush=True)
        return summ, divergent, ref, recs

    # ---- divergence discriminator -------------------------------------------
    def discriminator(divergent_recs, all_recs_in_batch, ref):
        """For each divergent request A, gather the facts that separate benign
        batch-shape fp-nondeterminism (non-batch-invariant GEMM/attn kernels
        flipping a near-tie; NOT a bug) from true cross-request contamination (B's
        data leaking into A; a bug):
          * batch_determinism: same batch run twice -> is A byte-reproducible?
            (contamination via a data race would typically break this; a stable
            kernel-fp flip is reproducible).
          * companion_dependence: A batched with DIFFERENT companions -> does A
            change? (both benign shape-fp AND contamination are companion/shape
            dependent, so this alone does NOT decide; the deciders are gate3 state
            isolation + whether the flip is a single near-tie value token with the
            exact_arguments verdict preserved).
        The verdict here is deliberately NOT self-adjudicating; gate3 decides."""
        out = []
        pool = [r for r in all_recs if r["global_turn"] not in
                {x[0] for x in divergent_recs}]
        for gt, A in divergent_recs:
            comp1 = [c for c in all_recs_in_batch if c["global_turn"] != gt][:3]
            comp2 = [c for c in pool if c["global_turn"] != gt][:3]
            if len(comp2) < 1:
                comp2 = comp1[::-1]
            b1 = gen_batch([A] + comp1, temp=0.0, reset=True)
            b1b = gen_batch([A] + comp1, temp=0.0, reset=True)  # same batch twice
            b2 = gen_batch([A] + comp2, temp=0.0, reset=True)
            a1, a1b, a2 = b1[0][0], b1b[0][0], b2[0][0]
            out.append({
                "global_turn": gt,
                "companions_1": [c["global_turn"] for c in comp1],
                "companions_2": [c["global_turn"] for c in comp2],
                "batch_determinism_same_batch_twice": a1 == a1b,
                "A_out_batch1_eq_batch2": a1 == a2,
                "A_batch1_eq_bs1": a1 == ref[gt],
                "A_batch2_eq_bs1": a2 == ref[gt],
                "note": "companion/shape dependent; adjudicate via gate3 isolation "
                        "+ near-tie value-token class (see gate2.jsonl exact_arguments)",
            })
        return out

    # Select gate-1 turns: 8 across families, moderate prompts (heavy co-batch).
    by_len = sorted(all_recs, key=lambda r: r["prompt_len"])
    fams = {}
    for r in all_recs:
        fams.setdefault(r["source_family"], []).append(r)
    # gate1: 8 moderate turns (prompt_len ~ 500..1200), diverse families
    mod = [r for r in by_len if 500 <= r["prompt_len"] <= 1300]
    gate1_turns = [r["global_turn"] for r in mod[:8]] if len(mod) >= 8 else \
        [r["global_turn"] for r in by_len[:8]]
    # gate2: 8 turns spanning the full length range (very different lengths)
    idxs = np.linspace(0, len(by_len) - 1, 8).astype(int).tolist()
    gate2_turns = [by_len[i]["global_turn"] for i in idxs]

    s1, div1, ref1, recs1 = invariance_gate("gate1_invariance", gate1_turns,
                                             OUTDIR / f"{CONFIG}_gate1.jsonl")
    disc1 = discriminator(div1, recs1, ref1) if div1 else []
    s1["discriminator"] = disc1
    (OUTDIR / f"{CONFIG}_gate1_summary.json").write_text(json.dumps(s1, indent=2))
    if disc1:
        print(f"[bg] gate1 discriminator: {json.dumps(disc1)}", flush=True)

    s2, div2, ref2, recs2 = invariance_gate("gate2_mixedlen", gate2_turns,
                                             OUTDIR / f"{CONFIG}_gate2.jsonl")
    disc2 = discriminator(div2, recs2, ref2) if div2 else []
    s2["discriminator"] = disc2
    (OUTDIR / f"{CONFIG}_gate2_summary.json").write_text(json.dumps(s2, indent=2))
    if disc2:
        print(f"[bg] gate2 discriminator: {json.dumps(disc2)}", flush=True)

    # =====================================================================
    # GATE 3: GDN state isolation probe (2 concurrent requests).
    # =====================================================================
    def run_gdn_probe(name, turns):
        g3recs = [records[t] for t in turns]
        GDN["records"].clear()
        GDN["on"] = True
        _ = gen_batch(g3recs, temp=0.0, reset=True)
        GDN["on"] = False
        recs3 = list(GDN["records"])
        with (OUTDIR / f"{CONFIG}_{name}_forwards.jsonl").open("w") as fh:
            for rec in recs3:
                fh.write(json.dumps(rec) + "\n")
        ok_recs = [r for r in recs3 if "error" not in r]
        errs = [r for r in recs3 if "error" in r]
        pure = [r for r in ok_recs if r["pure_denoise"]]
        pure_multi = [r for r in pure if r["num_reqs"] > 1]
        mixed = [r for r in ok_recs if not r["pure_denoise"]]
        cross_anchor = [r for r in ok_recs if not r["isolated"]]          # authoritative
        cross_band = [r for r in ok_recs if not r["isolated_band"]]       # loose (overlap FP)
        maxreq = max((r["num_reqs"] for r in ok_recs), default=0)
        res = {
            "name": name, "turns": turns,
            "total_forwards_probed": len(ok_recs),
            "n_errors": len(errs), "error_examples": errs[:2],
            "max_batch_probed": maxreq,
            "multireq_forwards": sum(1 for r in ok_recs if r["num_reqs"] > 1),
            "pure_denoise_forwards": len(pure),
            "pure_denoise_multireq_forwards": len(pure_multi),
            "pure_denoise_cache_identical": sum(1 for r in pure if r["pure_denoise_cache_identical"]),
            "pure_denoise_multireq_cache_identical": sum(
                1 for r in pure_multi if r["pure_denoise_cache_identical"]),
            "mixed_forwards": len(mixed),
            "cross_write_anchor_forwards": len(cross_anchor),
            "cross_write_band_forwards_FP": len(cross_band),
            "PASS_no_cross_write": len(cross_anchor) == 0 and len(errs) == 0,
            "PASS_pure_denoise_advance0": (
                all(r["pure_denoise_cache_identical"] for r in pure)
                if pure else None),
            "cross_write_anchor_examples": cross_anchor[:3],
        }
        print(f"[bg] {name}: probed={len(ok_recs)} maxbatch={maxreq} "
              f"pure_denoise={len(pure)}(multi={len(pure_multi)}) "
              f"advance0={res['pure_denoise_cache_identical']}/{len(pure)} "
              f"mixed={len(mixed)} cross_anchor={len(cross_anchor)} "
              f"cross_band_FP={len(cross_band)} errs={len(errs)} "
              f"PASS={res['PASS_no_cross_write']}", flush=True)
        return res

    # (a) task-literal 2-request probe; (b) the ACTUAL diverging 8-req gate2 batch.
    g3a = run_gdn_probe("gate3_2req", gate1_turns[:2])
    g3b = run_gdn_probe("gate3_gate2batch", gate2_turns)
    g3 = {
        "gate": "gate3_gdn_isolation", "config": CONFIG,
        "probe_2req": g3a, "probe_gate2batch_8req": g3b,
        "PASS_no_cross_write": g3a["PASS_no_cross_write"] and g3b["PASS_no_cross_write"],
        "PASS_pure_denoise_advance0": (
            (g3a["PASS_pure_denoise_advance0"] in (True, None))
            and (g3b["PASS_pure_denoise_advance0"] in (True, None))),
    }
    (OUTDIR / f"{CONFIG}_gate3_summary.json").write_text(json.dumps(g3, indent=2))

    # =====================================================================
    # GATE 5: seeded temp=0.7 reproducibility at bs=8 (production only, but
    #         run in both; determinism should hold regardless of config).
    # =====================================================================
    g5_turns = gate1_turns
    g5recs = [records[t] for t in g5_turns]
    per_seeds = [SEED + i for i in range(len(g5recs))]  # per-request distinct seeds
    RELEASED_STATS.clear()
    r_a = gen_batch(g5recs, temp=0.7, seeds=per_seeds, reset=True)
    audit5 = per_req_audit(r_a)
    r_b = gen_batch(g5recs, temp=0.7, seeds=per_seeds, reset=True)
    rows5 = []
    n_repro = 0
    for (ida, fina, rida), (idb, finb, ridb), r in zip(r_a, r_b, g5recs):
        repro = ida == idb
        n_repro += int(repro)
        txt = trim_scored_assistant(decode_text(tok, torch.tensor(ida, dtype=torch.long)))
        sc = score_tool_calls(txt, r["tools"], r["gold_block"])
        rows5.append({
            "gate": "gate5_seeded_temp07", "global_turn": r["global_turn"],
            "source_family": r["source_family"], "seed": per_seeds[g5recs.index(r)],
            "n_a": len(ida), "n_b": len(idb),
            "byte_reproducible_a_eq_b": repro,
            "bounded": (fina == "stop" and len(ida) <= r["n_ref"] + MARGIN),
            "valid_tool_call": bool(sc.get("valid_tool_call")),
            "finish_a": fina, "finish_b": finb,
        })
    with (OUTDIR / f"{CONFIG}_gate5.jsonl").open("w") as fh:
        for row in rows5:
            fh.write(json.dumps(row) + "\n")
    g5 = {
        "gate": "gate5_seeded_temp07", "config": CONFIG, "n_turns": len(g5recs),
        "byte_reproducible": n_repro, "total": len(g5recs),
        "all_bounded": all(x["bounded"] for x in rows5),
        "all_valid": all(x["valid_tool_call"] for x in rows5),
        "PASS_seeded_repro": n_repro == len(g5recs),
        "per_req_audit": audit5,
    }
    (OUTDIR / f"{CONFIG}_gate5_summary.json").write_text(json.dumps(g5, indent=2))
    print(f"[bg] gate5: byte_reproducible={n_repro}/{len(g5recs)} "
          f"bounded_all={g5['all_bounded']} valid_all={g5['all_valid']}", flush=True)

    # =====================================================================
    # GATE 6: greedy DETERMINISM probe -- is the gate2 (gt176) divergence
    #   batch-introduced kernel nondeterminism (atomic reduction) or intrinsic?
    #   bs=1 x3 (each cold) vs the SAME bs=8 gate2 batch x3 (each cold). A stable
    #   deterministic contamination would reproduce; a nondeterministic kernel
    #   reduction at batch>1 would not. Reports which requests / positions flip.
    # =====================================================================
    torch.cuda.empty_cache()
    det_turns = gate2_turns
    detrecs = [records[t] for t in det_turns]
    # bs=1 x3 per turn
    bs1_runs = {t: [] for t in det_turns}
    for _ in range(3):
        for r in detrecs:
            res = gen_batch([r], temp=0.0, reset=True)
            bs1_runs[r["global_turn"]].append(res[0][0])
    # bs=8 (same batch) x3
    bs8_runs = []
    for _ in range(3):
        bres = gen_batch(detrecs, temp=0.0, reset=True)
        bs8_runs.append({r["global_turn"]: ids for (ids, _, _), r in zip(bres, detrecs)})
    det_rows = []
    for r in detrecs:
        t = r["global_turn"]
        a, b, c = bs1_runs[t]
        bs1_det = (a == b == c)
        x, y, z = bs8_runs[0][t], bs8_runs[1][t], bs8_runs[2][t]
        bs8_det = (x == y == z)
        # first divergence among the 3 bs8 runs (if any)
        fd8 = None
        if not bs8_det:
            m = min(len(x), len(y), len(z))
            fd8 = next((i for i in range(m) if not (x[i] == y[i] == z[i])), m)
        det_rows.append({
            "global_turn": t, "source_family": r["source_family"],
            "prompt_len": r["prompt_len"], "n_ref": r["n_ref"],
            "bs1_deterministic_x3": bs1_det,
            "bs8_deterministic_x3": bs8_det,
            "bs8_first_nondet_pos": fd8,
        })
    with (OUTDIR / f"{CONFIG}_gate6_determinism.jsonl").open("w") as fh:
        for row in det_rows:
            fh.write(json.dumps(row) + "\n")
    g6 = {
        "gate": "gate6_determinism", "config": CONFIG, "turns": det_turns,
        "bs1_deterministic_all": all(x["bs1_deterministic_x3"] for x in det_rows),
        "bs8_deterministic_all": all(x["bs8_deterministic_x3"] for x in det_rows),
        "bs1_nondet_turns": [x["global_turn"] for x in det_rows if not x["bs1_deterministic_x3"]],
        "bs8_nondet_turns": [x["global_turn"] for x in det_rows if not x["bs8_deterministic_x3"]],
        "rows": det_rows,
    }
    (OUTDIR / f"{CONFIG}_gate6_summary.json").write_text(json.dumps(g6, indent=2))
    print(f"[bg] gate6 determinism: bs1_det_all={g6['bs1_deterministic_all']} "
          f"bs8_det_all={g6['bs8_deterministic_all']} "
          f"bs1_nondet={g6['bs1_nondet_turns']} bs8_nondet={g6['bs8_nondet_turns']}", flush=True)

    # =====================================================================
    # GATE 4: audits aggregated across gates 1/2/5.
    # =====================================================================
    def audit_pass(auditlist):
        if not auditlist:
            return {"n": 0}
        proj0 = all(a["proj0"] for a in auditlist)
        verok = all(a["verify"]["ok"] for a in auditlist)
        return {
            "n": len(auditlist),
            "all_proj0": proj0,
            "all_verify_ok": verok,
            "any_missing_stats": any(a["stats"] is None for a in auditlist),
        }
    g4 = {
        "gate": "gate4_audits", "config": CONFIG,
        "gate1_batched": audit_pass(s1["per_req_audit"]),
        "gate2_batched": audit_pass(s2["per_req_audit"]),
        "gate5_batched": audit_pass(audit5),
    }
    g4["PASS_audits"] = all(
        v.get("all_proj0", True) and v.get("all_verify_ok", True)
        and not v.get("any_missing_stats", False)
        for k, v in g4.items() if isinstance(v, dict) and v.get("n", 0) > 0
    )
    (OUTDIR / f"{CONFIG}_gate4_summary.json").write_text(json.dumps(g4, indent=2))
    print(f"[bg] gate4 audits: {json.dumps(g4)}", flush=True)

    # ---- overall ----
    overall = {
        "config": CONFIG,
        "gate1_invariance_PASS": s1["PASS_byte_invariance"],
        "gate1_byte_identical": f"{s1['byte_identical']}/{s1['total']}",
        "gate1_discriminator": disc1,
        "gate2_mixedlen_PASS": s2["PASS_byte_invariance"],
        "gate2_byte_identical": f"{s2['byte_identical']}/{s2['total']}",
        "gate2_discriminator": disc2,
        "gate3_no_cross_write_PASS": g3["PASS_no_cross_write"],
        "gate3_pure_denoise_advance0_PASS": g3["PASS_pure_denoise_advance0"],
        "gate3_2req": g3["probe_2req"],
        "gate3_gate2batch": g3["probe_gate2batch_8req"],
        "gate4_audits_PASS": g4["PASS_audits"],
        "gate5_seeded_repro_PASS": g5["PASS_seeded_repro"],
        "gate6_bs1_deterministic_all": g6["bs1_deterministic_all"],
        "gate6_bs8_deterministic_all": g6["bs8_deterministic_all"],
        "gate6_bs1_nondet_turns": g6["bs1_nondet_turns"],
        "gate6_bs8_nondet_turns": g6["bs8_nondet_turns"],
        "gate1_ledger": s1["batch_ledger"],
        "gate2_ledger": s2["batch_ledger"],
    }
    (OUTDIR / f"{CONFIG}_OVERALL.json").write_text(json.dumps(overall, indent=2))
    print(f"[bg] OVERALL {CONFIG}: {json.dumps(overall)}", flush=True)
    print("[bg] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[bg] ERROR:", repr(e), flush=True)
        traceback.print_exc()
        raise
