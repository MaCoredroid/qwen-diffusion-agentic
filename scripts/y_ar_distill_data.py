#!/usr/bin/env python
# coding=utf-8
"""SECTION Y — full-trajectory AR-self-distillation DATA build (Y.1 / binding lessons).

The X.1/X.2 pilots proved the read-arg-ONLY splice is corrosive (X.2 KILL-T1 49->0/63).
Y is the principled fix: distill the diffusion twin's K=1 clean-stream readout onto the
SAME weights' deterministic AR-greedy conditional across the WHOLE assistant trajectory,
at serving-like length (12288), with UNIFORM weight and coverage-by-SAMPLING (value/tool
always, reasoning sampled) -- NOT a narrow high-weight patch.

Two subcommands:

  targets   (GPU) reserve KL-probe episodes (disjoint), HARD leakage assert, then for each
            TRAIN window run the merged bf16 model's STRICTLY-CAUSAL (SDPA) teacher-forced
            forward and take argmax at every position -> the token the same-weights AR-greedy
            decode emits from that exact prefix (Y.1 PRIMARY = CE-on-AR-greedy token-ids;
            this is the clean-stream L_AR conditional serving's K=1 readout commits). Emit
            Y labels = AR-greedy target at each SUPERVISED position, kept by coverage:
              - TOOL-CALL region (<tool_call>..</tool_call>, incl. function name + every
                <parameter=..> value body) = ALWAYS covered (the emission conditional X.2
                collapsed);
              - REASONING prose = SAMPLED (p_reason, deterministic by seed) -- so late
                reasoning is trained near-cap without over-constraining the high-entropy
                policy to one greedy rollout.
            UNIFORM weight 1.0 (no per-class multiplier; differentiation is coverage). NO
            read-slice oversampling, NO narrow upweights (X.1 lesson). Writes the LMFlow
            text_only json (input_ids verbatim + Y labels) + manifest (leakage sha, class
            breakdown, ar!=keeper count).

  kl-probe  (CPU) build the s2_kl_probe.json-schema held probe from the RESERVED episodes'
            assistant turns, INCLUDING tool-call emission turns (binding lesson 3: assert
            >=1/3 of probe turns contain a tool call -- the X.2 probe watched only non-read
            turns and missed the multi-param arg-emission collapse).
"""
import os
import sys
import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(REPO / "scripts"))


def _rel(p):
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)

TOOL_CALL_OPEN = 248058   # <tool_call>
TOOL_CALL_CLOSE = 248059  # </tool_call>


# --------------------------------------------------------------------------- #
# eval-holdout reconstruction (byte-identical to x2_ar_self_distill.py)        #
# --------------------------------------------------------------------------- #
HERE = REPO / "runs/swe_datagen_s1"
MANIFEST = REPO / "data/swe_sft_pool/pool_manifest.json"
PIN = HERE / ".eval_holdout_sha256"
RING_SRC = {
    "tier0_20": REPO / "runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json",
    "tier1_100": Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/swe-bench-tier1-verified-instances-20260520.json"),
}


def _ids_from(path):
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        return set(d.get("instance_ids", []))
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return set(d)
        return {r.get("instance_id") for r in d if isinstance(r, dict) and r.get("instance_id")}
    return set()


def reconstruct_holdout():
    man = json.loads(MANIFEST.read_text())
    inner5 = set(man["held_out_rings"]["inner5"]["ids"])
    holdout = inner5 | _ids_from(RING_SRC["tier0_20"]) | _ids_from(RING_SRC["tier1_100"])
    sha = hashlib.sha256("\n".join(sorted(holdout)).encode()).hexdigest()
    pinned = PIN.read_text().strip()
    if sha != pinned:
        raise SystemExit(f"[y] HOLDOUT HASH MISMATCH: reconstructed {sha} != pinned {pinned}")
    return holdout, sha


def labels_from_row(r):
    """supervised label positions = assistant_spans (already ONLY the target turns this
    window owns; non-owned turns are loss-masked / not in assistant_spans -- see
    build_windowed_dataset.py)."""
    ids = r["input_ids"]
    labels = [-100] * len(ids)
    for sp in r.get("assistant_spans", []):
        a, b = (sp if isinstance(sp, (list, tuple)) else (sp["start"], sp["end"]))
        for k in range(a, b):
            if 0 <= k < len(ids):
                labels[k] = ids[k]
    return ids, labels


def tool_call_mask(ids):
    """True at positions inside a <tool_call>..</tool_call> region (inclusive of both
    markers). Covers function name, <parameter=..> markers and every value body."""
    n = len(ids)
    mask = [False] * n
    inside = False
    for i in range(n):
        t = ids[i]
        if t == TOOL_CALL_OPEN:
            inside = True
        mask[i] = inside
        if t == TOOL_CALL_CLOSE:
            inside = False
    return mask


def _pos_keep(seed, conv_id, pos, rate):
    """deterministic per-position reasoning keep decision."""
    h = hashlib.sha256(f"{seed}|{conv_id}|{pos}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / float(1 << 64)
    return x < rate


# --------------------------------------------------------------------------- #
# targets (GPU)                                                               #
# --------------------------------------------------------------------------- #
def cmd_targets(args):
    import torch
    from transformers import AutoModelForCausalLM
    from swe_sft_arm1_qlora_train import install_sdpa_attention

    rng = random.Random(args.seed)
    src = REPO / args.src
    holdout, holdout_sha = reconstruct_holdout()

    rows_by_ep = defaultdict(list)
    all_eps = []
    for line in open(src):
        r = json.loads(line)
        ep = r["episode_id"]
        if ep not in rows_by_ep:
            all_eps.append(ep)
        rows_by_ep[ep].append(r)

    # HARD leakage assert (1): no windowed episode is an eval-holdout id
    leaked = sorted({ep for ep in all_eps if ep.strip() in holdout})
    if leaked:
        raise SystemExit(f"[y] LEAKAGE: {len(leaked)} windowed episodes in eval holdout: {leaked[:10]}")

    # reserve KL-probe episodes (disjoint), deterministic
    eps_sorted = sorted(all_eps)
    rng.shuffle(eps_sorted)
    kl_probe_eps = set(eps_sorted[:args.kl_probe_episodes])
    train_eps = [ep for ep in eps_sorted if ep not in kl_probe_eps]
    assert not (set(train_eps) & kl_probe_eps), "[y] train/kl-probe episode overlap"

    outrun = REPO / args.out_run
    outrun.mkdir(parents=True, exist_ok=True)
    # reserve KL-probe episode rows for the kl-probe subcommand
    with open(outrun / "kl_probe_episode_rows.jsonl", "w") as fh:
        for ep in sorted(kl_probe_eps):
            for r in rows_by_ep[ep]:
                fh.write(json.dumps({"episode_id": ep, "conversation_id": r["conversation_id"],
                                     "input_ids": r["input_ids"],
                                     "assistant_spans": r.get("assistant_spans", [])}) + "\n")

    # ---- load teacher (same weights, strictly-causal SDPA = clean-stream L_AR) ----
    print(f"[y] loading teacher (bf16, SDPA-causal) {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    install_sdpa_attention(model)
    model.eval()
    inner = model.model
    lm_head = model.lm_head
    device = next(model.parameters()).device
    C = args.logits_chunk

    @torch.no_grad()
    def ar_greedy(ids):
        """teacher-forced argmax at every position: greedy[i] predicts token i+1."""
        x = torch.tensor([ids], dtype=torch.long, device=device)
        hs = inner(input_ids=x, use_cache=False).last_hidden_state[0]  # [L,H]
        out = torch.empty(hs.size(0), dtype=torch.long, device="cpu")
        for s in range(0, hs.size(0), C):
            logits = lm_head(hs[s:s + C])          # [c, V] bf16
            out[s:s + C] = logits.argmax(dim=-1).cpu()
        return out.tolist()

    instances = []
    n_win = 0
    n_sup = n_kept = n_tool = n_reason_kept = n_reason_total = 0
    ar_diff = ar_same = 0
    lens = []
    t_start = __import__("time").time()
    if args.limit > 0:
        train_eps = train_eps[:args.limit]
    for wi, ep in enumerate(train_eps):
        for r in rows_by_ep[ep]:
            ids, orig = labels_from_row(r)
            L = len(ids)
            if not any(l != -100 for l in orig):
                continue
            if L > args.max_len:  # windowed pool caps at 12286; guard anyway
                ids = ids[:args.max_len]
                orig = orig[:args.max_len]
                L = args.max_len
            greedy = ar_greedy(ids)                # greedy[i] predicts i+1
            tmask = tool_call_mask(ids)
            ylab = [-100] * L
            conv = r["conversation_id"]
            for p in range(1, L):
                if orig[p] == -100:
                    continue
                n_sup += 1
                in_tool = tmask[p]
                if in_tool:
                    keep = True
                    n_tool += 1
                else:
                    n_reason_total += 1
                    keep = _pos_keep(args.seed, conv, p, args.reason_rate)
                    if keep:
                        n_reason_kept += 1
                if not keep:
                    continue
                tgt = greedy[p - 1]                 # AR-greedy target for position p
                ylab[p] = tgt
                n_kept += 1
                if tgt == ids[p]:
                    ar_same += 1
                else:
                    ar_diff += 1
            if not any(l != -100 for l in ylab):
                continue
            instances.append({"text": "", "input_ids": ids, "labels": ylab})
            lens.append(L)
            n_win += 1
        if (wi + 1) % 25 == 0:
            el = __import__("time").time() - t_start
            print(f"[y] {wi + 1}/{len(train_eps)} eps  windows={n_win} kept_labels={n_kept} "
                  f"elapsed={el:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()

    rng.shuffle(instances)
    outdir = REPO / args.out_data
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "y_train.json"
    out_json.write_text(json.dumps({"type": "text_only", "instances": instances}))
    sha = hashlib.sha256(out_json.read_bytes()).hexdigest()
    lens.sort()

    manifest = {
        "spec": "k_raise_campaign_design.md SECTION Y.1 (full-trajectory AR-self-distillation, targets)",
        "objective": "CE on same-weights AR-greedy (strictly-causal clean-stream L_AR) targets over ALL "
                     "supervised assistant positions; uniform weight 1.0; coverage by sampling "
                     "(tool-call region always, reasoning sampled).",
        "teacher": "offline teacher-forced argmax on the merged bf16 model with SDPA strictly-causal "
                   "attention == the clean-stream next-token conditional serving's K=1 readout commits "
                   "(Y.1); deterministic, reproducible, no server.",
        "weight_note": "NO per-class weight (X.1 5.0 / X.2 2.0 both dropped). NO read-slice oversample. "
                       "Differentiation is COVERAGE only (value/tool always vs reasoning sampled).",
        "src": _rel(src),
        "model": args.model,
        "out_json": _rel(out_json), "out_json_sha256": sha,
        "max_len": args.max_len, "reason_rate": args.reason_rate, "seed": args.seed,
        "holdout_sha256": holdout_sha, "holdout_n": len(holdout),
        "leakage_train_eps_in_holdout": 0,
        "n_windowed_episodes_total": len(all_eps),
        "n_train_episodes": len(train_eps),
        "n_kl_probe_episodes": len(kl_probe_eps),
        "kl_probe_episodes": sorted(kl_probe_eps),
        "train_kl_disjoint": True,
        "n_train_windows": n_win,
        "n_supervised_positions": n_sup,
        "n_kept_labels": n_kept,
        "coverage": {"tool_call_always": n_tool,
                     "reasoning_total": n_reason_total, "reasoning_kept": n_reason_kept,
                     "reasoning_keep_frac": round(n_reason_kept / max(1, n_reason_total), 4)},
        "ar_target_equals_keeper": ar_same,
        "ar_target_differs_from_keeper": ar_diff,
        "ar_differ_frac": round(ar_diff / max(1, n_kept), 4),
        "instance_len_min": lens[0] if lens else 0,
        "instance_len_med": lens[len(lens) // 2] if lens else 0,
        "instance_len_max": lens[-1] if lens else 0,
    }
    (REPO / args.out_run / "y_targets_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# kl-probe (CPU)                                                              #
# --------------------------------------------------------------------------- #
def _span_has_tool_call(span_ids):
    return TOOL_CALL_OPEN in span_ids


def cmd_kl_probe(args):
    outrun = REPO / args.out_run
    rows = [json.loads(l) for l in open(outrun / "kl_probe_episode_rows.jsonl")]
    ctx_cap = args.ctx_cap
    tool_probe, plain_probe = [], []
    for r in rows:
        ids = r["input_ids"]
        spans = r.get("assistant_spans", [])
        n_ep_kept = 0
        for sp in spans:
            a, b = (sp if isinstance(sp, (list, tuple)) else (sp["start"], sp["end"]))
            span_ids = ids[a:b]
            if b - a < args.min_answer or b - a > args.max_answer:
                continue
            prompt_ids = ids[max(0, a - ctx_cap):a]
            answer_ids = ids[a:b]
            if len(prompt_ids) < 8 or not answer_ids:
                continue
            rec = {"prompt_ids": prompt_ids, "answer_ids": answer_ids,
                   "prompt_len": len(prompt_ids), "episode_id": r["episode_id"],
                   "has_tool_call": _span_has_tool_call(span_ids)}
            (tool_probe if rec["has_tool_call"] else plain_probe).append(rec)
            n_ep_kept += 1
            if n_ep_kept >= args.per_episode:
                break
    # binding lesson 3: >=1/3 of probe turns contain a tool call. Take at least
    # ceil(max_probes/3) tool-call turns, fill the rest with plain (drift breadth).
    import math
    n_tool_min = math.ceil(args.max_probes / 3)
    random.Random(args.seed).shuffle(tool_probe)
    random.Random(args.seed + 1).shuffle(plain_probe)
    take_tool = tool_probe[:max(n_tool_min, min(len(tool_probe), args.max_probes // 2))]
    remaining = args.max_probes - len(take_tool)
    take_plain = plain_probe[:max(0, remaining)]
    probe = take_tool + take_plain
    random.Random(args.seed + 2).shuffle(probe)
    probe = probe[:args.max_probes]
    n_tool = sum(1 for p in probe if p["has_tool_call"])
    frac = n_tool / max(1, len(probe))
    assert frac >= 1.0 / 3.0, f"[y] KL probe tool-call coverage {frac:.3f} < 1/3 (binding lesson 3)"
    outpath = outrun / "y_kl_probe.json"
    outpath.write_text(json.dumps({
        "probe": [{k: v for k, v in p.items() if k != "has_tool_call"} for p in probe],
        "spec": "SECTION Y held KL-to-base drift probe (reserved-episode assistant turns; "
                "INCLUDES tool-call emission turns per binding lesson 3)",
        "n_probe": len(probe), "n_tool_call_turns": n_tool, "tool_call_frac": round(frac, 4)}))
    print(json.dumps({"n_probe": len(probe), "n_tool_call_turns": n_tool,
                      "tool_call_frac": round(frac, 4),
                      "episodes": sorted({p["episode_id"] for p in probe}),
                      "out": _rel(outpath)}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("targets")
    t.add_argument("--src", default="data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl")
    t.add_argument("--model", default="models/qwen3.5-9b-fastdllm-mswe-S-iter2-merged")
    t.add_argument("--out-run", default="runs/kraise_reconvert_iter2_y")
    t.add_argument("--out-data", default="data/swe_y_ar_distill")
    t.add_argument("--max-len", type=int, default=12288)
    t.add_argument("--reason-rate", type=float, default=0.4)
    t.add_argument("--kl-probe-episodes", type=int, default=8)
    t.add_argument("--logits-chunk", type=int, default=2048)
    t.add_argument("--limit", type=int, default=0, help="cap train episodes (smoke); 0=all")
    t.add_argument("--seed", type=int, default=71201)
    t.set_defaults(func=cmd_targets)

    k = sub.add_parser("kl-probe")
    k.add_argument("--out-run", default="runs/kraise_reconvert_iter2_y")
    k.add_argument("--ctx-cap", type=int, default=1536)
    k.add_argument("--min-answer", type=int, default=8)
    k.add_argument("--max-answer", type=int, default=320)
    k.add_argument("--per-episode", type=int, default=4)
    k.add_argument("--max-probes", type=int, default=30)
    k.add_argument("--seed", type=int, default=71201)
    k.set_defaults(func=cmd_kl_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
