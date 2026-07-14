#!/usr/bin/env python
# coding=utf-8
"""X.2 — AR-self-distillation of the read-phase conditional (SECTION X.2).

Three subcommands, one leakage-safe pipeline:

  requests   read the KEEPER windowed pool -> X.1(b)-style read-arg-centered windows;
             for each SUPERVISED read_file limit/offset value slot emit ONE generation
             request (the token prefix ENDING at the <parameter=limit|offset> marker,
             i.e. the exact "read-phase state" the arg is derived from). Also reserves a
             handful of keeper episodes as a HELD KL-probe set (excluded from training).
             HARD leakage asserts: (1) reconstruct the 113-id eval holdout byte-identically
             + sha256 == pinned, (2) NO training/probe episode_id in the eval holdout,
             (3) train episodes and KL-probe episodes are DISJOINT.

  assemble   splice each slot's AR-self-distilled value tokens (from x2_gen_client.py) in
             place of the keeper's own value, oversample high-context read windows, add a
             small keeper general-coverage retention slice, write the LMFlow text_only
             pretok json + manifest (leakage re-asserted, sha256, weight note).

  kl-probe   build the s2_kl_probe.json-schema held probe from the RESERVED keeper
             episodes' NON-READ assistant turns (drift detector for the broad policy — the
             X.1 loop-halt pathology — NOT the read conditional we intend to move).

X.1 LESSON (binding): no narrow high-weight patch. X.2's lever is the ON-POLICY target
(the same-weights AR conditional, deterministic) + conservative weight, NOT a 5x span
weight. The read-window value class is trained at the STANDARD O2 derived-value weight
(2.0); the differential up-weight X.1 used (5.0) is deliberately dropped.
"""
import os
import re
import sys
import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path("/home/mark/qwen_diffusion")

# native qwen3_xml <parameter=NAME>value</parameter> marker subsequences (Qwen3.5 tokenizer)
LIM = (27, 15704, 28, 9226, 29)   # <parameter=limit>
OFF = (27, 15704, 28, 3075, 29)   # <parameter=offset>
END = (510, 15704, 29)            # </parameter>
SLOT_KEY = {LIM: "limit", OFF: "offset"}
STARTS = (LIM, OFF)


def match_at(seq, sub, i):
    return tuple(seq[i:i + len(sub)]) == sub


def find_read_slots(ids, labels):
    """Yield (marker_start, val_start, val_end, close_end, slot_key) for each read-window
    arg whose VALUE is supervised. val_start = first value token (after the marker),
    val_end = index of the </parameter> that closes it, close_end = after </parameter>."""
    n = len(ids)
    i = 0
    while i < n:
        opened = None
        for s in STARTS:
            if match_at(ids, s, i):
                opened = s
                break
        if opened is None:
            i += 1
            continue
        val_start = i + len(opened)
        j = val_start
        val_end = n
        while j < n:
            if match_at(ids, END, j):
                val_end = j
                break
            j += 1
        value_supervised = any(labels[k] != -100 for k in range(val_start, min(val_end, n)))
        close_end = (val_end + len(END)) if val_end < n else n
        if value_supervised:
            yield (i, val_start, val_end, close_end, SLOT_KEY[opened])
        i = close_end


def labels_from_row(r):
    ids = r["input_ids"]
    labels = r.get("labels")
    if labels is None:
        labels = [-100] * len(ids)
        for sp in r.get("assistant_spans", []):
            a, b = (sp if isinstance(sp, (list, tuple)) else (sp["start"], sp["end"]))
            for k in range(a, b):
                if 0 <= k < len(ids):
                    labels[k] = ids[k]
    return ids, labels


# ---------------------------------------------------------------------------
# eval-holdout reconstruction (byte-identical to leakage_audit.py / expand_frontier.py)
# ---------------------------------------------------------------------------
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
        raise SystemExit(f"[x2] HOLDOUT HASH MISMATCH: reconstructed {sha} != pinned {pinned}")
    return holdout, sha


def episode_to_instance_id(episode_id):
    """keeper episode_id 'python__mypy-10154' -> swe-bench instance id 'python__mypy-10154'.
    The keeper pool episode_id IS the instance id (repo__name-number). Normalize defensively."""
    return episode_id.strip()


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def cmd_requests(args):
    rng = random.Random(args.seed)
    MAXLEN = args.max_len
    src = REPO / args.src
    holdout, holdout_sha = reconstruct_holdout()

    # group rows by episode so we can reserve whole episodes for the KL probe
    rows_by_ep = defaultdict(list)
    all_eps = []
    for line in open(src):
        r = json.loads(line)
        ep = r["episode_id"]
        if ep not in rows_by_ep:
            all_eps.append(ep)
        rows_by_ep[ep].append(r)

    # HARD leakage assert (1): no keeper episode is an eval-holdout id
    leaked = sorted({ep for ep in all_eps if episode_to_instance_id(ep) in holdout})
    if leaked:
        raise SystemExit(f"[x2] LEAKAGE: {len(leaked)} keeper episodes in eval holdout: {leaked[:10]}")

    # reserve KL-probe episodes: pick episodes that CONTAIN non-read assistant turns
    # (so the probe has general-policy content), deterministically.
    eps_sorted = sorted(all_eps)
    rng.shuffle(eps_sorted)
    kl_probe_eps = set(eps_sorted[:args.kl_probe_episodes])
    train_eps = [ep for ep in eps_sorted if ep not in kl_probe_eps]
    assert not (set(train_eps) & kl_probe_eps), "[x2] train/kl-probe episode overlap"

    reqs = []
    total_slots = 0
    depths = []
    slot_keys = Counter()
    n_read_windows = 0
    for ep in train_eps:
        for r in rows_by_ep[ep]:
            ids, labels = labels_from_row(r)
            for (mstart, vs, ve, cend, key) in find_read_slots(ids, labels):
                total_slots += 1
                slot_keys[key] += 1
                end = cend
                start = max(0, end - MAXLEN)
                sub_ids = ids[start:end]
                sub_labels = labels[start:end]
                if not any(l != -100 for l in sub_labels):
                    continue
                # slot offsets within the sub-window
                v0, v1, m0 = vs - start, ve - start, mstart - start
                if v0 < 0 or v1 > len(sub_ids) or v0 >= v1:
                    continue
                depth = m0  # within-window context position of the read call
                depths.append(depth)
                prefix_ids = sub_ids[:v0]  # ends AT the <parameter=...> marker -> the read-phase state
                if len(prefix_ids) < 4:
                    continue
                req_id = f"{ep}::{r['conversation_id']}::{mstart}::{key}"
                reqs.append({
                    "req_id": req_id,
                    "episode_id": ep,
                    "conversation_id": r["conversation_id"],
                    "slot_key": key,
                    "depth": depth,
                    "prefix_ids": prefix_ids,
                    "sub_ids": sub_ids,
                    "sub_labels": sub_labels,
                    "v0": v0, "v1": v1,
                    "keeper_value_ids": sub_ids[v0:v1],
                })
                n_read_windows += 1

    outdir = REPO / args.out_run
    outdir.mkdir(parents=True, exist_ok=True)
    reqpath = outdir / "gen_requests.jsonl"
    with open(reqpath, "w") as fh:
        for rq in reqs:
            fh.write(json.dumps(rq) + "\n")
    # separate light-weight request file (prefix only) for the generation client
    genpath = outdir / "gen_prefixes.jsonl"
    with open(genpath, "w") as fh:
        for rq in reqs:
            fh.write(json.dumps({"req_id": rq["req_id"], "slot_key": rq["slot_key"],
                                 "prefix_ids": rq["prefix_ids"]}) + "\n")
    # reserve KL-probe episode rows
    klpath = outdir / "kl_probe_episode_rows.jsonl"
    with open(klpath, "w") as fh:
        for ep in sorted(kl_probe_eps):
            for r in rows_by_ep[ep]:
                fh.write(json.dumps({"episode_id": ep, "conversation_id": r["conversation_id"],
                                     "input_ids": r["input_ids"],
                                     "assistant_spans": r.get("assistant_spans", [])}) + "\n")

    depths.sort()
    manifest = {
        "spec": "k_raise_campaign_design.md SECTION X.2 (AR-self-distillation, requests phase)",
        "src": str(src.relative_to(REPO)),
        "max_len": MAXLEN,
        "holdout_sha256": holdout_sha, "holdout_n": len(holdout),
        "leakage_assert_train_eps_in_holdout": 0,
        "n_keeper_episodes_total": len(all_eps),
        "n_train_episodes": len(train_eps),
        "n_kl_probe_episodes": len(kl_probe_eps),
        "kl_probe_episodes": sorted(kl_probe_eps),
        "train_kl_disjoint": True,
        "total_supervised_read_slots": total_slots,
        "n_generation_requests": len(reqs),
        "slot_keys": dict(slot_keys),
        "readarg_depth_min": depths[0] if depths else 0,
        "readarg_depth_med": depths[len(depths) // 2] if depths else 0,
        "readarg_depth_max": depths[-1] if depths else 0,
        "gen_requests": str(reqpath.relative_to(REPO)),
        "gen_prefixes": str(genpath.relative_to(REPO)),
    }
    (outdir / "x2_requests_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def cmd_assemble(args):
    rng = random.Random(args.seed)
    MAXLEN = args.max_len
    outrun = REPO / args.out_run
    reqs = {json.loads(l)["req_id"]: json.loads(l) for l in open(outrun / "gen_requests.jsonl")}
    targets = {}
    for l in open(outrun / "ar_targets.jsonl"):
        t = json.loads(l)
        targets[t["req_id"]] = t
    holdout, holdout_sha = reconstruct_holdout()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / args.tokenizer), trust_remote_code=True)

    read_instances = []      # (ids, labels, depth)
    used_eps = set()
    n_ok = n_drop_nogen = n_drop_parse = n_drop_unbounded = 0
    ar_vs_keeper_same = 0
    for req_id, rq in reqs.items():
        t = targets.get(req_id)
        if t is None or not t.get("completion_text"):
            n_drop_nogen += 1
            continue
        val_text = _extract_value_text(t["completion_text"], rq["slot_key"])
        if val_text is None:
            n_drop_parse += 1
            continue
        # tokenize the AR value chunk (boundaries are newlines -> stable standalone tokenization)
        ar_val_ids = tok.encode(val_text, add_special_tokens=False)
        if not ar_val_ids:
            n_drop_parse += 1
            continue
        sub_ids = rq["sub_ids"]; sub_labels = rq["sub_labels"]
        v0, v1 = rq["v0"], rq["v1"]
        if sub_ids[v0:v1] == ar_val_ids:
            ar_vs_keeper_same += 1
        new_ids = sub_ids[:v0] + ar_val_ids + sub_ids[v1:]
        new_labels = sub_labels[:v0] + list(ar_val_ids) + sub_labels[v1:]
        # left-truncate to MAXLEN (keep the read call at the right edge)
        if len(new_ids) > MAXLEN:
            cut = len(new_ids) - MAXLEN
            new_ids = new_ids[cut:]
            new_labels = new_labels[cut:]
        if not any(l != -100 for l in new_labels):
            continue
        read_instances.append((new_ids, new_labels, rq["depth"]))
        used_eps.add(rq["episode_id"])
        n_ok += 1

    # oversample high-context read windows (X.1(b) curriculum)
    expanded = []
    for (ids, labs, depth) in read_instances:
        reps = args.oversample if depth >= args.hictx else 1
        for _ in range(reps):
            expanded.append((ids, labs))

    # general-coverage retention tiles (keeper, unchanged) from the SAME train episodes
    general = []
    for req_id, rq in reqs.items():
        pass
    # build general tiles from the source rows (head tile) restricted to train episodes
    src = REPO / args.src
    train_eps = used_eps  # only episodes that contributed a read window
    for line in open(src):
        r = json.loads(line)
        if r["episode_id"] not in train_eps:
            continue
        ids, labels = labels_from_row(r)
        seg_ids = ids[:MAXLEN]; seg_labels = labels[:MAXLEN]
        if any(l != -100 for l in seg_labels):
            general.append((seg_ids, seg_labels))
    rng.shuffle(general)
    n_gen_target = int(len(expanded) * args.general_frac)
    general = general[:n_gen_target]

    instances = []
    for (ids, labs) in expanded:
        instances.append({"text": "", "input_ids": ids, "labels": labs})
    for (ids, labs) in general:
        instances.append({"text": "", "input_ids": ids, "labels": labs})
    rng.shuffle(instances)

    # HARD leakage re-assert on the final dataset
    leaked = sorted({ep for ep in used_eps if episode_to_instance_id(ep) in holdout})
    if leaked:
        raise SystemExit(f"[x2] LEAKAGE in assembled dataset: {leaked[:10]}")

    outdir = REPO / args.out_data
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "x2_train.json"
    payload = {"type": "text_only", "instances": instances}
    out_json.write_text(json.dumps(payload))
    sha = hashlib.sha256(out_json.read_bytes()).hexdigest()

    lens = sorted(len(x["input_ids"]) for x in instances)
    manifest = {
        "spec": "k_raise_campaign_design.md SECTION X.2 AR-self-distillation (assemble phase)",
        "objective": "read_file limit/offset value slots carry the SAME-WEIGHTS AR teacher's "
                     "greedy value (on-policy, deterministic); keeper structure/file_path kept; "
                     "read turn stays K=1 sequential supervised.",
        "distill_weight_note": "READ_WINDOW_ARG loss weight = 2.0 (== O2 standard derived-value "
                               "weight); X.1 used 5.0 (narrow high-weight patch) — dropped per the "
                               "X.1 lesson. The lever is the on-policy target, not the weight.",
        "src": str(src.relative_to(REPO)),
        "out_json": str(out_json.relative_to(REPO)), "out_json_sha256": sha,
        "max_len": MAXLEN, "hictx": args.hictx, "oversample": args.oversample,
        "general_frac": args.general_frac,
        "holdout_sha256": holdout_sha, "leakage_used_eps_in_holdout": 0,
        "n_train_episodes_used": len(used_eps),
        "n_read_slots_ok": n_ok,
        "n_read_windows_after_oversample": len(expanded),
        "n_general_tiles": len(general),
        "total_instances": len(instances),
        "drops": {"no_generation": n_drop_nogen, "parse_fail": n_drop_parse,
                  "unbounded": n_drop_unbounded},
        "ar_target_equals_keeper_value": ar_vs_keeper_same,
        "ar_target_differs_from_keeper": n_ok - ar_vs_keeper_same,
        "instance_len_min": lens[0] if lens else 0,
        "instance_len_med": lens[len(lens) // 2] if lens else 0,
        "instance_len_max": lens[-1] if lens else 0,
    }
    # NOTE: manifest MUST NOT be a .json in the dataset dir — the lmflow loader ingests
    # every *.json in DATASET_DIR and rejects any without a "type" key. Use .meta (X.1
    # convention) and mirror a copy into the run dir for the record.
    (outdir / "x2_mix_manifest.meta").write_text(json.dumps(manifest, indent=2))
    (REPO / args.out_run / "x2_mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _extract_value_text(completion_text, slot_key):
    """From an AR completion that begins right after <parameter=limit|offset>, return the
    value chunk (the tokens between the marker and the next </parameter>), preserving the
    surrounding newlines that the native format uses (<parameter=limit>\\nVALUE\\n</parameter>).
    Require a bounded numeric value for limit (the grounding target). Returns None on failure."""
    txt = completion_text
    idx = txt.find("</parameter>")
    if idx == -1:
        return None
    chunk = txt[:idx]
    # the numeric value must be present and integer-like (the read window arg is numeric)
    m = re.search(r"-?\d+", chunk)
    if m is None:
        return None
    # keep native shape: leading/trailing single newline around the value
    return "\n" + m.group(0) + "\n"


# ---------------------------------------------------------------------------
# kl-probe
# ---------------------------------------------------------------------------
def cmd_kl_probe(args):
    """Build s2_kl_probe.json-schema probe from RESERVED keeper episodes' NON-READ assistant
    turns. Measures KL-to-base drift on the broad policy (the X.1 loop-halt pathology), NOT
    on the read conditional we intend to move."""
    outrun = REPO / args.out_run
    rows = [json.loads(l) for l in open(outrun / "kl_probe_episode_rows.jsonl")]
    probe = []
    ctx_cap = args.ctx_cap
    for r in rows:
        ids = r["input_ids"]
        labels = [-100] * len(ids)
        spans = r.get("assistant_spans", [])
        for sp in spans:
            a, b = (sp if isinstance(sp, (list, tuple)) else (sp["start"], sp["end"]))
            for k in range(a, b):
                if 0 <= k < len(ids):
                    labels[k] = ids[k]
        for sp in spans:
            a, b = (sp if isinstance(sp, (list, tuple)) else (sp["start"], sp["end"]))
            span_ids = ids[a:b]
            # SKIP read turns (this probe watches the OTHER behavior)
            if any(match_at(span_ids, s, i) for i in range(max(0, len(span_ids) - 5)) for s in STARTS):
                continue
            if b - a < args.min_answer or b - a > args.max_answer:
                continue
            prompt_ids = ids[max(0, a - ctx_cap):a]
            answer_ids = ids[a:b]
            if len(prompt_ids) < 8 or not answer_ids:
                continue
            probe.append({"prompt_ids": prompt_ids, "answer_ids": answer_ids,
                          "prompt_len": len(prompt_ids), "episode_id": r["episode_id"]})
            if len([p for p in probe if p["episode_id"] == r["episode_id"]]) >= args.per_episode:
                break
    # cap total
    probe = probe[:args.max_probes]
    outpath = outrun / "x2_kl_probe.json"
    outpath.write_text(json.dumps({"probe": probe,
                                   "spec": "X.2 held KL-to-base drift probe (non-read keeper turns from reserved episodes)"}))
    print(json.dumps({"n_probe": len(probe),
                      "episodes": sorted({p["episode_id"] for p in probe}),
                      "ctx_cap": ctx_cap, "out": str(outpath.relative_to(REPO)),
                      "answer_len_med": sorted(len(p["answer_ids"]) for p in probe)[len(probe) // 2] if probe else 0}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("requests")
    r.add_argument("--src", default="data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl")
    r.add_argument("--out-run", default="runs/kraise_reconvert_iter2_x2")
    r.add_argument("--max-len", type=int, default=4096)
    r.add_argument("--kl-probe-episodes", type=int, default=8)
    r.add_argument("--seed", type=int, default=81102)
    r.set_defaults(func=cmd_requests)

    a = sub.add_parser("assemble")
    a.add_argument("--src", default="data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl")
    a.add_argument("--out-run", default="runs/kraise_reconvert_iter2_x2")
    a.add_argument("--out-data", default="data/swe_x2_ar_distill_mix")
    a.add_argument("--tokenizer", default="models/qwen3.5-9b-fastdllm-mswe-S-iter2-vllm-bf16")
    a.add_argument("--max-len", type=int, default=4096)
    a.add_argument("--hictx", type=int, default=2048)
    a.add_argument("--oversample", type=int, default=3)
    a.add_argument("--general-frac", type=float, default=0.25)
    a.add_argument("--seed", type=int, default=81102)
    a.set_defaults(func=cmd_assemble)

    k = sub.add_parser("kl-probe")
    k.add_argument("--out-run", default="runs/kraise_reconvert_iter2_x2")
    k.add_argument("--ctx-cap", type=int, default=1536)
    k.add_argument("--min-answer", type=int, default=8)
    k.add_argument("--max-answer", type=int, default=320)
    k.add_argument("--per-episode", type=int, default=3)
    k.add_argument("--max-probes", type=int, default=24)
    k.set_defaults(func=cmd_kl_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
