#!/usr/bin/env python
# coding=utf-8
"""X.1(b) — high-context read-arg curriculum: build the read-grounding conversion mix.

SPEC: k_raise_campaign_design.md SECTION X.1(b). The plain iter-2 conversion mix
(flare_redesign_run1_copy_retention_mix) carries ZERO read_file content (measured: 0/5055
instances contain read_file), so the plain conversion's DENOISE stream never trains on
read-arg positions — which is the drift X.1 attacks. This builder produces a read-heavy,
high-context conversion dataset from the SWE keeper windowed pool (real read_file limit/offset
calls, up to ~12k context) as an LMFlow text_only pretok json (input_ids+labels) consumable by
the FLARE two-stream trainer (train_s2_finetune.py passthrough).

Two window classes:
  (A) READ-ARG-CENTERED (the X.1(b) curriculum): for each SUPERVISED limit/offset value in a
      source window, cut a <=MAXLEN sub-window ENDING just after the read call's </parameter>,
      so the read call sits at the HIGHEST within-window context MAXLEN allows (the derivation
      signal — file length + prior reads — is the preceding ~MAXLEN tokens). High-context
      instances (read-arg depth >= HICTX) are oversampled OVERSAMPLE x.
  (B) GENERAL COVERAGE (tool-format retention): head <=MAXLEN tile of each source window, so
      general agentic tool behavior (edit/write/bash) stays in the conversion (KILL-T1 guard).

MAXLEN is bounded by the FLARE two-stream logits materialization [2L, vocab=248320]: at L=8192
the logits alone are 16 GB (OOM on 32 GB); L=4096 keeps them ~8 GB (fits). This is itself an
X.1(b) feasibility ceiling: the two-stream objective cannot train reads near the 28-30k failure
regime on this card; 4096 is 8x the plain-conversion 512 and the max feasible dose.
"""
import os
import json
import argparse
import hashlib
import random
from collections import Counter

LIM = (27, 15704, 28, 9226, 29)   # <parameter=limit>
OFF = (27, 15704, 28, 3075, 29)   # <parameter=offset>
END = (510, 15704, 29)            # </parameter>
STARTS = (LIM, OFF)


def match_at(seq, sub, i):
    return tuple(seq[i:i + len(sub)]) == sub


def find_read_args(ids, labels):
    """Yield (marker_start, value_end_exclusive) for each read-window arg whose VALUE is
    supervised. value_end_exclusive = index just AFTER the closing </parameter>."""
    n = len(ids)
    i = 0
    while i < n:
        opened = None
        for s in STARTS:
            if match_at(ids, s, i):
                opened = len(s)
                break
        if opened is None:
            i += 1
            continue
        val_start = i + opened
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
            yield (i, close_end)
        i = close_end
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/swe_sft_pool/train_swe_sft_windowed.tokenized.jsonl")
    ap.add_argument("--out-dir", default="data/swe_x1_read_grounding_mix")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--hictx", type=int, default=2048, help="read-arg depth >= this -> oversample")
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--general-frac", type=float, default=0.25, help="general tiles as frac of read windows")
    ap.add_argument("--seed", type=int, default=81101)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    MAXLEN = args.max_len

    read_windows = []      # (ids, labels, depth)
    general_windows = []
    src_rows = 0
    total_readargs = 0
    for line in open(args.src):
        r = json.loads(line)
        src_rows += 1
        ids = r["input_ids"]
        labels = r.get("labels")
        if labels is None:
            labels = [-100] * len(ids)
            for sp in r.get("assistant_spans", []):
                a, b = (sp if isinstance(sp, (list, tuple)) else (sp["start"], sp["end"]))
                for k in range(a, b):
                    labels[k] = ids[k]
        # (A) read-arg-centered
        for (mstart, cend) in find_read_args(ids, labels):
            total_readargs += 1
            end = cend
            start = max(0, end - MAXLEN)
            sub_ids = ids[start:end]
            sub_labels = labels[start:end]
            if not any(l != -100 for l in sub_labels):
                continue
            depth = mstart - start  # within-window context position of the read call
            read_windows.append((sub_ids, sub_labels, depth))
        # (B) general coverage: head tile
        seg_ids = ids[:MAXLEN]
        seg_labels = labels[:MAXLEN]
        if any(l != -100 for l in seg_labels):
            general_windows.append((seg_ids, seg_labels))

    # oversample high-context read windows
    expanded = []
    for (sub_ids, sub_labels, depth) in read_windows:
        reps = args.oversample if depth >= args.hictx else 1
        for _ in range(reps):
            expanded.append((sub_ids, sub_labels))
    read_final = expanded

    # subsample general to general_frac of read windows
    n_gen_target = int(len(read_final) * args.general_frac)
    rng.shuffle(general_windows)
    gen_final = general_windows[:n_gen_target]

    instances = []
    for (ids, labs) in read_final:
        instances.append({"text": "", "input_ids": ids, "labels": labs, "_kind": "read"})
    for (ids, labs) in gen_final:
        instances.append({"text": "", "input_ids": ids, "labels": labs, "_kind": "general"})
    rng.shuffle(instances)
    # drop the _kind tag before writing (keep clean text_only schema)
    kinds = Counter(x.pop("_kind") for x in instances)

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "x1_train.json")
    payload = {"type": "text_only", "instances": instances}
    with open(out_json, "w") as fh:
        json.dump(payload, fh)
    sha = hashlib.sha256(open(out_json, "rb").read()).hexdigest()

    depths = sorted(d for (_, _, d) in read_windows)
    lens = sorted(len(x["input_ids"]) for x in instances)
    def med(a):
        return a[len(a) // 2] if a else 0
    manifest = {
        "spec": "k_raise_campaign_design.md SECTION X.1(b) high-context read-arg curriculum",
        "src": args.src, "out_json": out_json, "out_json_sha256": sha,
        "max_len": MAXLEN, "bd_size_note": "trainer pads each row to next multiple of bd_size (32)",
        "hictx_threshold": args.hictx, "oversample": args.oversample, "general_frac": args.general_frac,
        "src_rows": src_rows, "total_supervised_readargs": total_readargs,
        "unique_read_windows": len(read_windows), "read_windows_after_oversample": len(read_final),
        "general_windows": len(gen_final), "total_instances": len(instances),
        "instance_kinds": dict(kinds),
        "readarg_depth_min": depths[0] if depths else 0, "readarg_depth_med": med(depths),
        "readarg_depth_max": depths[-1] if depths else 0,
        "readarg_depth_ge_hictx": sum(d >= args.hictx for d in depths),
        "instance_len_min": lens[0] if lens else 0, "instance_len_med": med(lens), "instance_len_max": lens[-1] if lens else 0,
    }
    with open(os.path.join(args.out_dir, "x1_mix_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
