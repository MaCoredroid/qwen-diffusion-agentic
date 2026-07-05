#!/usr/bin/env python
"""S2 pilot — convert the self-trajectory corpus into a pre-tokenized LMFlow
text_only dataset for two-stream FLARE trajectory-consistency distillation.

FAITHFULNESS CONTRACT (spec s2_pilot_design.md sec.2):
  * Targets are the cached K=1 self-trajectory tokens (answer_ids) VERBATIM.
    We NEVER re-tokenize a decoded string (BPE re-tokenization of generated
    numeric content is non-idempotent). input_ids/labels carry the exact ids.
  * input_ids = prompt_ids + answer_ids
  * labels    = [-100]*prompt_len + answer_ids   (prompt clean / loss-masked;
                answer region is both the L_AR target and the M_S mask pool)
  * The dataset is loaded by a monkeypatched passthrough tokenize (see
    train_s2_finetune.py); no LMFlow tokenizer touches these ids.

Also emits a leakage-safe KL-to-base probe (held-out GSM8K-train rows disjoint
from the training corpus and from the 30-gate / 20-retention eval sets).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/s2_pilot/s2_traj_corpus.jsonl")
    ap.add_argument("--manifest", default="runs/s2_pilot/s2_traj_corpus.manifest.json")
    ap.add_argument("--raw-gen", default="runs/s2_pilot/train_gen.jsonl")
    ap.add_argument("--pool", default="runs/s2_pilot/gsm8k_train_prompts.json")
    ap.add_argument("--out-dir", default="runs/s2_pilot/s2_flare_dataset")
    ap.add_argument("--kl-probe-out", default="runs/s2_pilot/s2_kl_probe.json")
    ap.add_argument("--kl-probe-n", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=1152,
                    help="Max sample length; every example must fit (assert). 36*32.")
    ap.add_argument("--bd-size", type=int, default=32)
    ap.add_argument("--mask-id", type=int, default=248077)
    args = ap.parse_args()

    root = Path.cwd()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load corpus ----
    rows = []
    corpus_q = set()
    with open(args.corpus) as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(d)
            corpus_q.add(d["q_norm_sha256"])
    n = len(rows)
    assert n > 0, "empty corpus"

    if args.block_size % args.bd_size != 0:
        raise SystemExit(f"block_size {args.block_size} not divisible by bd_size {args.bd_size}")

    # ---- build instances (pre-tokenized) ----
    instances = []
    max_len = 0
    n_answer_tokens = 0
    for d in rows:
        pids = list(d["prompt_ids"])
        aids = list(d["answer_ids"])
        pl = int(d["prompt_len"])
        na = int(d["n_answer"])
        # hard structural asserts
        assert len(pids) == pl, f"prompt_len mismatch id={d['id']}"
        assert len(aids) == na, f"n_answer mismatch id={d['id']}"
        assert na >= 1, f"empty answer id={d['id']}"
        input_ids = pids + aids
        labels = [-100] * pl + aids
        assert len(input_ids) == len(labels) == pl + na
        total = pl + na
        if total > args.block_size:
            raise SystemExit(
                f"example id={d['id']} len={total} exceeds block_size={args.block_size}; "
                f"raise --block-size (max seen so far {max_len})"
            )
        max_len = max(max_len, total)
        n_answer_tokens += na
        instances.append({"text": "", "input_ids": input_ids, "labels": labels})

    dataset = {"type": "text_only", "instances": instances}
    out_json = out_dir / "s2_train.json"
    with open(out_json, "w") as fh:
        json.dump(dataset, fh)
    ds_sha = sha256_file(out_json)

    # ---- KL-to-base probe (leakage-safe, held-out) ----
    pool = json.load(open(args.pool))
    pool_by_q = {p["q_norm_sha256"]: p for p in pool}
    probe = []
    seen_q = set()
    with open(args.raw_gen) as fh:
        for line in fh:
            if len(probe) >= args.kl_probe_n:
                break
            g = json.loads(line)
            q = g["q_norm_sha256"]
            if q in corpus_q or q in seen_q:
                continue  # disjoint from training corpus + no dup
            if not g.get("correct") or g.get("finish_reason") != "stop":
                continue
            p = pool_by_q.get(q)
            if p is None:
                continue
            pids = list(p["prompt_ids"])
            aids = list(g["answer_ids"])
            if len(pids) + len(aids) > args.block_size:
                continue
            seen_q.add(q)
            probe.append({
                "q_norm_sha256": q,
                "train_idx": g.get("train_idx"),
                "prompt_ids": pids,
                "prompt_len": len(pids),
                "answer_ids": aids,
                "n_answer": len(aids),
            })
    # verify probe disjointness
    assert all(pr["q_norm_sha256"] not in corpus_q for pr in probe), "KL probe leaks into corpus"
    with open(root / args.kl_probe_out, "w") as fh:
        json.dump({"probe": probe, "source": args.raw_gen,
                   "note": "held-out GSM8K-train, disjoint from training corpus + gate/retention"}, fh, indent=2)

    # ---- manifest ----
    src_manifest = json.load(open(args.manifest)) if os.path.exists(args.manifest) else {}
    manifest = {
        "artifact": "s2_flare_dataset",
        "purpose": "pre-tokenized two-stream FLARE trajectory-consistency training set",
        "source_corpus": args.corpus,
        "source_corpus_sha256": src_manifest.get("hashes", {}).get("training_corpus_jsonl_sha256"),
        "n_examples": n,
        "block_size": args.block_size,
        "bd_size": args.bd_size,
        "mask_id": args.mask_id,
        "max_example_len": max_len,
        "total_answer_target_tokens": n_answer_tokens,
        "dataset_json": str(out_json.relative_to(root)),
        "dataset_json_sha256": ds_sha,
        "labels_rule": "input_ids=prompt_ids+answer_ids; labels=-100*prompt_len + answer_ids (no re-tokenization)",
        "leakage_dedupe_inherited": src_manifest.get("leakage_dedupe", {}),
        "kl_probe": {
            "path": args.kl_probe_out,
            "n": len(probe),
            "disjoint_from_corpus": True,
        },
    }
    # NOTE: the dataset dir is globbed for *.json by LMFlow; keep ONLY the
    # training json inside it. Manifest goes to the parent dir.
    man_path = out_dir.parent / (out_dir.name + "_manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps({
        "n_examples": n,
        "max_example_len": max_len,
        "block_size": args.block_size,
        "dataset_json": str(out_json),
        "dataset_json_sha256": ds_sha[:16],
        "total_answer_target_tokens": n_answer_tokens,
        "kl_probe_n": len(probe),
        "kl_probe_qs": [pr["q_norm_sha256"][:12] for pr in probe],
    }, indent=2))


if __name__ == "__main__":
    main()
