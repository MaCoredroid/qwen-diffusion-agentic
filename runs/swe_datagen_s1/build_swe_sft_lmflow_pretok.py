#!/usr/bin/env python
"""SWE-SFT arm-1 — convert the serve-exact pre-tokenized keeper dataset into a
pre-tokenized LMFlow `text_only` dataset for the certified two-stream FLARE
trainer (train_s2_finetune.py + FASTDLLM_S2_PRETOK=1).

FAITHFULNESS CONTRACT (mirrors scripts/build_s2_flare_dataset.py):
  * Consumes `data/swe_sft_pool/train_swe_sft.tokenized.jsonl` which stores, per
    keeper, `input_ids` (rendered under the 9B SERVING chat_template, native
    qwen3_xml) and `assistant_spans` = list of [start,end) half-open token index
    ranges whose union is the assistant-target (loss) region.
  * labels[i] = input_ids[i] for i inside any assistant span, else -100.
    (verified: sum(e-s over spans) == n_label_tokens in the source file.)
  * We NEVER re-tokenize a decoded string. The ids are carried verbatim, so the
    SFT distribution == the generation distribution == the eval distribution
    (native-format rule). This bypasses the trainer preset `fast_dllm_v2_native`
    whitespace divergence flagged in the dataset manifest.

VRAM-FORCED LENGTH CAP (measure-not-assume): the design's block_size "up to
32768" materialises full-vocab logits (vocab 248320) ~16 GB/stream at 32k, which
does not fit the 5090's 32 GB alongside a second (noisy) stream. So sequences are
LEFT-truncated to --max-len (keeps the final edit turns, the highest-value SWE
targets; drops the earliest problem-statement/exploration context). Spans are
remapped and clipped; fully-dropped spans are removed. The chosen --max-len is
set == the trainer block_size, so the pretok pad path's `len<=block_size` assert
always holds.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_left_trunc(input_ids, spans, max_len):
    """Left-truncate input_ids to the last max_len tokens; remap [start,end)
    spans into the truncated frame, clipping at 0 and dropping empties."""
    n = len(input_ids)
    if n <= max_len:
        return list(input_ids), [(int(s), int(e)) for s, e in spans]
    off = n - max_len
    ids = list(input_ids[off:])
    out = []
    for s, e in spans:
        ns, ne = int(s) - off, int(e) - off
        ns = max(0, ns)
        if ne <= 0 or ne <= ns:
            continue
        out.append((ns, ne))
    return ids, out


def to_labels(input_ids, spans):
    labels = [-100] * len(input_ids)
    for s, e in spans:
        for i in range(s, e):
            labels[i] = input_ids[i]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized", default="data/swe_sft_pool/train_swe_sft.tokenized.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, required=True,
                    help="== trainer block_size; sequences left-truncated to this.")
    ap.add_argument("--bd-size", type=int, default=32)
    ap.add_argument("--limit-longest", type=int, default=0,
                    help="if >0, keep only the N longest rows (worst-case VRAM smoke).")
    args = ap.parse_args()

    if args.max_len % args.bd_size != 0:
        raise SystemExit(f"max_len {args.max_len} not divisible by bd_size {args.bd_size}")

    root = Path.cwd()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.tokenized) as fh:
        for line in fh:
            rows.append(json.loads(line))

    if args.limit_longest > 0:
        rows.sort(key=lambda r: len(r["input_ids"]), reverse=True)
        rows = rows[: args.limit_longest]

    instances = []
    n_label_before = 0
    n_label_after = 0
    max_len_seen = 0
    n_trunc = 0
    n_zero_label = 0
    for r in rows:
        ids0 = r["input_ids"]
        spans0 = r["assistant_spans"]
        n_label_before += sum(int(e) - int(s) for s, e in spans0)
        ids, spans = remap_left_trunc(ids0, spans0, args.max_len)
        if len(ids0) > args.max_len:
            n_trunc += 1
        kept = sum(e - s for s, e in spans)
        n_label_after += kept
        if kept == 0:
            n_zero_label += 1
        labels = to_labels(ids, spans)
        assert len(ids) == len(labels) <= args.max_len
        assert len(ids) >= 1
        max_len_seen = max(max_len_seen, len(ids))
        instances.append({"text": "", "input_ids": ids, "labels": labels,
                          "conversation_id": r.get("conversation_id", "")})

    # LMFlow text_only schema (consumed by the s2-pretok passthrough tokenize).
    dataset = {"type": "text_only",
               "instances": [{"text": i["text"], "input_ids": i["input_ids"], "labels": i["labels"]}
                             for i in instances]}
    out_json = out_dir / "swe_sft_train.json"
    with open(out_json, "w") as fh:
        json.dump(dataset, fh)
    ds_sha = sha256_file(out_json)

    manifest = {
        "source": args.tokenized,
        "out_json": str(out_json.relative_to(root)),
        "out_json_sha256": ds_sha,
        "n_instances": len(instances),
        "max_len_cap": args.max_len,
        "bd_size": args.bd_size,
        "max_seq_len_seen": max_len_seen,
        "n_rows_left_truncated": n_trunc,
        "n_rows_zero_label_after_trunc": n_zero_label,
        "assistant_label_tokens_before": n_label_before,
        "assistant_label_tokens_after": n_label_after,
        "assistant_label_retention": (n_label_after / n_label_before) if n_label_before else 0.0,
        "limit_longest": args.limit_longest,
        "conversation_ids": [i["conversation_id"] for i in instances],
    }
    # NOTE: the dataset dir must contain ONLY the LMFlow dataset json (the loader
    # scans every *.json in the dir and requires each to carry "type"). Write the
    # manifest as a SIBLING of the dir.
    manifest_path = out_dir.parent / f"{out_dir.name}.manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps({k: v for k, v in manifest.items() if k != "conversation_ids"}, indent=2))


if __name__ == "__main__":
    main()
