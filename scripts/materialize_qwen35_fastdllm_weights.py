#!/usr/bin/env python3
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_CANDIDATE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


def hf_snapshot(model_id):
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--"))
    ref = root / "refs/main"
    if ref.exists():
        snap = root / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snap.exists():
            return snap
    snapshots = sorted((root / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def raw_to_candidate_key(key):
    if key.startswith("model.language_model."):
        return "model." + key.removeprefix("model.language_model.")
    if key == "lm_head.weight":
        return key
    if key.startswith("model.visual."):
        return None
    if key.startswith("mtp."):
        return None
    raise ValueError(f"Unhandled Qwen3.5 weight key: {key}")


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def build_plan(raw_snapshot, candidate_dir):
    index_path = raw_snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing raw index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    raw_map = index.get("weight_map") or {}
    kept = {}
    dropped = Counter()
    source_for_candidate = {}
    by_raw_shard = defaultdict(list)

    for raw_key, shard in sorted(raw_map.items()):
        candidate_key = raw_to_candidate_key(raw_key)
        if candidate_key is None:
            if raw_key.startswith("model.visual."):
                dropped["vision"] += 1
            elif raw_key.startswith("mtp."):
                dropped["mtp"] += 1
            else:
                dropped["other"] += 1
            continue
        kept[candidate_key] = shard
        source_for_candidate[candidate_key] = raw_key
        by_raw_shard[shard].append((raw_key, candidate_key))

    expected_shards = sorted(set(raw_map.values()))
    candidate_shards = sorted(set(kept.values()))
    missing_shards = [name for name in expected_shards if not (raw_snapshot / name).exists()]

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_snapshot": str(raw_snapshot),
        "candidate_dir": str(candidate_dir),
        "raw_index": str(index_path),
        "raw_key_count": len(raw_map),
        "kept_key_count": len(kept),
        "dropped_key_count": sum(dropped.values()),
        "dropped_counts": dict(sorted(dropped.items())),
        "expected_raw_shards": expected_shards,
        "candidate_shards": candidate_shards,
        "missing_raw_shards": missing_shards,
        "raw_total_size": (index.get("metadata") or {}).get("total_size"),
        "weight_map": kept,
        "source_for_candidate": source_for_candidate,
        "by_raw_shard": {name: pairs for name, pairs in sorted(by_raw_shard.items())},
    }


def write_candidate_shards(plan):
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except Exception as exc:
        raise RuntimeError(f"safetensors is required: {exc}") from exc

    raw_snapshot = Path(plan["raw_snapshot"])
    candidate_dir = Path(plan["candidate_dir"])
    candidate_dir.mkdir(parents=True, exist_ok=True)

    new_index = {
        "metadata": {"format": "pt", "source": plan["raw_snapshot"]},
        "weight_map": {},
    }
    total_size = 0

    for shard, pairs in plan["by_raw_shard"].items():
        raw_path = raw_snapshot / shard
        out_path = candidate_dir / shard
        tensors = {}
        with safe_open(raw_path, framework="pt", device="cpu") as handle:
            for raw_key, candidate_key in pairs:
                tensor = handle.get_tensor(raw_key)
                tensors[candidate_key] = tensor
                total_size += tensor_nbytes(tensor)
                new_index["weight_map"][candidate_key] = shard
        save_file(tensors, out_path, metadata={"format": "pt"})

    new_index["metadata"]["total_size"] = total_size
    (candidate_dir / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2) + "\n",
        encoding="utf-8",
    )
    return new_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-model", default=DEFAULT_MODEL)
    parser.add_argument("--raw-snapshot", type=Path)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--write", action="store_true", help="Write remapped safetensor shards into candidate-dir.")
    args = parser.parse_args()

    raw_snapshot = args.raw_snapshot or hf_snapshot(args.raw_model)
    if raw_snapshot is None:
        raise SystemExit(f"No local HF snapshot found for {args.raw_model}")

    plan = build_plan(raw_snapshot, args.candidate_dir)
    plan_out = args.plan_out or (args.candidate_dir / "weight_remap_plan.json")
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(json.dumps({k: v for k, v in plan.items() if k != "by_raw_shard"}, indent=2) + "\n", encoding="utf-8")

    if args.write:
        if plan["missing_raw_shards"]:
            print(json.dumps(plan, indent=2))
            raise SystemExit("Cannot write remapped weights; raw safetensor shards are missing.")
        index = write_candidate_shards(plan)
        print(json.dumps({"wrote": True, "candidate_dir": str(args.candidate_dir), "keys": len(index["weight_map"])}, indent=2))
        return

    print(
        json.dumps(
            {
                "wrote": False,
                "plan_out": str(plan_out),
                "raw_snapshot": plan["raw_snapshot"],
                "raw_key_count": plan["raw_key_count"],
                "kept_key_count": plan["kept_key_count"],
                "dropped_counts": plan["dropped_counts"],
                "expected_raw_shards": plan["expected_raw_shards"],
                "missing_raw_shards": plan["missing_raw_shards"],
                "raw_total_size": plan["raw_total_size"],
            },
            indent=2,
        )
    )
    if plan["missing_raw_shards"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
