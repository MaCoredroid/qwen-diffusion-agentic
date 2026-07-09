#!/usr/bin/env python3
"""Export a Fast-dLLM converted base to vLLM bf16 WITHOUT any adapter merge.

Produces the ARM-2 pre-SFT anchor reference: the stock `qwen3.5-9b-fastdllm-init`
converted base served AR, through the *identical* export pipeline the post-SFT
`M_swe_T` candidate uses (flywheel export_qwen35_9b_fastdllm_vllm), the only
difference being no LoRA merge (lora_pairs = {}). This gives the tightest paired
comparability for the KILL-T1 anchor McNemar: pre = init(+no adapter), post =
init(+SWE adapter), exactly mirroring arm-1 (pre = init+RLv2, post = init+SWE).

Reuses the flywheel module's own functions so the tensor replacement / metadata
copy / index write are bit-for-bit the same code path as the candidate export.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FLYWHEEL = Path("/home/mark/shared/lumoFlyWheel_codex_fork")
sys.path.insert(0, str(FLYWHEEL / "scripts"))
import export_qwen35_9b_fastdllm_vllm as fx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-model", type=Path, default=fx.DEFAULT_OFFICIAL)
    ap.add_argument("--converted-model", type=Path,
                    default=Path("/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    official = args.official_model.resolve()
    converted = args.converted_model.resolve()
    output = args.output.resolve()

    official_index = fx.load_index(official)
    converted_index = fx.load_index(converted)
    official_weight_map = official_index["weight_map"]
    converted_weight_map = converted_index["weight_map"]
    lora_pairs: dict = {}  # NO ADAPTER

    mapped = {
        official_key: converted_key
        for official_key in official_weight_map
        if (converted_key := fx.official_to_converted_key(official_key)) is not None
    }
    missing = sorted(v for v in mapped.values() if v not in converted_weight_map)
    if missing:
        raise RuntimeError(f"official text keys map to missing converted tensors: {missing[:20]}")

    print(f"official={official}")
    print(f"converted={converted}")
    print(f"adapter=<none>")
    print(f"output={output}")
    print(f"mapped_text_tensors={len(mapped)} lora_targets=0 lora_scale=0")

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    copied_metadata = fx.copy_metadata_files(official, converted, output)

    replacement_count = 0
    shard_names = sorted(set(official_weight_map.values()))
    started = time.time()
    for shard_name in shard_names:
        shard_path = official / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors: dict[str, torch.Tensor] = {}
            for key in handle.keys():
                official_tensor = handle.get_tensor(key)
                converted_key = mapped.get(key)
                tensor, replaced, _merged = fx.merge_tensor(
                    key=key,
                    converted_key=converted_key,
                    official_tensor=official_tensor,
                    converted_root=converted,
                    converted_weight_map=converted_weight_map,
                    lora_pairs=lora_pairs,
                    lora_scale=0.0,
                )
                tensors[key] = tensor
                replacement_count += int(replaced)
        save_file(tensors, output / shard_name, metadata=metadata)
        print(f"wrote {shard_name} tensors={len(tensors)} elapsed_s={time.time()-started:.1f}")

    (output / "model.safetensors.index.json").write_text(
        json.dumps(official_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "lumo.qwen35_9b_fastdllm_vllm_export.noadapter.v1",
        "created_at_unix": time.time(),
        "official_model": str(official),
        "converted_model": str(converted),
        "adapter": None,
        "strategy": "official_qwen35_conditional_layout_with_converted_language_model_no_adapter",
        "mapped_text_tensors": len(mapped),
        "replacement_count": replacement_count,
        "lora_targets": [],
        "lora_merge_count": 0,
        "lora_scale": 0.0,
        "copied_metadata_files": copied_metadata,
    }
    (output / "lumo_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done replacement_count={replacement_count} lora_merge_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
