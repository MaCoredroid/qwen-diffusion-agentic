#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_component(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("component must be ALPHA:PATH or NAME=ALPHA:PATH")
    name, rest = text.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError("component must be NAME=ALPHA:PATH")
    alpha_text, path_text = rest.split(":", 1)
    try:
        alpha = float(alpha_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid alpha {alpha_text!r}") from exc
    return name, alpha, Path(path_text)


def load_adapter(path):
    tensor_path = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return load_file(tensor_path), json.loads(config_path.read_text(encoding="utf-8"))


def assert_compatible(base_tensors, component_tensors, name):
    base_keys = set(base_tensors)
    component_keys = set(component_tensors)
    if base_keys != component_keys:
        missing = sorted(base_keys - component_keys)[:20]
        extra = sorted(component_keys - base_keys)[:20]
        raise ValueError(f"{name}: tensor key mismatch missing={missing} extra={extra}")
    for key, value in base_tensors.items():
        other = component_tensors[key]
        if value.shape != other.shape:
            raise ValueError(f"{name}: shape mismatch for {key}: {value.shape} != {other.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument(
        "--component",
        action="append",
        type=parse_component,
        required=True,
        help="NAME=ALPHA:/path/to/adapter_model_dir. Output = base + sum(alpha * (component - base)).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy-tokenizer-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    base_tensors, base_config = load_adapter(args.base_adapter)
    components = []
    for name, alpha, path in args.component:
        tensors, config = load_adapter(path)
        assert_compatible(base_tensors, tensors, name)
        for field in ("r", "lora_alpha", "peft_type", "task_type"):
            if base_config.get(field) != config.get(field):
                raise ValueError(f"{name}: config {field} mismatch: {base_config.get(field)!r} != {config.get(field)!r}")
        components.append((name, alpha, path, tensors))

    blended = {}
    for key, base_value in base_tensors.items():
        out = base_value.detach().clone().to(torch.float32)
        for _name, alpha, _path, tensors in components:
            out.add_((tensors[key].to(torch.float32) - base_value.to(torch.float32)) * alpha)
        blended[key] = out.to(base_value.dtype)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(blended, args.out_dir / "adapter_model.safetensors")
    for filename in ("adapter_config.json", "README.md"):
        src = args.base_adapter / filename
        if src.exists():
            shutil.copy2(src, args.out_dir / filename)
    if args.copy_tokenizer_files:
        for src in args.base_adapter.iterdir():
            if src.name in {"adapter_model.safetensors", "adapter_config.json", "README.md"}:
                continue
            if src.is_file():
                shutil.copy2(src, args.out_dir / src.name)

    manifest = {
        "base_adapter": str(args.base_adapter),
        "components": [
            {"name": name, "alpha": alpha, "adapter": str(path)}
            for name, alpha, path, _tensors in components
        ],
        "tensor_count": len(blended),
        "out_dir": str(args.out_dir),
        "formula": "base + sum(alpha * (component - base))",
    }
    (args.out_dir / "blend_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
