#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
MASK_TOKEN = "|<MASK>|"


def clip(text, limit=6000):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... clipped {len(text) - limit} chars"


def run_json(python, code, args=None, timeout=90):
    args = args or []
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        proc = subprocess.run(
            [str(python), "-c", code, *map(str, args)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stdout": "",
            "stderr": "",
        }

    payload = {
        "returncode": proc.returncode,
        "stdout": clip(proc.stdout),
        "stderr": clip(proc.stderr),
    }
    if proc.returncode != 0:
        payload["ok"] = False
        return payload
    try:
        decoded = json.loads(proc.stdout)
    except Exception as exc:
        payload["ok"] = False
        payload["error_type"] = type(exc).__name__
        payload["error"] = f"stdout was not JSON: {exc}"
        return payload
    if isinstance(decoded, dict):
        decoded.setdefault("ok", True)
        decoded.setdefault("stderr", clip(proc.stderr))
        return decoded
    return {"ok": True, "value": decoded, "stderr": clip(proc.stderr)}


def hf_cache_root(model_id):
    return Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--"))


def latest_snapshot(model_id):
    root = hf_cache_root(model_id)
    if not root.exists():
        return None
    ref = root / "refs/main"
    if ref.exists():
        sha = ref.read_text(encoding="utf-8").strip()
        snap = root / "snapshots" / sha
        if snap.exists():
            return snap
    snapshots = [p for p in (root / "snapshots").glob("*") if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def read_config(config_root):
    path = Path(config_root) / "config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def config_summary(cfg):
    if not cfg:
        return {"present": False}
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    layer_types = text_cfg.get("layer_types") or []
    return {
        "present": True,
        "model_type": cfg.get("model_type") or text_cfg.get("model_type"),
        "architectures": cfg.get("architectures") or text_cfg.get("architectures"),
        "vocab_size": text_cfg.get("vocab_size"),
        "hidden_size": text_cfg.get("hidden_size"),
        "intermediate_size": text_cfg.get("intermediate_size"),
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "num_attention_heads": text_cfg.get("num_attention_heads"),
        "num_key_value_heads": text_cfg.get("num_key_value_heads"),
        "head_dim": text_cfg.get("head_dim"),
        "bd_size": cfg.get("bd_size") or text_cfg.get("bd_size"),
        "diffusion_bridge_status": cfg.get("diffusion_bridge_status") or text_cfg.get("diffusion_bridge_status"),
        "gdn_mode": cfg.get("gdn_mode") or text_cfg.get("gdn_mode"),
        "mask_token_id": cfg.get("mask_token_id") or text_cfg.get("mask_token_id"),
        "layer_type_counts": dict(Counter(layer_types)),
        "layer_types_head": layer_types[:8],
        "has_linear_attention_layers": "linear_attention" in layer_types,
        "has_full_attention_layers": "full_attention" in layer_types,
    }


def weight_cache_status(snapshot):
    if snapshot is None:
        return {"snapshot": None, "has_snapshot": False, "complete": False, "weight_files": 0}
    snapshot = Path(snapshot)
    weight_files = sorted(
        p.name
        for p in snapshot.iterdir()
        if p.is_file() and (p.name.endswith(".safetensors") or p.name.endswith(".bin"))
    )
    index = snapshot / "model.safetensors.index.json"
    index_bin = snapshot / "pytorch_model.bin.index.json"
    index_path = index if index.exists() else index_bin if index_bin.exists() else None
    missing = []
    expected_count = None
    if index_path:
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            expected = sorted(set(data.get("weight_map", {}).values()))
            expected_count = len(expected)
            missing = [name for name in expected if not (snapshot / name).exists()]
        except Exception as exc:
            missing = [f"index_read_error:{type(exc).__name__}:{exc}"]
    complete = bool(weight_files) and not missing
    return {
        "snapshot": str(snapshot),
        "has_snapshot": True,
        "complete": complete,
        "weight_files": len(weight_files),
        "expected_weight_files": expected_count,
        "missing_weight_files": missing[:20],
        "index_file": str(index_path) if index_path else None,
    }


def dataset_status(path):
    path = Path(path)
    out = {"path": str(path), "exists": path.exists(), "count": 0}
    if not path.exists():
        return out
    try:
        if path.suffix == ".jsonl":
            out["count"] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            out["count"] = len(obj.get("instances", []))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def bridge_status(root):
    roots = [
        root / "models",
        root / "fast-dllm/v2",
        root / "fast-dllm/fast_ddrive",
        root / "fast-dllm/third_party/lmflow/models",
    ]
    hits = []
    statuses = {}
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lower = text.lower()
            has_fastdllm = "fast_dllm" in lower or "fast-dllm" in lower
            has_qwen35 = "qwen3_5" in lower or "qwen35" in lower or "qwen3.5" in lower
            has_gdn = "gated deltanet" in lower or "gateddeltanet" in lower or "delta_net" in lower
            if has_fastdllm and (has_qwen35 or has_gdn):
                rel = str(path.relative_to(root))
                hits.append(rel)
                match = re.search(r"FAST_DLLM_QWEN3_5_BRIDGE_STATUS\s*=\s*['\"]([^'\"]+)['\"]", text)
                if match:
                    statuses[rel] = match.group(1)
                elif "notimplementederror" in lower or "scaffold" in lower:
                    statuses[rel] = "scaffold"
                else:
                    statuses[rel] = "unknown"
    implemented_hits = [path for path, status in statuses.items() if status == "implemented"]
    return {
        "found": bool(hits),
        "implemented": bool(implemented_hits),
        "hits": hits[:20],
        "statuses": statuses,
        "implemented_hits": implemented_hits[:20],
        "searched_roots": [str(p.relative_to(root)) for p in roots if p.exists()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--candidate-model-path", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--training-python", type=Path, default=ROOT / ".venv-fastdllm/bin/python")
    parser.add_argument("--metadata-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    candidate = args.candidate_model_path
    if not candidate.is_absolute():
        candidate = root / candidate

    snapshot = latest_snapshot(args.model)
    model_source = snapshot if snapshot is not None else args.model
    raw_cfg = read_config(snapshot) if snapshot is not None else None
    candidate_cfg = read_config(candidate)

    package_probe = run_json(
        args.training_python,
        """
import importlib.metadata as md
import json
import sys
names = ['torch','transformers','peft','bitsandbytes','deepspeed','lmflow','huggingface_hub']
packages = {}
for name in names:
    try:
        packages[name] = md.version(name)
    except Exception as exc:
        packages[name] = 'MISSING:' + type(exc).__name__
print(json.dumps({'python': sys.executable, 'python_version': sys.version.split()[0], 'packages': packages}))
""",
    )

    config_support_source = candidate if candidate.exists() else model_source
    config_support_probe = run_json(
        args.training_python,
        """
import json
import sys
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
print(json.dumps({'model_type': getattr(cfg, 'model_type', None), 'architectures': getattr(cfg, 'architectures', None)}))
""",
        [config_support_source],
    )

    tokenizer_probe = run_json(
        args.metadata_python,
        """
import json
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
ids = tok.encode(sys.argv[2], add_special_tokens=False)
vocab = tok.get_vocab()
print(json.dumps({
    'tokenizer_len': len(tok),
    'mask_token': sys.argv[2],
    'mask_in_vocab': sys.argv[2] in vocab,
    'mask_id': vocab.get(sys.argv[2]),
    'mask_is_special': sys.argv[2] in set(tok.all_special_tokens),
    'encoded_ids': ids,
    'single_token': len(ids) == 1,
}))
""",
        [model_source, MASK_TOKEN],
    )

    candidate_tokenizer_probe = None
    if candidate.exists():
        candidate_tokenizer_probe = run_json(
            args.metadata_python,
            """
import json
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
ids = tok.encode(sys.argv[2], add_special_tokens=False)
vocab = tok.get_vocab()
print(json.dumps({
    'tokenizer_len': len(tok),
    'mask_token': sys.argv[2],
    'mask_in_vocab': sys.argv[2] in vocab,
    'mask_id': vocab.get(sys.argv[2]),
    'mask_is_special': sys.argv[2] in set(tok.all_special_tokens),
    'encoded_ids': ids,
    'single_token': len(ids) == 1,
}))
""",
            [candidate, MASK_TOKEN],
        )

    datasets = {
        "public_toolcall_train": dataset_status(root / "data/fastdllm_toolcall_train/train_toolcall.json"),
        "synthetic_onecall_train": dataset_status(root / "data/synthetic_onecall_train/train_synthetic_onecall.json"),
        "synthetic_toolresult_train": dataset_status(root / "data/synthetic_toolresult_train/train_synthetic_toolresult.json"),
        "repo_edit_tasks": dataset_status(root / "data/repo_edit_eval/tiny_repo_edit_5.jsonl"),
        "repo_edit_qwen36_results": dataset_status(
            root / "data/repo_edit_eval/tiny_repo_edit_qwen_code_q36_8k_requiredall_512_tools12_5.jsonl"
        ),
    }

    bridge = bridge_status(root)
    weights = weight_cache_status(snapshot)
    candidate_weights = weight_cache_status(candidate) if candidate.exists() else None
    raw_summary = config_summary(raw_cfg)
    candidate_summary = config_summary(candidate_cfg)
    candidate_arch = candidate_summary.get("architectures") or []
    if isinstance(candidate_arch, str):
        candidate_arch = [candidate_arch]
    candidate_is_fastdllm = any("Fast_dLLM" in str(item) for item in candidate_arch) or (
        "fast_dllm" in str(candidate_summary.get("model_type", "")).lower()
    )

    blockers = []
    warnings = []

    if not args.training_python.exists():
        blockers.append(f"training python not found: {args.training_python}")
    elif not package_probe.get("ok"):
        blockers.append("training python package probe failed")
    else:
        packages = package_probe.get("packages", {})
        for name in ["torch", "transformers", "peft", "bitsandbytes", "lmflow"]:
            if str(packages.get(name, "")).startswith("MISSING"):
                blockers.append(f"training environment missing {name}")

    if not raw_summary.get("present"):
        blockers.append(f"raw {args.model} config is not cached locally")

    if not config_support_probe.get("ok"):
        blockers.append(f"Fast-dLLM training env cannot load model config: {config_support_source}")

    if not tokenizer_probe.get("ok"):
        warnings.append("could not inspect raw Qwen3.5 tokenizer")
    elif not tokenizer_probe.get("single_token") or not tokenizer_probe.get("mask_in_vocab"):
        warnings.append("raw Qwen3.5 tokenizer does not have a single |<MASK>| token; conversion must add one")

    if not candidate.exists():
        blockers.append(f"converted diffusion candidate missing: {candidate}")
    else:
        if not candidate_summary.get("present"):
            blockers.append("converted candidate has no readable config.json")
        if not candidate_summary.get("bd_size"):
            blockers.append("converted candidate config is missing bd_size")
        if not candidate_is_fastdllm:
            blockers.append("converted candidate is not marked as a Fast_dLLM model")
        if not candidate_tokenizer_probe or not candidate_tokenizer_probe.get("ok"):
            blockers.append("converted candidate tokenizer could not be inspected")
        elif not candidate_tokenizer_probe.get("single_token") or not candidate_tokenizer_probe.get("mask_in_vocab"):
            blockers.append("converted candidate tokenizer does not have a single |<MASK>| token")

    if not bridge.get("found"):
        blockers.append("no Fast_dLLM Qwen3.5/GDN modeling bridge found")
    elif not bridge.get("implemented"):
        blockers.append("Fast_dLLM Qwen3.5/GDN bridge is present but not implemented")

    if candidate.exists() and candidate_weights and not candidate_weights.get("complete"):
        blockers.append("converted candidate has no local model weights")

    missing_datasets = [name for name, status in datasets.items() if not status.get("exists") or not status.get("count")]
    if missing_datasets:
        blockers.append("missing or empty curriculum inputs: " + ", ".join(missing_datasets))

    if not weights.get("complete"):
        warnings.append("raw Qwen3.5-9B weights are not fully cached locally")

    if candidate.exists():
        if candidate_weights and not candidate_weights.get("complete"):
            first_next_action = "Download/cache Qwen3.5-9B raw safetensor shards, then materialize text-only candidate weights with scripts/materialize_qwen35_fastdllm_weights.py --write."
        elif bridge.get("implemented"):
            first_next_action = "Run the guarded Qwen3.5-9B diffusion QLoRA pilot."
        else:
            first_next_action = "Implement the Qwen3.5 GDN/full-attention Fast_dLLM bridge, starting with Option A."
    else:
        first_next_action = "Create a converted local Qwen3.5-9B Fast_dLLM init with bd_size and a real |<MASK>| token."

    result = {
        "ready": not blockers,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "model": args.model,
        "model_source": str(model_source),
        "candidate_model_path": str(candidate),
        "checks": {
            "training_python": str(args.training_python),
            "metadata_python": str(args.metadata_python),
            "training_env_config_support_source": str(config_support_source),
            "training_packages": package_probe,
            "training_env_config_support": config_support_probe,
            "raw_config": raw_summary,
            "raw_tokenizer": tokenizer_probe,
            "candidate_config": candidate_summary,
            "candidate_is_fastdllm": candidate_is_fastdllm,
            "candidate_tokenizer": candidate_tokenizer_probe,
            "qwen35_weight_cache": weights,
            "candidate_weight_cache": candidate_weights,
            "fastdllm_qwen35_gdn_bridge": bridge,
            "datasets": datasets,
        },
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": [
            first_next_action,
            "Run scripts/run_fastdllm_qwen35_9b_agentic_qlora_pilot.sh only after readiness is clean.",
        ],
    }

    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
