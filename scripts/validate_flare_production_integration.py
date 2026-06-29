#!/usr/bin/env python3
"""Production integration gates for the FLARE two-stream forward.

Runs CPU/fp32 checks only:
1. flag-gated production forward vs the already validated standalone helper;
2. clean stream vs the existing AR/causal forward;
3. LMFlow doc_id packing snapshot and no-cross-doc mask assertions.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


@contextlib.contextmanager
def patched_env(**updates):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_stage1_module():
    script_path = Path(__file__).resolve().with_name("validate_flare_two_stream_forward.py")
    spec = importlib.util.spec_from_file_location("_flare_stage1_forward_validator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Stage-1 validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 and right.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def test_production_matches_helper(stage1, config_module, modeling_module, *, seed: int, atol: float):
    block_size = 2
    model = stage1.make_tiny_model(config_module, modeling_module, seed=seed, block_size=block_size)
    input_ids = torch.tensor([[5, 8, 13, 21, 7, 11, 17, 19]], dtype=torch.long)
    labels = input_ids.clone()
    doc_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    mask_indices = torch.tensor([[False, True, False, True, True, False, True, False]])

    with torch.no_grad():
        model.train()
        helper = stage1.flare_two_stream_forward(
            model,
            modeling_module,
            input_ids,
            doc_ids,
            mask_indices,
            block_size=block_size,
        )
        with patched_env(FASTDLLM_FLARE_TWO_STREAM="1", FLARE_TWO_STREAM=None):
            production = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                doc_ids=doc_ids,
                flare_mask_indices=mask_indices,
            )
            padded_input_ids = torch.tensor([[10, 11, 12, 3, 20, 21, 22, 23]], dtype=torch.long)
            padded_labels = torch.tensor([[10, 11, 12, -100, 20, 21, 22, 23]], dtype=torch.long)
            padded_attention = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.long)
            padded_doc_ids = torch.tensor([[0, 0, 0, -1, 1, 1, 1, 1]], dtype=torch.long)
            padded_masks = torch.tensor([[False, True, False, False, True, False, True, False]])
            padded_production = model(
                input_ids=padded_input_ids,
                attention_mask=padded_attention,
                labels=padded_labels,
                doc_ids=padded_doc_ids,
                flare_mask_indices=padded_masks,
            )
        model.eval()
        golden_ar = model(input_ids=input_ids).logits

    prod_helper_logits_diff = max_abs_diff(production.logits, helper.clean_logits)
    prod_helper_loss_diff = abs(float(production.loss.item() - helper.losses.total_loss.item()))
    clean_ar_diff = max_abs_diff(production.logits, golden_ar)
    padded_finite = bool(
        torch.isfinite(padded_production.loss).item()
        and torch.isfinite(padded_production.logits).all().item()
    )
    passed = (
        prod_helper_logits_diff <= atol
        and prod_helper_loss_diff <= atol
        and clean_ar_diff <= atol
        and padded_finite
    )
    detail = (
        f"prod_vs_helper_logits={prod_helper_logits_diff:.6g} "
        f"prod_vs_helper_loss={prod_helper_loss_diff:.6g} "
        f"clean_vs_ar_logits={clean_ar_diff:.6g} "
        f"padded_forward_finite={padded_finite}"
    )
    return passed, detail


class _MainProcessFirst:
    def __call__(self, desc=None):
        return contextlib.nullcontext()


def test_doc_id_packing_snapshot(root: Path, modeling_module):
    sys.path.insert(0, str(root / "fast-dllm" / "third_party"))
    cuda_root = root / ".venv-fastdllm" / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu13"
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    from datasets import Dataset as HFDataset
    from lmflow.pipeline.finetuner import Finetuner

    tokenized = HFDataset.from_dict(
        {
            "input_ids": [[10, 11, 12], [20, 21, 22, 23]],
            "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
            "labels": [[10, 11, 12], [20, 21, 22, 23]],
        }
    )
    finetuner = Finetuner.__new__(Finetuner)
    finetuner.data_args = SimpleNamespace(
        block_size=8,
        pad_mask_token=False,
        bd_size=2,
        mask_id=3,
        disable_group_texts=False,
        streaming=False,
        preprocessing_num_workers=None,
        overwrite_cache=True,
        group_texts_batch_size=1000,
    )
    finetuner.finetuner_args = SimpleNamespace(main_process_first=_MainProcessFirst())
    finetuner.model_args = SimpleNamespace(truncate_to_model_max_length=True)

    with patched_env(FASTDLLM_FLARE_TWO_STREAM="1", FLARE_TWO_STREAM=None):
        packed = finetuner.group_text(tokenized, model_max_length=16)
    row = packed[0]
    expected_doc_ids = [0, 0, 0, -1, 1, 1, 1, 1]
    doc_ids_ok = row["doc_ids"] == expected_doc_ids
    pad_ok = row["input_ids"][3] == 3 and row["attention_mask"][3] == 0 and row["labels"][3] == -100

    doc_ids = torch.tensor([row["doc_ids"]], dtype=torch.long)
    mask = modeling_module.flare_two_stream_bool_mask(doc_ids, block_size=2)[0, 0]
    seq_len = doc_ids.shape[1]
    no_cross_doc = (
        not bool(mask[4, 2])
        and not bool(mask[2, 4])
        and not bool(mask[seq_len + 4, 2])
        and not bool(mask[seq_len + 2, seq_len + 4])
    )
    pad_isolated = (
        bool(mask[3, 3])
        and bool(mask[seq_len + 3, seq_len + 3])
        and int(mask[3].sum().item()) == 1
        and int(mask[:, 3].sum().item()) == 1
        and int(mask[seq_len + 3].sum().item()) == 1
        and int(mask[:, seq_len + 3].sum().item()) == 1
    )
    passed = doc_ids_ok and pad_ok and no_cross_doc and pad_isolated
    detail = (
        f"doc_ids={row['doc_ids']} input_ids={row['input_ids']} "
        f"attention_mask={row['attention_mask']} labels={row['labels']} "
        f"no_cross_doc={no_cross_doc} pad_isolated={pad_isolated}"
    )
    return passed, detail


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/qwen3.5-9b-fastdllm-init")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    root = Path(__file__).resolve().parents[1]
    stage1 = load_stage1_module()
    config_module, modeling_module = stage1.load_local_bridge((root / args.model_dir).resolve())

    prod_ok, prod_detail = test_production_matches_helper(
        stage1,
        config_module,
        modeling_module,
        seed=args.seed,
        atol=args.atol,
    )
    pack_ok, pack_detail = test_doc_id_packing_snapshot(root, modeling_module)

    print("FLARE production integration validation")
    print(f"device=cpu dtype=float32 threads={torch.get_num_threads()} seed={args.seed}")
    print(f"{'PASS' if prod_ok else 'FAIL'}\tproduction-vs-helper + clean==AR\t{prod_detail}")
    print(f"{'PASS' if pack_ok else 'FAIL'}\tdoc-id packing snapshot\t{pack_detail}")
    final = prod_ok and pack_ok
    print("FINAL:", "PASS" if final else "FAIL")
    return 0 if final else 1


if __name__ == "__main__":
    raise SystemExit(main())
