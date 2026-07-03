#!/usr/bin/env python3
"""S2.0 teacher sanity check on held-out masked parameter-value spans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import re
import types
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval_fastdllm_toolcall_cases import flare_two_stream_noisy_logits, resolve_token_ids


ROOT = Path("/home/mark/qwen_diffusion")
PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\s*\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    parser.add_argument(
        "--teacher-adapter",
        type=Path,
        default=ROOT / "runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16",
    )
    parser.add_argument(
        "--chat-template-path",
        type=Path,
        default=Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja"),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=ROOT / "data/toolcall_eval_native/heldout_seed_multicall_policy_targets_qwen_native.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-seq-tokens", type=int, default=1024)
    parser.add_argument("--partial-prefix-fraction", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--teacher-adapter-name", default="s2_teacher")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--no-4bit", action="store_true", help="Load bf16 full weights instead of the NF4 S2 path.")
    return parser.parse_args()


def configure_env() -> None:
    os.environ.setdefault("FASTDLLM_FLARE_GDN_ROUTE", "route_i")
    os.environ.setdefault("FASTDLLM_GDN_KERNEL", "torch")
    os.environ.setdefault("FASTDLLM_BATCH_FLARE_NOISY_GDN", "1")
    os.environ.setdefault("FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN", "1")


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_chat_template(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def render_prompt(tokenizer, row: dict[str, Any], chat_template: str | None) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    tools = row.get("tools") or None
    if tools:
        kwargs["tools"] = tools
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    return tokenizer.apply_chat_template(row.get("prompt_messages") or [], **kwargs)


def value_span_token_groups(tokenizer, assistant: str) -> list[dict[str, Any]]:
    spans = []
    for match in PARAMETER_RE.finditer(assistant or ""):
        start, end = match.span(2)
        while start < end and assistant[start] in "\r\n\t ":
            start += 1
        while end > start and assistant[end - 1] in "\r\n\t ":
            end -= 1
        if end > start:
            spans.append({"key": match.group(1), "start": start, "end": end})
    if not spans:
        return []
    encoded = tokenizer(
        assistant,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    groups: list[dict[str, Any]] = []
    for span_idx, span in enumerate(spans):
        token_indices = []
        for token_idx, (start, end) in enumerate(encoded.offset_mapping):
            if end <= start:
                continue
            if end > span["start"] and start < span["end"]:
                token_indices.append(token_idx)
        if token_indices:
            groups.append(
                {
                    "span_idx": span_idx,
                    "key": span["key"],
                    "char_start": span["start"],
                    "char_end": span["end"],
                    "token_indices": token_indices,
                    "token_count": len(token_indices),
                }
            )
    return groups


def active_adapter_names(model) -> list[str]:
    values = []
    for attr in ("active_adapters", "active_adapter"):
        if not hasattr(model, attr):
            continue
        value = getattr(model, attr)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
    return list(dict.fromkeys(values))


def load_teacher(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.no_4bit:
        base = AutoModelForCausalLM.from_pretrained(
            str(args.base_model),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(f"cuda:{args.gpu_index}")
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            str(args.base_model),
            trust_remote_code=True,
            local_files_only=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(args.gpu_index)},
        )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base,
        str(args.teacher_adapter),
        adapter_name=args.teacher_adapter_name,
        is_trainable=False,
    )
    model.set_adapter(args.teacher_adapter_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    repo_v2 = ROOT / "fast-dllm/v2"
    if str(repo_v2) not in sys.path:
        sys.path.insert(0, str(repo_v2))
    import generation_functions

    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )
    return model, tokenizer


def selected_mask_positions(
    groups: list[dict[str, Any]],
    *,
    prompt_len: int,
    crop_start: int,
    seq_len: int,
    partial_prefix_fraction: float,
) -> tuple[list[int], dict[int, list[int]], Counter]:
    selected = []
    span_positions: dict[int, list[int]] = {}
    counts = Counter()
    for group in groups:
        token_indices = list(group["token_indices"])
        if not token_indices:
            continue
        if len(token_indices) == 1:
            reveal_count = 0
            counts["single_token_spans"] += 1
        else:
            reveal_count = max(1, int(math.floor(len(token_indices) * partial_prefix_fraction)))
            reveal_count = min(reveal_count, len(token_indices) - 1)
        masked = token_indices[reveal_count:]
        if not masked:
            continue
        for assistant_idx in masked:
            absolute = prompt_len + int(assistant_idx)
            cropped = absolute - crop_start
            if 0 <= cropped < seq_len:
                selected.append(cropped)
                span_positions.setdefault(int(group["span_idx"]), []).append(cropped)
            else:
                counts["cropped_target_tokens"] += 1
        counts["value_spans"] += 1
        counts["value_tokens"] += len(token_indices)
        counts["revealed_prefix_tokens"] += reveal_count
        counts["masked_suffix_tokens_before_crop"] += len(masked)
    return selected, span_positions, counts


def score_rows(model, tokenizer, rows: list[dict[str, Any]], args: argparse.Namespace, chat_template: str | None):
    mask_id, _, _ = resolve_token_ids(model, tokenizer)
    device = next(model.parameters()).device
    token_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    totals = Counter()
    t0 = time.time()

    for row_idx, row in enumerate(rows):
        prompt = render_prompt(tokenizer, row, chat_template)
        assistant = str(row.get("gold_assistant") or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        assistant_ids = tokenizer(assistant, add_special_tokens=False).input_ids
        full_ids = prompt_ids + assistant_ids
        if len(full_ids) > args.max_seq_tokens:
            crop_start = len(full_ids) - int(args.max_seq_tokens)
            cropped_ids = full_ids[crop_start:]
        else:
            crop_start = 0
            cropped_ids = full_ids
        groups = value_span_token_groups(tokenizer, assistant)
        selected, span_positions, row_counts = selected_mask_positions(
            groups,
            prompt_len=len(prompt_ids),
            crop_start=crop_start,
            seq_len=len(cropped_ids),
            partial_prefix_fraction=float(args.partial_prefix_fraction),
        )
        totals.update(row_counts)
        totals["rows"] += 1
        totals["prompt_tokens_total"] += len(prompt_ids)
        totals["assistant_tokens_total"] += len(assistant_ids)
        totals["seq_tokens_after_crop_total"] += len(cropped_ids)
        totals["cropped_rows"] += int(crop_start > 0)
        if not selected:
            row_records.append(
                {
                    "row_idx": row_idx,
                    "id": row.get("id"),
                    "status": "skipped_no_uncropped_value_targets",
                    "value_spans": len(groups),
                    "prompt_tokens": len(prompt_ids),
                    "assistant_tokens": len(assistant_ids),
                    "crop_start": crop_start,
                }
            )
            totals["skipped_rows"] += 1
            continue

        clean = torch.tensor([cropped_ids], dtype=torch.long, device=device)
        noisy = clean.clone()
        selected_tensor = torch.tensor(sorted(set(selected)), dtype=torch.long, device=device)
        noisy[:, selected_tensor] = int(mask_id)
        with torch.inference_mode():
            logits = flare_two_stream_noisy_logits(
                model,
                clean,
                noisy,
                block_size=int(args.block_size),
                mask_id=int(mask_id),
            )[:1]
            shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            pos_logits = shifted[0, selected_tensor].float()
            pos_logits[:, int(mask_id)] = -torch.inf
            pred_ids = torch.argmax(pos_logits, dim=-1)
            gold_ids = clean[0, selected_tensor]
            matches = pred_ids.eq(gold_ids)

        pos_to_match = {
            int(pos): bool(match)
            for pos, match in zip(selected_tensor.detach().cpu().tolist(), matches.detach().cpu().tolist())
        }
        span_exact = 0
        scored_spans = 0
        for positions in span_positions.values():
            kept = [pos for pos in positions if pos in pos_to_match]
            if not kept:
                continue
            scored_spans += 1
            span_exact += int(all(pos_to_match[pos] for pos in kept))

        row_correct = int(matches.sum().item())
        row_total = int(selected_tensor.numel())
        totals["scored_rows"] += 1
        totals["scored_tokens"] += row_total
        totals["top1_correct_tokens"] += row_correct
        totals["scored_value_spans"] += scored_spans
        totals["top1_exact_value_spans"] += span_exact
        row_records.append(
            {
                "row_idx": row_idx,
                "id": row.get("id"),
                "status": "scored",
                "value_spans": len(groups),
                "scored_tokens": row_total,
                "top1_correct_tokens": row_correct,
                "top1_accuracy": row_correct / max(row_total, 1),
                "scored_value_spans": scored_spans,
                "top1_exact_value_spans": span_exact,
                "span_exact_accuracy": span_exact / max(scored_spans, 1),
                "prompt_tokens": len(prompt_ids),
                "assistant_tokens": len(assistant_ids),
                "crop_start": crop_start,
            }
        )
        for pos, pred_id, gold_id, match in zip(
            selected_tensor.detach().cpu().tolist(),
            pred_ids.detach().cpu().tolist(),
            gold_ids.detach().cpu().tolist(),
            matches.detach().cpu().tolist(),
        ):
            token_records.append(
                {
                    "row_idx": row_idx,
                    "id": row.get("id"),
                    "cropped_token_pos": int(pos),
                    "absolute_token_pos": int(pos + crop_start),
                    "gold_id": int(gold_id),
                    "pred_id": int(pred_id),
                    "match": bool(match),
                    "gold_text": tokenizer.decode([int(gold_id)], skip_special_tokens=False),
                    "pred_text": tokenizer.decode([int(pred_id)], skip_special_tokens=False),
                }
            )

    elapsed = time.time() - t0
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "limit": int(args.limit),
        "rows_loaded": len(rows),
        "base_model": str(args.base_model),
        "teacher_adapter": str(args.teacher_adapter),
        "teacher_adapter_name": args.teacher_adapter_name,
        "tokenizer": str(args.tokenizer),
        "chat_template_path": str(args.chat_template_path) if args.chat_template_path else None,
        "chat_template_sha256": sha256_file(args.chat_template_path),
        "block_size": int(args.block_size),
        "max_seq_tokens": int(args.max_seq_tokens),
        "partial_prefix_fraction": float(args.partial_prefix_fraction),
        "threshold": float(args.threshold),
        "mask_id": int(mask_id),
        "top1_accuracy": totals["top1_correct_tokens"] / max(totals["scored_tokens"], 1),
        "span_exact_accuracy": totals["top1_exact_value_spans"] / max(totals["scored_value_spans"], 1),
        "pass": totals["scored_tokens"] > 0
        and (totals["top1_correct_tokens"] / max(totals["scored_tokens"], 1)) >= float(args.threshold),
        "elapsed_sec": elapsed,
        "totals": dict(totals),
        "adapter_active_proof": {
            "peft_model": isinstance(model, PeftModel),
            "active_adapters": active_adapter_names(model),
            "expected_adapter": args.teacher_adapter_name,
            "disable_adapter_used": False,
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
            "quantization": "bf16_full" if args.no_4bit else "nf4_4bit",
        },
        "git_head": git_head(),
        "script_sha256": sha256_file(Path(__file__)),
        "row_summary_jsonl": str(args.out_dir / "teacher_value_top1_rows.jsonl"),
        "token_records_jsonl": str(args.out_dir / "teacher_value_top1_tokens.jsonl"),
    }
    return summary, row_records, token_records


def write_report(path: Path, summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    verdict = "PASS" if summary["pass"] else "FAIL"
    lines = [
        "# S2.0 Teacher Value-Span Precheck",
        "",
        f"Verdict: **{verdict}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Top-1 value-token accuracy | {summary['top1_accuracy']:.4f} |",
        f"| Threshold | {summary['threshold']:.4f} |",
        f"| Correct / scored tokens | {totals.get('top1_correct_tokens', 0)} / {totals.get('scored_tokens', 0)} |",
        f"| Exact masked value spans | {totals.get('top1_exact_value_spans', 0)} / {totals.get('scored_value_spans', 0)} |",
        f"| Scored rows | {totals.get('scored_rows', 0)} / {summary['rows_loaded']} |",
        f"| Cropped rows | {totals.get('cropped_rows', 0)} |",
        "",
        "## Teacher Lineage",
        "",
        f"- Base: `{summary['base_model']}`",
        f"- Teacher adapter: `{summary['teacher_adapter']}`",
        f"- Selected because v2 held matched-20 at 44/63 while v4 regressed to 37/63.",
        f"- Active adapters: `{summary['adapter_active_proof']['active_adapters']}`",
        f"- `disable_adapter` used: `{summary['adapter_active_proof']['disable_adapter_used']}`",
        f"- All parameters frozen: `{summary['adapter_active_proof']['all_parameters_frozen']}`",
        f"- Quantization: `{summary['adapter_active_proof']['quantization']}`",
        "",
        "## Pins",
        "",
        f"- Git HEAD: `{summary['git_head']}`",
        f"- Script SHA256: `{summary['script_sha256']}`",
        f"- Chat template: `{summary['chat_template_path']}` (`{summary['chat_template_sha256']}`)",
        f"- Tokenizer: `{summary['tokenizer']}`",
        f"- Input: `{summary['input_jsonl']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_env()
    torch.manual_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chat_template = load_chat_template(args.chat_template_path)
    rows = read_jsonl(args.input_jsonl, args.limit)
    model, tokenizer = load_teacher(args)
    summary, row_records, token_records = score_rows(model, tokenizer, rows, args, chat_template)
    write_json(args.out_dir / "summary.json", summary)
    write_jsonl(args.out_dir / "teacher_value_top1_rows.jsonl", row_records)
    write_jsonl(args.out_dir / "teacher_value_top1_tokens.jsonl", token_records)
    write_report(args.out_dir / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
