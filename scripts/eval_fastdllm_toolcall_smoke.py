#!/usr/bin/env python3
import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_toolcall_jsonl import extract_json_objects


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_BASE = ROOT / "models/qwen2.5-1.5b-fastdllm-init"
DEFAULT_EVAL = ROOT / "data/toolcall_eval/fastdllm_toolcall_smoke.jsonl"
MASK_ID = 151665
STOP_TOKEN_ID = 151645


def called_tool_names(text):
    names = []
    invalid = 0
    for obj in extract_json_objects(text):
        if not isinstance(obj, dict):
            invalid += 1
            continue
        name = obj.get("name") or obj.get("function") or obj.get("tool_name")
        if isinstance(name, dict):
            name = name.get("name")
        if name:
            names.append(str(name))
    return names, invalid


def constrained_repair(text, available_names):
    ordered = []
    for name in sorted(available_names, key=len, reverse=True):
        pos = text.find(name)
        if pos >= 0:
            ordered.append((pos, name))
    ordered.sort()

    names = []
    seen = set()
    for _, name in ordered:
        if name not in seen:
            names.append(name)
            seen.add(name)

    if not names:
        return "", []

    repaired = []
    for name in names:
        payload = {"name": name, "arguments": {}}
        repaired.append("<tool_call>\n" + json.dumps(payload, separators=(",", ": ")) + "\n</tool_call>")
    return "\n".join(repaired), names


def load_cases(path, limit):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def resolve_chat_template(name):
    if not name:
        return None
    third_party = ROOT / "fast-dllm/third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from lmflow.utils.conversation_template import PRESET_TEMPLATES

    if name not in PRESET_TEMPLATES:
        raise ValueError(f"Unknown conversation template {name!r}")
    return PRESET_TEMPLATES[name]


def make_prompt(tokenizer, case, chat_template=None):
    messages = case["prompt_messages"]
    tools = case.get("tools") or None
    kwargs = {
        "tools": tools,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(base_model, adapter, merge_adapter):
    repo_v2 = ROOT / "fast-dllm/v2"
    sys.path.insert(0, str(repo_v2))
    import generation_functions

    tokenizer_path = adapter or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if adapter:
        model = PeftModel.from_pretrained(base, adapter)
        if merge_adapter:
            model = model.merge_and_unload()
    else:
        model = base
    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )
    model.to("cuda").eval()
    return model, tokenizer


def resolve_token_ids(model, tokenizer):
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        converted = tokenizer.convert_tokens_to_ids("|<MASK>|")
        if converted != tokenizer.unk_token_id:
            mask_id = converted
    if mask_id is None:
        mask_id = MASK_ID

    stop_token_id = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id
    if isinstance(stop_token_id, (list, tuple)):
        stop_token_id = stop_token_id[0] if stop_token_id else None
    if stop_token_id is None:
        stop_token_id = STOP_TOKEN_ID

    return int(mask_id), int(stop_token_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repair-mode", choices=["none", "known-name"], default="none")
    parser.add_argument("--conversation-template", default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    args = parser.parse_args()
    chat_template = resolve_chat_template(args.conversation_template)

    model, tokenizer = load_model(
        str(args.base_model),
        str(args.adapter) if args.adapter else None,
        merge_adapter=not args.no_merge_adapter,
    )
    mask_id, stop_token_id = resolve_token_ids(model, tokenizer)
    cases = load_cases(args.eval_jsonl, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "valid_tool_json": 0,
        "any_tool_call": 0,
        "exact_tool_name_set": 0,
        "contains_all_gold_names": 0,
        "called_known_tool": 0,
        "mentions_any_gold_name": 0,
        "mentions_all_gold_names": 0,
        "mentions_known_tool": 0,
        "repaired_any_tool_call": 0,
        "repaired_exact_tool_name_set": 0,
        "repaired_contains_all_gold_names": 0,
        "repaired_called_known_tool": 0,
        "unresolved_mask_examples": 0,
    }
    total_new_tokens = 0
    start = time.time()

    with args.out.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            prompt = make_prompt(tokenizer, case, chat_template=chat_template)
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            seq_len = torch.tensor([input_ids.shape[1]], device="cuda")

            sample_start = time.time()
            with torch.no_grad():
                generated = model.mdm_sample(
                    input_ids,
                    tokenizer=tokenizer,
                    block_size=args.block_size,
                    small_block_size=args.small_block_size,
                    max_new_tokens=args.max_new_tokens,
                    mask_id=mask_id,
                    stop_token=stop_token_id,
                    min_len=input_ids.shape[1],
                    seq_len=seq_len,
                    threshold=args.threshold,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )[0]
            sample_seconds = time.time() - sample_start

            new_ids = generated[input_ids.shape[1] :]
            mask_count = int((new_ids == mask_id).sum().item())
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            called_names, invalid_json_count = called_tool_names(text)
            gold_names = [str(name) for name in case["gold_tool_names"]]
            available_names = set(case.get("available_tool_names") or [])
            mentioned_gold_names = [name for name in gold_names if name in text]
            mentioned_available_names = [name for name in sorted(available_names) if name in text]
            repaired_text = ""
            repaired_names = []
            if args.repair_mode == "known-name":
                repaired_text, repaired_names = constrained_repair(text, available_names)

            called_set = set(called_names)
            gold_set = set(gold_names)
            mentioned_gold_set = set(mentioned_gold_names)
            repaired_set = set(repaired_names)
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "gold_tool_names": gold_names,
                "called_tool_names": called_names,
                "mentioned_gold_names": mentioned_gold_names,
                "mentioned_available_names": mentioned_available_names,
                "invalid_tool_json_count": invalid_json_count,
                "exact_tool_name_set": called_set == gold_set,
                "contains_all_gold_names": gold_set.issubset(called_set),
                "called_known_tool": bool(called_set & available_names),
                "mentions_any_gold_name": bool(mentioned_gold_names),
                "mentions_all_gold_names": gold_set.issubset(mentioned_gold_set),
                "mentions_known_tool": bool(mentioned_available_names),
                "repair_mode": args.repair_mode,
                "repaired_assistant": repaired_text,
                "repaired_tool_names": repaired_names,
                "repaired_any_tool_call": bool(repaired_names),
                "repaired_exact_tool_name_set": repaired_set == gold_set,
                "repaired_contains_all_gold_names": gold_set.issubset(repaired_set),
                "repaired_called_known_tool": bool(repaired_set & available_names),
                "generated": text,
                "mask_count": mask_count,
                "seconds": sample_seconds,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            totals["records"] += 1
            totals["valid_tool_json"] += int(bool(called_names) and invalid_json_count == 0)
            totals["any_tool_call"] += int(bool(called_names))
            totals["exact_tool_name_set"] += int(row["exact_tool_name_set"])
            totals["contains_all_gold_names"] += int(row["contains_all_gold_names"])
            totals["called_known_tool"] += int(row["called_known_tool"])
            totals["mentions_any_gold_name"] += int(row["mentions_any_gold_name"])
            totals["mentions_all_gold_names"] += int(row["mentions_all_gold_names"])
            totals["mentions_known_tool"] += int(row["mentions_known_tool"])
            totals["repaired_any_tool_call"] += int(row["repaired_any_tool_call"])
            totals["repaired_exact_tool_name_set"] += int(row["repaired_exact_tool_name_set"])
            totals["repaired_contains_all_gold_names"] += int(row["repaired_contains_all_gold_names"])
            totals["repaired_called_known_tool"] += int(row["repaired_called_known_tool"])
            totals["unresolved_mask_examples"] += int(mask_count > 0)
            total_new_tokens += int((new_ids != mask_id).sum().item())
            print(
                f"{idx + 1}/{len(cases)} exact={totals['exact_tool_name_set']} "
                f"contains={totals['contains_all_gold_names']} called={called_names}"
            )

    elapsed = time.time() - start
    summary = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "eval_jsonl": str(args.eval_jsonl),
        "output_jsonl": str(args.out),
        "totals": totals,
        "rates": {
            key: value / totals["records"] if totals["records"] else 0.0
            for key, value in totals.items()
            if key != "records"
        },
        "elapsed_seconds": elapsed,
        "generated_tokens": total_new_tokens,
        "generated_tokens_per_second": total_new_tokens / elapsed if elapsed else 0.0,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repair_mode": args.repair_mode,
        "mask_id": mask_id,
        "stop_token_id": stop_token_id,
        "conversation_template": args.conversation_template,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
