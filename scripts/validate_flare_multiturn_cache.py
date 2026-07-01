#!/usr/bin/env python3
"""Validate FLARE cache threading across chat turns and tool-result tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import flare_two_stream_noisy_logits, load_model, resolve_token_ids
from eval_toolcall_jsonl import qwen_native_tool_call_text
from flare_hf_cache import RequestDiffusionState


DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--no-merge-adapter", action="store_true")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=10)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out-json", type=Path, default=ROOT / "runs/agentic_eval/multiturn_cache_parity.json")
    return parser.parse_args()


def tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a new task for a user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["user_id", "title"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_task_status",
                "description": "Update the status of a task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "completed"]},
                    },
                    "required": ["task_id", "status"],
                },
            },
        },
    ]


def multiturn_messages(tool_result_json: str) -> list[dict]:
    created_call = qwen_native_tool_call_text(
        [
            {
                "name": "create_task",
                "arguments": {
                    "user_id": "user_1",
                    "title": "Project Review",
                    "description": "Review Q4 project status",
                },
            }
        ]
    )
    update_call = qwen_native_tool_call_text(
        [{"name": "update_task_status", "arguments": {"task_id": "task_2", "status": "completed"}}]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a task-management support agent. Use tools for database changes. "
                "Follow the mock-domain policy: tasks need titles, statuses are pending or completed, "
                "and only existing users can create tasks."
            ),
        },
        {"role": "user", "content": "Create a task for the project review meeting for user_1."},
        {"role": "assistant", "content": created_call},
        {"role": "user", "content": f"<tool_response>\n{tool_result_json}\n</tool_response>"},
        {
            "role": "assistant",
            "content": "Created task_2 for the project review meeting. It is currently pending.",
        },
        {"role": "user", "content": "Now mark task_2 as completed."},
        {"role": "assistant", "content": update_call},
        {
            "role": "user",
            "content": (
                "<tool_response>\n"
                '{"task_id":"task_2","title":"Project Review","description":"Review Q4 project status",'
                '"status":"completed"}\n'
                "</tool_response>"
            ),
        },
        {"role": "assistant", "content": "task_2 is completed."},
    ]


def render_prompt(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def token_span_for_substring(tokenizer, prompt: str, substring: str) -> dict[str, int | str]:
    char_start = prompt.find(substring)
    if char_start < 0:
        return {"status": "missing", "token_start": -1, "token_end": -1}
    char_end = char_start + len(substring)
    before_ids = tokenizer(prompt[:char_start], add_special_tokens=False).input_ids
    sub_ids = tokenizer(prompt[char_start:char_end], add_special_tokens=False).input_ids
    return {
        "status": "ok",
        "char_start": int(char_start),
        "char_end": int(char_end),
        "token_start": int(len(before_ids)),
        "token_end": int(len(before_ids) + len(sub_ids)),
    }


def shifted_reference(model, x_t: torch.Tensor, block_size: int, mask_id: int, active_start: int) -> torch.Tensor:
    noisy_logits = flare_two_stream_noisy_logits(model, x_t, x_t, block_size=block_size, mask_id=mask_id)[: x_t.shape[0]]
    shifted = torch.cat([noisy_logits[:, :1, :], noisy_logits[:, :-1, :]], dim=1)
    return shifted[:, active_start:]


def intersecting_blocks(span: dict, block_size: int, full_blocks: int) -> list[int]:
    if span.get("status") != "ok":
        return []
    start = int(span["token_start"])
    end = int(span["token_end"])
    blocks = []
    for block_idx in range(full_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size
        if max(start, block_start) < min(end, block_end):
            blocks.append(block_idx)
    return blocks


@torch.no_grad()
def greedy_argmax_parity(
    model,
    prompt_ids: torch.Tensor,
    *,
    block_size: int,
    mask_id: int,
    stop_token_ids: list[int],
    new_tokens: int,
) -> dict:
    output_ids = prompt_ids.clone()
    original_len = int(output_ids.shape[1])
    state = RequestDiffusionState.reset(model, output_ids, block_size)
    steps = []
    argmax_flips = 0
    max_logit_abs_diff = 0.0
    stopped = False
    stop_token_id = None

    while output_ids.shape[1] - original_len < new_tokens and not stopped:
        remaining = new_tokens - (output_ids.shape[1] - original_len)
        active_len = output_ids.shape[1] - state.block_start
        if active_len < 0 or active_len >= block_size:
            raise RuntimeError(
                f"invalid active_len={active_len} for output_len={output_ids.shape[1]} "
                f"block_start={state.block_start}"
            )
        block_pad = min(block_size - active_len, remaining)
        masks = torch.full(
            (output_ids.shape[0], block_pad),
            int(mask_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        x_t = torch.cat([output_ids, masks], dim=1)
        while bool((x_t[:, -block_pad:] == mask_id).any().item()):
            window_start = state.block_start
            window = x_t[:, window_start:]
            cache_logits = state.shifted_active_logits(model, x_t)
            ref_logits = shifted_reference(model, x_t, block_size, mask_id, active_start=window_start)
            mask_positions = (window[0] == mask_id).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                break
            local_idx = int(mask_positions[0].item())
            abs_idx = window_start + local_idx
            cache_token = int(cache_logits[0, local_idx].argmax().item())
            ref_token = int(ref_logits[0, local_idx].argmax().item())
            diff = float((cache_logits.float() - ref_logits.float()).abs().max().item())
            max_logit_abs_diff = max(max_logit_abs_diff, diff)
            equal = cache_token == ref_token
            argmax_flips += int(not equal)
            steps.append(
                {
                    "generated_index": int(abs_idx - original_len),
                    "abs_index": int(abs_idx),
                    "cache_token": cache_token,
                    "reference_token": ref_token,
                    "equal": equal,
                    "active_window_start": int(window_start),
                    "active_window_len": int(window.shape[1]),
                    "logit_max_abs_diff": diff,
                }
            )
            x_t[:, abs_idx] = ref_token
            if ref_token in stop_token_ids:
                stopped = True
                stop_token_id = ref_token
                break

        active_block = x_t[:, state.block_start :]
        if active_block.shape[1] == block_size and not bool((active_block == mask_id).any().item()):
            state.advance(model, active_block)
        output_ids = x_t

    return {
        "new_tokens": int(new_tokens),
        "tokens_compared": int(len(steps)),
        "stopped_on_token": None if stop_token_id is None else int(stop_token_id),
        "argmax_flips": int(argmax_flips),
        "logit_max_abs_diff": max_logit_abs_diff,
        "cache_stats": state.stats(),
        "steps": steps,
    }


@torch.no_grad()
def run_validation(args: argparse.Namespace) -> dict:
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and args.adapter.exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    mask_id, _, stop_token_ids = resolve_token_ids(model, tokenizer)
    block_size = int(args.block_size)
    if hasattr(model, "config"):
        setattr(model.config, "bd_size", block_size)

    tool_result = (
        '{"task_id":"task_2","title":"Project Review","description":"Review Q4 project status",'
        '"status":"pending"}'
    )
    prompt = render_prompt(tokenizer, multiturn_messages(tool_result), tool_schema())
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")

    tool_span = token_span_for_substring(tokenizer, prompt, tool_result)
    min_required_tokens = int(tool_span.get("token_end", 0)) + block_size if tool_span.get("status") == "ok" else 0
    if ids.shape[1] < min_required_tokens:
        pad_text = "\nAdditional context: the tool result above is authoritative for the next turn."
        extra_ids = tokenizer(pad_text, add_special_tokens=False, return_tensors="pt").input_ids.to(ids.device)
        while ids.shape[1] < min_required_tokens:
            ids = torch.cat([ids, extra_ids], dim=1)

    full_blocks = int(ids.shape[1] // block_size)
    if full_blocks < 2:
        raise RuntimeError(f"need at least 2 full blocks for multi-turn cache parity, got {full_blocks}")

    tool_blocks = intersecting_blocks(tool_span, block_size, full_blocks)
    committed_prompt_tokens = full_blocks * block_size
    tool_result_committed = bool(tool_blocks) and int(tool_span.get("token_end", 10**9)) <= committed_prompt_tokens
    generation = greedy_argmax_parity(
        model,
        ids,
        block_size=block_size,
        mask_id=mask_id,
        stop_token_ids=[int(item) for item in stop_token_ids],
        new_tokens=int(args.new_tokens),
    )
    return {
        "status": "pass" if generation["argmax_flips"] == 0 and tool_result_committed else "fail",
        "name": "multiturn_tool_result_cache_parity",
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "merge_adapter": not args.no_merge_adapter,
        "block_size": block_size,
        "prompt_tokens": int(ids.shape[1]),
        "prompt_full_blocks": full_blocks,
        "prompt_committed_tokens_on_reset": int(committed_prompt_tokens),
        "tool_result_span": tool_span,
        "tool_result_blocks": tool_blocks,
        "tool_result_committed_on_reset": bool(tool_result_committed),
        "generation_argmax_parity": generation,
        "argmax_flips": int(generation["argmax_flips"]),
        "logit_max_abs_diff": float(generation["logit_max_abs_diff"]),
        "mask_id": int(mask_id),
        "stop_token_ids": [int(item) for item in stop_token_ids],
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    result = run_validation(args)
    result["elapsed_seconds"] = time.time() - start
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
