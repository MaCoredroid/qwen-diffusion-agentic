#!/usr/bin/env python3
import argparse
import copy
import difflib
import json
import re
import sys
import time
import types
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnose_toolcall_json_completability import JsonPrefixParser
from eval_toolcall_jsonl import (
    extract_qwen_function_calls,
    extract_tool_calls,
    qwen_native_tool_call_text,
    score_tool_calls,
    tool_schema_by_name,
)


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_BASE = ROOT / "models/qwen2.5-1.5b-fastdllm-init"
DEFAULT_INPUT = ROOT / "data/toolcall_eval/synthetic_onecall_smoke.jsonl"
DEFAULT_OUT = ROOT / "runs/fastdllm_qwen25_1p5b_diffusion_baseline/synthetic_onecall.jsonl"
MASK_ID = 151665
STOP_TOKEN_ID = 151645
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


def load_cases(path, limit):
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def load_sampler_schedules(path):
    if not path:
        return {}, {}
    schedules = {}
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            key = case_key(row, idx)
            rows[key] = row
            schedules[key] = row.get("schedule") or []
    return schedules, rows


def parse_kind_set(raw):
    if not raw:
        return set()
    items = set()
    for item in re.split(r"[,\s]+", str(raw)):
        item = item.strip()
        if item:
            items.add(item)
    return items


def resolve_single_token_ids(tokenizer, strings):
    ids = set()
    for text in strings:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            ids.add(int(token_ids[0]))
    return sorted(ids)


def is_json_boundary_piece(text):
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if any(ch.isalnum() or ch == "_" for ch in stripped):
        return False
    return any(ch in stripped for ch in ['"', ",", "{", "}", "[", "]"])


def target_boundary_mask(tokenizer, token_ids, cache):
    mask = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id not in cache:
            cache[token_id] = is_json_boundary_piece(tokenizer.decode([token_id]))
        mask.append(cache[token_id])
    return mask


def contiguous_decoded_prefix(tokenizer, token_ids, mask_id):
    visible = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == mask_id:
            break
        visible.append(token_id)
    if not visible:
        return ""
    return tokenizer.decode(visible, skip_special_tokens=True)


def active_tool_call_body(text):
    open_idx = text.rfind(TOOL_OPEN)
    if open_idx < 0:
        return None
    close_idx = text.rfind(TOOL_CLOSE)
    if close_idx > open_idx:
        return None
    return text[open_idx + len(TOOL_OPEN) :]


def json_body_complete(body):
    stripped = body.lstrip()
    if not stripped:
        return False
    json_start = body.find("{")
    if json_start < 0:
        return False
    if body[:json_start].strip():
        return False

    parser = JsonPrefixParser(body[json_start:])
    pos = parser.skip_ws(0)
    status, pos = parser.parse_value(pos)
    if status != "complete":
        return False
    trailing = parser.text[parser.skip_ws(pos) :]
    return not trailing


def active_tool_call_json_started(text):
    body = active_tool_call_body(text)
    if body is None:
        return False
    json_start = body.find("{")
    if json_start < 0:
        return False
    return bool(body[json_start:].strip())


def active_tool_call_json_complete(text):
    body = active_tool_call_body(text)
    if body is None:
        return True
    return json_body_complete(body)


def json_prefix_completable(body):
    stripped = body.lstrip()
    if not stripped:
        return True
    json_start = body.find("{")
    if json_start < 0:
        return "{".startswith(stripped)
    if body[:json_start].strip():
        return False

    parser = JsonPrefixParser(body[json_start:])
    pos = parser.skip_ws(0)
    status, pos = parser.parse_value(pos)
    if status == "incomplete":
        return True
    if status != "complete":
        return False

    trailing = parser.text[parser.skip_ws(pos) :]
    if not trailing:
        return True
    return TOOL_CLOSE.startswith(trailing.lstrip())


def tool_json_prefix_completable(text):
    cursor = 0
    while True:
        open_idx = text.find(TOOL_OPEN, cursor)
        if open_idx < 0:
            return True
        body_start = open_idx + len(TOOL_OPEN)
        close_idx = text.find(TOOL_CLOSE, body_start)
        if close_idx < 0:
            return json_prefix_completable(text[body_start:])
        if not json_body_complete(text[body_start:close_idx]):
            return False
        cursor = close_idx + len(TOOL_CLOSE)


def tool_call_mode_force_mask(
    tokenizer,
    x_t,
    current_mask,
    original_len,
    args,
    schedule_events,
):
    force_mask = current_mask.clone()
    for row_idx in range(x_t.shape[0]):
        if not current_mask[row_idx].any():
            continue
        generated = x_t[row_idx, original_len:].detach().tolist()
        text = contiguous_decoded_prefix(tokenizer, generated, args.mask_id)
        if active_tool_call_json_started(text) and not active_tool_call_json_complete(text):
            force_mask[row_idx, :] = False
            schedule_events["tool_call_mode_close_deferred_interval_visits"] += 1
            schedule_events["tool_call_mode_close_deferred_token_visits"] += int(
                current_mask[row_idx].sum().item()
            )
    return force_mask


def proposal_keeps_tool_json_prefix(tokenizer, sequence, original_len, abs_idx, token_id, mask_id):
    proposal = sequence.clone()
    proposal[abs_idx] = int(token_id)
    generated = proposal[original_len:].detach().tolist()
    text = contiguous_decoded_prefix(tokenizer, generated, mask_id)
    return tool_json_prefix_completable(text)


def tool_json_live_prefix_active(text):
    body = active_tool_call_body(text)
    if body is not None:
        return True
    stripped = str(text or "").lstrip()
    return bool(stripped and TOOL_OPEN.startswith(stripped))


def live_tool_json_top_token(
    tokenizer,
    sequence,
    logits,
    original_len,
    abs_idx,
    mask_id,
    topk,
):
    topk = max(1, min(int(topk), logits.shape[-1]))
    for token_id in torch.topk(logits, k=topk).indices.detach().tolist():
        token_id = int(token_id)
        if proposal_keeps_tool_json_prefix(tokenizer, sequence, original_len, abs_idx, token_id, mask_id):
            return token_id, True
    return int(torch.argmax(logits).item()), False


def apply_live_tool_json_grammar(
    tokenizer,
    x_t,
    logits,
    current_mask,
    window_abs_start,
    span_start,
    original_len,
    args,
    schedule_events,
):
    if not current_mask.any():
        return None, None, False

    x_1 = torch.full(
        current_mask.shape,
        args.mask_id,
        dtype=torch.long,
        device=x_t.device,
    )
    unmask_idx = torch.zeros_like(current_mask, dtype=torch.bool)
    active_rows = 0
    rejected = 0
    unsafe = 0

    for row_idx in range(x_t.shape[0]):
        if not current_mask[row_idx].any():
            continue
        generated = x_t[row_idx, original_len:].detach().tolist()
        text = contiguous_decoded_prefix(tokenizer, generated, args.mask_id)
        if not tool_json_live_prefix_active(text):
            continue

        active_rows += 1
        local_pos = int(current_mask[row_idx].nonzero(as_tuple=False)[0, 0].item())
        abs_idx = window_abs_start + span_start + local_pos
        token_id, safe = live_tool_json_top_token(
            tokenizer,
            x_t[row_idx].clone(),
            logits[row_idx, local_pos],
            original_len,
            abs_idx,
            args.mask_id,
            args.live_tool_json_topk,
        )
        if not safe:
            unsafe += 1
        original_top = int(torch.argmax(logits[row_idx, local_pos]).item())
        if token_id != original_top:
            rejected += 1
        x_1[row_idx, local_pos] = token_id
        unmask_idx[row_idx, local_pos] = True

    if not active_rows:
        return None, None, False

    dropped = int(current_mask.sum().item()) - int(unmask_idx.sum().item())
    schedule_events["live_tool_json_grammar_interval_visits"] += 1
    schedule_events["live_tool_json_grammar_active_rows"] += active_rows
    schedule_events["live_tool_json_grammar_token_visits"] += int(unmask_idx.sum().item())
    schedule_events["live_tool_json_grammar_left_to_right_dropped_token_visits"] += dropped
    schedule_events["live_tool_json_grammar_replacement_token_visits"] += rejected
    schedule_events["live_tool_json_grammar_unsafe_fallback_token_visits"] += unsafe
    return x_1, unmask_idx, True


def apply_tool_json_prefix_guard(
    tokenizer,
    x_t,
    logits,
    x_1,
    unmask_idx,
    current_mask,
    window_abs_start,
    window_len,
    span_start,
    original_len,
    target_token_ids,
    args,
    schedule_events,
):
    if args.json_prefix_guard_left_to_right:
        leftmost = current_mask & (current_mask.cumsum(dim=-1) == 1)
        dropped = int((unmask_idx & ~leftmost).sum().item())
        if dropped:
            schedule_events["json_prefix_guard_left_to_right_dropped_token_visits"] += dropped
        unmask_idx = unmask_idx & leftmost
        missing_rows = leftmost.any(dim=-1) & ~unmask_idx.any(dim=-1)
        if missing_rows.any():
            unmask_idx = unmask_idx | (leftmost & missing_rows.unsqueeze(-1))

    if not unmask_idx.any():
        return x_1, unmask_idx

    schedule_events["json_prefix_guard_interval_visits"] += 1
    for row_idx, pos_idx in unmask_idx.nonzero(as_tuple=False).tolist():
        sequence = x_t[row_idx].clone()
        abs_idx = window_abs_start + span_start + int(pos_idx)
        original_token = int(x_1[row_idx, pos_idx].item())
        if proposal_keeps_tool_json_prefix(
            tokenizer,
            sequence,
            original_len,
            abs_idx,
            original_token,
            args.mask_id,
        ):
            schedule_events["json_prefix_guard_accepted_token_visits"] += 1
            continue

        schedule_events["json_prefix_guard_rejected_token_visits"] += 1
        topk = min(int(args.json_prefix_guard_topk), logits.shape[-1])
        replacement = None
        if topk > 0:
            for token_id in torch.topk(logits[row_idx, pos_idx], k=topk).indices.detach().tolist():
                token_id = int(token_id)
                if proposal_keeps_tool_json_prefix(
                    tokenizer,
                    sequence,
                    original_len,
                    abs_idx,
                    token_id,
                    args.mask_id,
                ):
                    replacement = token_id
                    break
        if replacement is None and args.json_prefix_guard_target_fallback and len(target_token_ids) == (logits.shape[1]):
            target_token = int(target_token_ids[pos_idx])
            if proposal_keeps_tool_json_prefix(
                tokenizer,
                sequence,
                original_len,
                abs_idx,
                target_token,
                args.mask_id,
            ):
                replacement = target_token
                schedule_events["json_prefix_guard_target_fallback_token_visits"] += 1

        if replacement is None:
            schedule_events["json_prefix_guard_unsafe_fallback_token_visits"] += 1
            continue

        if replacement != original_token:
            schedule_events["json_prefix_guard_replacement_token_visits"] += 1
        x_1[row_idx, pos_idx] = replacement

    return x_1, unmask_idx


def generation_instruction(case):
    return case.get("teacher_instruction") or (
        "Return the necessary Qwen tool call or calls for the request above. "
        "Use only this format and no prose:\n"
        "<tool_call>\n"
        "<function=tool_name>\n"
        "<parameter=argument_name>\n"
        "argument value\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def model_repair_instruction(raw_text):
    return (
        "The previous assistant draft below may have malformed tool-call syntax, "
        "wrong JSON punctuation, missing argument keys, or extra prose. Rewrite it "
        "using the same user request and available tools. Return only valid Qwen "
        "tool-call block(s) in this exact shape, with no prose before or after:\n"
        "<tool_call>\n"
        "<function=tool_name>\n"
        "<parameter=argument_name>\n"
        "argument value\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "Previous assistant draft:\n"
        f"{raw_text}"
    )


def model_repair_case(case, raw_text):
    repaired = copy.deepcopy(case)
    repaired["prompt_messages"] = list(copy.deepcopy(case.get("prompt_messages") or []))
    repaired["prompt_messages"].append({"role": "user", "content": model_repair_instruction(raw_text)})
    return repaired


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


def apply_chat_template(tokenizer, messages, tools, chat_template=None):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    return tokenizer.apply_chat_template(messages, **kwargs)


def make_prompt(tokenizer, case, append_instruction, chat_template=None):
    messages = list(case["prompt_messages"])
    if append_instruction:
        messages.append({"role": "user", "content": generation_instruction(case)})
    return apply_chat_template(tokenizer, messages, case.get("tools") or None, chat_template=chat_template)


def load_model(base_model, adapter, merge_adapter, tokenizer_path=None):
    repo_v2 = ROOT / "fast-dllm/v2"
    sys.path.insert(0, str(repo_v2))
    import generation_functions

    tokenizer_path = tokenizer_path or adapter or base_model
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

    stop_token_ids = []

    def add_stop(value):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add_stop(item)
            return
        item = int(value)
        if item not in stop_token_ids:
            stop_token_ids.append(item)

    add_stop(tokenizer.eos_token_id)
    add_stop(getattr(model.config, "eos_token_id", None))
    for text in ("<|im_end|>", "<|im_start|>"):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            add_stop(token_ids[0])
    if not stop_token_ids:
        add_stop(STOP_TOKEN_ID)

    return int(mask_id), int(stop_token_ids[0]), stop_token_ids


def function_name(tool):
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    return str(fn.get("name")) if isinstance(fn, dict) and fn.get("name") else None


def normalize_for_match(text):
    return re.sub(r"[-_\s]+", "", str(text).lower())


def closest_tool_name(name, available_names):
    if not name:
        return None
    if name in available_names:
        return name
    normalized = normalize_for_match(name)
    for candidate in available_names:
        candidate_norm = normalize_for_match(candidate)
        if normalized and (candidate_norm.startswith(normalized) or normalized.startswith(candidate_norm)):
            return candidate
    matches = difflib.get_close_matches(str(name), sorted(available_names), n=1, cutoff=0.72)
    return matches[0] if matches else None


def ordered_tool_mentions(text, available_names):
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
    return names


def regex_name_mentions(text):
    names = []
    for pattern in [
        r'"name"\s*:\s*"([^"]+)"',
        r'"function"\s*:\s*"([^"]+)"',
        r"<function=([^>\s]+)>",
    ]:
        for match in re.finditer(pattern, text):
            names.append(match.group(1).strip())
    return names


def tool_call_bodies(text):
    bodies = []
    cursor = 0
    while True:
        start = text.find("<tool_call>", cursor)
        if start < 0:
            break
        content_start = start + len("<tool_call>")
        end = text.find("</tool_call>", content_start)
        if end < 0:
            bodies.append({"body": text[content_start:], "complete": False})
            break
        bodies.append({"body": text[content_start:end], "complete": True})
        cursor = end + len("</tool_call>")
    return bodies


def trim_after_tool_call_closers(text, max_calls):
    if not max_calls or max_calls <= 0:
        return text, False
    cursor = 0
    end = -1
    for _ in range(max_calls):
        end = text.find("</tool_call>", cursor)
        if end < 0:
            return text, False
        cursor = end + len("</tool_call>")
    trimmed = text[:cursor].rstrip()
    return trimmed, trimmed != text.rstrip()


def parse_jsonish_call(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    snippet = text[start : end + 1]
    candidates = [snippet]
    fixed = snippet
    fixed = fixed.replace('arguments":":', 'arguments":')
    fixed = fixed.replace('arguments": ":', 'arguments":')
    fixed = re.sub(r'([,{]\s*)([A-Za-z_][\w-]*)"\s*:', r'\1"\2":', fixed)
    fixed = re.sub(r'"([A-Za-z_][\w-]*)\s+"', r'"\1": "', fixed)
    fixed = re.sub(r'"([A-Za-z_][\w-]*)"\s+"', r'"\1": "', fixed)
    candidates.append(fixed)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalize_arg_key(key, properties):
    if key in properties:
        return key
    stripped = str(key).strip().strip("_")
    if stripped in properties:
        return stripped
    stripped_norm = normalize_for_match(stripped)
    for prop in properties:
        prop_norm = normalize_for_match(prop)
        if stripped_norm and (prop_norm.endswith(stripped_norm) or stripped_norm.endswith(prop_norm)):
            return prop
    matches = difflib.get_close_matches(stripped, sorted(properties), n=1, cutoff=0.7)
    return matches[0] if matches else str(key)


def parsed_arguments_from_text(text):
    payload = parse_jsonish_call(text) or {}
    arguments = payload.get("arguments", {}) if isinstance(payload, dict) else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if isinstance(arguments, dict) and arguments:
        return arguments
    qwen_calls = extract_qwen_function_calls(text)
    if qwen_calls:
        return qwen_calls[0].get("arguments") or {}
    return arguments if isinstance(arguments, dict) else {}


def candidate_call_from_tool_body(body, available_names):
    payload = parse_jsonish_call(body) or {}
    name = None
    arguments = {}
    qwen_calls = extract_qwen_function_calls(body)
    if qwen_calls:
        call = qwen_calls[0]
        name = closest_tool_name(call.get("name"), available_names)
        arguments = call.get("arguments") or {}
    if isinstance(payload, dict):
        raw_name = payload.get("name") or payload.get("function") or payload.get("tool_name")
        if isinstance(raw_name, dict):
            arguments = raw_name.get("arguments") or payload.get("arguments") or {}
            raw_name = raw_name.get("name")
        else:
            arguments = payload.get("arguments") or {}
        name = closest_tool_name(raw_name, available_names)
    if not name:
        for raw_name in regex_name_mentions(body):
            name = closest_tool_name(raw_name, available_names)
            if name:
                break
    if not name:
        mentions = ordered_tool_mentions(body, available_names)
        name = mentions[0] if mentions else None
    if not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name, "arguments": arguments}


def case_context_text(case):
    parts = []
    for message in case.get("prompt_messages") or []:
        role = message.get("role")
        if role in {"user", "tool", "assistant"}:
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(content)
    return "\n".join(parts)


def enum_value_from_text(text, values):
    normalized_text = normalize_for_match(text)
    for value in values:
        raw = str(value)
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(raw) + r"(?![A-Za-z0-9_])", text, flags=re.IGNORECASE):
            return value
        normalized = normalize_for_match(value)
        if normalized and re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(normalized) + r"(?![A-Za-z0-9_])",
            normalized_text,
            flags=re.IGNORECASE,
        ):
            return value
    return None


def string_value_from_text(text, key):
    separator = value_separator_pattern()
    prefix = r"(?<![A-Za-z0-9_])"
    for text_key in property_text_keys(key):
        key_pattern = key_pattern_for_text(text_key)
        patterns = [
            prefix + r'"?' + key_pattern + r'"?' + separator + r'"([^"]+)"',
            prefix + r'"' + key_pattern + r'\s+"([^"]+)"',
            prefix + r'"?' + key_pattern + r'"?' + separator + r"([^,}\]\n<]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip().strip('"').strip()
            if value and value not in {":", ": "}:
                return value
    return None


def property_text_keys(key):
    key = str(key)
    keys = [key]
    alias_map = {
        "body": ["email_body", "message_body"],
        "date": ["callback_date", "appointment_date", "delivery_date"],
        "recipient": ["email", "customer_email"],
        "subject": ["email_subject", "message_subject"],
        "time": ["callback_time", "appointment_time", "delivery_time"],
    }
    for alias in alias_map.get(key.lower(), []):
        if alias not in keys:
            keys.append(alias)
    return keys


def number_value_from_text(text, key, expected):
    key_pattern = key_pattern_for_text(key)
    separator = value_separator_pattern()
    prefix = r"(?<![A-Za-z0-9_])"
    patterns = [
        prefix + r'"?' + key_pattern + r'"?' + separator + r'"?(-?\d+(?:\.\d+)?)',
        prefix + r'"?' + key_pattern + r'"?' + separator + r'"?(-?\d(?:[\s_]\d)+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1)
        if re.search(r"\d[\s_]\d", raw):
            raw = re.sub(r"[\s_]+", "", raw)
        try:
            return int(float(raw)) if expected == "integer" else float(raw)
        except Exception:
            continue
    return None


def boolean_value_from_text(text, key):
    key_pattern = key_pattern_for_text(key)
    separator = value_separator_pattern()
    match = re.search(
        r"(?<![A-Za-z0-9_])" + r'"?' + key_pattern + r'"?' + separator + r"(true|false)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower() == "true"
    return None


def extract_balanced_fragment(text, start_char, end_char, start_index):
    start = text.find(start_char, start_index)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def json_value_after_key(text, key, expected):
    key_match = re.search(re.escape(str(key)), text, flags=re.IGNORECASE)
    if not key_match:
        return None
    if expected == "array":
        fragment = extract_balanced_fragment(text, "[", "]", key_match.end())
    elif expected == "object":
        fragment = extract_balanced_fragment(text, "{", "}", key_match.end())
    else:
        return None
    if not fragment:
        return None
    try:
        return json.loads(fragment)
    except Exception:
        return None


def normalize_table_header(header):
    return normalize_for_match(header)


def markdown_tables(text):
    lines = text.splitlines()
    tables = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "|" not in line or idx + 1 >= len(lines) or "|" not in lines[idx + 1]:
            idx += 1
            continue
        separator = lines[idx + 1]
        if not re.search(r"\|\s*:?-{2,}:?\s*(?:\||$)", separator):
            idx += 1
            continue
        headers = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(headers) < 2:
            idx += 1
            continue
        rows = []
        row_idx = idx + 2
        while row_idx < len(lines) and "|" in lines[row_idx]:
            parts = [part.strip() for part in lines[row_idx].strip().strip("|").split("|")]
            if len(parts) == len(headers):
                rows.append(parts)
            row_idx += 1
        title_lines = []
        back = idx - 1
        while back >= 0 and len(title_lines) < 3:
            previous = lines[back].strip().strip("*").strip()
            if previous and not previous.startswith("```"):
                title_lines.append(previous)
            back -= 1
        if rows:
            tables.append({"title": " ".join(reversed(title_lines)), "headers": headers, "rows": rows})
        idx = max(row_idx, idx + 1)
    return tables


def coerce_schema_value(raw, schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
    value = str(raw).strip()
    if expected == "integer":
        try:
            return int(float(value.replace(",", "")))
        except Exception:
            return None
    if expected == "number":
        try:
            number = float(value.replace(",", ""))
            return int(number) if number.is_integer() else number
        except Exception:
            return None
    if expected == "boolean":
        if value.lower() in {"true", "yes"}:
            return True
        if value.lower() in {"false", "no"}:
            return False
        return None
    return value.strip('"').strip()


def plural_variants(token):
    variants = {token}
    if token.endswith("y") and len(token) > 1:
        variants.add(token[:-1] + "ies")
    elif token.endswith("s"):
        variants.add(token[:-1])
    else:
        variants.add(token + "s")
    return variants


def token_match_score(prop, table):
    prop_tokens = [token for token in re.split(r"[_\W]+", str(prop).lower()) if len(token) >= 3]
    haystack = " ".join([table.get("title") or "", " ".join(table.get("headers") or [])]).lower()
    score = 0
    for token in prop_tokens:
        score += int(any(variant in haystack for variant in plural_variants(token)))
    return score


def object_array_from_markdown_table(text, key, schema):
    item_schema = (schema or {}).get("items") if isinstance(schema, dict) else {}
    if not isinstance(item_schema, dict):
        return None
    item_properties = (item_schema.get("properties") or {}) if isinstance(item_schema.get("properties"), dict) else {}
    if not item_properties:
        return None
    required = set(item_schema.get("required") or [])
    candidates = []
    for table in markdown_tables(text):
        header_map = {}
        for header_idx, header in enumerate(table["headers"]):
            normalized_header = normalize_table_header(header)
            for prop in item_properties:
                if normalized_header == normalize_for_match(prop):
                    header_map[prop] = header_idx
                    break
        if required and not required.issubset(header_map):
            continue
        if len(header_map) < max(1, min(2, len(item_properties))):
            continue
        rows = []
        for parts in table["rows"]:
            row = {}
            for prop, header_idx in header_map.items():
                value = coerce_schema_value(parts[header_idx], item_properties.get(prop, {}))
                if value is not None:
                    row[prop] = value
            if row:
                rows.append(row)
        if rows:
            candidates.append((token_match_score(key, table), len(header_map), rows))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    return candidates[0][2]


def clean_list_item(item):
    item = re.sub(r"^[\-\*\d\.)\s]+", "", str(item)).strip()
    item = re.sub(r"\s+", " ", item)
    return item.strip(" ,.;")


def string_array_from_bullets(text, key):
    lines = text.splitlines()
    prop_tokens = [token for token in re.split(r"[_\W]+", str(key).lower()) if len(token) >= 4]
    groups = []
    idx = 0
    while idx < len(lines):
        if not re.match(r"^\s*[-*]\s+", lines[idx]):
            idx += 1
            continue
        start = idx
        items = []
        while idx < len(lines) and re.match(r"^\s*[-*]\s+", lines[idx]):
            item = clean_list_item(lines[idx])
            if item:
                items.append(item)
            idx += 1
        context = "\n".join(lines[max(0, start - 3) : start]).lower()
        score = sum(int(any(variant in context for variant in plural_variants(token))) for token in prop_tokens)
        if items:
            groups.append((score, items))
        continue
    if not groups:
        return None
    groups.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return groups[0][1]


def string_array_from_inline_list(text, key):
    prop_tokens = [token for token in re.split(r"[_\W]+", str(key).lower()) if len(token) >= 4]
    if prop_tokens and not any(any(variant in text.lower() for variant in plural_variants(token)) for token in prop_tokens):
        return None
    patterns = [
        r"(?:such as|including|include|languages)\s+([A-Z][A-Za-z0-9+# .-]*(?:,\s*[A-Z#][A-Za-z0-9+# .-]*)*(?:,\s*and\s*[A-Z#][A-Za-z0-9+# .-]*| and [A-Z#][A-Za-z0-9+# .-]*)?)",
        r"(?:for|with)\s+(?:the\s+)?(?:[\w\s]+?\s+)?([A-Z][A-Za-z0-9+# .-]*(?:,\s*[A-Z#][A-Za-z0-9+# .-]*)+(?:,\s*and\s*[A-Z#][A-Za-z0-9+# .-]*| and [A-Z#][A-Za-z0-9+# .-]*)?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(1)
            raw = re.split(r"\s+(?:which|that|as well as|to provide|to acquire)\b", raw, maxsplit=1)[0]
            parts = [clean_list_item(part) for part in re.split(r",|\band\b", raw) if clean_list_item(part)]
            parts = [part for part in parts if len(part) <= 80 and not part.lower().startswith("the ")]
            if len(parts) >= 2:
                return parts
    return None


def complex_value_from_context(text, key, schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
    if expected == "array":
        weekly_value = weekly_temperature_schedule_from_context(text, key, schema)
        if weekly_value is not None:
            return weekly_value
        item_schema = schema.get("items") if isinstance(schema, dict) else {}
        item_type = item_schema.get("type") if isinstance(item_schema, dict) else None
        if item_type == "object" or isinstance(item_schema, dict) and item_schema.get("properties"):
            table_value = object_array_from_markdown_table(text, key, schema)
            if table_value is not None:
                return table_value
        if item_type in {None, "string"}:
            bullet_value = string_array_from_bullets(text, key)
            if bullet_value is not None:
                return bullet_value
            inline_value = string_array_from_inline_list(text, key)
            if inline_value is not None:
                return inline_value
    return None


def day_temperature_time_properties(schema):
    item_schema = schema.get("items") if isinstance(schema, dict) else {}
    item_props = item_schema.get("properties") if isinstance(item_schema, dict) else {}
    if not isinstance(item_props, dict):
        return None
    day_prop = next((prop for prop in item_props if "day" in prop.lower()), None)
    temp_prop = next((prop for prop in item_props if "temp" in prop.lower()), None)
    time_prop = next((prop for prop in item_props if "time" in prop.lower()), None)
    if day_prop and temp_prop and time_prop:
        return day_prop, temp_prop, time_prop
    return None


def normalize_clock_time(hour, minute):
    return f"{int(hour):02d}:{int(minute):02d}"


def clock_times_from_text(text):
    times = []
    for match in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text):
        times.append((int(match.group(1)), int(match.group(2))))
    for match in re.finditer(r"\b(1[0-2]|0?[1-9])\s*(AM|PM)\b", text, flags=re.IGNORECASE):
        hour = int(match.group(1))
        suffix = match.group(2).lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        times.append((hour, 0))
    out = []
    for item in times:
        if item not in out:
            out.append(item)
    return out


def temperatures_from_text(text):
    values = []
    for match in re.finditer(r"\b(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|fahrenheit|f\b)", text, flags=re.IGNORECASE):
        number = float(match.group(1))
        value = int(number) if number.is_integer() else number
        if value not in values:
            values.append(value)
    return values


def weekly_temperature_schedule_from_context(text, key, schema):
    if "schedule" not in str(key).lower():
        return None
    props = day_temperature_time_properties(schema)
    if not props:
        return None
    lower = text.lower()
    if not all(token in lower for token in ["weekday", "weekend"]):
        return None
    temperatures = temperatures_from_text(text)
    times = clock_times_from_text(text)
    if len(temperatures) < 2 or len(times) < 2:
        return None

    day_prop, temp_prop, time_prop = props
    warm_temp, cool_temp = temperatures[0], temperatures[1]
    weekday_warm_time = normalize_clock_time(*times[0])
    cool_time = normalize_clock_time(*times[1])
    weekend_warm_time = normalize_clock_time(*times[2]) if len(times) >= 3 else weekday_warm_time

    rows = []
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        rows.append({day_prop: day, temp_prop: warm_temp, time_prop: weekday_warm_time})
        rows.append({day_prop: day, temp_prop: cool_temp, time_prop: cool_time})
    for day in ["Saturday", "Sunday"]:
        rows.append({day_prop: day, temp_prop: warm_temp, time_prop: weekend_warm_time})
        rows.append({day_prop: day, temp_prop: cool_temp, time_prop: cool_time})
    return rows


def contextual_string_value_from_text(text, key, schema):
    if not text:
        return None
    lower_key = str(key).lower()
    if is_id_like_property_name(lower_key):
        value = id_value_from_context(text, lower_key)
        if value is not None:
            return value
    if lower_key == "voice_command":
        match = re.search(r"\bvoice\s+command\s+['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if lower_key == "music_playlist":
        patterns = [
            r"\bplaylist(?:\s+titled|\s+named)?\s+['\"]([^'\"]+)['\"]",
            r"['\"]([^'\"]+)['\"]\s+playlist\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    if lower_key == "lighting_scene":
        match = re.search(r"\bto\s+a\s+([^,\n.]+?)\s+ambiance\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    if lower_key == "function_type" and re.search(r"\bperiodic\s+functions?\b", text, flags=re.IGNORECASE):
        return "periodic"
    return string_value_from_text(text, key)


def is_id_like_property_name(prop):
    lower = str(prop).lower()
    return lower == "id" or lower.endswith("_id") or lower.endswith("id")


def id_value_from_context(text, prop):
    if not text:
        return None
    key_prefix = re.sub(r"(?:_?id)$", "", str(prop).lower()).replace("_", r"[\s_-]+")
    patterns = []
    if key_prefix:
        patterns.append(
            rf"\b{key_prefix}\b[^.\n]{{0,80}}\b(?:id|identifier)\b\s*(?:as|is|:)?\s*['\"]?([A-Za-z0-9][A-Za-z0-9_.:-]*)['\"]?"
        )
    patterns.extend(
        [
            r"\b(?:device\s+)?(?:id|identifier)\b\s*(?:as|is|:)?\s*['\"]?([A-Za-z0-9][A-Za-z0-9_.:-]*)['\"]?",
            r"\b(?:unique\s+)?(?:device\s+)?identifier\s+as\s+['\"]([^'\"]+)['\"]",
        ]
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1).strip().strip(".,;")
            if re.search(r"\d", value):
                return value
    return None


def should_prefer_contextual_string(prop, parsed_value, contextual_value):
    if contextual_value is None:
        return False
    if parsed_value is None:
        return True
    if not isinstance(parsed_value, str):
        return False
    parsed_clean = clean_repeated_scalar_value(parsed_value)
    contextual_clean = clean_repeated_scalar_value(contextual_value)
    if not parsed_clean:
        return True
    if parsed_clean == contextual_clean:
        return False
    lower_prop = str(prop).lower()
    if lower_prop in {"voice_command", "music_playlist", "lighting_scene", "function_type"}:
        return True
    if is_id_like_property_name(lower_prop) and re.search(r"\d", str(contextual_value)):
        return True
    if parsed_clean != parsed_value:
        return True
    return False


def constrained_value_from_text(text, key, schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
    if isinstance(schema, dict) and "enum" in schema:
        return enum_value_from_text(text, schema["enum"])
    if expected in {"integer", "number"}:
        return number_value_from_text(text, key, expected)
    if expected == "boolean":
        return boolean_value_from_text(text, key)
    if expected in {"array", "object"}:
        contextual = complex_value_from_context(text, key, schema)
        if contextual is not None:
            return contextual
        parsed = json_value_after_key(text, key, expected)
        if parsed is not None:
            return parsed
        return None
    return contextual_string_value_from_text(text, key, schema)


def key_pattern_for_text(key):
    parts = [part for part in re.split(r"[_\s-]+", str(key)) if part]
    if not parts:
        return re.escape(str(key))
    return r"[_\s-]+".join(re.escape(part) for part in parts)


def value_separator_pattern():
    return r"\s*(?::|=|is|of|to|as|at)\s*"


def has_tool_result_evidence(text):
    return bool(re.search(r"\bTool result for\b|\brole=tool\b|<tool_response>", str(text or ""), flags=re.IGNORECASE))


def clean_repeated_scalar_value(value):
    if not isinstance(value, str):
        return value
    parts = re.split(r"([\s_-]+)", value)
    cleaned = []
    last_word = None
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[\s_-]+", part):
            cleaned.append(part)
            continue
        norm = part.lower()
        if norm == last_word:
            if cleaned and re.fullmatch(r"[\s_-]+", cleaned[-1]):
                cleaned.pop()
            continue
        cleaned.append(part)
        last_word = norm
    out = "".join(cleaned).strip()
    out = re.sub(r"\b([A-Za-z0-9]+)(?:[ _-]+\1\b)+", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b([A-Za-z]{3,})\1\b", r"\1", out, flags=re.IGNORECASE)
    for width in (3, 2):
        pattern = r"\b((?:[A-Za-z0-9]+[ _-]+){" + str(width - 1) + r"}[A-Za-z0-9]+)(?:[ _-]+\1\b)+"
        out = re.sub(pattern, r"\1", out, flags=re.IGNORECASE)
    return out


def clean_repeated_values(value):
    if isinstance(value, dict):
        return {key: clean_repeated_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_repeated_values(item) for item in value]
    return clean_repeated_scalar_value(value)


def should_assume_utc_z(prop, prop_schema):
    if not isinstance(prop_schema, dict):
        return False
    text = " ".join(
        str(part or "")
        for part in [
            prop,
            prop_schema.get("description"),
            prop_schema.get("format"),
            prop_schema.get("title"),
        ]
    ).lower()
    return (
        "iso 8601" in text
        or "date-time" in text
        or "datetime" in text
        or str(prop).lower().endswith("_time")
    )


def normalize_naive_iso8601_datetime(value, prop, prop_schema, assume_utc_z=False):
    if not assume_utc_z or not isinstance(value, str):
        return value
    if not should_assume_utc_z(prop, prop_schema):
        return value
    stripped = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?", stripped):
        return stripped + "Z"
    return value


def tool_context_window(context_text, name, radius=700):
    if not context_text or not name:
        return ""
    candidates = [name, name.replace("_", " "), name.replace("_", "-")]
    spans = []
    for candidate in candidates:
        for match in re.finditer(re.escape(candidate), context_text, flags=re.IGNORECASE):
            start = max(0, match.start() - radius // 3)
            end = min(len(context_text), match.end() + radius)
            spans.append(context_text[start:end])
    return "\n".join(spans)


def normalize_parsed_arguments(arguments, properties):
    if not isinstance(arguments, dict):
        return {}, []
    normalized = {}
    unknown = []
    for key, value in arguments.items():
        normalized_key = normalize_arg_key(key, properties)
        if normalized_key in properties:
            normalized[normalized_key] = value
        else:
            unknown.append(value)
    return normalized, unknown


def choose_constrained_argument_value(
    prop,
    prop_schema,
    parsed_value,
    scoped_context,
    full_context,
    generated_text,
    assume_utc_z=False,
):
    scoped = constrained_value_from_text(scoped_context, prop, prop_schema) if scoped_context else None
    full = constrained_value_from_text(full_context, prop, prop_schema) if full_context else None

    expected = prop_schema.get("type") if isinstance(prop_schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)

    if isinstance(prop_schema, dict) and "enum" in prop_schema:
        return scoped if scoped is not None else full if full is not None else parsed_value

    if expected in {"integer", "number", "boolean"}:
        return scoped if scoped is not None else parsed_value if parsed_value is not None else full

    if expected in {"array", "object"}:
        value = scoped if scoped is not None else full if full is not None else parsed_value
        return clean_repeated_values(value)

    contextual = scoped if scoped is not None else full
    if expected == "string" and should_prefer_contextual_string(prop, parsed_value, contextual):
        value = contextual
    else:
        value = parsed_value if parsed_value is not None else contextual
    value = clean_repeated_scalar_value(value)
    if isinstance(value, str):
        if not value.strip():
            value = scoped if scoped is not None else full
            value = clean_repeated_scalar_value(value)
        full_evidence = "\n".join(part for part in [scoped_context, full_context, generated_text] if part)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value) and value + "Z" in full_evidence:
            value = value + "Z"
        value = normalize_naive_iso8601_datetime(value, prop, prop_schema, assume_utc_z=assume_utc_z)
    return value


def sequence_preserving_constrained_tool_call_text(
    text,
    tools,
    context_text="",
    max_calls=0,
    assume_utc_z=False,
):
    schemas = tool_schema_by_name(tools)
    available_names = set(schemas)
    if not available_names:
        return ""

    calls = []
    for item in tool_call_bodies(text):
        if not item.get("complete"):
            continue
        call = candidate_call_from_tool_body(item.get("body") or "", available_names)
        if call:
            calls.append(call)

    if not calls:
        strict_calls, _ = extract_tool_calls(text)
        for call in strict_calls:
            repaired = closest_tool_name(call.get("name"), available_names)
            if repaired:
                calls.append({"name": repaired, "arguments": call.get("arguments") or {}})

    if not calls:
        for raw_name in regex_name_mentions(text):
            repaired = closest_tool_name(raw_name, available_names)
            if repaired:
                calls.append({"name": repaired, "arguments": {}})

    if not calls:
        calls = [{"name": name, "arguments": {}} for name in ordered_tool_mentions(text, available_names)]

    if max_calls and max_calls > 0:
        calls = calls[:max_calls]
    if not calls:
        return ""

    constrained_calls = []
    full_context = "\n".join(part for part in [context_text, text] if part)
    for call in calls:
        name = call["name"]
        schema = schemas.get(name) or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        required = set((schema or {}).get("required") or [])
        normalized_parsed, unknown_parsed_values = normalize_parsed_arguments(call.get("arguments") or {}, properties)
        scoped_context = tool_context_window(context_text, name) or context_text
        string_props = [
            prop
            for prop, prop_schema in properties.items()
            if (prop_schema.get("type") if isinstance(prop_schema, dict) else None) == "string"
        ]

        arguments = {}
        for prop, prop_schema in properties.items():
            parsed_value = normalized_parsed.get(prop)
            value = choose_constrained_argument_value(
                prop,
                prop_schema,
                parsed_value,
                scoped_context,
                full_context,
                text,
                assume_utc_z=assume_utc_z,
            )
            if value is None and prop in string_props and len(string_props) == 1:
                string_values = [item for item in unknown_parsed_values if isinstance(item, str) and item.strip()]
                if len(string_values) == 1:
                    value = clean_repeated_scalar_value(string_values[0].strip())
            if value is not None or prop in required:
                if value is not None:
                    arguments[prop] = value

        constrained_calls.append(qwen_native_tool_call_text([{"name": name, "arguments": arguments}]))
    return "\n".join(constrained_calls)


def scalar_value_from_text(text, key, schema, fallback_numbers):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
    if isinstance(schema, dict) and "enum" in schema:
        return enum_value_from_text(text, schema["enum"])
    key_pattern = re.escape(str(key))
    if expected in {"integer", "number"}:
        match = re.search(key_pattern + r'[^0-9-]{0,24}(-?\d+(?:\.\d+)?)', text)
        if match:
            return int(float(match.group(1))) if expected == "integer" else float(match.group(1))
        if fallback_numbers:
            value = fallback_numbers.pop(0)
            return int(float(value)) if expected == "integer" else float(value)
    match = re.search(key_pattern + r'[^"\n]{0,32}"([^"]+)"', text)
    if match:
        return match.group(1)
    return None


def constrained_tool_call_text(text, tools, context_text="", max_calls=0, assume_utc_z=False):
    schemas = tool_schema_by_name(tools)
    available_names = set(schemas)
    if not available_names:
        return ""

    names = []
    strict_calls, _ = extract_tool_calls(text)
    for call in strict_calls:
        repaired = closest_tool_name(call.get("name"), available_names)
        if repaired and repaired not in names:
            names.append(repaired)
    for raw_name in regex_name_mentions(text):
        repaired = closest_tool_name(raw_name, available_names)
        if repaired and repaired not in names:
            names.append(repaired)
    if not names:
        names = ordered_tool_mentions(text, available_names)
    if not names:
        if max_calls == 1 and len(available_names) == 1:
            names = [next(iter(available_names))]
        else:
            return ""
    if max_calls and max_calls > 0:
        names = names[:max_calls]

    generated_args = parsed_arguments_from_text(text)
    evidence = "\n".join(part for part in [text, context_text] if part)
    constrained_calls = []
    for name in names:
        schema = schemas.get(name) or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        required = set((schema or {}).get("required") or [])
        arguments = {}
        normalized_parsed = {}
        unknown_parsed_values = []
        for key, value in generated_args.items():
            normalized_key = normalize_arg_key(key, properties)
            if normalized_key in properties:
                normalized_parsed[normalized_key] = value
            else:
                unknown_parsed_values.append(value)
        string_props = [
            prop
            for prop, prop_schema in properties.items()
            if (prop_schema.get("type") if isinstance(prop_schema, dict) else None) == "string"
        ]
        for prop, prop_schema in properties.items():
            value = None
            expected = prop_schema.get("type") if isinstance(prop_schema, dict) else None
            if isinstance(expected, list):
                expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
            if expected in {"array", "object"}:
                value = constrained_value_from_text(evidence, prop, prop_schema)
            if prop in normalized_parsed:
                parsed_value = normalized_parsed[prop]
                contextual_value = constrained_value_from_text(context_text, prop, prop_schema) if expected == "string" else None
                if expected == "string" and should_prefer_contextual_string(prop, parsed_value, contextual_value):
                    value = contextual_value
                else:
                    value = value if value is not None else parsed_value
            if value is None:
                if expected == "string" and has_tool_result_evidence(context_text):
                    value = constrained_value_from_text(context_text, prop, prop_schema)
                if value is None:
                    value = constrained_value_from_text(evidence, prop, prop_schema)
            if value is None and prop in string_props and len(string_props) == 1:
                string_values = [item for item in unknown_parsed_values if isinstance(item, str) and item.strip()]
                if len(string_values) == 1:
                    value = string_values[0].strip()
            value = normalize_naive_iso8601_datetime(value, prop, prop_schema, assume_utc_z=assume_utc_z)
            if value is not None or prop in required:
                if value is not None:
                    arguments[prop] = value
        constrained_calls.append(qwen_native_tool_call_text([{"name": name, "arguments": arguments}]))
    return "\n".join(constrained_calls)


def repaired_tool_call_text(text, tools):
    schemas = tool_schema_by_name(tools)
    available_names = set(schemas)
    if not available_names:
        return ""

    strict_calls, _ = extract_tool_calls(text)
    names = []
    for call in strict_calls:
        repaired = closest_tool_name(call.get("name"), available_names)
        if repaired and repaired not in names:
            names.append(repaired)
    if not names:
        for raw_name in regex_name_mentions(text):
            repaired = closest_tool_name(raw_name, available_names)
            if repaired and repaired not in names:
                names.append(repaired)
    if not names:
        names = ordered_tool_mentions(text, available_names)
    if not names:
        return ""

    parsed_args = parsed_arguments_from_text(text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    repaired_calls = []
    for name in names:
        schema = schemas.get(name) or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        arguments = {}
        normalized_parsed = {}
        for key, value in parsed_args.items():
            normalized_parsed[normalize_arg_key(key, properties)] = value
        for prop, prop_schema in properties.items():
            value = None
            if prop in normalized_parsed:
                value = normalized_parsed[prop]
                if isinstance(prop_schema, dict) and "enum" in prop_schema and value not in prop_schema["enum"]:
                    value = enum_value_from_text(str(value) + "\n" + text, prop_schema["enum"])
            if value is None:
                value = scalar_value_from_text(text, prop, prop_schema, numbers)
            if value is not None:
                arguments[prop] = value
        repaired_calls.append(qwen_native_tool_call_text([{"name": name, "arguments": arguments}]))
    return "\n".join(repaired_calls)


def sample_with_top_p(model, logits, top_p, temperature):
    if hasattr(model, "sample_with_top_p"):
        return model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
    if temperature <= 0:
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
    probs = probs.masked_fill(indices_to_remove, 0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
    return sampled.reshape(probs.shape[:-1]), probs


def default_window_intervals(window_len, small_block_size):
    return default_intervals_between(0, window_len, small_block_size)


def default_intervals_between(start, end, small_block_size):
    intervals = []
    cursor = start
    while cursor < end:
        next_cursor = min(end, cursor + small_block_size)
        intervals.append(
            {
                "start": cursor,
                "end": next_cursor,
                "kind": "default",
                "scheduled": False,
            }
        )
        cursor = next_cursor
    return intervals


def scheduled_window_intervals(
    window_len,
    window_abs_start,
    original_len,
    block_pad,
    sampler_schedule,
    small_block_size,
):
    if not sampler_schedule:
        return default_window_intervals(window_len, small_block_size)

    mask_abs_start = window_abs_start + window_len - block_pad
    mask_abs_end = window_abs_start + window_len
    scheduled = []
    for item in sampler_schedule:
        abs_start = original_len + int(item.get("token_start", 0))
        abs_end = original_len + int(item.get("token_end", 0))
        start = max(mask_abs_start, abs_start)
        end = min(mask_abs_end, abs_end)
        if start >= end:
            continue
        target_token_ids = item.get("target_token_ids") or []
        clipped_target_token_ids = []
        if target_token_ids:
            offset = start - abs_start
            clipped_target_token_ids = target_token_ids[offset : offset + (end - start)]
        candidate_allowed = item.get("candidate_allowed_token_ids_by_offset") or []
        clipped_candidate_allowed = []
        if candidate_allowed:
            offset = start - abs_start
            clipped_candidate_allowed = candidate_allowed[offset : offset + (end - start)]
        selected_candidate = item.get("selected_candidate_token_ids_by_offset") or []
        clipped_selected_candidate = []
        if selected_candidate:
            offset = start - abs_start
            clipped_selected_candidate = selected_candidate[offset : offset + (end - start)]
        candidate_sequences = item.get("candidate_sequence_token_ids_by_offset") or []
        clipped_candidate_sequences = []
        if candidate_sequences:
            offset = start - abs_start
            clipped_candidate_sequences = [
                sequence[offset : offset + (end - start)]
                for sequence in candidate_sequences
            ]
        scheduled.append(
            {
                "start": start - window_abs_start,
                "end": end - window_abs_start,
                "kind": item.get("kind") or "scheduled",
                "scheduled": True,
                "denoise_steps": item.get("denoise_steps"),
                "constraint": item.get("constraint"),
                "target_token_ids": clipped_target_token_ids,
                "candidate_allowed_token_ids_by_offset": clipped_candidate_allowed,
                "selected_candidate_token_ids_by_offset": clipped_selected_candidate,
                "candidate_sequence_token_ids_by_offset": clipped_candidate_sequences,
                "candidate_sequence_values": item.get("candidate_sequence_values") or [],
                "candidate_group_id": "|".join(
                    str(part)
                    for part in [
                        item.get("tool_call_index"),
                        item.get("json_key"),
                        item.get("target_text"),
                    ]
                ),
            }
        )

    if not scheduled:
        return default_window_intervals(window_len, small_block_size)

    intervals = []
    cursor = 0
    for item in sorted(scheduled, key=lambda row: (row["start"], row["end"])):
        if cursor < item["start"]:
            intervals.extend(default_intervals_between(cursor, item["start"], small_block_size))
        if item["end"] > cursor:
            clipped = dict(item)
            clipped["start"] = max(item["start"], cursor)
            intervals.append(clipped)
            cursor = clipped["end"]
    if cursor < window_len:
        intervals.extend(default_intervals_between(cursor, window_len, small_block_size))

    return [item for item in intervals if item["end"] > item["start"]]


def full_context_sample(model, input_ids, tokenizer, args, sampler_schedule=None, original_len_override=None):
    output_ids = input_ids
    original_len = int(original_len_override) if original_len_override is not None else input_ids.shape[1]
    stop_token_ids = torch.tensor(
        getattr(args, "stop_token_ids", [args.stop_token_id]),
        dtype=torch.long,
        device=input_ids.device,
    )

    def truncate_if_stopped(sequence):
        generated = sequence[:, original_len:]
        if generated.numel() == 0:
            return None
        stop_mask = torch.isin(generated, stop_token_ids)
        if not bool(stop_mask.any().item()):
            return None
        first_stop = int(stop_mask.nonzero(as_tuple=False)[0, 1].item())
        prefix = generated[:, :first_stop]
        if bool((prefix == args.mask_id).any().item()):
            return None
        return sequence[:, : original_len + first_stop + 1]

    schedule_events = {
        "scheduled_interval_visits": 0,
        "default_interval_visits": 0,
        "scheduled_token_visits": 0,
        "default_token_visits": 0,
        "forced_schedule_interval_visits": 0,
        "forced_schedule_token_visits": 0,
        "forced_argument_boundary_token_visits": 0,
        "argument_boundary_ban_interval_visits": 0,
        "argument_boundary_ban_token_visits": 0,
        "argument_candidate_constraint_interval_visits": 0,
        "argument_candidate_constraint_token_visits": 0,
        "selected_candidate_force_interval_visits": 0,
        "selected_candidate_force_token_visits": 0,
        "candidate_sequence_force_interval_visits": 0,
        "candidate_sequence_force_token_visits": 0,
        "candidate_sequence_choice_count": 0,
        "candidate_sequence_deferred_choice_count": 0,
        "candidate_sequence_shared_prefix_force_token_visits": 0,
        "tool_value_candidate_force_interval_visits": 0,
        "tool_value_candidate_force_token_visits": 0,
        "tool_value_candidate_choice_count": 0,
        "tool_value_candidate_deferred_choice_count": 0,
        "tool_value_candidate_shared_prefix_force_token_visits": 0,
        "tool_name_sequence_force_interval_visits": 0,
        "tool_name_sequence_force_token_visits": 0,
        "tool_name_sequence_choice_count": 0,
        "tool_name_sequence_deferred_choice_count": 0,
        "tool_name_sequence_shared_prefix_force_token_visits": 0,
        "tool_name_candidate_guard_force_interval_visits": 0,
        "tool_name_candidate_guard_force_token_visits": 0,
        "tool_name_candidate_guard_choice_count": 0,
        "tool_name_candidate_guard_deferred_choice_count": 0,
        "tool_name_candidate_guard_shared_prefix_force_token_visits": 0,
        "tool_call_mode_force_interval_visits": 0,
        "tool_call_mode_force_token_visits": 0,
        "tool_call_mode_close_deferred_interval_visits": 0,
        "tool_call_mode_close_deferred_token_visits": 0,
        "json_prefix_guard_interval_visits": 0,
        "json_prefix_guard_accepted_token_visits": 0,
        "json_prefix_guard_rejected_token_visits": 0,
        "json_prefix_guard_replacement_token_visits": 0,
        "json_prefix_guard_target_fallback_token_visits": 0,
        "json_prefix_guard_unsafe_fallback_token_visits": 0,
        "json_prefix_guard_left_to_right_dropped_token_visits": 0,
        "live_tool_json_grammar_interval_visits": 0,
        "live_tool_json_grammar_active_rows": 0,
        "live_tool_json_grammar_token_visits": 0,
        "live_tool_json_grammar_left_to_right_dropped_token_visits": 0,
        "live_tool_json_grammar_replacement_token_visits": 0,
        "live_tool_json_grammar_unsafe_fallback_token_visits": 0,
    }
    candidate_group_choices = {}
    candidate_group_allowed_indices = {}
    while output_ids.shape[1] - original_len < args.max_new_tokens:
        remaining = args.max_new_tokens - (output_ids.shape[1] - original_len)
        if getattr(args, "fresh_generation_blocks", False):
            block_pad = args.block_size
        else:
            block_pad = args.block_size - (output_ids.shape[1] % args.block_size)
            if block_pad == 0:
                block_pad = args.block_size
        block_pad = min(block_pad, remaining)
        masks = torch.full(
            (output_ids.shape[0], block_pad),
            args.mask_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        x_t = torch.cat([output_ids, masks], dim=1)
        while (x_t[:, -block_pad:] == args.mask_id).any():
            window_len = min(args.block_size, x_t.shape[1])
            window_abs_start = x_t.shape[1] - window_len
            intervals = scheduled_window_intervals(
                window_len,
                window_abs_start,
                original_len,
                block_pad,
                sampler_schedule,
                args.small_block_size,
            )
            for interval in intervals:
                start = interval["start"]
                end = interval["end"]
                while True:
                    mask_idx = x_t[:, -window_len:] == args.mask_id
                    current_mask = mask_idx[:, start:end]
                    if current_mask.sum() == 0:
                        break
                    target_token_ids = interval.get("target_token_ids") or []
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "argument_value"
                        and args.force_argument_boundary_target_tokens
                        and len(target_token_ids) == end - start
                    ):
                        boundary_mask = target_boundary_mask(
                            tokenizer,
                            target_token_ids,
                            args._argument_boundary_target_cache,
                        )
                        if any(boundary_mask):
                            boundary_mask_tensor = torch.tensor(
                                boundary_mask,
                                dtype=torch.bool,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            force_mask = current_mask & boundary_mask_tensor
                            if force_mask.any():
                                target = torch.tensor(
                                    target_token_ids,
                                    dtype=torch.long,
                                    device=x_t.device,
                                ).unsqueeze(0).expand(x_t.shape[0], -1)
                                window = x_t[:, -window_len:]
                                span = window[:, start:end].clone()
                                span[force_mask] = target[force_mask]
                                window[:, start:end] = span
                                x_t[:, -window_len:] = window
                                forced_count = int(force_mask.sum().item())
                                schedule_events["forced_argument_boundary_token_visits"] += forced_count
                                mask_idx = x_t[:, -window_len:] == args.mask_id
                                current_mask = mask_idx[:, start:end]
                                if current_mask.sum() == 0:
                                    break
                    selected_candidate = interval.get("selected_candidate_token_ids_by_offset") or []
                    candidate_sequences = interval.get("candidate_sequence_token_ids_by_offset") or []
                    candidate_group_id = interval.get("candidate_group_id")
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "tool_name"
                        and (args.force_best_tool_name_sequence or args.guard_tool_name_candidates)
                        and candidate_sequences
                        and all(len(sequence) == end - start for sequence in candidate_sequences)
                    ):
                        chosen_idx = candidate_group_choices.get(candidate_group_id)
                        sequence_tuples = [tuple(int(token_id) for token_id in sequence) for sequence in candidate_sequences]
                        allowed_indices = candidate_group_allowed_indices.get(candidate_group_id)
                        if allowed_indices is None:
                            allowed_indices = list(range(len(sequence_tuples)))
                        else:
                            allowed_indices = [idx for idx in allowed_indices if idx < len(sequence_tuples)]
                            if not allowed_indices:
                                allowed_indices = list(range(len(sequence_tuples)))
                        allowed_sequence_tuples = [sequence_tuples[idx] for idx in allowed_indices]
                        if chosen_idx is None and len(set(allowed_sequence_tuples)) == 1:
                            target = torch.tensor(
                                allowed_sequence_tuples[0],
                                dtype=torch.long,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            window = x_t[:, -window_len:]
                            span = window[:, start:end].clone()
                            span[current_mask] = target[current_mask]
                            window[:, start:end] = span
                            x_t[:, -window_len:] = window
                            forced_count = int(current_mask.sum().item())
                            schedule_events["tool_name_sequence_force_interval_visits"] += 1
                            schedule_events["tool_name_sequence_force_token_visits"] += forced_count
                            schedule_events["tool_name_sequence_shared_prefix_force_token_visits"] += forced_count
                            if args.guard_tool_name_candidates:
                                schedule_events["tool_name_candidate_guard_force_interval_visits"] += 1
                                schedule_events["tool_name_candidate_guard_force_token_visits"] += forced_count
                                schedule_events["tool_name_candidate_guard_shared_prefix_force_token_visits"] += forced_count
                            mask_idx = x_t[:, -window_len:] == args.mask_id
                            current_mask = mask_idx[:, start:end]
                            if current_mask.sum() == 0:
                                break
                        else:
                            target_sequence = None
                            if chosen_idx is None:
                                logits = model(input_ids=x_t, use_cache=False).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, -window_len:][:, start:end]
                                log_probs = torch.log_softmax(logits, dim=-1)
                                scores = []
                                for sequence in allowed_sequence_tuples:
                                    score = torch.zeros(x_t.shape[0], dtype=log_probs.dtype, device=x_t.device)
                                    for pos, token_id in enumerate(sequence):
                                        score = score + torch.where(
                                            current_mask[:, pos],
                                            log_probs[:, pos, token_id],
                                            torch.zeros_like(score),
                                        )
                                    scores.append(score)
                                stacked = torch.stack(scores, dim=-1)
                                local_idx = int(stacked.argmax(dim=-1)[0].item())
                                target_sequence = allowed_sequence_tuples[local_idx]
                                matching_indices = [
                                    idx
                                    for idx in allowed_indices
                                    if sequence_tuples[idx] == target_sequence
                                ]
                                if len(matching_indices) == 1:
                                    chosen_idx = matching_indices[0]
                                    candidate_group_choices[candidate_group_id] = chosen_idx
                                    schedule_events["tool_name_sequence_choice_count"] += 1
                                    if args.guard_tool_name_candidates:
                                        schedule_events["tool_name_candidate_guard_choice_count"] += 1
                                else:
                                    candidate_group_allowed_indices[candidate_group_id] = matching_indices
                                    schedule_events["tool_name_sequence_deferred_choice_count"] += 1
                                    if args.guard_tool_name_candidates:
                                        schedule_events["tool_name_candidate_guard_deferred_choice_count"] += 1
                            if target_sequence is None:
                                target_sequence = sequence_tuples[chosen_idx]
                            target = torch.tensor(
                                target_sequence,
                                dtype=torch.long,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            window = x_t[:, -window_len:]
                            span = window[:, start:end].clone()
                            span[current_mask] = target[current_mask]
                            window[:, start:end] = span
                            x_t[:, -window_len:] = window
                            forced_count = int(current_mask.sum().item())
                            schedule_events["tool_name_sequence_force_interval_visits"] += 1
                            schedule_events["tool_name_sequence_force_token_visits"] += forced_count
                            if args.guard_tool_name_candidates:
                                schedule_events["tool_name_candidate_guard_force_interval_visits"] += 1
                                schedule_events["tool_name_candidate_guard_force_token_visits"] += forced_count
                            mask_idx = x_t[:, -window_len:] == args.mask_id
                            current_mask = mask_idx[:, start:end]
                            if current_mask.sum() == 0:
                                break
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "argument_value"
                        and args.force_selected_candidate_tokens
                        and len(selected_candidate) == end - start
                    ):
                        selected_mask = [bool(ids) for ids in selected_candidate]
                        if any(selected_mask):
                            selected_mask_tensor = torch.tensor(
                                selected_mask,
                                dtype=torch.bool,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            force_mask = current_mask & selected_mask_tensor
                            if force_mask.any():
                                target = torch.full(
                                    (x_t.shape[0], end - start),
                                    args.mask_id,
                                    dtype=torch.long,
                                    device=x_t.device,
                                )
                                for pos, ids in enumerate(selected_candidate):
                                    if ids:
                                        target[:, pos] = int(ids[0])
                                window = x_t[:, -window_len:]
                                span = window[:, start:end].clone()
                                span[force_mask] = target[force_mask]
                                window[:, start:end] = span
                                x_t[:, -window_len:] = window
                                forced_count = int(force_mask.sum().item())
                                schedule_events["selected_candidate_force_interval_visits"] += 1
                                schedule_events["selected_candidate_force_token_visits"] += forced_count
                                mask_idx = x_t[:, -window_len:] == args.mask_id
                                current_mask = mask_idx[:, start:end]
                                if current_mask.sum() == 0:
                                    break
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "argument_value"
                        and (args.force_best_candidate_sequence or args.guard_tool_value_candidates)
                        and candidate_sequences
                        and all(len(sequence) == end - start for sequence in candidate_sequences)
                    ):
                        chosen_idx = candidate_group_choices.get(candidate_group_id)
                        sequence_tuples = [tuple(int(token_id) for token_id in sequence) for sequence in candidate_sequences]
                        allowed_indices = candidate_group_allowed_indices.get(candidate_group_id)
                        if allowed_indices is None:
                            allowed_indices = list(range(len(sequence_tuples)))
                        else:
                            allowed_indices = [idx for idx in allowed_indices if idx < len(sequence_tuples)]
                            if not allowed_indices:
                                allowed_indices = list(range(len(sequence_tuples)))
                        allowed_sequence_tuples = [sequence_tuples[idx] for idx in allowed_indices]
                        if chosen_idx is None and len(set(allowed_sequence_tuples)) == 1:
                            target = torch.tensor(
                                allowed_sequence_tuples[0],
                                dtype=torch.long,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            window = x_t[:, -window_len:]
                            span = window[:, start:end].clone()
                            span[current_mask] = target[current_mask]
                            window[:, start:end] = span
                            x_t[:, -window_len:] = window
                            forced_count = int(current_mask.sum().item())
                            schedule_events["candidate_sequence_force_interval_visits"] += 1
                            schedule_events["candidate_sequence_force_token_visits"] += forced_count
                            schedule_events["candidate_sequence_shared_prefix_force_token_visits"] += forced_count
                            if args.guard_tool_value_candidates:
                                schedule_events["tool_value_candidate_force_interval_visits"] += 1
                                schedule_events["tool_value_candidate_force_token_visits"] += forced_count
                                schedule_events["tool_value_candidate_shared_prefix_force_token_visits"] += forced_count
                            mask_idx = x_t[:, -window_len:] == args.mask_id
                            current_mask = mask_idx[:, start:end]
                            if current_mask.sum() == 0:
                                break
                        else:
                            target_sequence = None
                            if chosen_idx is None:
                                logits = model(input_ids=x_t, use_cache=False).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, -window_len:][:, start:end]
                                log_probs = torch.log_softmax(logits, dim=-1)
                                scores = []
                                for sequence in allowed_sequence_tuples:
                                    score = torch.zeros(x_t.shape[0], dtype=log_probs.dtype, device=x_t.device)
                                    for pos, token_id in enumerate(sequence):
                                        score = score + torch.where(
                                            current_mask[:, pos],
                                            log_probs[:, pos, token_id],
                                            torch.zeros_like(score),
                                        )
                                    scores.append(score)
                                stacked = torch.stack(scores, dim=-1)
                                local_idx = int(stacked.argmax(dim=-1)[0].item())
                                target_sequence = allowed_sequence_tuples[local_idx]
                                matching_indices = [
                                    idx
                                    for idx in allowed_indices
                                    if sequence_tuples[idx] == target_sequence
                                ]
                                if len(matching_indices) == 1:
                                    chosen_idx = matching_indices[0]
                                    candidate_group_choices[candidate_group_id] = chosen_idx
                                    schedule_events["candidate_sequence_choice_count"] += 1
                                    if args.guard_tool_value_candidates:
                                        schedule_events["tool_value_candidate_choice_count"] += 1
                                else:
                                    candidate_group_allowed_indices[candidate_group_id] = matching_indices
                                    schedule_events["candidate_sequence_deferred_choice_count"] += 1
                                    if args.guard_tool_value_candidates:
                                        schedule_events["tool_value_candidate_deferred_choice_count"] += 1
                            if target_sequence is None:
                                target_sequence = sequence_tuples[chosen_idx]
                            target = torch.tensor(
                                target_sequence,
                                dtype=torch.long,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            window = x_t[:, -window_len:]
                            span = window[:, start:end].clone()
                            span[current_mask] = target[current_mask]
                            window[:, start:end] = span
                            x_t[:, -window_len:] = window
                            forced_count = int(current_mask.sum().item())
                            schedule_events["candidate_sequence_force_interval_visits"] += 1
                            schedule_events["candidate_sequence_force_token_visits"] += forced_count
                            if args.guard_tool_value_candidates:
                                schedule_events["tool_value_candidate_force_interval_visits"] += 1
                                schedule_events["tool_value_candidate_force_token_visits"] += forced_count
                            mask_idx = x_t[:, -window_len:] == args.mask_id
                            current_mask = mask_idx[:, start:end]
                            if current_mask.sum() == 0:
                                break
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "tool_tag"
                        and args.guard_tool_call_mode
                        and len(target_token_ids) == end - start
                    ):
                        force_mask = tool_call_mode_force_mask(
                            tokenizer,
                            x_t,
                            current_mask,
                            original_len,
                            args,
                            schedule_events,
                        )
                        if not force_mask.any():
                            pass
                        else:
                            target = torch.tensor(
                                target_token_ids,
                                dtype=torch.long,
                                device=x_t.device,
                            ).unsqueeze(0).expand(x_t.shape[0], -1)
                            window = x_t[:, -window_len:]
                            span = window[:, start:end].clone()
                            span[force_mask] = target[force_mask]
                            window[:, start:end] = span
                            x_t[:, -window_len:] = window
                            forced_count = int(force_mask.sum().item())
                            schedule_events["tool_call_mode_force_interval_visits"] += 1
                            schedule_events["tool_call_mode_force_token_visits"] += forced_count
                            schedule_events["scheduled_interval_visits"] += 1
                            schedule_events["scheduled_token_visits"] += forced_count
                            break
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") in args.force_schedule_token_kinds
                        and not (
                            interval.get("kind") == "tool_tag"
                            and args.guard_tool_call_mode
                        )
                        and len(target_token_ids) == end - start
                    ):
                        target = torch.tensor(
                            target_token_ids,
                            dtype=torch.long,
                            device=x_t.device,
                        ).unsqueeze(0).expand(x_t.shape[0], -1)
                        window = x_t[:, -window_len:]
                        span = window[:, start:end].clone()
                        span[current_mask] = target[current_mask]
                        window[:, start:end] = span
                        x_t[:, -window_len:] = window
                        forced_count = int(current_mask.sum().item())
                        schedule_events["forced_schedule_interval_visits"] += 1
                        schedule_events["forced_schedule_token_visits"] += forced_count
                        schedule_events["scheduled_interval_visits"] += 1
                        schedule_events["scheduled_token_visits"] += forced_count
                        break
                    logits = model(input_ids=x_t, use_cache=False).logits
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    logits = logits[:, -window_len:][:, start:end]
                    candidate_allowed = interval.get("candidate_allowed_token_ids_by_offset") or []
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "argument_value"
                        and args.constrain_argument_candidate_tokens
                        and len(candidate_allowed) == end - start
                    ):
                        for pos, allowed_ids in enumerate(candidate_allowed):
                            if not allowed_ids:
                                continue
                            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                            mask[allowed_ids] = False
                            logits[:, pos, :] = logits[:, pos, :].masked_fill(mask, -torch.inf)
                        schedule_events["argument_candidate_constraint_interval_visits"] += 1
                        schedule_events["argument_candidate_constraint_token_visits"] += int(current_mask.sum().item())
                    if (
                        interval.get("scheduled")
                        and interval.get("kind") == "argument_value"
                        and args.ban_argument_boundary_tokens
                    ):
                        banned_ids = set(args.argument_boundary_token_ids)
                        if args.ban_argument_json_boundary_tokens and target_token_ids:
                            boundary_mask = target_boundary_mask(
                                tokenizer,
                                target_token_ids,
                                args._argument_boundary_target_cache,
                            )
                            banned_ids.update(
                                int(token_id)
                                for token_id, is_boundary in zip(target_token_ids, boundary_mask)
                                if is_boundary
                            )
                        if args.ban_argument_newline_tokens:
                            banned_ids.update(args.argument_newline_token_ids)
                        if banned_ids:
                            logits[:, :, sorted(banned_ids)] = -torch.inf
                            schedule_events["argument_boundary_ban_interval_visits"] += 1
                            schedule_events["argument_boundary_ban_token_visits"] += int(current_mask.sum().item())
                    logits = logits.clone()
                    logits[..., int(args.mask_id)] = torch.finfo(logits.dtype).min
                    if args.live_tool_json_grammar:
                        x_1, unmask_idx, handled_by_live_grammar = apply_live_tool_json_grammar(
                            tokenizer,
                            x_t,
                            logits,
                            current_mask,
                            window_abs_start,
                            start,
                            original_len,
                            args,
                            schedule_events,
                        )
                    else:
                        handled_by_live_grammar = False
                    if not handled_by_live_grammar:
                        x_1, p_1t = sample_with_top_p(model, logits, args.top_p, args.temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(current_mask, x1_p, -torch.inf)
                        unmask_idx = x1_p > args.threshold
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), max_prob_idx] = True
                        unmask_idx = unmask_idx & current_mask
                    if (
                        args.guard_tool_json_prefix
                        and interval.get("scheduled")
                        and interval.get("kind") in args.json_prefix_guard_kinds
                    ):
                        x_1, unmask_idx = apply_tool_json_prefix_guard(
                            tokenizer,
                            x_t,
                            logits,
                            x_1,
                            unmask_idx,
                            current_mask,
                            window_abs_start,
                            window_len,
                            start,
                            original_len,
                            target_token_ids,
                            args,
                            schedule_events,
                        )
                    window = x_t[:, -window_len:]
                    span = window[:, start:end].clone()
                    span[unmask_idx] = x_1[unmask_idx]
                    window[:, start:end] = span
                    x_t[:, -window_len:] = window
                    stopped = truncate_if_stopped(x_t)
                    if stopped is not None:
                        args._last_sampler_schedule_events = schedule_events
                        return stopped[0]
                    if interval.get("scheduled"):
                        schedule_events["scheduled_interval_visits"] += 1
                        schedule_events["scheduled_token_visits"] += int(current_mask.sum().item())
                    else:
                        schedule_events["default_interval_visits"] += 1
                        schedule_events["default_token_visits"] += int(current_mask.sum().item())
            if (x_t[:, -block_pad:] == args.mask_id).all():
                break
        output_ids = x_t
        stopped = truncate_if_stopped(output_ids)
        if stopped is not None:
            args._last_sampler_schedule_events = schedule_events
            return stopped[0]
    args._last_sampler_schedule_events = schedule_events
    return output_ids[0]


def empty_totals():
    return {
        "records": 0,
        "ok": 0,
        "valid_tool_json": 0,
        "exact_tool_name_set": 0,
        "exact_tool_sequence": 0,
        "exact_tool_name_multiset": 0,
        "same_tool_call_count": 0,
        "exact_arguments": 0,
        "all_schema_valid": 0,
        "all_required_args_present": 0,
        "records_with_extra_calls": 0,
        "records_with_missing_calls": 0,
        "records_with_repeated_calls": 0,
        "total_extra_calls": 0,
        "total_missing_calls": 0,
        "total_repeated_calls": 0,
        "repaired_valid_tool_json": 0,
        "repaired_exact_tool_name_set": 0,
        "repaired_exact_tool_sequence": 0,
        "repaired_exact_arguments": 0,
        "repaired_all_schema_valid": 0,
        "repaired_all_required_args_present": 0,
        "constrained_valid_tool_json": 0,
        "constrained_exact_tool_name_set": 0,
        "constrained_exact_tool_sequence": 0,
        "constrained_exact_arguments": 0,
        "constrained_all_schema_valid": 0,
        "constrained_all_required_args_present": 0,
        "model_repair_valid_tool_json": 0,
        "model_repair_exact_tool_name_set": 0,
        "model_repair_exact_tool_sequence": 0,
        "model_repair_exact_arguments": 0,
        "model_repair_all_schema_valid": 0,
        "model_repair_all_required_args_present": 0,
        "unresolved_mask_examples": 0,
        "sampler_schedule_used": 0,
        "sampler_scheduled_interval_visits": 0,
        "sampler_default_interval_visits": 0,
        "sampler_scheduled_token_visits": 0,
        "sampler_default_token_visits": 0,
        "sampler_forced_schedule_interval_visits": 0,
        "sampler_forced_schedule_token_visits": 0,
        "sampler_forced_argument_boundary_token_visits": 0,
        "sampler_argument_boundary_ban_interval_visits": 0,
        "sampler_argument_boundary_ban_token_visits": 0,
        "sampler_argument_candidate_constraint_interval_visits": 0,
        "sampler_argument_candidate_constraint_token_visits": 0,
        "sampler_selected_candidate_force_interval_visits": 0,
        "sampler_selected_candidate_force_token_visits": 0,
        "sampler_candidate_sequence_force_interval_visits": 0,
        "sampler_candidate_sequence_force_token_visits": 0,
        "sampler_candidate_sequence_choice_count": 0,
        "sampler_candidate_sequence_deferred_choice_count": 0,
        "sampler_candidate_sequence_shared_prefix_force_token_visits": 0,
        "sampler_tool_value_candidate_force_interval_visits": 0,
        "sampler_tool_value_candidate_force_token_visits": 0,
        "sampler_tool_value_candidate_choice_count": 0,
        "sampler_tool_value_candidate_deferred_choice_count": 0,
        "sampler_tool_value_candidate_shared_prefix_force_token_visits": 0,
        "sampler_tool_name_sequence_force_interval_visits": 0,
        "sampler_tool_name_sequence_force_token_visits": 0,
        "sampler_tool_name_sequence_choice_count": 0,
        "sampler_tool_name_sequence_deferred_choice_count": 0,
        "sampler_tool_name_sequence_shared_prefix_force_token_visits": 0,
        "sampler_tool_name_candidate_guard_force_interval_visits": 0,
        "sampler_tool_name_candidate_guard_force_token_visits": 0,
        "sampler_tool_name_candidate_guard_choice_count": 0,
        "sampler_tool_name_candidate_guard_deferred_choice_count": 0,
        "sampler_tool_name_candidate_guard_shared_prefix_force_token_visits": 0,
        "sampler_tool_call_mode_force_interval_visits": 0,
        "sampler_tool_call_mode_force_token_visits": 0,
        "sampler_tool_call_mode_close_deferred_interval_visits": 0,
        "sampler_tool_call_mode_close_deferred_token_visits": 0,
        "sampler_json_prefix_guard_interval_visits": 0,
        "sampler_json_prefix_guard_accepted_token_visits": 0,
        "sampler_json_prefix_guard_rejected_token_visits": 0,
        "sampler_json_prefix_guard_replacement_token_visits": 0,
        "sampler_json_prefix_guard_target_fallback_token_visits": 0,
        "sampler_json_prefix_guard_unsafe_fallback_token_visits": 0,
        "sampler_json_prefix_guard_left_to_right_dropped_token_visits": 0,
        "sampler_live_tool_json_grammar_interval_visits": 0,
        "sampler_live_tool_json_grammar_active_rows": 0,
        "sampler_live_tool_json_grammar_token_visits": 0,
        "sampler_live_tool_json_grammar_left_to_right_dropped_token_visits": 0,
        "sampler_live_tool_json_grammar_replacement_token_visits": 0,
        "sampler_live_tool_json_grammar_unsafe_fallback_token_visits": 0,
        "stop_boundary_guard_trimmed": 0,
        "errors": 0,
    }


def parse_eval_spec(spec):
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "--eval must be name:input_jsonl:out_jsonl or name:input_jsonl:out_jsonl:limit"
        )
    name, input_jsonl, out_jsonl = parts[:3]
    limit = int(parts[3]) if len(parts) == 4 and parts[3] else 0
    if not name:
        raise argparse.ArgumentTypeError("--eval name cannot be empty")
    return name, Path(input_jsonl), Path(out_jsonl), limit


def generate_case(model, tokenizer, case, args, sampler_schedule=None):
    prompt = make_prompt(tokenizer, case, args.append_instruction, chat_template=args.chat_template)
    prompt_input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
    input_ids = prompt_input_ids
    forced_prefix = args.forced_assistant_prefix or ""
    if args.force_tool_call_prefix:
        forced_prefix = "<tool_call>\n" + forced_prefix
    if forced_prefix:
        prefix_ids = tokenizer(forced_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)
    with torch.no_grad():
        if args.full_context_sampling:
            generated = full_context_sample(
                model,
                input_ids,
                tokenizer,
                args,
                sampler_schedule=sampler_schedule,
                original_len_override=prompt_input_ids.shape[1],
            )
        else:
            seq_len = torch.tensor([input_ids.shape[1]], device="cuda")
            generated = model.mdm_sample(
                input_ids,
                tokenizer=tokenizer,
                block_size=args.block_size,
                small_block_size=args.small_block_size,
                max_new_tokens=args.max_new_tokens,
                mask_id=args.mask_id,
                stop_token=args.stop_token_id,
                min_len=input_ids.shape[1],
                seq_len=seq_len,
                threshold=args.threshold,
                temperature=args.temperature,
                top_p=args.top_p,
                use_block_cache=args.use_block_cache,
            )[0]
    new_ids = generated[prompt_input_ids.shape[1] :]
    mask_count = int((new_ids == args.mask_id).sum().item())
    generated_token_count = int((new_ids != args.mask_id).sum().item())
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, mask_count, generated_token_count


def gold_tool_call_count(case):
    if isinstance(case.get("gold_tool_calls"), list):
        return len(case["gold_tool_calls"])
    calls, _ = extract_tool_calls(case.get("gold_assistant") or "")
    return len(calls)


def stop_boundary_target_count(args, case, sampler_schedule_row):
    if args.stop_after_tool_calls and args.stop_after_tool_calls > 0:
        return args.stop_after_tool_calls
    if args.stop_after_schedule_tool_calls and sampler_schedule_row:
        count = sampler_schedule_row.get("tool_call_count")
        if isinstance(count, int) and count > 0:
            return count
    if args.stop_after_gold_tool_calls:
        return gold_tool_call_count(case)
    return 0


def run_eval(model, tokenizer, args, eval_name, input_jsonl, out_jsonl, limit):
    cases = load_cases(input_jsonl, limit)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals = empty_totals()
    generated_tokens = 0
    model_repair_generated_tokens = 0
    start = time.time()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            row = {
                "idx": idx,
                "source": case.get("source"),
                "id": case.get("id"),
                "gold_tool_names": case.get("gold_tool_names") or [],
                "available_tool_names": case.get("available_tool_names") or [],
            }
            try:
                sample_start = time.time()
                key = case_key(case, idx)
                sampler_schedule = args.sampler_schedules.get(key) if args.full_context_sampling else None
                sampler_schedule_row = args.sampler_schedule_rows.get(key) if args.full_context_sampling else None
                generation_case = case
                if args.strip_gold_for_generation:
                    generation_case = copy.deepcopy(case)
                    for gold_key in ("gold_assistant", "gold_tool_names", "gold_tool_calls"):
                        generation_case.pop(gold_key, None)
                text, mask_count, token_count = generate_case(
                    model,
                    tokenizer,
                    generation_case,
                    args,
                    sampler_schedule=sampler_schedule,
                )
                raw_generated_text = text
                stop_target_count = stop_boundary_target_count(args, case, sampler_schedule_row)
                text, stop_guard_trimmed = trim_after_tool_call_closers(text, stop_target_count)
                sample_seconds = time.time() - sample_start
                metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
                repaired_text = ""
                repaired_metrics = None
                if args.repair_mode != "none":
                    repaired_text = repaired_tool_call_text(text, case.get("tools") or [])
                    repaired_metrics = score_tool_calls(repaired_text, case.get("tools") or [], case.get("gold_assistant"))
                constrained_text = ""
                constrained_metrics = None
                if args.constrained_tool_decoding:
                    constrained_fn = (
                        sequence_preserving_constrained_tool_call_text
                        if args.constrained_sequence_preserving
                        else constrained_tool_call_text
                    )
                    constrained_text = constrained_fn(
                        text,
                        case.get("tools") or [],
                        context_text=case_context_text(case),
                        max_calls=args.constrained_max_calls,
                        assume_utc_z=args.constrained_assume_utc_z,
                    )
                    constrained_metrics = score_tool_calls(
                        constrained_text,
                        case.get("tools") or [],
                        case.get("gold_assistant"),
                    )
                model_repair_text = ""
                model_repair_metrics = None
                model_repair_mask_count = 0
                model_repair_token_count = 0
                model_repair_seconds = 0.0
                if args.model_repair_pass:
                    repair_args = copy.copy(args)
                    repair_args.append_instruction = False
                    if args.model_repair_max_new_tokens:
                        repair_args.max_new_tokens = args.model_repair_max_new_tokens
                    repair_case = model_repair_case(case, text)
                    repair_start = time.time()
                    model_repair_text, model_repair_mask_count, model_repair_token_count = generate_case(
                        model,
                        tokenizer,
                        repair_case,
                        repair_args,
                    )
                    model_repair_seconds = time.time() - repair_start
                    model_repair_metrics = score_tool_calls(
                        model_repair_text,
                        case.get("tools") or [],
                        case.get("gold_assistant"),
                    )
                row.update(
                    {
                        "status": "ok",
                        "assistant": text,
                        "called_names": metrics["called_names"],
                        "calls": metrics["calls"],
                        "invalid_tool_json_count": metrics["invalid_tool_call_count"],
                        "valid_tool_json": metrics["valid_tool_call"],
                        "valid_tool_call": metrics["valid_tool_call"],
                        "exact_tool_name_set": metrics.get("exact_tool_name_set"),
                        "exact_tool_name_multiset": metrics.get("exact_tool_name_multiset"),
                        "exact_tool_sequence": metrics.get("exact_tool_sequence"),
                        "same_tool_call_count": metrics.get("same_tool_call_count"),
                        "exact_arguments": metrics.get("exact_arguments"),
                        "all_schema_valid": metrics["all_schema_valid"],
                        "all_required_args_present": metrics["all_required_args_present"],
                        "schema_valid_count": metrics["schema_valid_count"],
                        "required_args_count": metrics["required_args_count"],
                        "extra_call_count": metrics.get("extra_call_count"),
                        "missing_call_count": metrics.get("missing_call_count"),
                        "repeated_call_count": metrics.get("repeated_call_count"),
                        "extra_call_names": metrics.get("extra_call_names"),
                        "missing_call_names": metrics.get("missing_call_names"),
                        "repeated_call_names": metrics.get("repeated_call_names"),
                        "call_errors": metrics["call_errors"],
                        "mask_count": mask_count,
                        "generated_token_count": token_count,
                        "seconds": sample_seconds,
                        "sampler_schedule_used": bool(sampler_schedule),
                        "sampler_schedule_events": getattr(args, "_last_sampler_schedule_events", {}),
                        "stop_boundary_target_count": stop_target_count,
                        "stop_boundary_guard_trimmed": stop_guard_trimmed,
                        "forced_assistant_prefix": (
                            ("<tool_call>\n" if args.force_tool_call_prefix else "")
                            + (args.forced_assistant_prefix or "")
                        ),
                    }
                )
                if stop_guard_trimmed:
                    row["pre_stop_boundary_assistant"] = raw_generated_text
                if repaired_metrics is not None:
                    row.update(
                        {
                            "repair_mode": args.repair_mode,
                            "repaired_assistant": repaired_text,
                            "repaired_called_names": repaired_metrics["called_names"],
                            "repaired_calls": repaired_metrics["calls"],
                            "repaired_valid_tool_json": repaired_metrics["valid_tool_call"],
                            "repaired_exact_tool_name_set": repaired_metrics.get("exact_tool_name_set"),
                            "repaired_exact_tool_sequence": repaired_metrics.get("exact_tool_sequence"),
                            "repaired_exact_arguments": repaired_metrics.get("exact_arguments"),
                            "repaired_all_schema_valid": repaired_metrics["all_schema_valid"],
                            "repaired_all_required_args_present": repaired_metrics["all_required_args_present"],
                            "repaired_call_errors": repaired_metrics["call_errors"],
                        }
                    )
                if constrained_metrics is not None:
                    row.update(
                        {
                            "constrained_assistant": constrained_text,
                            "constrained_max_calls": args.constrained_max_calls,
                            "constrained_sequence_preserving": args.constrained_sequence_preserving,
                            "constrained_assume_utc_z": args.constrained_assume_utc_z,
                            "constrained_called_names": constrained_metrics["called_names"],
                            "constrained_calls": constrained_metrics["calls"],
                            "constrained_valid_tool_json": constrained_metrics["valid_tool_call"],
                            "constrained_exact_tool_name_set": constrained_metrics.get("exact_tool_name_set"),
                            "constrained_exact_tool_sequence": constrained_metrics.get("exact_tool_sequence"),
                            "constrained_exact_arguments": constrained_metrics.get("exact_arguments"),
                            "constrained_all_schema_valid": constrained_metrics["all_schema_valid"],
                            "constrained_all_required_args_present": constrained_metrics["all_required_args_present"],
                            "constrained_call_errors": constrained_metrics["call_errors"],
                        }
                    )
                if model_repair_metrics is not None:
                    row.update(
                        {
                            "model_repair_assistant": model_repair_text,
                            "model_repair_called_names": model_repair_metrics["called_names"],
                            "model_repair_calls": model_repair_metrics["calls"],
                            "model_repair_valid_tool_json": model_repair_metrics["valid_tool_call"],
                            "model_repair_exact_tool_name_set": model_repair_metrics.get("exact_tool_name_set"),
                            "model_repair_exact_tool_sequence": model_repair_metrics.get("exact_tool_sequence"),
                            "model_repair_exact_arguments": model_repair_metrics.get("exact_arguments"),
                            "model_repair_all_schema_valid": model_repair_metrics["all_schema_valid"],
                            "model_repair_all_required_args_present": model_repair_metrics["all_required_args_present"],
                            "model_repair_call_errors": model_repair_metrics["call_errors"],
                            "model_repair_mask_count": model_repair_mask_count,
                            "model_repair_generated_token_count": model_repair_token_count,
                            "model_repair_seconds": model_repair_seconds,
                        }
                    )
                totals["ok"] += 1
                totals["valid_tool_json"] += int(row["valid_tool_json"])
                totals["exact_tool_name_set"] += int(bool(row["exact_tool_name_set"]))
                totals["exact_tool_sequence"] += int(bool(row["exact_tool_sequence"]))
                totals["exact_tool_name_multiset"] += int(bool(row["exact_tool_name_multiset"]))
                totals["same_tool_call_count"] += int(bool(row["same_tool_call_count"]))
                totals["exact_arguments"] += int(bool(row["exact_arguments"]))
                totals["all_schema_valid"] += int(bool(row["all_schema_valid"]))
                totals["all_required_args_present"] += int(bool(row["all_required_args_present"]))
                schedule_events = row.get("sampler_schedule_events") or {}
                totals["sampler_schedule_used"] += int(bool(row.get("sampler_schedule_used")))
                totals["sampler_scheduled_interval_visits"] += int(
                    schedule_events.get("scheduled_interval_visits") or 0
                )
                totals["sampler_default_interval_visits"] += int(
                    schedule_events.get("default_interval_visits") or 0
                )
                totals["sampler_scheduled_token_visits"] += int(schedule_events.get("scheduled_token_visits") or 0)
                totals["sampler_default_token_visits"] += int(schedule_events.get("default_token_visits") or 0)
                totals["sampler_forced_schedule_interval_visits"] += int(
                    schedule_events.get("forced_schedule_interval_visits") or 0
                )
                totals["sampler_forced_schedule_token_visits"] += int(
                    schedule_events.get("forced_schedule_token_visits") or 0
                )
                totals["sampler_forced_argument_boundary_token_visits"] += int(
                    schedule_events.get("forced_argument_boundary_token_visits") or 0
                )
                totals["sampler_argument_boundary_ban_interval_visits"] += int(
                    schedule_events.get("argument_boundary_ban_interval_visits") or 0
                )
                totals["sampler_argument_boundary_ban_token_visits"] += int(
                    schedule_events.get("argument_boundary_ban_token_visits") or 0
                )
                totals["sampler_argument_candidate_constraint_interval_visits"] += int(
                    schedule_events.get("argument_candidate_constraint_interval_visits") or 0
                )
                totals["sampler_argument_candidate_constraint_token_visits"] += int(
                    schedule_events.get("argument_candidate_constraint_token_visits") or 0
                )
                totals["sampler_selected_candidate_force_interval_visits"] += int(
                    schedule_events.get("selected_candidate_force_interval_visits") or 0
                )
                totals["sampler_selected_candidate_force_token_visits"] += int(
                    schedule_events.get("selected_candidate_force_token_visits") or 0
                )
                totals["sampler_candidate_sequence_force_interval_visits"] += int(
                    schedule_events.get("candidate_sequence_force_interval_visits") or 0
                )
                totals["sampler_candidate_sequence_force_token_visits"] += int(
                    schedule_events.get("candidate_sequence_force_token_visits") or 0
                )
                totals["sampler_candidate_sequence_choice_count"] += int(
                    schedule_events.get("candidate_sequence_choice_count") or 0
                )
                totals["sampler_candidate_sequence_deferred_choice_count"] += int(
                    schedule_events.get("candidate_sequence_deferred_choice_count") or 0
                )
                totals["sampler_candidate_sequence_shared_prefix_force_token_visits"] += int(
                    schedule_events.get("candidate_sequence_shared_prefix_force_token_visits") or 0
                )
                totals["sampler_tool_value_candidate_force_interval_visits"] += int(
                    schedule_events.get("tool_value_candidate_force_interval_visits") or 0
                )
                totals["sampler_tool_value_candidate_force_token_visits"] += int(
                    schedule_events.get("tool_value_candidate_force_token_visits") or 0
                )
                totals["sampler_tool_value_candidate_choice_count"] += int(
                    schedule_events.get("tool_value_candidate_choice_count") or 0
                )
                totals["sampler_tool_value_candidate_deferred_choice_count"] += int(
                    schedule_events.get("tool_value_candidate_deferred_choice_count") or 0
                )
                totals["sampler_tool_value_candidate_shared_prefix_force_token_visits"] += int(
                    schedule_events.get("tool_value_candidate_shared_prefix_force_token_visits") or 0
                )
                totals["sampler_tool_name_sequence_force_interval_visits"] += int(
                    schedule_events.get("tool_name_sequence_force_interval_visits") or 0
                )
                totals["sampler_tool_name_sequence_force_token_visits"] += int(
                    schedule_events.get("tool_name_sequence_force_token_visits") or 0
                )
                totals["sampler_tool_name_sequence_choice_count"] += int(
                    schedule_events.get("tool_name_sequence_choice_count") or 0
                )
                totals["sampler_tool_name_sequence_deferred_choice_count"] += int(
                    schedule_events.get("tool_name_sequence_deferred_choice_count") or 0
                )
                totals["sampler_tool_name_sequence_shared_prefix_force_token_visits"] += int(
                    schedule_events.get("tool_name_sequence_shared_prefix_force_token_visits") or 0
                )
                totals["sampler_tool_name_candidate_guard_force_interval_visits"] += int(
                    schedule_events.get("tool_name_candidate_guard_force_interval_visits") or 0
                )
                totals["sampler_tool_name_candidate_guard_force_token_visits"] += int(
                    schedule_events.get("tool_name_candidate_guard_force_token_visits") or 0
                )
                totals["sampler_tool_name_candidate_guard_choice_count"] += int(
                    schedule_events.get("tool_name_candidate_guard_choice_count") or 0
                )
                totals["sampler_tool_name_candidate_guard_deferred_choice_count"] += int(
                    schedule_events.get("tool_name_candidate_guard_deferred_choice_count") or 0
                )
                totals["sampler_tool_name_candidate_guard_shared_prefix_force_token_visits"] += int(
                    schedule_events.get("tool_name_candidate_guard_shared_prefix_force_token_visits") or 0
                )
                totals["sampler_tool_call_mode_force_interval_visits"] += int(
                    schedule_events.get("tool_call_mode_force_interval_visits") or 0
                )
                totals["sampler_tool_call_mode_force_token_visits"] += int(
                    schedule_events.get("tool_call_mode_force_token_visits") or 0
                )
                totals["sampler_tool_call_mode_close_deferred_interval_visits"] += int(
                    schedule_events.get("tool_call_mode_close_deferred_interval_visits") or 0
                )
                totals["sampler_tool_call_mode_close_deferred_token_visits"] += int(
                    schedule_events.get("tool_call_mode_close_deferred_token_visits") or 0
                )
                totals["sampler_json_prefix_guard_interval_visits"] += int(
                    schedule_events.get("json_prefix_guard_interval_visits") or 0
                )
                totals["sampler_json_prefix_guard_accepted_token_visits"] += int(
                    schedule_events.get("json_prefix_guard_accepted_token_visits") or 0
                )
                totals["sampler_json_prefix_guard_rejected_token_visits"] += int(
                    schedule_events.get("json_prefix_guard_rejected_token_visits") or 0
                )
                totals["sampler_json_prefix_guard_replacement_token_visits"] += int(
                    schedule_events.get("json_prefix_guard_replacement_token_visits") or 0
                )
                totals["sampler_json_prefix_guard_target_fallback_token_visits"] += int(
                    schedule_events.get("json_prefix_guard_target_fallback_token_visits") or 0
                )
                totals["sampler_json_prefix_guard_unsafe_fallback_token_visits"] += int(
                    schedule_events.get("json_prefix_guard_unsafe_fallback_token_visits") or 0
                )
                totals["sampler_json_prefix_guard_left_to_right_dropped_token_visits"] += int(
                    schedule_events.get("json_prefix_guard_left_to_right_dropped_token_visits") or 0
                )
                totals["sampler_live_tool_json_grammar_interval_visits"] += int(
                    schedule_events.get("live_tool_json_grammar_interval_visits") or 0
                )
                totals["sampler_live_tool_json_grammar_active_rows"] += int(
                    schedule_events.get("live_tool_json_grammar_active_rows") or 0
                )
                totals["sampler_live_tool_json_grammar_token_visits"] += int(
                    schedule_events.get("live_tool_json_grammar_token_visits") or 0
                )
                totals["sampler_live_tool_json_grammar_left_to_right_dropped_token_visits"] += int(
                    schedule_events.get("live_tool_json_grammar_left_to_right_dropped_token_visits") or 0
                )
                totals["sampler_live_tool_json_grammar_replacement_token_visits"] += int(
                    schedule_events.get("live_tool_json_grammar_replacement_token_visits") or 0
                )
                totals["sampler_live_tool_json_grammar_unsafe_fallback_token_visits"] += int(
                    schedule_events.get("live_tool_json_grammar_unsafe_fallback_token_visits") or 0
                )
                totals["stop_boundary_guard_trimmed"] += int(bool(row.get("stop_boundary_guard_trimmed")))
                totals["records_with_extra_calls"] += int((row["extra_call_count"] or 0) > 0)
                totals["records_with_missing_calls"] += int((row["missing_call_count"] or 0) > 0)
                totals["records_with_repeated_calls"] += int((row["repeated_call_count"] or 0) > 0)
                totals["total_extra_calls"] += int(row["extra_call_count"] or 0)
                totals["total_missing_calls"] += int(row["missing_call_count"] or 0)
                totals["total_repeated_calls"] += int(row["repeated_call_count"] or 0)
                if repaired_metrics is not None:
                    totals["repaired_valid_tool_json"] += int(bool(row["repaired_valid_tool_json"]))
                    totals["repaired_exact_tool_name_set"] += int(bool(row["repaired_exact_tool_name_set"]))
                    totals["repaired_exact_tool_sequence"] += int(bool(row["repaired_exact_tool_sequence"]))
                    totals["repaired_exact_arguments"] += int(bool(row["repaired_exact_arguments"]))
                    totals["repaired_all_schema_valid"] += int(bool(row["repaired_all_schema_valid"]))
                    totals["repaired_all_required_args_present"] += int(bool(row["repaired_all_required_args_present"]))
                if constrained_metrics is not None:
                    totals["constrained_valid_tool_json"] += int(bool(row["constrained_valid_tool_json"]))
                    totals["constrained_exact_tool_name_set"] += int(bool(row["constrained_exact_tool_name_set"]))
                    totals["constrained_exact_tool_sequence"] += int(bool(row["constrained_exact_tool_sequence"]))
                    totals["constrained_exact_arguments"] += int(bool(row["constrained_exact_arguments"]))
                    totals["constrained_all_schema_valid"] += int(bool(row["constrained_all_schema_valid"]))
                    totals["constrained_all_required_args_present"] += int(bool(row["constrained_all_required_args_present"]))
                if model_repair_metrics is not None:
                    totals["model_repair_valid_tool_json"] += int(bool(row["model_repair_valid_tool_json"]))
                    totals["model_repair_exact_tool_name_set"] += int(bool(row["model_repair_exact_tool_name_set"]))
                    totals["model_repair_exact_tool_sequence"] += int(bool(row["model_repair_exact_tool_sequence"]))
                    totals["model_repair_exact_arguments"] += int(bool(row["model_repair_exact_arguments"]))
                    totals["model_repair_all_schema_valid"] += int(bool(row["model_repair_all_schema_valid"]))
                    totals["model_repair_all_required_args_present"] += int(
                        bool(row["model_repair_all_required_args_present"])
                    )
                    model_repair_generated_tokens += model_repair_token_count
                totals["unresolved_mask_examples"] += int(mask_count > 0)
                generated_tokens += token_count
            except Exception as exc:
                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            totals["records"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"{eval_name} {idx + 1}/{len(cases)} "
                f"seq={totals['exact_tool_sequence']} args={totals['exact_arguments']} "
                f"valid={totals['valid_tool_json']}",
                flush=True,
            )

    elapsed = time.time() - start
    summary = {
        "eval_name": eval_name,
        "input_jsonl": str(input_jsonl),
        "out_jsonl": str(out_jsonl),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "merge_adapter": not args.no_merge_adapter,
        "append_instruction": args.append_instruction,
        "totals": totals,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "model_repair_generated_tokens": model_repair_generated_tokens,
        "model_repair_generated_tokens_per_second": model_repair_generated_tokens / elapsed if elapsed else 0.0,
        "block_size": args.block_size,
        "small_block_size": args.small_block_size,
        "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_block_cache": args.use_block_cache,
        "full_context_sampling": args.full_context_sampling,
        "sampler_schedule_jsonl": str(args.sampler_schedule_jsonl) if args.sampler_schedule_jsonl else None,
        "sampler_schedule_count": len(args.sampler_schedules),
        "force_schedule_token_kinds": sorted(args.force_schedule_token_kinds),
        "force_argument_boundary_target_tokens": args.force_argument_boundary_target_tokens,
        "constrain_argument_candidate_tokens": args.constrain_argument_candidate_tokens,
        "force_selected_candidate_tokens": args.force_selected_candidate_tokens,
        "force_best_candidate_sequence": args.force_best_candidate_sequence,
        "guard_tool_value_candidates": args.guard_tool_value_candidates,
        "force_best_tool_name_sequence": args.force_best_tool_name_sequence,
        "guard_tool_name_candidates": args.guard_tool_name_candidates,
        "ban_argument_boundary_tokens": args.ban_argument_boundary_tokens,
        "ban_argument_json_boundary_tokens": args.ban_argument_json_boundary_tokens,
        "ban_argument_newline_tokens": args.ban_argument_newline_tokens,
        "guard_tool_call_mode": args.guard_tool_call_mode,
        "guard_tool_json_prefix": args.guard_tool_json_prefix,
        "json_prefix_guard_kinds": sorted(args.json_prefix_guard_kinds),
        "json_prefix_guard_topk": args.json_prefix_guard_topk,
        "json_prefix_guard_left_to_right": args.json_prefix_guard_left_to_right,
        "json_prefix_guard_target_fallback": args.json_prefix_guard_target_fallback,
        "live_tool_json_grammar": args.live_tool_json_grammar,
        "live_tool_json_topk": args.live_tool_json_topk,
        "live_tool_json_mode": "hybrid_diffusion_nl_constrained_ar_json" if args.live_tool_json_grammar else "off",
        "strip_gold_for_generation": args.strip_gold_for_generation,
        "argument_boundary_token_ids": args.argument_boundary_token_ids,
        "argument_newline_token_ids": args.argument_newline_token_ids,
        "force_tool_call_prefix": args.force_tool_call_prefix,
        "forced_assistant_prefix": args.forced_assistant_prefix,
        "stop_after_tool_calls": args.stop_after_tool_calls,
        "stop_after_schedule_tool_calls": args.stop_after_schedule_tool_calls,
        "stop_after_gold_tool_calls": args.stop_after_gold_tool_calls,
        "repair_mode": args.repair_mode,
        "constrained_tool_decoding": args.constrained_tool_decoding,
        "constrained_max_calls": args.constrained_max_calls,
        "constrained_sequence_preserving": args.constrained_sequence_preserving,
        "constrained_assume_utc_z": args.constrained_assume_utc_z,
        "model_repair_pass": args.model_repair_pass,
        "model_repair_max_new_tokens": args.model_repair_max_new_tokens or args.max_new_tokens,
        "mask_id": args.mask_id,
        "stop_token_id": args.stop_token_id,
        "conversation_template": args.conversation_template,
    }
    if torch.cuda.is_available():
        summary["cuda_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        summary["cuda_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer source. Useful when --adapter points to a checkpoint-only PEFT folder.",
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--eval",
        dest="eval_specs",
        action="append",
        type=parse_eval_spec,
        default=[],
        help="Run an eval after a single model load: name:input_jsonl:out_jsonl[:limit]",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--append-instruction", action="store_true")
    parser.add_argument(
        "--force-tool-call-prefix",
        action="store_true",
        help="Force the assistant generation to start with '<tool_call>\\n'.",
    )
    parser.add_argument(
        "--forced-assistant-prefix",
        default="",
        help="Literal assistant-prefix text to append as fixed context before sampling.",
    )
    parser.add_argument("--conversation-template", default=None)
    parser.add_argument("--use-block-cache", action="store_true")
    parser.add_argument("--full-context-sampling", action="store_true")
    parser.add_argument(
        "--fresh-generation-blocks",
        action="store_true",
        help="Start generation from a fresh fully masked block instead of filling only the prompt-tail remainder.",
    )
    parser.add_argument(
        "--sampler-schedule-jsonl",
        type=Path,
        default=None,
        help="Optional token-sensitive sampler schedule JSONL keyed by case id. Only used with --full-context-sampling.",
    )
    parser.add_argument(
        "--force-schedule-token-kinds",
        default="",
        help=(
            "Comma/space separated schedule kinds whose target_token_ids should be "
            "hard-filled during full-context sampling."
        ),
    )
    parser.add_argument(
        "--force-argument-boundary-target-tokens",
        action="store_true",
        help="Within argument_value intervals, hard-fill target tokens that decode as JSON boundary pieces.",
    )
    parser.add_argument(
        "--constrain-argument-candidate-tokens",
        action="store_true",
        help="Within argument_value intervals, mask logits to per-position candidate token ids from the schedule.",
    )
    parser.add_argument(
        "--force-selected-candidate-tokens",
        action="store_true",
        help="Within argument_value intervals, hard-fill the schedule's selected candidate token ids when present.",
    )
    parser.add_argument(
        "--force-best-candidate-sequence",
        action="store_true",
        help="Within argument_value intervals, choose one whole candidate sequence by model score and force it consistently.",
    )
    parser.add_argument(
        "--guard-tool-value-candidates",
        action="store_true",
        help=(
            "Within argument_value intervals, choose and force one whole candidate sequence as a named "
            "schema/value infill guard. Uses the same candidate sequences as --force-best-candidate-sequence "
            "but reports separate value-guard counters."
        ),
    )
    parser.add_argument(
        "--force-best-tool-name-sequence",
        action="store_true",
        help="Within tool_name intervals, choose one compatible available-tool name by model score and force it consistently.",
    )
    parser.add_argument(
        "--guard-tool-name-candidates",
        action="store_true",
        help=(
            "Within tool_name intervals, choose and force one compatible tool-name candidate as a named "
            "route/name guard. Uses the same candidate sequences as --force-best-tool-name-sequence "
            "but reports separate tool-name guard counters."
        ),
    )
    parser.add_argument(
        "--ban-argument-boundary-tokens",
        action="store_true",
        help="Within argument_value intervals, mask tool/role boundary token ids from logits.",
    )
    parser.add_argument(
        "--ban-argument-json-boundary-tokens",
        action="store_true",
        help="With --ban-argument-boundary-tokens, also mask boundary-like target token ids such as '\",\"'.",
    )
    parser.add_argument(
        "--ban-argument-newline-tokens",
        action="store_true",
        help="With --ban-argument-boundary-tokens, also mask newline token ids inside argument_value intervals.",
    )
    parser.add_argument(
        "--guard-tool-call-mode",
        action="store_true",
        help=(
            "Force scheduled tool_tag target tokens as a named tool-call mode/sentinel guard. "
            "Use with --guard-tool-json-prefix so prose cannot bypass the active JSON checker."
        ),
    )
    parser.add_argument(
        "--guard-tool-json-prefix",
        action="store_true",
        help=(
            "Within scheduled JSON/tool intervals, keep commits left-to-right and "
            "replace top tokens that make the active <tool_call> JSON prefix unrecoverable."
        ),
    )
    parser.add_argument(
        "--json-prefix-guard-kinds",
        default="tool_tag,json_structure,json_key,tool_name,argument_value",
        help="Comma/space separated schedule kinds protected by --guard-tool-json-prefix.",
    )
    parser.add_argument(
        "--json-prefix-guard-topk",
        type=int,
        default=32,
        help="Top-k logits scan width for a JSON-prefix-safe replacement token.",
    )
    parser.add_argument(
        "--json-prefix-guard-target-fallback",
        action="store_true",
        help="If top-k has no safe token, try the schedule target token before allowing the original unsafe token.",
    )
    parser.add_argument(
        "--no-json-prefix-guard-left-to-right",
        action="store_true",
        help="Disable left-to-right commit restriction inside JSON-prefix-guarded intervals.",
    )
    parser.add_argument(
        "--live-tool-json-grammar",
        action="store_true",
        help=(
            "Hybrid sampler mode: normal diffusion outside tool JSON spans, but inside an active "
            "<tool_call> body commit the leftmost masked token using the highest-logit token that "
            "keeps the JSON prefix grammar-completable. Label-free: uses no gold target tokens."
        ),
    )
    parser.add_argument(
        "--live-tool-json-topk",
        type=int,
        default=128,
        help="Top-k logits scan width for --live-tool-json-grammar legality selection.",
    )
    parser.add_argument(
        "--strip-gold-for-generation",
        action="store_true",
        help=(
            "Leakage proof mode: remove gold_assistant/gold_tool_names/gold_tool_calls from the case "
            "before prompt construction and sampling. Scoring still uses the original case."
        ),
    )
    parser.add_argument(
        "--stop-after-tool-calls",
        type=int,
        default=0,
        help="When >0, trim decoded assistant text after this many complete </tool_call> blocks before scoring.",
    )
    parser.add_argument(
        "--stop-after-schedule-tool-calls",
        action="store_true",
        help="Trim decoded assistant text after the schedule row's planned tool_call_count before scoring.",
    )
    parser.add_argument(
        "--stop-after-gold-tool-calls",
        action="store_true",
        help="Eval-only oracle stop guard: trim after the gold assistant's tool-call count.",
    )
    parser.add_argument("--repair-mode", choices=["none", "schema"], default="none")
    parser.add_argument("--constrained-tool-decoding", action="store_true")
    parser.add_argument(
        "--constrained-sequence-preserving",
        action="store_true",
        help="Use block-by-block tool-call repair so constrained projection preserves generated call order.",
    )
    parser.add_argument(
        "--constrained-assume-utc-z",
        action="store_true",
        help="For constrained projection, append Z to naive ISO-8601 datetime strings in time-like fields.",
    )
    parser.add_argument(
        "--constrained-max-calls",
        type=int,
        default=0,
        help="When >0, cap constrained projection to this many tool calls. Use 1 for one-call eval slices.",
    )
    parser.add_argument(
        "--model-repair-pass",
        action="store_true",
        help="Run a second model generation pass that rewrites the raw draft into valid Qwen tool-call block(s).",
    )
    parser.add_argument(
        "--model-repair-max-new-tokens",
        type=int,
        default=0,
        help="Override max_new_tokens for --model-repair-pass; 0 reuses --max-new-tokens.",
    )
    parser.add_argument("--no-merge-adapter", action="store_true")
    args = parser.parse_args()
    args.chat_template = resolve_chat_template(args.conversation_template)
    args.sampler_schedules, args.sampler_schedule_rows = load_sampler_schedules(args.sampler_schedule_jsonl)
    args.force_schedule_token_kinds = parse_kind_set(args.force_schedule_token_kinds)
    args.json_prefix_guard_kinds = parse_kind_set(args.json_prefix_guard_kinds)
    args.json_prefix_guard_left_to_right = not args.no_json_prefix_guard_left_to_right
    token_id_tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path or args.adapter or args.base_model),
        trust_remote_code=True,
    )
    args.argument_boundary_token_ids = resolve_single_token_ids(
        token_id_tokenizer,
        ["<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>"],
    )
    args.argument_newline_token_ids = resolve_single_token_ids(
        token_id_tokenizer,
        ["\n", "\r\n"],
    )
    args._argument_boundary_target_cache = {}

    model, tokenizer = load_model(
        str(args.base_model),
        str(args.adapter) if args.adapter else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
    )
    args.mask_id, args.stop_token_id, args.stop_token_ids = resolve_token_ids(model, tokenizer)
    print(
        "[token_ids] "
        + json.dumps({"mask_id": args.mask_id, "stop_token_ids": args.stop_token_ids}, sort_keys=True),
        flush=True,
    )
    eval_specs = args.eval_specs or [("default", args.input_jsonl, args.out_jsonl, args.limit)]
    summaries = [
        run_eval(model, tokenizer, args, eval_name, input_jsonl, out_jsonl, limit)
        for eval_name, input_jsonl, out_jsonl, limit in eval_specs
    ]
    if len(summaries) > 1:
        print(json.dumps({"suite": [item["out_jsonl"] for item in summaries]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
