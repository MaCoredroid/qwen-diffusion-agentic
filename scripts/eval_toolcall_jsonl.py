#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import re
from pathlib import Path


def parse_first_json_object(text):
    start = text.find("{")
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
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except Exception:
                    return None
    return None


def extract_json_objects(text):
    objects = []
    cursor = 0
    while True:
        start = text.find("<tool_call>", cursor)
        if start < 0:
            break
        content_start = start + len("<tool_call>")
        end = text.find("</tool_call>", content_start)
        if end < 0:
            objects.append(None)
            break
        objects.append(parse_first_json_object(text[content_start:end]))
        cursor = end + len("</tool_call>")
    if objects:
        return objects
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return [parse_first_json_object(stripped)]
    return []


def parse_scalar_value(text):
    value = str(text).strip()
    if not value:
        return ""
    if value[0] in "[{\"":
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def normalize_json_tool_call(obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool_name")
    arguments = obj.get("arguments", {})
    if isinstance(name, dict):
        function = name
        name = function.get("name")
        arguments = function.get("arguments", arguments)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"__raw_arguments__": arguments}
    if not isinstance(arguments, dict):
        arguments = {"__raw_arguments__": arguments}
    if not name:
        return None
    return {"name": str(name), "arguments": arguments, "format": "json"}


def normalize_tool_call_object(obj):
    """Normalize JSON-style or OpenAI-native tool-call objects."""
    return normalize_json_tool_call(obj)


def extract_tool_call_objects(tool_calls):
    calls = []
    invalid = 0
    if not isinstance(tool_calls, list):
        return calls, 1
    for item in tool_calls:
        call = normalize_tool_call_object(item)
        if call is None:
            invalid += 1
        else:
            call["format"] = "openai_tool_call" if isinstance(item, dict) and item.get("type") == "function" else call["format"]
            calls.append(call)
    return calls, invalid


def extract_qwen_function_calls(text):
    calls = []
    for match in re.finditer(r"<function=([^>\s]+)>(.*?)</function>", text, re.DOTALL):
        name = match.group(1).strip()
        body = match.group(2)
        arguments = {}
        for param in re.finditer(r"<parameter=([^>\s]+)>(.*?)</parameter>", body, re.DOTALL):
            key = param.group(1).strip()
            arguments[key] = parse_scalar_value(param.group(2))
        calls.append({"name": name, "arguments": arguments, "format": "qwen_function"})
    return calls


def extract_tool_calls(text):
    calls = []
    invalid = 0
    cursor = 0
    found_tool_tag = False
    while True:
        start = text.find("<tool_call>", cursor)
        if start < 0:
            break
        found_tool_tag = True
        content_start = start + len("<tool_call>")
        end = text.find("</tool_call>", content_start)
        if end < 0:
            invalid += 1
            break
        content = text[content_start:end]
        qwen_calls = extract_qwen_function_calls(content)
        if qwen_calls:
            calls.extend(qwen_calls)
        else:
            obj = parse_first_json_object(content)
            call = normalize_json_tool_call(obj)
            if call is None:
                invalid += 1
            else:
                calls.append(call)
        cursor = end + len("</tool_call>")

    if found_tool_tag:
        return calls, invalid

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        call = normalize_json_tool_call(parse_first_json_object(stripped))
        if call is None:
            return [], 1
        return [call], 0
    return [], 0


def required_tool_names(tools):
    names = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
    return names


def tool_schema_by_name(tools):
    schemas = {}
    if not isinstance(tools, list):
        return schemas
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if isinstance(fn, dict) and fn.get("name"):
            schemas[str(fn["name"])] = fn.get("parameters") or {}
    return schemas


def coerce_value(value, schema):
    if not isinstance(schema, dict):
        return value
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        expected_type = next((item for item in expected_type if item != "null"), expected_type[0] if expected_type else None)

    if isinstance(value, str):
        stripped = value.strip()
        if expected_type in {"integer", "number"}:
            try:
                return int(stripped) if expected_type == "integer" else float(stripped)
            except Exception:
                return value
        if expected_type == "boolean":
            if stripped.lower() == "true":
                return True
            if stripped.lower() == "false":
                return False
            return value
        if expected_type in {"array", "object"} and stripped[:1] in "[{":
            try:
                value = json.loads(stripped)
            except Exception:
                return value

    if expected_type == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        return {key: coerce_value(item, props.get(key, {})) for key, item in value.items()}
    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items") or {}
        return [coerce_value(item, item_schema) for item in value]
    return value


def coerce_arguments(arguments, schema):
    if not isinstance(arguments, dict):
        return arguments
    props = (schema or {}).get("properties") or {}
    return {key: coerce_value(value, props.get(key, {})) for key, value in arguments.items()}


def schema_errors(value, schema, path="$"):
    if not isinstance(schema, dict) or not schema:
        return []
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value not in enum")

    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        type_errors = []
        for item_type in expected_type:
            type_errors = schema_errors(value, {**schema, "type": item_type}, path)
            if not type_errors:
                return []
        return type_errors

    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: missing required")
        props = schema.get("properties") or {}
        for key, item in value.items():
            if key in props:
                errors.extend(schema_errors(item, props[key], f"{path}.{key}"))
        return errors
    if expected_type == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        item_schema = schema.get("items") or {}
        for idx, item in enumerate(value):
            errors.extend(schema_errors(item, item_schema, f"{path}[{idx}]"))
        return errors
    if expected_type == "string" and not isinstance(value, str):
        errors.append(f"{path}: expected string")
    elif expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        errors.append(f"{path}: expected integer")
    elif expected_type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        errors.append(f"{path}: expected number")
    elif expected_type == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}: expected boolean")
    return errors


def normalize_call_for_compare(call, schemas):
    schema = schemas.get(call.get("name"), {})
    return {
        "name": call.get("name"),
        "arguments": coerce_arguments(call.get("arguments") or {}, schema),
    }


def count_extra_missing(called_names, gold_names):
    called = Counter(called_names)
    gold = Counter(gold_names)
    extra = called - gold
    missing = gold - called
    repeated = Counter({name: count - 1 for name, count in called.items() if count > 1})
    return {
        "extra_call_count": sum(extra.values()),
        "missing_call_count": sum(missing.values()),
        "repeated_call_count": sum(repeated.values()),
        "extra_call_names": sorted(extra.elements()),
        "missing_call_names": sorted(missing.elements()),
        "repeated_call_names": sorted(repeated.elements()),
    }


def score_normalized_tool_calls(calls, invalid, tools, gold_calls=None, gold_invalid=0):
    schemas = tool_schema_by_name(tools)
    called_names = [call["name"] for call in calls]
    schema_valid_count = 0
    required_args_count = 0
    call_errors = []
    for call in calls:
        schema = schemas.get(call["name"], {})
        coerced_args = coerce_arguments(call.get("arguments") or {}, schema)
        errors = schema_errors(coerced_args, schema)
        call_errors.append(errors)
        if not errors and schema:
            schema_valid_count += 1
        required = (schema or {}).get("required") or []
        if all(key in coerced_args for key in required):
            required_args_count += 1

    metrics = {
        "tool_call_count": len(calls) + invalid,
        "recognized_tool_call_count": len(calls),
        "invalid_tool_call_count": invalid,
        "valid_tool_call": bool(calls) and invalid == 0,
        "called_names": called_names,
        "called_name_set": sorted(set(called_names)),
        "known_tool_call": bool(set(called_names) & set(schemas)) if schemas else None,
        "schema_valid_count": schema_valid_count,
        "all_schema_valid": bool(calls) and schema_valid_count == len(calls),
        "required_args_count": required_args_count,
        "all_required_args_present": bool(calls) and required_args_count == len(calls),
        "call_errors": call_errors,
        "calls": calls,
    }

    if gold_calls is not None:
        gold_names = [call["name"] for call in gold_calls]
        normalized_calls = [normalize_call_for_compare(call, schemas) for call in calls]
        normalized_gold = [normalize_call_for_compare(call, schemas) for call in gold_calls]
        count_metrics = count_extra_missing(called_names, gold_names)
        metrics.update(
            {
                "gold_tool_call_count": len(gold_calls) + gold_invalid,
                "gold_invalid_tool_call_count": gold_invalid,
                "gold_called_names": gold_names,
                "exact_tool_sequence": called_names == gold_names,
                "exact_tool_name_multiset": Counter(called_names) == Counter(gold_names),
                "exact_tool_name_set": set(called_names) == set(gold_names),
                "same_tool_call_count": len(calls) == len(gold_calls),
                "exact_arguments": normalized_calls == normalized_gold,
                **count_metrics,
            }
        )
    return metrics


def score_tool_calls(text, tools, gold_text=None):
    calls, invalid = extract_tool_calls(text)
    gold_calls = None
    gold_invalid = 0
    if gold_text is not None:
        gold_calls, gold_invalid = extract_tool_calls(gold_text)
    return score_normalized_tool_calls(calls, invalid, tools, gold_calls, gold_invalid)


def score_tool_call_objects(tool_calls, tools, gold_tool_calls=None, gold_text=None):
    calls, invalid = extract_tool_call_objects(tool_calls)
    gold_calls = None
    gold_invalid = 0
    if gold_tool_calls is not None:
        gold_calls, gold_invalid = extract_tool_call_objects(gold_tool_calls)
    elif gold_text is not None:
        gold_calls, gold_invalid = extract_tool_calls(gold_text)
    return score_normalized_tool_calls(calls, invalid, tools, gold_calls, gold_invalid)


def score_record(record):
    messages = record.get("messages") or []
    assistant_text = "\n".join(
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "assistant"
    )
    tool_names = required_tool_names(record.get("tools"))
    metrics = score_tool_calls(assistant_text, record.get("tools"))
    return {
        "has_assistant": bool(assistant_text.strip()),
        "tool_call_count": metrics["tool_call_count"],
        "valid_json_count": metrics["recognized_tool_call_count"],
        "valid_json": metrics["valid_tool_call"],
        "known_tool_call": bool(set(metrics["called_names"]) & tool_names) if tool_names else None,
        "called_names": sorted(set(metrics["called_names"])),
        "available_tool_count": len(tool_names),
        "schema_valid_count": metrics["schema_valid_count"],
        "all_schema_valid": metrics["all_schema_valid"],
        "all_required_args_present": metrics["all_required_args_present"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    totals = {
        "records": 0,
        "has_assistant": 0,
        "records_with_tool_call": 0,
        "records_with_valid_json": 0,
        "records_with_known_tool_call": 0,
        "records_with_tools": 0,
    }
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            score = score_record(record)
            score["line_no"] = line_no
            score["source"] = record.get("source")
            rows.append(score)
            totals["records"] += 1
            totals["has_assistant"] += int(score["has_assistant"])
            totals["records_with_tool_call"] += int(score["tool_call_count"] > 0)
            totals["records_with_valid_json"] += int(score["valid_json"])
            if score["available_tool_count"]:
                totals["records_with_tools"] += 1
                totals["records_with_known_tool_call"] += int(bool(score["known_tool_call"]))

    summary = {"input": str(args.jsonl), "totals": totals, "rows": rows}
    out = args.out or args.jsonl.with_suffix(".eval.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
