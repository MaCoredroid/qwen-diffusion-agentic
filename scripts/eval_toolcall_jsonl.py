#!/usr/bin/env python3
import argparse
import json
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


def score_record(record):
    messages = record.get("messages") or []
    assistant_text = "\n".join(
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "assistant"
    )
    parsed = extract_json_objects(assistant_text)
    valid = [obj for obj in parsed if isinstance(obj, dict)]
    tool_names = required_tool_names(record.get("tools"))
    called_names = set()
    for obj in valid:
        name = obj.get("name") or obj.get("function") or obj.get("tool_name")
        if isinstance(name, dict):
            name = name.get("name")
        if name:
            called_names.add(str(name))
    return {
        "has_assistant": bool(assistant_text.strip()),
        "tool_call_count": len(parsed),
        "valid_json_count": len(valid),
        "valid_json": bool(parsed) and len(valid) == len(parsed),
        "known_tool_call": bool(called_names & tool_names) if tool_names else None,
        "called_names": sorted(called_names),
        "available_tool_count": len(tool_names),
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
