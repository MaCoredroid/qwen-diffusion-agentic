#!/usr/bin/env python3
import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls, tool_schema_by_name


ROOT = Path("/home/mark/qwen_diffusion")
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def index_cases(cases):
    return {case_key(case, idx): case for idx, case in enumerate(cases)}


def case_context_text(case):
    chunks = []
    for message in case.get("prompt_messages") or []:
        if message.get("role") in {"system", "user"}:
            content = str(message.get("content") or "").strip()
            if content:
                chunks.append(content)
    return "\n\n".join(chunks)


def compact_call(name, arguments):
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def compact_calls(calls):
    return "\n".join(compact_call(call.get("name"), call.get("arguments") or {}) for call in calls if call.get("name"))


def schema_type(schema):
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
    return expected


def schema_properties(schema):
    props = (schema or {}).get("properties") if isinstance(schema, dict) else {}
    return props if isinstance(props, dict) else {}


def required_properties(schema):
    required = (schema or {}).get("required") if isinstance(schema, dict) else []
    return set(required if isinstance(required, list) else [])


def scalar_properties(schema, arguments):
    props = schema_properties(schema)
    keys = list(props) if props else sorted((arguments or {}).keys())
    out = []
    for key in keys:
        expected = schema_type(props.get(key, {}) if isinstance(props, dict) else {})
        if expected not in {"array", "object"}:
            out.append(key)
    return out


def numbered_line_context(text, call_index):
    target = str(call_index + 1)
    lines = text.splitlines()
    out = []
    for idx, line in enumerate(lines):
        if re.match(rf"^\s*{re.escape(target)}[\.)]\s+", line):
            out.append(line)
    return "\n\n".join(out)


def function_name_context(text, name):
    if not name:
        return ""
    lines = text.splitlines()
    out = []
    for line in lines:
        if name in line:
            out.append(line)
    return "\n\n".join(out)


def segment_context(text, call_index, call_count, radius=900):
    if call_count <= 0:
        return text[:radius]
    center = int((call_index + 0.5) * len(text) / call_count)
    return text[max(0, center - radius // 2) : min(len(text), center + radius // 2)]


def call_context(text, name, call_index, call_count):
    chunks = [
        function_section_context(text, name),
        function_name_context(text, name),
        numbered_line_context(text, call_index),
        segment_context(text, call_index, call_count),
    ]
    seen = set()
    out = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            out.append(chunk)
    return "\n...\n".join(out)


def function_section_context(text, name):
    if not name:
        return ""
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if name in line:
            start = idx
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if re.search(r"`functions\.[A-Za-z0-9_]+`|`[A-Za-z0-9_]+` tool", line) and name not in line:
            end = idx
            break
    return "\n".join(lines[start:end])


def quoted_values(text):
    values = []
    for pattern in [r'"([^"\n]{1,120})"', r"'([^'\n]{1,120})'"]:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            if value and value not in values:
                values.append(value)
    return values


def parse_dates(text):
    dates = []
    for match in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text):
        dates.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    month_pattern = "|".join(MONTHS)
    for match in re.finditer(rf"\b({month_pattern})\s+(\d{{1,2}}),\s*(20\d{{2}})\b", text, flags=re.IGNORECASE):
        dates.append((int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2))))
    out = []
    for date in dates:
        if date not in out:
            out.append(date)
    return out


def parse_times(text):
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


def iso_datetime_candidates(text):
    dates = parse_dates(text)
    times = parse_times(text)
    if not dates or not times:
        return []
    year, month, day = dates[-1]
    return [f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z" for hour, minute in times]


def is_id_like_property(prop):
    lower = prop.lower()
    return lower == "id" or lower.endswith("_id") or lower.endswith("id") or lower in {"camera", "device"}


def plausible_id_candidate(prop, original, candidate):
    lower_prop = prop.lower()
    if lower_prop in {"camera_id", "cameraid"}:
        return True
    original_text = str(original or "")
    candidate_has_id_marker = bool(re.search(r"[\d-]", candidate))
    original_has_id_marker = bool(re.search(r"[\d-]", original_text))
    if original_has_id_marker and not candidate_has_id_marker:
        return False
    return candidate_has_id_marker


def choose_id_candidate(prop, original, context):
    chunks = [chunk.strip() for chunk in context.split("\n...\n") if chunk.strip()] or [context]
    candidates = []
    for chunk in chunks:
        chunk_candidates = [
            item
            for item in quoted_values(chunk)
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", item)
            and not item.lower().startswith("http")
        ]
        original_text = str(original or "")
        original_in_chunk = original_text and original_text.lower() in chunk.lower()
        alternatives = [item for item in chunk_candidates if item != original_text and plausible_id_candidate(prop, original, item)]
        if not original_in_chunk and len(alternatives) == 1:
            return alternatives[0], "specific_context_single_quoted_id"
        candidates.extend(item for item in chunk_candidates if item not in candidates)
    if not candidates:
        return None, None
    original_text = str(original or "")
    original_in_context = original_text and original_text.lower() in context.lower()
    alternatives = [item for item in candidates if item != original_text and plausible_id_candidate(prop, original, item)]
    if not alternatives:
        return None, None
    if not original_in_context and len(alternatives) == 1:
        return alternatives[0], "single_quoted_id_not_original"
    prop_tokens = [token for token in re.split(r"[_\W]+", prop.lower()) if len(token) >= 4 and token not in {"device", "camera"}]
    scored = []
    for item in alternatives:
        lower = item.lower()
        score = sum(int(token in lower) for token in prop_tokens)
        scored.append((score, item))
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1], "property_token_id_match"
    return None, None


def choose_datetime_candidate(prop, original, context):
    lower = prop.lower()
    if not any(token in lower for token in ["time", "date"]):
        return None, None
    candidates = iso_datetime_candidates(context)
    if not candidates:
        return None, None
    if "start" in lower:
        candidate = candidates[0]
    elif "end" in lower:
        candidate = candidates[-1]
    else:
        return None, None
    if candidate != original:
        return candidate, "datetime_from_call_context"
    return None, None


def property_line_values(prop, context):
    values = []
    for key in property_context_keys(prop):
        for value in property_line_values_for_key(key, context):
            if value not in values:
                values.append(value)
    return values


def property_context_keys(prop):
    prop = str(prop)
    keys = [prop]
    alias_map = {
        "body": ["email_body", "message_body"],
        "date": ["callback_date", "appointment_date", "delivery_date"],
        "recipient": ["email", "customer_email"],
        "subject": ["email_subject", "message_subject"],
        "time": ["callback_time", "appointment_time", "delivery_time"],
    }
    for key in alias_map.get(prop.lower(), []):
        if key not in keys:
            keys.append(key)
    return keys


def property_line_values_for_key(prop, context):
    escaped = re.escape(prop)
    values = []
    patterns = [
        rf"(?:`{escaped}`|['\"]{escaped}['\"]|{escaped})\s*:\s*\"([^\"\n]+)\"",
        rf"(?:`{escaped}`|['\"]{escaped}['\"]|{escaped})\s*:\s*'([^'\n]+)'",
        rf"(?:`{escaped}`|['\"]{escaped}['\"]|{escaped})\s*:\s*([^\n,;]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, context, flags=re.IGNORECASE):
            value = match.group(1).strip().strip("`").strip()
            if value and value not in values:
                values.append(value)
    return values


def coerce_context_value(raw, expected):
    if raw is None:
        return None
    value = str(raw).strip().strip(".").strip("\"'")
    if expected == "integer":
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return None
    if expected == "number":
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
            number = float(value)
            return int(number) if number.is_integer() else number
        return None
    if expected == "boolean":
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return None
    if expected == "string":
        return value
    return None


def choose_malformed_string_cleanup(prop, original, context):
    value = str(original or "").strip()
    if not value:
        return None, None
    candidates = []
    for candidate in [
        value.lstrip("\"'"),
        value.rstrip("\"'"),
        value.strip("\"'"),
        value.replace('\\"', '"').strip("\"'"),
    ]:
        candidate = candidate.strip()
        if candidate and candidate != value and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate and candidate.lower() in context.lower():
            return candidate, "malformed_quote_cleanup"
    return None, None


def choose_explicit_property_candidate(prop, expected, original, context):
    values = property_line_values(prop, context)
    candidates = []
    for value in values:
        candidate = coerce_context_value(value, expected)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    if len(candidates) == 1 and candidates[0] != original:
        return candidates[0], "explicit_property_value"
    return None, None


def project_call(call, schema, context):
    projected = copy.deepcopy(call)
    arguments = copy.deepcopy(call.get("arguments") or {})
    replacements = []
    props_by_name = schema_properties(schema)
    required = required_properties(schema)
    props = scalar_properties(schema, arguments)
    for prop in props:
        prop_schema = props_by_name.get(prop, {})
        original = arguments.get(prop)
        expected = schema_type(prop_schema)
        candidate, reason = None, None
        if prop not in arguments or original is None or original == "":
            if prop in required:
                candidate, reason = choose_explicit_property_candidate(prop, expected, original, context)
        elif isinstance(original, str):
            candidate, reason = choose_explicit_property_candidate(prop, expected, original, context)
            if candidate is not None:
                reason = "explicit_property_value"
            elif is_id_like_property(prop):
                candidate, reason = choose_id_candidate(prop, original, context)
                if candidate is None:
                    candidate, reason = choose_malformed_string_cleanup(prop, original, context)
            else:
                candidate, reason = choose_malformed_string_cleanup(prop, original, context)
            if candidate is None:
                candidate, reason = choose_datetime_candidate(prop, original, context)
        if candidate is not None and candidate != original:
            arguments[prop] = candidate
            replacements.append({"property": prop, "from": original, "to": candidate, "reason": reason})
    projected["arguments"] = arguments
    return projected, replacements


def metric_totals():
    return {
        "valid_tool_json": 0,
        "exact_tool_name_set": 0,
        "exact_tool_name_multiset": 0,
        "exact_tool_sequence": 0,
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
    }


def add_metric_totals(totals, metrics):
    totals["valid_tool_json"] += int(bool(metrics.get("valid_tool_call")))
    totals["exact_tool_name_set"] += int(bool(metrics.get("exact_tool_name_set")))
    totals["exact_tool_name_multiset"] += int(bool(metrics.get("exact_tool_name_multiset")))
    totals["exact_tool_sequence"] += int(bool(metrics.get("exact_tool_sequence")))
    totals["same_tool_call_count"] += int(bool(metrics.get("same_tool_call_count")))
    totals["exact_arguments"] += int(bool(metrics.get("exact_arguments")))
    totals["all_schema_valid"] += int(bool(metrics.get("all_schema_valid")))
    totals["all_required_args_present"] += int(bool(metrics.get("all_required_args_present")))
    extra = int(metrics.get("extra_call_count") or 0)
    missing = int(metrics.get("missing_call_count") or 0)
    repeated = int(metrics.get("repeated_call_count") or 0)
    totals["records_with_extra_calls"] += int(extra > 0)
    totals["records_with_missing_calls"] += int(missing > 0)
    totals["records_with_repeated_calls"] += int(repeated > 0)
    totals["total_extra_calls"] += extra
    totals["total_missing_calls"] += missing
    totals["total_repeated_calls"] += repeated


def add_row_metrics(row, prefix, metrics):
    row[f"{prefix}_called_names"] = metrics.get("called_names") or []
    row[f"{prefix}_calls"] = metrics.get("calls") or []
    row[f"{prefix}_valid_tool_json"] = bool(metrics.get("valid_tool_call"))
    row[f"{prefix}_exact_tool_sequence"] = bool(metrics.get("exact_tool_sequence"))
    row[f"{prefix}_exact_arguments"] = bool(metrics.get("exact_arguments"))
    row[f"{prefix}_all_schema_valid"] = bool(metrics.get("all_schema_valid"))
    row[f"{prefix}_all_required_args_present"] = bool(metrics.get("all_required_args_present"))
    row[f"{prefix}_extra_call_count"] = metrics.get("extra_call_count")
    row[f"{prefix}_missing_call_count"] = metrics.get("missing_call_count")
    row[f"{prefix}_repeated_call_count"] = metrics.get("repeated_call_count")
    row[f"{prefix}_call_errors"] = metrics.get("call_errors") or []


def run(args):
    cases = load_jsonl(args.cases_jsonl)
    rows = load_jsonl(args.input_jsonl)
    cases_by_key = index_cases(cases)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals = {"records": 0, "ok": 0, "errors": 0, "input": metric_totals(), "projected": metric_totals()}
    replacement_counts = Counter()
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            out = dict(row)
            try:
                case = cases_by_key.get(case_key(row, idx)) or (cases[idx] if idx < len(cases) else None)
                if case is None:
                    raise KeyError(f"no case for row {idx}")
                text = str(row.get(args.text_field) or "")
                calls, invalid = extract_tool_calls(text)
                schemas = tool_schema_by_name(case.get("tools") or [])
                context_text = case_context_text(case)
                projected_calls = []
                replacements = []
                for call_index, call in enumerate(calls):
                    context = call_context(context_text, call.get("name"), call_index, len(calls))
                    projected, call_replacements = project_call(call, schemas.get(call.get("name"), {}), context)
                    projected_calls.append(projected)
                    for replacement in call_replacements:
                        replacement = {"call_index": call_index, "tool_name": call.get("name"), **replacement}
                        replacements.append(replacement)
                        replacement_counts[replacement["reason"]] += 1
                projected_text = compact_calls(projected_calls)
                input_metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
                projected_metrics = score_tool_calls(projected_text, case.get("tools") or [], case.get("gold_assistant"))
                out.update(
                    {
                        "contextual_projection_assistant": projected_text,
                        "contextual_projection_replacements": replacements,
                        "contextual_projection_replacement_count": len(replacements),
                        "contextual_projection_input_invalid_tool_json_count": invalid,
                        "status": "ok",
                    }
                )
                add_row_metrics(out, "contextual_projection_input", input_metrics)
                add_row_metrics(out, "contextual_projection", projected_metrics)
                add_metric_totals(totals["input"], input_metrics)
                add_metric_totals(totals["projected"], projected_metrics)
                totals["ok"] += 1
            except Exception as exc:
                out.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            totals["records"] += 1
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "text_field": args.text_field,
        "totals": totals,
        "replacement_counts": dict(sorted(replacement_counts.items())),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--text-field", default="scalar_repair_constrained_assistant")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
