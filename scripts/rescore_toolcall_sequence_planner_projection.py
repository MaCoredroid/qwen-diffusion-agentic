#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from eval_fastdllm_toolcall_cases import (
    choose_constrained_argument_value,
    clean_repeated_values,
    normalize_for_match,
    normalize_parsed_arguments,
)
from eval_toolcall_jsonl import extract_tool_calls, score_tool_calls, tool_schema_by_name


ROOT = Path("/home/mark/qwen_diffusion")

STOPWORDS = {
    "able",
    "about",
    "after",
    "against",
    "also",
    "and",
    "any",
    "are",
    "assist",
    "based",
    "been",
    "being",
    "call",
    "calls",
    "can",
    "could",
    "current",
    "details",
    "each",
    "enable",
    "every",
    "feature",
    "from",
    "function",
    "functions",
    "given",
    "have",
    "into",
    "looking",
    "manage",
    "need",
    "please",
    "provided",
    "request",
    "should",
    "specified",
    "system",
    "task",
    "that",
    "the",
    "these",
    "this",
    "through",
    "using",
    "want",
    "with",
}

GENERIC_NAME_TOKENS = {
    "activate",
    "add",
    "change",
    "check",
    "configure",
    "control",
    "create",
    "get",
    "install",
    "list",
    "manage",
    "process",
    "record",
    "retrieve",
    "schedule",
    "set",
    "start",
    "submit",
    "track",
    "update",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def index_cases(cases):
    return {case_key(case, idx): case for idx, case in enumerate(cases)}


def user_context_text(case):
    chunks = []
    for message in case.get("prompt_messages") or []:
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content:
                chunks.append(content)
    return "\n\n".join(chunks)


def compact_call(name, arguments):
    payload = {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def compact_calls(calls):
    return "\n".join(compact_call(call.get("name"), call.get("arguments") or {}) for call in calls if call.get("name"))


def unique(items):
    out = []
    seen = set()
    for item in items:
        if item is None:
            continue
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def clean_candidate(value):
    return str(value).strip().strip("`").strip().rstrip("):,.;")


def words(text):
    return [item for item in re.split(r"[^A-Za-z0-9]+", str(text).lower()) if item]


def content_words(text):
    return [item for item in words(text) if len(item) >= 3 and item not in STOPWORDS]


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


def function_payload(tool):
    if not isinstance(tool, dict):
        return {}
    return tool.get("function", tool) if isinstance(tool.get("function", tool), dict) else {}


def tool_name(tool):
    fn = function_payload(tool)
    name = fn.get("name")
    return str(name) if name else None


def tool_tokens(name):
    return [item for item in words(name.replace("_", " ")) if item and item not in STOPWORDS]


def tool_description_tokens(tool):
    fn = function_payload(tool)
    return content_words(fn.get("description") or "")


def table_sections(text):
    lines = text.splitlines(keepends=True)
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    sections = []
    idx = 0
    while idx < len(lines):
        heading = lines[idx].strip()
        if not (heading.startswith("**") and heading.endswith(":**")):
            idx += 1
            continue
        start = idx
        end = idx + 1
        seen_table = False
        while end < len(lines):
            current = lines[end].strip()
            if end > start and current.startswith("**") and current.endswith(":**"):
                break
            if "|" in current:
                seen_table = True
            if seen_table and current.startswith("```") and end > start + 1:
                end += 1
                break
            if seen_table and not current and end + 1 < len(lines) and lines[end + 1].strip().startswith("**"):
                break
            end += 1
        if seen_table:
            content = "".join(lines[start:end]).strip()
            if content:
                sections.append({"start": starts[start], "text": content, "kind": "table_section"})
        idx = max(end, idx + 1)
    return sections


def list_runs(text):
    lines = text.splitlines(keepends=True)
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    runs = []
    current = []
    idx = 0
    while idx < len(lines):
        if not re.match(r"^\s*(?:[-*]|\d+[\.)])\s+", lines[idx]):
            if len(current) >= 2:
                runs.append(current)
            current = []
            idx += 1
            continue
        start = idx
        end = idx + 1
        while end < len(lines):
            line = lines[end]
            if re.match(r"^\s*(?:[-*]|\d+[\.)])\s+", line):
                break
            if not line.strip():
                break
            if re.match(r"^\S", line):
                break
            end += 1
        content = "".join(lines[start:end]).strip()
        if content:
            current.append({"start": starts[start], "text": content, "kind": "list_item"})
        idx = max(end, idx + 1)
    if len(current) >= 2:
        runs.append(current)
    return runs


def paragraph_segments(text):
    segments = []
    offset = 0
    for match in re.finditer(r"\S.*?(?:\n\s*\n|$)", text, flags=re.DOTALL):
        chunk = match.group(0).strip()
        if len(chunk) >= 60:
            segments.append({"start": match.start(), "text": chunk, "kind": "paragraph"})
        offset = match.end()
    if not segments and text.strip():
        segments.append({"start": offset, "text": text.strip(), "kind": "full"})
    return segments


def task_segments(text):
    sections = table_sections(text)
    if len(sections) >= 2:
        return sections
    runs = [run for run in list_runs(text) if len(run) >= 2]
    if runs:
        return runs[-1]
    return paragraph_segments(text)


def has_phrase(haystack, phrase):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", haystack, flags=re.IGNORECASE) is not None


def token_variants(token):
    variants = {str(token).lower()}
    token = str(token).lower()
    if token.endswith("y") and len(token) > 1:
        variants.add(token[:-1] + "ies")
    elif token.endswith("s"):
        variants.add(token[:-1])
    else:
        variants.add(token + "s")
    if token.endswith("e"):
        variants.add(token[:-1] + "ing")
    else:
        variants.add(token + "ing")
    variants.add(token + "ed")
    return variants


def token_present(haystack, token):
    return any(has_phrase(haystack, variant) for variant in token_variants(token))


def quoted_values(text):
    values = []
    for pattern in [r'"([^"\n]{1,160})"', r"'([^'\n]{1,160})'"]:
        values.extend(clean_candidate(match.group(1)) for match in re.finditer(pattern, text))
    return unique(values)


def parenthesized_values(text):
    values = []
    for match in re.finditer(r"\(([^()\n]{1,80})\)", text):
        value = clean_candidate(match.group(1))
        if re.search(r"[A-Za-z]", value) and re.search(r"\d", value):
            values.append(value)
    return unique(values)


def anchor_values(segment):
    anchors = []
    anchors.extend(quoted_values(segment))
    anchors.extend(parenthesized_values(segment))
    anchors.extend(match.group(0).strip() for match in re.finditer(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,5}\b", segment))
    anchors = [item for item in unique(anchors) if len(item) >= 4 and item.lower() not in STOPWORDS]
    anchors.sort(key=len, reverse=True)
    return anchors[:5]


def significant_tokens(text):
    return {
        token
        for token in content_words(text)
        if token not in GENERIC_NAME_TOKENS and token not in {"command", "commands", "device", "devices"}
    }


def infer_voice_device_type(text):
    lower = str(text or "").lower()
    if re.search(r"\bcameras?\b|\bsecurity\s+cameras?\b", lower):
        return "camera"
    if re.search(r"\blights?\b", lower):
        return "light"
    if re.search(r"\bthermostat\b|\btemperature\b", lower):
        return "thermostat"
    return None


def explicit_location_value(text):
    match = re.search(r"\blocation\s*(?::|=|is|as)?\s*[\"']([^\"'\n]+)[\"']", str(text or ""), flags=re.IGNORECASE)
    if match:
        return clean_candidate(match.group(1))
    return None


def infer_camera_location(command, segment):
    explicit = explicit_location_value(segment)
    if explicit is not None:
        return explicit
    for source in [command, segment]:
        for match in re.finditer(r"\b([A-Za-z][A-Za-z -]{1,40}?)\s+cameras?\b", str(source or ""), flags=re.IGNORECASE):
            candidate = clean_candidate(match.group(1))
            candidate = re.sub(
                r"^(?:(?:activate|activating|arm|arming|enable|enabling|start|starting|set|setting|prepare|preparing|the|for)\s+)+",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()
            if candidate and candidate.lower() not in {"security", "the security", "security camera", "security cameras"}:
                return candidate.replace("-", " ")
    return ""


def quoted_command_candidates(text):
    candidates = []
    for match in re.finditer(r"[\"']([^\"'\n]{3,160})[\"']", str(text or "")):
        command = clean_candidate(match.group(1))
        start, end = match.span()
        window = text[max(0, start - 220) : min(len(text), end + 160)]
        if not re.search(
            r"\b(?:voice\s+commands?|spoken|by\s+saying|saying|command(?:\s+it)?\s+with|command\s*:)",
            window,
            flags=re.IGNORECASE,
        ):
            continue
        candidates.append({"command": command, "start": start, "end": end, "window": window})
    return candidates


def voice_command_evidence_for_segment(full_text, segment):
    if not re.search(r"\b(?:voice\s+commands?|by\s+saying|spoken)\b", full_text, flags=re.IGNORECASE):
        return None
    segment_device = infer_voice_device_type(segment)
    if segment_device != "camera":
        return None
    segment_tokens = significant_tokens(segment)
    best = None
    for candidate in quoted_command_candidates(full_text):
        command = candidate["command"]
        command_device = infer_voice_device_type(command)
        if command_device != "camera":
            continue
        command_tokens = significant_tokens(command)
        overlap = segment_tokens & command_tokens
        score = len(overlap)
        if "camera" in command_tokens or "cameras" in command_tokens:
            score += 1.0
        if re.search(r"\bby\s+saying\b", candidate["window"], flags=re.IGNORECASE):
            score += 1.0
        if score < 3.0:
            continue
        item = {
            "command": command,
            "device_type": "camera",
            "location": infer_camera_location(command, segment),
            "score": score,
            "overlap": sorted(overlap),
        }
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def nearby_context(full_context, segment, radius=420):
    chunks = [segment]
    for anchor in anchor_values(segment):
        for match in re.finditer(re.escape(anchor), full_context, flags=re.IGNORECASE):
            start = max(0, match.start() - radius)
            end = min(len(full_context), match.end() + radius)
            chunks.append(full_context[start:end])
            break
    return "\n...\n".join(unique(chunks))


def id_like_property(prop):
    lower = str(prop).lower()
    return lower == "id" or lower.endswith("_id") or lower.endswith("id") or lower in {"appliance", "device"}


def code_like_property(prop):
    return "code" in str(prop).lower() or str(prop).lower().endswith("token")


def candidate_code_values(text, anchors=None):
    if anchors:
        anchored = []
        for chunk in str(text).split("\n...\n"):
            for anchor in anchors:
                match = re.search(re.escape(anchor), chunk, flags=re.IGNORECASE)
                if not match:
                    continue
                window = chunk[match.end() : min(len(chunk), match.end() + 280)]
                anchored.extend(candidate_code_values(window))
                if anchored:
                    return unique(anchored)[:1]
    candidates = []
    for pattern in [
        r"(?:code|token)(?:[^\"'\n]{0,90})(?:is|as|:)?\s*[\"']([^\"'\n]+)[\"']",
        r"[\"']([A-Z0-9][A-Z0-9_.:-]{3,})[\"']",
    ]:
        candidates.extend(clean_candidate(match.group(1)) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return [item for item in unique(candidates) if re.search(r"\d", item)]


def candidate_id_values(text):
    candidates = quoted_values(text) + parenthesized_values(text)
    for pattern in [
        r"\b(?:business|device|appliance|camera)\s+id\s*:\s*([A-Za-z0-9_.:-]+)",
        r"\b(?:Business|Device|Appliance|Camera)\s+ID\s*:\s*([A-Za-z0-9_.:-]+)",
    ]:
        candidates.extend(clean_candidate(match.group(1)) for match in re.finditer(pattern, text))
    return [item for item in unique(candidates) if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", item) and re.search(r"\d", item)]


def candidate_model_values(text):
    candidates = []
    nouns = r"(?:motion detectors?|smart locks?|lock|fridge|washing machine|smart lights?|lights?|thermostat|camera)"
    for match in re.finditer(rf"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){{0,5}})\s+{nouns}\b", text):
        value = clean_candidate(match.group(1))
        value = re.sub(r"^(?:Configure|Install|Activate|For|The|My)\s+", "", value).strip()
        if value and value.lower() not in {"for", "the", "my"}:
            candidates.append(value)
    return unique(candidates)


def segment_specific_value(prop, prop_schema, segment_context, anchors=None):
    expected = schema_type(prop_schema if isinstance(prop_schema, dict) else {})
    if expected not in {None, "string"}:
        return None
    prop_lower = str(prop).lower()
    candidates = []
    if id_like_property(prop):
        candidates = candidate_id_values(segment_context)
    elif code_like_property(prop):
        candidates = candidate_code_values(segment_context, anchors=anchors)
    elif prop_lower == "model":
        candidates = candidate_model_values(segment_context)
    if len(candidates) == 1:
        return candidates[0]
    return None


def property_match_score(segment, prop, prop_schema):
    lower = segment.lower()
    prop_spaced = str(prop).replace("_", " ")
    score = 0.0
    if has_phrase(lower, str(prop).lower()) or has_phrase(lower, prop_spaced.lower()):
        score += 3.0
    prop_tokens = [token for token in words(prop_spaced) if len(token) >= 3 and token not in STOPWORDS]
    for token in prop_tokens:
        if token_present(lower, token):
            score += 0.8
    if isinstance(prop_schema, dict):
        for value in prop_schema.get("enum") or []:
            if has_phrase(lower, str(value).lower()):
                score += 2.0
        desc_tokens = content_words(prop_schema.get("description") or "")
        for token in desc_tokens[:8]:
            if token_present(lower, token):
                score += 0.25
    return score


def tool_score(segment, tool, schema):
    lower = segment.lower()
    name = tool_name(tool) or ""
    name_spaced = name.replace("_", " ")
    score = 0.0
    if has_phrase(lower, name.lower()) or has_phrase(lower, name_spaced.lower()):
        score += 24.0
    for token in tool_tokens(name):
        if token_present(lower, token):
            score += 1.5 if token in GENERIC_NAME_TOKENS else 4.0
    desc_tokens = tool_description_tokens(tool)
    for token in desc_tokens[:12]:
        if token_present(lower, token):
            score += 0.55

    props = schema_properties(schema)
    required = required_properties(schema)
    matched_required = 0
    for prop, prop_schema in props.items():
        prop_score = property_match_score(segment, prop, prop_schema)
        score += prop_score
        if prop in required and prop_score >= 2.0:
            matched_required += 1
    if required and matched_required == len(required):
        score += 4.0
    if "voice command" in lower and ("voice" in desc_tokens or "command" in props):
        score += 6.0
    return score


def choose_tool_for_segment(segment, tools, schemas):
    scored = []
    for tool in tools:
        name = tool_name(tool)
        if not name or name not in schemas:
            continue
        scored.append((tool_score(segment, tool, schemas[name]), name))
    scored.sort(reverse=True)
    if not scored:
        return None, 0.0, 0.0
    best_score, best_name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 5.0:
        return None, best_score, runner_up
    if runner_up and best_score - runner_up < 1.0 and best_score < 14.0:
        return None, best_score, runner_up
    return best_name, best_score, runner_up


def voice_command_tool_override(full_text, segment, chosen_name, schemas):
    if "activate_voice_command" not in schemas:
        return None
    if chosen_name == "activate_voice_command":
        return None
    if not chosen_name or "camera" not in chosen_name.replace("_", " ").lower():
        return None
    return voice_command_evidence_for_segment(full_text, segment)


def planned_sequence(case):
    text = user_context_text(case)
    schemas = tool_schema_by_name(case.get("tools") or [])
    segments = task_segments(text)
    planned = []
    segment_audit = []
    for segment in segments:
        name, score, runner_up = choose_tool_for_segment(segment["text"], case.get("tools") or [], schemas)
        voice_evidence = voice_command_tool_override(text, segment["text"], name, schemas)
        original_name = name
        if voice_evidence:
            name = "activate_voice_command"
        audit_row = {
            "kind": segment["kind"],
            "start": segment["start"],
            "text": segment["text"][:500],
            "chosen_name": name,
            "score": round(score, 3),
            "runner_up": round(runner_up, 3),
        }
        if voice_evidence:
            audit_row["original_chosen_name"] = original_name
            audit_row["voice_command_override"] = voice_evidence
        segment_audit.append(audit_row)
        if name:
            planned.append(
                {
                    "name": name,
                    "segment": segment,
                    "score": score,
                    "runner_up": runner_up,
                    "voice_command_evidence": voice_evidence,
                }
            )
    return planned, segment_audit


def draft_call_queues(text):
    queues = defaultdict(deque)
    for call in extract_tool_calls(text)[0]:
        name = call.get("name")
        if name:
            queues[name].append(call)
    return queues


def draft_names(text):
    return [call.get("name") for call in extract_tool_calls(text)[0] if call.get("name")]


def sequence_mismatch_is_safe(audit, min_score=14.0, min_margin=2.0):
    if not audit:
        return False
    for item in audit:
        if not item.get("chosen_name"):
            return False
        score = float(item.get("score") or 0.0)
        runner_up = float(item.get("runner_up") or 0.0)
        if score < min_score or score - runner_up < min_margin:
            return False
    return True


def has_targeted_sequence_override(audit):
    return any(bool(item.get("voice_command_override")) for item in audit or [])


def should_use_plan(
    input_names,
    planned_names,
    min_input_calls,
    use_plan_on_sequence_mismatch=False,
    use_safe_plan_on_sequence_mismatch=False,
    safe_sequence_mismatch=False,
    targeted_sequence_override=False,
):
    if not planned_names:
        return False
    if len(input_names) < min_input_calls:
        return False
    if not input_names:
        return True
    if planned_names == input_names:
        return True
    if targeted_sequence_override and len(planned_names) == len(input_names):
        return True
    if use_plan_on_sequence_mismatch and len(planned_names) == len(input_names):
        return True
    if use_safe_plan_on_sequence_mismatch and safe_sequence_mismatch and len(planned_names) == len(input_names):
        return True
    if len(planned_names) > len(input_names):
        return True
    if len(planned_names) == len(input_names) and Counter(planned_names) == Counter(input_names):
        return True
    return False


def choose_draft_call(name, queues):
    if queues.get(name):
        return queues[name].popleft()
    return {"name": name, "arguments": {}}


def planned_argument_value(
    prop,
    prop_schema,
    parsed_value,
    scoped_context,
    segment_context,
    full_context,
    generated_text,
    prefer_segment_args,
):
    if prefer_segment_args:
        value = choose_constrained_argument_value(
            prop,
            prop_schema,
            None,
            scoped_context,
            "",
            "",
        )
        if value is None:
            value = segment_specific_value(prop, prop_schema, scoped_context)
        if value is None:
            value = segment_specific_value(prop, prop_schema, segment_context, anchors=anchor_values(scoped_context))
        if value is not None:
            return value
    return choose_constrained_argument_value(
        prop,
        prop_schema,
        parsed_value,
        segment_context,
        full_context,
        generated_text,
    )


def planned_call_text(case, source_text, prefer_segment_args=True):
    schemas = tool_schema_by_name(case.get("tools") or [])
    planned, audit = planned_sequence(case)
    if not planned:
        return "", audit, []

    queues = draft_call_queues(source_text)
    full_context = "\n".join(part for part in [user_context_text(case), source_text] if part)
    generated_text = source_text
    out_calls = []
    for item in planned:
        name = item["name"]
        schema = schemas.get(name) or {}
        properties = schema_properties(schema)
        required = required_properties(schema)
        draft = choose_draft_call(name, queues)
        normalized_parsed, unknown_values = normalize_parsed_arguments(draft.get("arguments") or {}, properties)
        string_props = [
            prop
            for prop, prop_schema in properties.items()
            if schema_type(prop_schema if isinstance(prop_schema, dict) else {}) == "string"
        ]
        arguments = {}
        scoped_context = item["segment"]["text"]
        segment_context = nearby_context(full_context, scoped_context)
        voice_evidence = item.get("voice_command_evidence") or {}
        for prop, prop_schema in properties.items():
            parsed_value = normalized_parsed.get(prop)
            if name == "activate_voice_command" and prop in voice_evidence:
                value = voice_evidence[prop]
            else:
                value = planned_argument_value(
                    prop,
                    prop_schema,
                    parsed_value,
                    scoped_context,
                    segment_context,
                    full_context,
                    generated_text,
                    prefer_segment_args,
                )
            if value is None and prop in string_props and len(string_props) == 1:
                string_values = [value for value in unknown_values if isinstance(value, str) and value.strip()]
                if len(string_values) == 1:
                    value = string_values[0].strip()
            if value is not None:
                arguments[prop] = clean_repeated_values(value)
            elif prop in required:
                continue
        out_calls.append({"name": name, "arguments": arguments})
    return compact_calls(out_calls), audit, [call["name"] for call in out_calls]


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
    row[f"{prefix}_exact_tool_name_set"] = bool(metrics.get("exact_tool_name_set"))
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
    totals = {"records": 0, "ok": 0, "errors": 0, "input": metric_totals(), "planned": metric_totals()}
    planned_lengths = Counter()
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            out = dict(row)
            try:
                case = cases_by_key.get(case_key(row, idx)) or (cases[idx] if idx < len(cases) else None)
                if case is None:
                    raise KeyError(f"no case for row {idx}")
                text = str(row.get(args.text_field) or "")
                planned_text, audit, planned_names = planned_call_text(
                    case,
                    text,
                    prefer_segment_args=args.prefer_segment_args,
                )
                input_names = draft_names(text)
                targeted_sequence_override = has_targeted_sequence_override(audit)
                safe_sequence_mismatch = sequence_mismatch_is_safe(
                    audit,
                    min_score=args.sequence_mismatch_min_score,
                    min_margin=args.sequence_mismatch_min_margin,
                )
                used_plan = should_use_plan(
                    input_names,
                    planned_names,
                    args.min_input_calls_for_plan,
                    use_plan_on_sequence_mismatch=args.use_plan_on_sequence_mismatch,
                    use_safe_plan_on_sequence_mismatch=args.use_safe_plan_on_sequence_mismatch,
                    safe_sequence_mismatch=safe_sequence_mismatch,
                    targeted_sequence_override=targeted_sequence_override,
                )
                if not planned_text or not used_plan:
                    planned_text = text
                input_metrics = score_tool_calls(text, case.get("tools") or [], case.get("gold_assistant"))
                planned_metrics = score_tool_calls(planned_text, case.get("tools") or [], case.get("gold_assistant"))
                planned_lengths[str(len(planned_names))] += 1
                out.update(
                    {
                        "sequence_planner_status": "ok",
                        "sequence_planner_text_field": args.text_field,
                        "sequence_planner_assistant": planned_text,
                        "sequence_planner_names": planned_names,
                        "sequence_planner_input_names": input_names,
                        "sequence_planner_used_plan": used_plan,
                        "sequence_planner_targeted_sequence_override": targeted_sequence_override,
                        "sequence_planner_safe_sequence_mismatch": safe_sequence_mismatch,
                        "sequence_planner_segment_audit": audit,
                    }
                )
                add_row_metrics(out, "sequence_planner_input", input_metrics)
                add_row_metrics(out, "sequence_planner", planned_metrics)
                add_metric_totals(totals["input"], input_metrics)
                add_metric_totals(totals["planned"], planned_metrics)
                totals["ok"] += 1
            except Exception as exc:
                out.update({"sequence_planner_status": "error", "sequence_planner_error": f"{type(exc).__name__}: {exc}"})
                totals["errors"] += 1
            totals["records"] += 1
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "text_field": args.text_field,
        "min_input_calls_for_plan": args.min_input_calls_for_plan,
        "use_plan_on_sequence_mismatch": args.use_plan_on_sequence_mismatch,
        "use_safe_plan_on_sequence_mismatch": args.use_safe_plan_on_sequence_mismatch,
        "sequence_mismatch_min_score": args.sequence_mismatch_min_score,
        "sequence_mismatch_min_margin": args.sequence_mismatch_min_margin,
        "prefer_segment_args": args.prefer_segment_args,
        "totals": totals,
        "planned_lengths": dict(sorted(planned_lengths.items())),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--text-field", default="contextual_projection_assistant")
    parser.add_argument(
        "--min-input-calls-for-plan",
        type=int,
        default=2,
        help="Only replace the input when the rescored input already has at least this many tool calls.",
    )
    parser.add_argument(
        "--use-plan-on-sequence-mismatch",
        action="store_true",
        help="Opt in to replacing same-length draft sequences when the request planner disagrees with the draft tool names.",
    )
    parser.add_argument(
        "--use-safe-plan-on-sequence-mismatch",
        action="store_true",
        help="Replace same-length draft sequences only when every planned segment clears the score and margin thresholds.",
    )
    parser.add_argument("--sequence-mismatch-min-score", type=float, default=14.0)
    parser.add_argument("--sequence-mismatch-min-margin", type=float, default=2.0)
    parser.add_argument(
        "--no-prefer-segment-args",
        dest="prefer_segment_args",
        action="store_false",
        help="Keep stale draft arguments ahead of segment-local request evidence.",
    )
    parser.set_defaults(prefer_segment_args=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
