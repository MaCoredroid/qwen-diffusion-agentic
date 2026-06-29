#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


POLICIES = {
    "prose": {
        "block": "large",
        "role": "low-risk natural language",
        "suggested_steps": 4,
        "constrain": "none",
    },
    "tool_tag": {
        "block": "tiny",
        "role": "tool boundary",
        "suggested_steps": 1,
        "constrain": "literal",
    },
    "json_key": {
        "block": "tiny",
        "role": "schema key",
        "suggested_steps": 1,
        "constrain": "schema-key-enum",
    },
    "tool_name": {
        "block": "tiny",
        "role": "function selector",
        "suggested_steps": 2,
        "constrain": "tool-name-enum",
    },
    "argument_value": {
        "block": "small",
        "role": "grounded argument value",
        "suggested_steps": 8,
        "constrain": "json-schema-and-request-evidence",
    },
    "json_structure": {
        "block": "small",
        "role": "JSON punctuation/container structure",
        "suggested_steps": 3,
        "constrain": "json-grammar",
    },
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def split_chunks(start, end, kind, max_chars):
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + max_chars)
        yield {"start": cursor, "end": chunk_end, "kind": kind}
        cursor = chunk_end


def is_escaped(text, idx):
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return bool(backslashes % 2)


def scan_json_tokens(text):
    tokens = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch.isspace():
            idx += 1
            continue
        if ch == '"':
            start = idx
            idx += 1
            while idx < len(text):
                if text[idx] == '"' and not is_escaped(text, idx):
                    idx += 1
                    break
                idx += 1
            raw = text[start:idx]
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw.strip('"')
            tokens.append({"type": "string", "start": start, "end": idx, "value": value})
            continue
        if ch in "{}[]:,":
            tokens.append({"type": ch, "start": idx, "end": idx + 1, "value": ch})
            idx += 1
            continue
        match = re.match(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null", text[idx:])
        if match:
            raw = match.group(0)
            tokens.append({"type": "literal", "start": idx, "end": idx + len(raw), "value": raw})
            idx += len(raw)
            continue
        tokens.append({"type": "unknown", "start": idx, "end": idx + 1, "value": ch})
        idx += 1
    return tokens


def path_to_string(parts):
    rendered = ""
    normalized = list(parts)
    if normalized and normalized[0] == "arguments":
        normalized = normalized[1:]
    for part in normalized:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            if rendered:
                rendered += "."
            rendered += str(part)
    return rendered


def json_token_paths(tokens):
    paths = {}

    def skip_commas(idx):
        while idx < len(tokens) and tokens[idx]["type"] == ",":
            idx += 1
        return idx

    def parse_value(idx, path):
        if idx >= len(tokens):
            return idx
        token = tokens[idx]
        if token["type"] == "{":
            return parse_object(idx, path)
        if token["type"] == "[":
            return parse_array(idx, path)
        if token["type"] in {"string", "literal"}:
            paths[(token["start"], token["end"])] = path_to_string(path)
        return idx + 1

    def parse_object(idx, path):
        idx += 1
        while idx < len(tokens):
            idx = skip_commas(idx)
            if idx >= len(tokens) or tokens[idx]["type"] == "}":
                return idx + 1
            key_token = tokens[idx]
            if key_token["type"] != "string":
                idx += 1
                continue
            key = key_token["value"]
            key_path = path + [key]
            paths[(key_token["start"], key_token["end"])] = path_to_string(key_path)
            idx += 1
            if idx < len(tokens) and tokens[idx]["type"] == ":":
                idx += 1
            idx = parse_value(idx, key_path)
            if idx < len(tokens) and tokens[idx]["type"] == ",":
                idx += 1
        return idx

    def parse_array(idx, path):
        idx += 1
        array_idx = 0
        while idx < len(tokens):
            idx = skip_commas(idx)
            if idx >= len(tokens) or tokens[idx]["type"] == "]":
                return idx + 1
            idx = parse_value(idx, path + [array_idx])
            array_idx += 1
            if idx < len(tokens) and tokens[idx]["type"] == ",":
                idx += 1
        return idx

    parse_value(0, [])
    return paths


def next_nonspace(text, idx):
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return text[idx] if idx < len(text) else ""


def previous_nonspace(text, idx):
    idx -= 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    return text[idx] if idx >= 0 else ""


def token_sensitive_spans(json_text, base_offset):
    tokens = scan_json_tokens(json_text)
    token_paths = json_token_paths(tokens)
    spans = []
    last_key = None
    for token in tokens:
        kind = None
        if token["type"] == "string" and next_nonspace(json_text, token["end"]) == ":":
            last_key = token["value"]
            kind = "json_key"
        elif token["type"] in {"string", "literal"} and previous_nonspace(json_text, token["start"]) == ":":
            json_path = token_paths.get((token["start"], token["end"]))
            kind = "tool_name" if json_path == "name" else "argument_value"
        if kind:
            json_path = token_paths.get((token["start"], token["end"]))
            spans.append(
                {
                    "start": base_offset + token["start"],
                    "end": base_offset + token["end"],
                    "kind": kind,
                    "json_key": last_key,
                    "json_path": json_path,
                    "argument_path": json_path if kind == "argument_value" else None,
                    "text": json_text[token["start"] : token["end"]],
                }
            )
    return spans


def add_structural_gaps(segments, start, end, sensitive_spans, max_chars):
    cursor = start
    for span in sorted(sensitive_spans, key=lambda item: (item["start"], item["end"])):
        if cursor < span["start"]:
            segments.extend(split_chunks(cursor, span["start"], "json_structure", max_chars))
        segments.append(span)
        cursor = max(cursor, span["end"])
    if cursor < end:
        segments.extend(split_chunks(cursor, end, "json_structure", max_chars))


def plan_text(text, max_prose_chars, max_json_structure_chars):
    segments = []
    cursor = 0
    for tool_call_idx, match in enumerate(TOOL_CALL_RE.finditer(text)):
        if cursor < match.start():
            segments.extend(split_chunks(cursor, match.start(), "prose", max_prose_chars))

        before_tool_segments = len(segments)
        body_start, body_end = match.span(1)
        segments.append({"start": match.start(), "end": body_start, "kind": "tool_tag"})
        body = text[body_start:body_end]
        sensitive_spans = token_sensitive_spans(body, body_start)
        add_structural_gaps(
            segments,
            body_start,
            body_end,
            sensitive_spans,
            max_json_structure_chars,
        )
        segments.append({"start": body_end, "end": match.end(), "kind": "tool_tag"})
        for segment in segments[before_tool_segments:]:
            segment["tool_call_index"] = tool_call_idx
        cursor = match.end()

    if cursor < len(text):
        segments.extend(split_chunks(cursor, len(text), "prose", max_prose_chars))

    for idx, segment in enumerate(segments):
        segment["idx"] = idx
        segment["chars"] = segment["end"] - segment["start"]
        segment["policy"] = POLICIES[segment["kind"]]
    return segments


def load_tokenizer(tokenizer_path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for --tokenizer-path") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit(f"tokenizer does not provide offset mapping: {tokenizer_path}")
    return tokenizer


def token_offsets(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    return {
        "input_ids": encoded["input_ids"],
        "offsets": [(int(start), int(end)) for start, end in offsets],
    }


def overlapping_token_range(offsets, start, end):
    token_indexes = [
        idx
        for idx, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    if not token_indexes:
        return None
    return min(token_indexes), max(token_indexes) + 1


def add_token_spans(record, tokenizer, include_token_ids=False):
    tokenized = token_offsets(tokenizer, record["text"])
    offsets = tokenized["offsets"]
    input_ids = tokenized["input_ids"]
    record["token_count"] = len(input_ids)
    record["tokenizer_has_offsets"] = True
    record["token_blocks"] = token_blocks_for_segments(record, offsets)
    for segment in record["segments"]:
        token_range = overlapping_token_range(offsets, segment["start"], segment["end"])
        if token_range is None:
            segment["token_start"] = None
            segment["token_end"] = None
            segment["token_count"] = 0
            segment["token_boundary_exact"] = False
            continue
        token_start, token_end = token_range
        token_char_start = min(offsets[idx][0] for idx in range(token_start, token_end))
        token_char_end = max(offsets[idx][1] for idx in range(token_start, token_end))
        segment["token_start"] = token_start
        segment["token_end"] = token_end
        segment["token_count"] = token_end - token_start
        segment["token_char_start"] = token_char_start
        segment["token_char_end"] = token_char_end
        segment["token_boundary_exact"] = (
            token_char_start == segment["start"] and token_char_end == segment["end"]
        )
        if include_token_ids:
            segment["token_ids"] = input_ids[token_start:token_end]
    if include_token_ids:
        for block in record["token_blocks"]:
            block["token_ids"] = input_ids[block["token_start"] : block["token_end"]]


def overlap_len(left_start, left_end, right_start, right_end):
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def token_blocks_for_segments(record, offsets):
    segments = record["segments"]
    if not segments:
        return []
    blocks = []
    active = None
    for token_idx, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        best_segment = None
        best_overlap = 0
        for segment in segments:
            current = overlap_len(token_start, token_end, segment["start"], segment["end"])
            if current > best_overlap:
                best_segment = segment
                best_overlap = current
        if best_segment is None or best_overlap <= 0:
            continue
        kind = best_segment["kind"]
        if active and active["kind"] == kind and active["token_end"] == token_idx:
            active["token_end"] = token_idx + 1
            active["char_end"] = token_end
            active["segment_indexes"].append(best_segment["idx"])
            continue
        active = {
            "token_start": token_idx,
            "token_end": token_idx + 1,
            "token_count": 1,
            "char_start": token_start,
            "char_end": token_end,
            "kind": kind,
            "policy": POLICIES[kind],
            "segment_indexes": [best_segment["idx"]],
        }
        blocks.append(active)
    for block in blocks:
        block["token_count"] = block["token_end"] - block["token_start"]
        block["segment_indexes"] = sorted(set(block["segment_indexes"]))
        source_segments = [segments[idx] for idx in block["segment_indexes"] if idx < len(segments)]
        for field in ["tool_call_index", "json_key", "json_path", "argument_path"]:
            values = {
                segment.get(field)
                for segment in source_segments
                if segment.get(field) is not None
            }
            if len(values) == 1:
                block[field] = next(iter(values))
        texts = []
        for segment in source_segments:
            text = segment.get("text")
            if text is not None and text not in texts:
                texts.append(text)
        if texts:
            block["segment_texts"] = texts
            if len(texts) == 1:
                block["target_text"] = texts[0]
    return blocks


def summarize(records):
    totals = Counter()
    for record in records:
        totals["records"] += 1
        totals["segments"] += len(record["segments"])
        totals["tool_calls"] += record.get("tool_call_count", 0)
        totals["records_with_tool_calls"] += int(record.get("tool_call_count", 0) > 0)
        totals["records_without_segments"] += int(not record["segments"])
        totals["tokens"] += record.get("token_count", 0)
        totals["token_blocks"] += len(record.get("token_blocks", []))
        for block in record.get("token_blocks", []):
            totals[f"token_blocks:{block['kind']}"] += 1
            totals[f"block_tokens:{block['kind']}"] += block["token_count"]
        for segment in record["segments"]:
            totals[f"kind:{segment['kind']}"] += 1
            totals[f"chars:{segment['kind']}"] += segment["chars"]
            if "token_count" in segment:
                totals[f"tokens:{segment['kind']}"] += segment["token_count"]
                totals["tokenized_segments"] += 1
                totals["token_boundary_mismatches"] += int(
                    not segment.get("token_boundary_exact")
                )
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--text-field", default="gold_assistant")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-prose-chars", type=int, default=512)
    parser.add_argument("--max-json-structure-chars", type=int, default=96)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--include-token-ids", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path) if args.tokenizer_path else None

    records = []
    for idx, case in enumerate(load_jsonl(args.input_jsonl)):
        if args.limit and idx >= args.limit:
            break
        text = case.get(args.text_field) or ""
        segments = plan_text(text, args.max_prose_chars, args.max_json_structure_chars)
        records.append(
            {
                "id": case.get("id") or str(idx),
                "source": case.get("source"),
                "text_field": args.text_field,
                "text": text,
                "text_chars": len(text),
                "tool_call_count": len(TOOL_CALL_RE.findall(text)),
                "segments": segments,
            }
        )
        if tokenizer:
            add_token_spans(records[-1], tokenizer, include_token_ids=args.include_token_ids)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "text_field": args.text_field,
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "policy": POLICIES,
        "totals": summarize(records),
    }

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        args.out_jsonl.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
