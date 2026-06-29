#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_CASES = ROOT / "data/toolcall_eval/heldout_seed_multicall_2to3_clean.jsonl"
DEFAULT_REQUIRED = ROOT / "runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_required.jsonl"
DEFAULT_AUTO = ROOT / "runs/qwen36_teacher_heldout_multicall_2to3_clean_toolchoice_auto.jsonl"
DEFAULT_HEURISTIC = ROOT / "runs/heldout_seed_multicall_2to3_clean/sequence_planner_from_empty.jsonl"
DEFAULT_OUT = ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_analysis.jsonl"


def load_jsonl(path):
    rows = []
    if not path or not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def case_key(row, fallback):
    return row.get("id") or row.get("case_id") or str(fallback)


def by_key(rows):
    return {case_key(row, idx): row for idx, row in enumerate(rows)}


def user_text(case):
    return "\n\n".join(
        str(message.get("content") or "").strip()
        for message in case.get("prompt_messages") or []
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    )


def function_payload(tool):
    if not isinstance(tool, dict):
        return {}
    fn = tool.get("function", tool)
    return fn if isinstance(fn, dict) else {}


def tool_name(tool):
    fn = function_payload(tool)
    name = fn.get("name")
    return str(name) if name else ""


def tool_description(tool):
    fn = function_payload(tool)
    return str(fn.get("description") or "")


def words(text):
    return [item for item in re.split(r"[^A-Za-z0-9]+", str(text).lower()) if item]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "function",
    "functions",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def content_tokens(text):
    return [token for token in words(text) if len(token) >= 3 and token not in STOPWORDS]


def variants(token):
    out = {token}
    if token.endswith("y") and len(token) > 1:
        out.add(token[:-1] + "ies")
    elif token.endswith("s"):
        out.add(token[:-1])
    else:
        out.add(token + "s")
    if token.endswith("e"):
        out.add(token[:-1] + "ing")
    else:
        out.add(token + "ing")
    out.add(token + "ed")
    return out


def token_present(haystack, token):
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(item) + r"(?![A-Za-z0-9])", haystack) for item in variants(token))


def tool_evidence(case, name):
    text = user_text(case)
    lower = text.lower()
    name_spaced = str(name).replace("_", " ").lower()
    explicit_name = bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(name_spaced) + r"(?![A-Za-z0-9])", lower))
    name_tokens = content_tokens(name_spaced)
    token_hits = [token for token in name_tokens if token_present(lower, token)]
    tool = next((tool for tool in case.get("tools") or [] if tool_name(tool) == name), None)
    desc_tokens = content_tokens(tool_description(tool))[:16] if tool else []
    desc_hits = [token for token in desc_tokens if token_present(lower, token)]
    strong = explicit_name or len(token_hits) >= max(2, min(3, len(name_tokens))) or (
        len(token_hits) >= 1 and len(desc_hits) >= 2
    )
    return {
        "tool_name": name,
        "explicit_name": explicit_name,
        "name_tokens": name_tokens,
        "name_token_hits": token_hits,
        "description_token_hits": desc_hits,
        "prompt_evidence": bool(strong),
    }


def repeated_split_needed(gold_names, candidate_names):
    rows = []
    gold_counts = Counter(gold_names)
    cand_counts = Counter(candidate_names)
    for name, gold_count in gold_counts.items():
        cand_count = cand_counts.get(name, 0)
        if gold_count > 1 and cand_count == 1:
            rows.append({"tool_name": name, "gold_count": gold_count, "candidate_count": cand_count})
    return rows


def flat_values(value, path=""):
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            out.extend(flat_values(item, child))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(flat_values(item, f"{path}[{idx}]"))
        return out
    return [{"path": path, "value": value}]


def argument_mismatch_tags(gold_calls, candidate_calls):
    tags = Counter()
    examples = []
    if len(gold_calls) != len(candidate_calls):
        return tags, examples
    for call_idx, (gold, candidate) in enumerate(zip(gold_calls, candidate_calls)):
        if gold.get("name") != candidate.get("name"):
            continue
        gold_flat = {row["path"]: row["value"] for row in flat_values(gold.get("arguments") or {})}
        cand_flat = {row["path"]: row["value"] for row in flat_values(candidate.get("arguments") or {})}
        for path in sorted(set(gold_flat) | set(cand_flat)):
            gold_value = gold_flat.get(path)
            cand_value = cand_flat.get(path)
            if gold_value == cand_value:
                continue
            text = " ".join(str(item) for item in [path, gold_value, cand_value]).lower()
            if re.search(r"\bgrowth|rate|percent|risk|weight|refund|policy|event_id|device_id|network_id\b", text):
                if "refund" in text or "policy" in text:
                    tag = "policy_threshold_or_enum"
                elif "growth" in text or "rate" in text or "percent" in text:
                    tag = "percentage_normalization"
                elif "weight" in text:
                    tag = "rounded_residual"
                elif "risk" in text:
                    tag = "alias_normalization"
                elif "id" in text:
                    tag = "id_grounding"
                else:
                    tag = "scalar_normalization"
            elif gold_value is None or cand_value is None:
                tag = "missing_or_extra_argument"
            else:
                tag = "value_grounding"
            tags[tag] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "tool_call_index": call_idx,
                        "tool_name": gold.get("name"),
                        "path": path,
                        "gold": gold_value,
                        "candidate": cand_value,
                        "tag": tag,
                    }
                )
    return tags, examples


def metric(row, key, default=False):
    if row is None:
        return default
    return bool(row.get(key))


def names(row, prefix=""):
    if row is None:
        return []
    field = f"{prefix}called_names" if prefix else "called_names"
    return list(row.get(field) or [])


def calls(row, prefix=""):
    if row is None:
        return []
    field = f"{prefix}calls" if prefix else "calls"
    return list(row.get(field) or [])


def classify(case, required, auto, heuristic):
    gold_names = list(case.get("gold_tool_names") or [])
    required_names = names(required)
    auto_names = names(auto)
    heuristic_names = names(heuristic, "sequence_planner_")
    tags = []
    recommended = "curate"

    if metric(required, "exact_tool_sequence") and metric(required, "exact_arguments"):
        tags.append("teacher_required_exact")
        recommended = "teacher_required_or_gold"
    elif metric(required, "exact_tool_sequence"):
        tags.append("teacher_required_sequence_exact_args_need_normalization")
        recommended = "teacher_required_sequence_plus_value_sidecars"
    else:
        tags.append("teacher_required_sequence_mismatch")

    if metric(auto, "exact_tool_sequence") and metric(auto, "exact_arguments"):
        tags.append("teacher_auto_exact")
    elif metric(auto, "exact_tool_sequence"):
        tags.append("teacher_auto_sequence_exact_args_need_normalization")
    else:
        tags.append("teacher_auto_sequence_mismatch")

    extra_required = list((required or {}).get("extra_call_names") or [])
    missing_required = list((required or {}).get("missing_call_names") or [])
    extra_evidence = [tool_evidence(case, name) for name in extra_required]
    missing_evidence = [tool_evidence(case, name) for name in missing_required]
    if extra_required:
        if extra_evidence and all(item["prompt_evidence"] for item in extra_evidence):
            tags.append("seed_gold_subset_ambiguous_teacher_overcalls_prompt_supported")
            recommended = "adjudicate_full_request_vs_seed_gold"
        else:
            tags.append("teacher_required_extra_calls")
    if missing_required:
        if any(item["prompt_evidence"] for item in missing_evidence):
            tags.append("teacher_required_undercalls_prompt_supported_gold")
            if recommended == "curate":
                recommended = "gold_sequence_decomposition_target"
        else:
            tags.append("teacher_required_missing_calls")

    split_rows = repeated_split_needed(gold_names, required_names)
    if split_rows:
        tags.append("split_call_policy_needed")
        if recommended == "curate":
            recommended = "gold_split_call_target"

    arg_tags, arg_examples = argument_mismatch_tags(case.get("gold_tool_calls") or [], calls(required))
    for tag, count in arg_tags.items():
        tags.append(f"arg_{tag}")
    if arg_tags and recommended == "curate":
        recommended = "value_normalization_sidecar_target"

    if metric(heuristic, "sequence_planner_exact_tool_sequence") and metric(heuristic, "sequence_planner_exact_arguments"):
        tags.append("heuristic_exact")
    elif metric(heuristic, "sequence_planner_exact_tool_sequence"):
        tags.append("heuristic_sequence_exact_args_wrong")
    else:
        tags.append("heuristic_sequence_mismatch")

    return {
        "tags": sorted(set(tags)),
        "recommended_policy_target": recommended,
        "gold_names": gold_names,
        "teacher_required_names": required_names,
        "teacher_auto_names": auto_names,
        "heuristic_names": heuristic_names,
        "teacher_required_extra_evidence": extra_evidence,
        "teacher_required_missing_evidence": missing_evidence,
        "split_call_policy": split_rows,
        "argument_mismatch_tags": dict(sorted(arg_tags.items())),
        "argument_mismatch_examples": arg_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--teacher-required-jsonl", type=Path, default=DEFAULT_REQUIRED)
    parser.add_argument("--teacher-auto-jsonl", type=Path, default=DEFAULT_AUTO)
    parser.add_argument("--heuristic-jsonl", type=Path, default=DEFAULT_HEURISTIC)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases = load_jsonl(args.cases_jsonl)
    required_by_key = by_key(load_jsonl(args.teacher_required_jsonl))
    auto_by_key = by_key(load_jsonl(args.teacher_auto_jsonl))
    heuristic_by_key = by_key(load_jsonl(args.heuristic_jsonl))

    rows = []
    totals = Counter()
    tag_counts = Counter()
    target_counts = Counter()
    for idx, case in enumerate(cases):
        key = case_key(case, idx)
        required = required_by_key.get(key)
        auto = auto_by_key.get(key)
        heuristic = heuristic_by_key.get(key)
        analysis = classify(case, required, auto, heuristic)
        row = {
            "id": key,
            "source": case.get("source"),
            "task": case.get("task"),
            "category": case.get("category"),
            "prompt_excerpt": " ".join(user_text(case).split())[:500],
            "teacher_required_metrics": {
                "valid_tool_json": metric(required, "valid_tool_json"),
                "exact_tool_sequence": metric(required, "exact_tool_sequence"),
                "exact_arguments": metric(required, "exact_arguments"),
                "all_schema_valid": metric(required, "all_schema_valid"),
                "all_required_args_present": metric(required, "all_required_args_present"),
                "extra_call_count": (required or {}).get("extra_call_count"),
                "missing_call_count": (required or {}).get("missing_call_count"),
                "repeated_call_count": (required or {}).get("repeated_call_count"),
            },
            "teacher_auto_metrics": {
                "valid_tool_json": metric(auto, "valid_tool_json"),
                "exact_tool_sequence": metric(auto, "exact_tool_sequence"),
                "exact_arguments": metric(auto, "exact_arguments"),
                "all_schema_valid": metric(auto, "all_schema_valid"),
                "all_required_args_present": metric(auto, "all_required_args_present"),
                "extra_call_count": (auto or {}).get("extra_call_count"),
                "missing_call_count": (auto or {}).get("missing_call_count"),
                "repeated_call_count": (auto or {}).get("repeated_call_count"),
            },
            "heuristic_metrics": {
                "valid_tool_json": metric(heuristic, "sequence_planner_valid_tool_json"),
                "exact_tool_sequence": metric(heuristic, "sequence_planner_exact_tool_sequence"),
                "exact_arguments": metric(heuristic, "sequence_planner_exact_arguments"),
                "all_schema_valid": metric(heuristic, "sequence_planner_all_schema_valid"),
                "all_required_args_present": metric(heuristic, "sequence_planner_all_required_args_present"),
                "extra_call_count": (heuristic or {}).get("sequence_planner_extra_call_count"),
                "missing_call_count": (heuristic or {}).get("sequence_planner_missing_call_count"),
                "repeated_call_count": (heuristic or {}).get("sequence_planner_repeated_call_count"),
            },
            **analysis,
        }
        rows.append(row)
        totals["records"] += 1
        target_counts[row["recommended_policy_target"]] += 1
        for tag in row["tags"]:
            tag_counts[tag] += 1

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "teacher_required_jsonl": str(args.teacher_required_jsonl),
        "teacher_auto_jsonl": str(args.teacher_auto_jsonl),
        "heuristic_jsonl": str(args.heuristic_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "totals": dict(totals),
        "recommended_policy_target_counts": dict(sorted(target_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
