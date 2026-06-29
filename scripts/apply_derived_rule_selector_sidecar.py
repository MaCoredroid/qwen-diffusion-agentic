#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_SELECTOR = (
    ROOT
    / "runs/candidate_ranking/"
    "heldout_seed_multicall_gold_evidence_selector_peerctx_rules_snippets_ckpt275_pairwise_tournament.jsonl"
)
DEFAULT_EXAMPLES = (
    ROOT
    / "data/candidate_ranking/"
    "heldout_seed_multicall_gold_evidence_selector_toolname_argument_ranking_evidence_peerctx_rules.jsonl"
)
DEFAULT_OUT = (
    ROOT
    / "runs/candidate_ranking/"
    "heldout_seed_multicall_gold_evidence_selector_peerctx_rules_snippets_derived_sidecar.jsonl"
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row):
    return (
        row.get("id"),
        row.get("kind"),
        row.get("tool_call_index"),
        row.get("json_key"),
        row.get("json_path") or row.get("argument_path") or row.get("miss_path"),
        row.get("schedule_token_start"),
        row.get("schedule_token_end"),
    )


def normalize(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.strip()
    return value


def values_equal(left, right):
    left = normalize(left)
    right = normalize(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def candidate_index(candidates, value):
    for idx, candidate in enumerate(candidates):
        if values_equal(candidate, value):
            return idx
    return -1


def context_from_prompt(prompt):
    marker = "\n\nSpan kind:"
    before_span = (prompt or "").split(marker, 1)[0]
    context_marker = "User/tool context:\n"
    if context_marker in before_span:
        return before_span.split(context_marker, 1)[1].strip()
    return before_span.strip()


INDEXED_PATH_RE = re.compile(r"^(?P<prefix>.+)\[(?P<idx>\d+)\]\.(?P<leaf>[^.\[]+)$")


def indexed_path(path):
    match = INDEXED_PATH_RE.match(path or "")
    if not match:
        return None
    return match.group("prefix"), int(match.group("idx")), match.group("leaf")


def peer_path(peer):
    return peer.get("json_path") or peer.get("argument_path") or peer.get("json_key") or ""


def all_peer_arguments(row):
    out = []
    seen = set()
    for field in ("local_peer_arguments", "same_call_peer_arguments"):
        for peer in row.get(field) or []:
            key = (peer_path(peer), json.dumps(peer.get("target"), sort_keys=True, ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(peer)
    return out


def equal_weight_residual(row, context):
    path = row.get("json_path") or row.get("argument_path") or ""
    parsed = indexed_path(path)
    if not parsed:
        return None
    prefix, current_idx, leaf = parsed
    if leaf != "weight":
        return None
    candidates = row.get("candidate_values") or []
    numeric_candidates = [(idx, numeric(value)) for idx, value in enumerate(candidates)]
    numeric_candidates = [(idx, value) for idx, value in numeric_candidates if value is not None]
    if len(numeric_candidates) < 2:
        return None

    weights = {}
    item_indices = {current_idx}
    for peer in all_peer_arguments(row):
        peer_parsed = indexed_path(peer_path(peer))
        if not peer_parsed:
            continue
        peer_prefix, peer_idx, peer_leaf = peer_parsed
        if peer_prefix != prefix:
            continue
        item_indices.add(peer_idx)
        if peer_leaf == "weight":
            value = numeric(peer.get("target"))
            if value is not None:
                weights[peer_idx] = value

    if len(item_indices) < 2 or current_idx != max(item_indices):
        return None
    if any(idx not in weights for idx in item_indices if idx != current_idx):
        return None

    known_sum = sum(weights[idx] for idx in item_indices if idx != current_idx)
    scored = []
    for idx, value in numeric_candidates:
        error = abs((known_sum + value) - 1.0)
        scored.append((error, idx, candidates[idx]))
    scored.sort(key=lambda item: (item[0], item[1]))
    if not scored or scored[0][0] > 0.005:
        return None
    if len(scored) > 1 and math.isclose(scored[0][0], scored[1][0], abs_tol=1e-9):
        return None
    return {
        "rule": "equal_weight_residual",
        "predicted_index": scored[0][1],
        "predicted_value": scored[0][2],
        "reason": {
            "array_prefix": prefix,
            "known_weight_sum": known_sum,
            "item_indices": sorted(item_indices),
            "residual_error": scored[0][0],
        },
    }


PERCENT_RANGE_RE = re.compile(
    r"(?P<a>-?\d+(?:\.\d+)?)\s*%\s*(?:to|-|–|—)\s*(?P<b>-?\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def snippet_around(context, anchor, window=220):
    if not anchor:
        return ""
    lower_context = context.lower()
    idx = lower_context.find(str(anchor).lower())
    if idx < 0:
        return ""
    start = max(0, idx - window)
    end = min(len(context), idx + len(str(anchor)) + window)
    return context[start:end]


def percentage_range_midpoint(row, context):
    key = str(row.get("json_key") or "").lower()
    path = str(row.get("json_path") or row.get("argument_path") or "").lower()
    if not any(term in f"{key} {path}" for term in ("rate", "growth", "percent")):
        return None
    candidates = row.get("candidate_values") or []
    numeric_candidates = [(idx, numeric(value)) for idx, value in enumerate(candidates)]
    numeric_candidates = [(idx, value) for idx, value in numeric_candidates if value is not None]
    if not numeric_candidates:
        return None

    anchors = [
        peer.get("target")
        for peer in all_peer_arguments(row)
        if isinstance(peer.get("target"), str) and len(peer.get("target")) >= 3
    ]
    anchors.append(row.get("json_key"))
    for anchor in anchors:
        snippet = snippet_around(context, anchor)
        if not snippet:
            continue
        match = PERCENT_RANGE_RE.search(snippet)
        if not match:
            continue
        left = float(match.group("a"))
        right = float(match.group("b"))
        midpoint = (left + right) / 2.0
        scored = sorted(
            (abs(value - midpoint), idx, candidates[idx])
            for idx, value in numeric_candidates
        )
        if scored and scored[0][0] <= 0.01:
            return {
                "rule": "percentage_range_midpoint",
                "predicted_index": scored[0][1],
                "predicted_value": scored[0][2],
                "reason": {
                    "anchor": anchor,
                    "range": [left, right],
                    "midpoint": midpoint,
                    "snippet": " ".join(snippet.split())[:320],
                },
            }
    return None


def refund_policy_threshold(row, context):
    key = str(row.get("json_key") or "").lower()
    path = str(row.get("json_path") or row.get("argument_path") or "").lower()
    if "refund" not in f"{key} {path}":
        return None
    candidates = row.get("candidate_values") or []
    normalized_candidates = {str(value).lower(): idx for idx, value in enumerate(candidates)}
    if not {"full", "partial", "none"}.issubset(set(normalized_candidates)):
        return None

    anchors = [
        peer.get("target")
        for peer in all_peer_arguments(row)
        if isinstance(peer.get("target"), str) and len(peer.get("target")) >= 3
    ]
    relevant = [snippet_around(context, anchor, window=260) for anchor in anchors]
    relevant = [snippet for snippet in relevant if snippet]
    request_text = " ".join(relevant) if relevant else context
    days_matches = [int(match.group(1)) for match in re.finditer(r"(\d+)\s+days?\s+before", request_text, re.I)]
    if not days_matches:
        return None
    days_before = days_matches[-1]

    full_match = re.search(r"full\s+refunds?.{0,80}?up to\s+(\d+)\s+days?\s+before", context, re.I | re.S)
    partial_match = re.search(
        r"(?:50\s*%|partial)\s+refunds?.{0,80}?up to\s+(\d+)\s+days?\s+before",
        context,
        re.I | re.S,
    )
    if not full_match and not partial_match:
        return None
    full_threshold = int(full_match.group(1)) if full_match else None
    partial_threshold = int(partial_match.group(1)) if partial_match else None

    if full_threshold is not None and days_before <= full_threshold and (
        partial_threshold is None or days_before > partial_threshold
    ):
        value = "full"
    elif partial_threshold is not None and days_before <= partial_threshold:
        value = "partial"
    else:
        value = "none"
    return {
        "rule": "refund_policy_threshold",
        "predicted_index": normalized_candidates[value],
        "predicted_value": candidates[normalized_candidates[value]],
        "reason": {
            "days_before": days_before,
            "full_threshold": full_threshold,
            "partial_threshold": partial_threshold,
        },
    }


RULES = [equal_weight_residual, percentage_range_midpoint, refund_policy_threshold]


def apply_rules(row, example):
    context = context_from_prompt((example or {}).get("prompt") or "")
    enriched = dict(row)
    if example:
        for field in ("prompt", "local_peer_arguments", "same_call_peer_arguments"):
            if field not in enriched and field in example:
                enriched[field] = example[field]
    for rule in RULES:
        decision = rule(enriched, context)
        if decision is not None:
            return decision
    return None


def recalculated_correct(row, predicted_index):
    target_index = row.get("target_index")
    try:
        return int(target_index) == int(predicted_index)
    except Exception:
        return values_equal(row.get("target"), row.get("predicted_value"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-jsonl", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    examples = {row_key(row): row for row in load_jsonl(args.examples_jsonl)}
    rows = []
    totals = Counter()
    for row in load_jsonl(args.selector_jsonl):
        totals["rows"] += 1
        totals[f"rows:{row.get('kind')}"] += 1
        totals["model_correct"] += int(bool(row.get("correct")))
        totals[f"model_correct:{row.get('kind')}"] += int(bool(row.get("correct")))
        out = dict(row)
        out["model_predicted_index"] = row.get("predicted_index")
        out["model_predicted_value"] = row.get("predicted_value")
        out["model_correct"] = bool(row.get("correct"))
        decision = apply_rules(row, examples.get(row_key(row)))
        if decision is not None:
            totals["rules_applied"] += 1
            totals[f"rules_applied:{decision['rule']}"] += 1
            totals[f"rules_applied_kind:{row.get('kind')}"] += 1
            out["predicted_index"] = decision["predicted_index"]
            out["predicted_value"] = decision["predicted_value"]
            out["correct"] = recalculated_correct(row, decision["predicted_index"])
            out["derived_rule_sidecar"] = {
                "applied": True,
                "rule": decision["rule"],
                "reason": decision.get("reason") or {},
            }
            if out["predicted_index"] != row.get("predicted_index"):
                totals["rules_changed_prediction"] += 1
                totals[f"rules_changed_prediction:{decision['rule']}"] += 1
            if bool(row.get("correct")) != bool(out["correct"]):
                totals["correctness_changed"] += 1
        else:
            out["derived_rule_sidecar"] = {"applied": False}
        totals["final_correct"] += int(bool(out.get("correct")))
        totals[f"final_correct:{row.get('kind')}"] += int(bool(out.get("correct")))
        rows.append(out)

    write_jsonl(args.out_jsonl, rows)
    summary = {
        "selector_jsonl": str(args.selector_jsonl),
        "examples_jsonl": str(args.examples_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "totals": dict(totals),
        "model_accuracy": totals["model_correct"] / totals["rows"] if totals["rows"] else 0.0,
        "final_accuracy": totals["final_correct"] / totals["rows"] if totals["rows"] else 0.0,
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
