#!/usr/bin/env python3
"""Shared per-group GRPO signal-quality metrics (engine + AR use IDENTICAL code).

Given a group of N same-prompt samples (each a list of generated token ids +
decoded assistant text), compute:
  * DIVERSITY: unique-output fraction (raw token-id sequences) and
    unique-argument-set fraction (canonical name+coerced-args), plus the
    GRPO-relevant unique *valid* arg-set fraction (distinct valid rollouts / N).
  * pass@1 (per-sample exact rate) and pass@N (any sample exact in the group).
  * valid fraction.

All scoring goes through the project's audited scorer (eval_toolcall_jsonl):
score_tool_calls (exact_arguments/valid) + normalize_call_for_compare (the same
coercion the exact-args comparison uses) for a canonical arg-set key.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/mark/qwen_diffusion/scripts")))
from eval_toolcall_jsonl import (  # noqa: E402
    extract_tool_calls,
    normalize_call_for_compare,
    score_tool_calls,
    tool_schema_by_name,
)


def argset_key(text, tools):
    """Canonical (name, coerced-args) key for a sample's tool call(s).

    Uses the SAME normalize_call_for_compare the exact-args scorer uses, so two
    samples share a key iff the exact-args comparison would treat them as the
    same call. Samples with no parseable call collapse to 'NOCALL'."""
    calls, _invalid = extract_tool_calls(text)
    if not calls:
        return "NOCALL"
    schemas = tool_schema_by_name(tools)
    norm = [normalize_call_for_compare(c, schemas) for c in calls]
    return json.dumps(norm, sort_keys=True, ensure_ascii=False)


def score_sample(text, tools, gold_block):
    sc = score_tool_calls(text, tools, gold_block)
    return {
        "exact": bool(sc.get("exact_arguments")),
        "valid": bool(sc.get("valid_tool_call")),
        "argset": argset_key(text, tools),
    }


def group_metrics(samples):
    """samples: list of dicts each with keys: token_ids (list[int]), text (str),
    exact (bool), valid (bool), argset (str), finish_reason, n_tok, seed."""
    N = len(samples)
    out_keys = [tuple(int(t) for t in s["token_ids"]) for s in samples]
    uniq_out = len(set(out_keys))
    argsets = [s["argset"] for s in samples]
    uniq_argset = len(set(argsets))
    valid_argsets = {s["argset"] for s in samples if s["valid"] and s["argset"] != "NOCALL"}
    n_valid = sum(1 for s in samples if s["valid"])
    n_exact = sum(1 for s in samples if s["exact"])
    return {
        "N": N,
        "n_valid": n_valid,
        "valid_frac": round(n_valid / N, 4),
        "unique_output_count": uniq_out,
        "unique_output_frac": round(uniq_out / N, 4),
        "unique_argset_count": uniq_argset,
        "unique_argset_frac": round(uniq_argset / N, 4),
        "unique_valid_argset_count": len(valid_argsets),
        "unique_valid_argset_frac": round(len(valid_argsets) / N, 4),
        "n_exact": n_exact,
        "pass1": round(n_exact / N, 4),
        "passN": 1 if n_exact > 0 else 0,
    }
