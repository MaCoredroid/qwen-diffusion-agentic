#!/usr/bin/env python3
"""Build Run 1 copy-grounded native FLARE training mix."""

from __future__ import annotations

import argparse
import copy
import json
import random
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_flare_agentic_mix_v1 import eval_row_to_instance, fingerprint, load_instances  # noqa: E402
from build_qwen35_planner_selector_retention_mix import (  # noqa: E402
    filter_eval_overlaps,
    resolve_chat_template,
    strip_training_metadata,
    summarize_audit,
    token_stats,
    write_jsonl,
)
from convert_toolcall_cases_to_qwen_native import convert_eval_row, native_instruction_text  # noqa: E402
from eval_toolcall_jsonl import qwen_native_parameter_value, qwen_native_tool_call_text  # noqa: E402


DEFAULT_MODEL = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_OUT_DIR = ROOT / "data/flare_redesign_run1_copy_retention_mix"
DEFAULT_RETENTION = ROOT / "data/flare_stage1_ab_pilot_train/train_agentic_mix.json"
DEFAULT_NATIVE_POOL = ROOT / "data/flare_agentic_mix_v2_native/train_agentic_mix.json"
DEFAULT_EVAL_EXCLUDES = [
    ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl",
    ROOT / "data/toolcall_eval/public_train_multicall_gold_cases.jsonl",
    ROOT / "data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl",
]
DEFAULT_POOL_JSONL = [
    ROOT / "data/toolcall_eval/synthetic_onecall_train_smoke.jsonl",
    ROOT / "data/toolcall_eval/synthetic_onecall_teacher_probe.jsonl",
    ROOT / "data/toolcall_eval/synthetic_toolresult_smoke.jsonl",
    ROOT / "data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl",
    ROOT / "data/toolcall_eval/public_train_multicall_gold_cases.jsonl",
    ROOT / "data/toolcall_eval/public_train_multicall_no_public_smoke_cases.jsonl",
    ROOT / "runs/planner_decomposition/heldout_seed_multicall_policy_targets.jsonl",
]


COPY_SYSTEM = (
    "You are a tool-call formatter. Return exactly one Qwen-native <tool_call> block "
    "using <function=...> and <parameter=...> tags with no prose."
)


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "read_project_file",
        "description": "Read an exact project file for a tenant.",
        "params": [
            ("tenant_id", "string", "tenant"),
            ("file_path", "string", "path"),
            ("revision_id", "string", "revision"),
        ],
    },
    {
        "name": "lookup_device_event",
        "description": "Look up an event from a device telemetry stream.",
        "params": [
            ("device_id", "string", "device"),
            ("event_id", "string", "event"),
            ("timestamp", "string", "timestamp"),
        ],
    },
    {
        "name": "schedule_field_visit",
        "description": "Schedule a field visit at a known site.",
        "params": [
            ("site_id", "string", "site"),
            ("visit_date", "string", "date"),
            ("technician_id", "string", "person"),
            ("priority", "string", "priority"),
        ],
    },
    {
        "name": "submit_invoice_packet",
        "description": "Submit an invoice packet for processing.",
        "params": [
            ("invoice_id", "string", "invoice"),
            ("customer_id", "string", "customer"),
            ("amount_cents", "integer", "amount"),
            ("attachment_path", "string", "path"),
        ],
    },
    {
        "name": "start_training_job",
        "description": "Start a training job from a manifest.",
        "params": [
            ("job_id", "string", "job"),
            ("model_ref", "string", "model_ref"),
            ("manifest_path", "string", "path"),
            ("dataset_sha", "string", "sha"),
        ],
    },
    {
        "name": "open_support_case",
        "description": "Open a support case with exact routing values.",
        "params": [
            ("case_id", "string", "case"),
            ("account_id", "string", "account"),
            ("severity", "string", "severity"),
            ("contact_email", "string", "email"),
        ],
    },
    {
        "name": "query_metric_window",
        "description": "Query an operational metric over an exact window.",
        "params": [
            ("metric_name", "string", "metric"),
            ("region", "string", "region"),
            ("window_start", "string", "timestamp"),
            ("window_end", "string", "timestamp"),
        ],
    },
    {
        "name": "apply_json_patch",
        "description": "Apply a JSON patch to an object.",
        "params": [
            ("object_id", "string", "object"),
            ("patch_path", "string", "json_pointer"),
            ("patch_value", "object", "json_object"),
        ],
    },
    {
        "name": "reserve_inventory",
        "description": "Reserve SKUs for an order.",
        "params": [
            ("order_id", "string", "order"),
            ("sku_list", "array", "sku_list"),
            ("warehouse_id", "string", "warehouse"),
        ],
    },
    {
        "name": "register_webhook",
        "description": "Register a webhook endpoint.",
        "params": [
            ("webhook_id", "string", "webhook"),
            ("callback_url", "string", "url"),
            ("secret_ref", "string", "secret"),
            ("events", "array", "event_list"),
        ],
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def token(size: int, rng: random.Random) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(size))


def slug(rng: random.Random) -> str:
    left = rng.choice(["alpha", "bravo", "cedar", "delta", "ember", "forge", "helix", "ion"])
    right = rng.choice(["ledger", "sensor", "route", "packet", "matrix", "beacon", "vault", "trace"])
    return f"{left}-{right}-{rng.randrange(10, 999)}"


def value_for_kind(kind: str, rng: random.Random, idx: int) -> Any:
    if kind == "tenant":
        return f"tenant_{token(6, rng)}"
    if kind == "path":
        return f"/srv/flare/{slug(rng)}/batch_{idx % 97:03d}/{token(8, rng).lower()}.json"
    if kind == "revision":
        return f"rev-{token(10, rng).lower()}"
    if kind == "device":
        return f"dev-{token(4, rng)}-{idx % 1000:03d}"
    if kind == "event":
        return f"evt_{token(12, rng).lower()}"
    if kind == "timestamp":
        day = 1 + (idx % 27)
        hour = rng.randrange(0, 24)
        minute = rng.randrange(0, 60)
        return f"2026-07-{day:02d}T{hour:02d}:{minute:02d}:00Z"
    if kind == "site":
        return f"SITE-{rng.randrange(1000, 9999)}-{token(3, rng)}"
    if kind == "date":
        return f"2026-08-{1 + idx % 28:02d}"
    if kind == "person":
        return f"tech_{token(5, rng).lower()}"
    if kind == "priority":
        return rng.choice(["low", "normal", "high", "urgent"])
    if kind == "invoice":
        return f"INV-{20260000 + idx}-{token(4, rng)}"
    if kind == "customer":
        return f"cust_{token(9, rng).lower()}"
    if kind == "amount":
        return rng.randrange(2500, 999999)
    if kind == "job":
        return f"job_{idx:04d}_{token(6, rng).lower()}"
    if kind == "model_ref":
        return f"qwen35-fastdllm/run1-{rng.randrange(100, 999)}"
    if kind == "sha":
        return token(16, rng).lower()
    if kind == "case":
        return f"CASE-{rng.randrange(100000, 999999)}"
    if kind == "account":
        return f"acct_{token(8, rng).lower()}"
    if kind == "severity":
        return rng.choice(["sev1", "sev2", "sev3"])
    if kind == "email":
        return f"{slug(rng).replace('-', '.')}.{idx}@example.net"
    if kind == "metric":
        return rng.choice(["agent.decode.latency_ms", "tool.value.exact_rate", "gdn.cache.hit_rate", "queue.depth"])
    if kind == "region":
        return rng.choice(["us-west-2", "us-east-1", "eu-central-1", "ap-southeast-2"])
    if kind == "object":
        return f"obj_{token(10, rng).lower()}"
    if kind == "json_pointer":
        return rng.choice(["/routing/primary", "/limits/max_tokens", "/metadata/source_id", "/flags/parallel"])
    if kind == "json_object":
        return {"source_id": f"src_{token(6, rng).lower()}", "enabled": bool(idx % 2), "rank": idx % 17}
    if kind == "order":
        return f"ORD-{token(5, rng)}-{idx:05d}"
    if kind == "sku_list":
        return [f"SKU-{token(5, rng)}" for _ in range(2 + idx % 3)]
    if kind == "warehouse":
        return f"WH-{rng.randrange(10, 99)}-{rng.choice(['A', 'B', 'C'])}"
    if kind == "webhook":
        return f"wh_{token(12, rng).lower()}"
    if kind == "url":
        return f"https://hooks.example.net/{slug(rng)}/{token(6, rng).lower()}"
    if kind == "secret":
        return f"secret/projects/{slug(rng)}/{token(8, rng).lower()}"
    if kind == "event_list":
        return rng.sample(["job.started", "job.finished", "case.created", "invoice.paid"], k=2)
    raise ValueError(f"unknown value kind={kind!r}")


def schema_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    required = []
    for name, schema_type, _ in spec["params"]:
        required.append(name)
        if schema_type == "array":
            properties[name] = {"type": "array", "items": {"type": "string"}}
        elif schema_type == "object":
            properties[name] = {"type": "object"}
        else:
            properties[name] = {"type": schema_type}
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def copy_instance(idx: int, rng: random.Random) -> dict[str, Any]:
    spec = rng.choice(TOOL_SPECS)
    arguments = {
        name: value_for_kind(kind, rng, idx)
        for name, _, kind in spec["params"]
    }
    assistant = qwen_native_tool_call_text([{"name": spec["name"], "arguments": arguments}])
    value_lines = []
    copy_spans = []
    for name, value in arguments.items():
        rendered = qwen_native_parameter_value(value)
        value_lines.append(f"- {name}: {rendered}")
        copy_spans.append(
            {
                "function": spec["name"],
                "parameter": name,
                "value_text": rendered,
                "context_occurrences": 1,
                "conditional_entropy_proxy": "C0_verbatim_prompt_copy",
            }
        )
    rng.shuffle(value_lines)
    user = (
        "Use the context block below. Copy every argument value verbatim into the native tool call. "
        "Do not normalize dates, paths, IDs, JSON objects, arrays, numbers, or casing.\n\n"
        f"Function to call: {spec['name']}\n"
        "Context values:\n"
        + "\n".join(value_lines)
    )
    return {
        "system": COPY_SYSTEM,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "tools": [schema_for_spec(spec)],
        "source": "redesign_run1_copy_synth",
        "copy_spans": copy_spans,
    }


def native_pool_instances(path: Path) -> list[dict[str, Any]]:
    instances = []
    if not path.exists():
        return instances
    for instance in load_instances(path):
        source = instance.get("source") or ""
        if source.startswith("general_retention_"):
            continue
        cloned = copy.deepcopy(instance)
        cloned["source"] = "prior_native_toolcall_pool:" + (source or path.stem)
        instances.append(cloned)
    return instances


def eval_pool_instances(paths: list[Path]) -> list[dict[str, Any]]:
    instances = []
    for path in paths:
        for row_idx, row in enumerate(load_jsonl(path)):
            try:
                native_row, _ = convert_eval_row(row)
                instance = eval_row_to_instance(native_row)
            except Exception:
                continue
            instance["source"] = f"eval_pool:{path.name}:{row_idx}"
            if "system" in instance:
                instance["system"] = native_instruction_text(instance["system"])
            instances.append(instance)
    return instances


def repeat_to_count(items: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    if target <= 0 or not items:
        return []
    out = []
    cycle = [copy.deepcopy(item) for item in items]
    cursor = 0
    while len(out) < target:
        if cursor % len(cycle) == 0:
            rng.shuffle(cycle)
        out.append(copy.deepcopy(cycle[cursor % len(cycle)]))
        cursor += 1
    return out


def row_for(instance: dict[str, Any], source_dataset: str, source_index: int, repeat: int = 0) -> dict[str, Any]:
    return {
        "instance": instance,
        "dataset_dir": source_dataset,
        "source_dataset": source_dataset,
        "source_index": source_index,
        "repeat": repeat,
        "source": instance.get("source") or source_dataset,
    }


def audit_candidate_rows(rows: list[dict[str, Any]], tokenizer, chat_template, args):
    accepted = []
    rejected = []
    for row in rows:
        stats = token_stats(tokenizer, chat_template, row["instance"], args.block_size, args.truncation_side)
        audit = {
            "source_dataset": row["source_dataset"],
            "dataset_dir": row["dataset_dir"],
            "source_index": row["source_index"],
            "repeat": row["repeat"],
            "source": row["source"],
            "tool_count": len(row["instance"].get("tools") or []),
            "copy_span_count": len(row["instance"].get("copy_spans") or []),
            **stats,
        }
        if stats["kept_labels"] < args.min_labels:
            rejected.append({**audit, "reject_reason": "too_few_labels"})
            continue
        if args.require_full_labels and not stats["full_labels_kept"]:
            rejected.append({**audit, "reject_reason": "partial_labels_after_truncation"})
            continue
        accepted.append({"instance": row["instance"], "audit": audit})
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conversation-template", default="fast_dllm_v2_native")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--native-pool", type=Path, default=DEFAULT_NATIVE_POOL)
    parser.add_argument("--pool-jsonl", type=Path, nargs="*", default=DEFAULT_POOL_JSONL)
    parser.add_argument("--exclude-eval-jsonl", type=Path, nargs="*", default=DEFAULT_EVAL_EXCLUDES)
    parser.add_argument("--copy-count", type=int, default=2048)
    parser.add_argument("--pool-target", type=int, default=512)
    parser.add_argument("--retention-target", type=int, default=-1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--require-full-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=71101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.conversation_template in {"tokenizer", "tokenizer_chat_template", "native"}:
        chat_template = tokenizer.chat_template
        if not chat_template:
            raise ValueError(f"{args.model} does not define tokenizer.chat_template")
    else:
        chat_template = resolve_chat_template(args.conversation_template)

    copy_instances = [copy_instance(idx, rng) for idx in range(args.copy_count)]
    pool_candidates = native_pool_instances(args.native_pool) + eval_pool_instances(args.pool_jsonl)
    pool_by_fp = {}
    for item in pool_candidates:
        pool_by_fp.setdefault(fingerprint(item), item)
    pool_candidates = list(pool_by_fp.values())
    rng.shuffle(pool_candidates)
    pool_instances = pool_candidates[: max(0, args.pool_target)]

    tool_rows = [
        row_for(instance, "redesign_run1_copy_synth", idx)
        for idx, instance in enumerate(copy_instances)
    ]
    tool_rows.extend(
        row_for(instance, "redesign_run1_public_toolcall_pool", idx)
        for idx, instance in enumerate(pool_instances)
    )

    retention_raw = load_instances(args.retention)
    retention_target = args.retention_target if args.retention_target >= 0 else len(tool_rows)
    retention_instances = repeat_to_count(retention_raw, retention_target, rng)
    retention_rows = [
        row_for(instance, "redesign_run1_gsm8k_mbpp_retention", idx, repeat=idx // max(1, len(retention_raw)))
        for idx, instance in enumerate(retention_instances)
    ]

    rows = tool_rows + retention_rows
    rows, eval_overlap_removed = filter_eval_overlaps(rows, args.exclude_eval_jsonl)
    accepted, rejected = audit_candidate_rows(rows, tokenizer, chat_template, args)
    order = list(range(len(accepted)))
    rng.shuffle(order)
    accepted = [accepted[idx] for idx in order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    audit_path = args.out_dir / "train_agentic_mix.audit.jsonl"
    rejected_path = args.out_dir / "train_agentic_mix.rejected.jsonl"
    span_path = args.out_dir / "copy_spans.jsonl"
    overlap_removed_path = args.out_dir / "eval_overlap_removed.jsonl"

    train_instances = [strip_training_metadata(item["instance"]) for item in accepted]
    write_json(train_path, {"type": "conversation", "instances": train_instances})
    write_jsonl(audit_path, [item["audit"] for item in accepted])
    write_jsonl(rejected_path, rejected)
    write_jsonl(overlap_removed_path, eval_overlap_removed)

    span_rows = []
    for item in accepted:
        spans = item["instance"].get("copy_spans") or []
        if not spans:
            continue
        for span in spans:
            span_rows.append(
                {
                    "source": item["audit"]["source"],
                    "source_index": item["audit"]["source_index"],
                    **span,
                }
            )
    write_jsonl(span_path, span_rows)

    audits = [item["audit"] for item in accepted]
    source_counts = Counter(row["source_dataset"] for row in audits)
    detail_counts = Counter(row.get("source") or "unknown" for row in audits)
    manifest = {
        "train_path": str(train_path),
        "audit_path": str(audit_path),
        "rejected_path": str(rejected_path),
        "copy_spans_path": str(span_path),
        "eval_overlap_removed_path": str(overlap_removed_path),
        "count": len(accepted),
        "candidate_count": len(rows) + len(eval_overlap_removed),
        "candidate_after_eval_filter_count": len(rows),
        "rejected_count": len(rejected),
        "eval_overlap_removed_count": len(eval_overlap_removed),
        "source_counts": dict(sorted(source_counts.items())),
        "source_detail_counts_top50": dict(detail_counts.most_common(50)),
        "copy_span_count": len(span_rows),
        "copy_count_requested": args.copy_count,
        "pool_target": args.pool_target,
        "pool_candidate_unique_count": len(pool_candidates),
        "pool_selected_count": len(pool_instances),
        "retention_target": retention_target,
        "retention_unique_count": len({fingerprint(item) for item in retention_raw}),
        "retention_path": str(args.retention),
        "native_pool_path": str(args.native_pool),
        "pool_jsonl": [str(path) for path in args.pool_jsonl],
        "exclude_eval_jsonl": [str(path) for path in args.exclude_eval_jsonl],
        "native_format": "qwen_native_function_parameter",
        "conversation_template": args.conversation_template,
        "span_label_contract": (
            "copy_spans rows mark C0 verbatim values; trainer consumes them through native "
            "<parameter=...> argument-span tokens plus VALUE_SPAN_TOKEN_IDS derived from this dataset."
        ),
        "schedule_contract": {
            "copy_spans": "wide masked via FASTDLLM_VALUE_SPAN_MASK_PROB=1.0",
            "non_copy": "near-AR/high-entropy blocks via FASTDLLM_FLARE_ADAPTIVE_COPY_SCHEDULE=1",
            "noise": "BD3 clipped mask rate U[0.3,0.8] on copy/value blocks",
        },
        "block_size": args.block_size,
        "truncation_side": args.truncation_side,
        "min_labels": args.min_labels,
        "require_full_labels": args.require_full_labels,
        "audit_summary": summarize_audit(audits),
        "rejected_summary": summarize_audit(rejected),
        "seed": args.seed,
    }
    manifest_path = args.out_dir / "train_agentic_mix.manifest"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
