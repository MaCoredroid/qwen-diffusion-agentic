#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def key(row):
    return (
        row.get("id"),
        row.get("kind"),
        row.get("tool_call_index"),
        row.get("json_key"),
        json.dumps(row.get("candidate_values") or [], ensure_ascii=False),
        json.dumps(row.get("target"), ensure_ascii=False),
    )


def brief(row):
    return {
        "id": row.get("id"),
        "kind": row.get("kind"),
        "tool_call_index": row.get("tool_call_index"),
        "json_key": row.get("json_key"),
        "target": row.get("target"),
        "predicted_value": row.get("predicted_value"),
        "candidate_values": row.get("candidate_values") or [],
        "target_margin": row.get("target_margin"),
        "candidate_count": row.get("candidate_count"),
    }


def accuracy(rows, kind=None):
    selected = [row for row in rows if kind is None or row.get("kind") == kind]
    if not selected:
        return 0, 0, 0.0
    correct = sum(1 for row in selected if row.get("correct"))
    return correct, len(selected), correct / len(selected)


def table_row(label, rows):
    total = accuracy(rows)
    tool = accuracy(rows, "tool_name")
    args = accuracy(rows, "argument_value")
    return (
        f"| {label} | {total[0]}/{total[1]} ({total[2]:.1%}) | "
        f"{tool[0]}/{tool[1]} ({tool[2]:.1%}) | "
        f"{args[0]}/{args[1]} ({args[2]:.1%}) |"
    )


def md_list(title, items, before_rows, after_rows):
    lines = [f"## {title}", ""]
    if not items:
        lines.extend(["- none", ""])
        return lines
    for item_key in items:
        before = before_rows[item_key]
        after = after_rows[item_key]
        lines.extend(
            [
                f"- id: `{after.get('id')}`",
                f"  - kind: `{after.get('kind')}`, call: `{after.get('tool_call_index')}`, key: `{after.get('json_key')}`",
                f"  - target: `{after.get('target')}`",
                f"  - before predicted: `{before.get('predicted_value')}`; after predicted: `{after.get('predicted_value')}`",
                f"  - margins before/after: `{before.get('target_margin')}` / `{after.get('target_margin')}`",
                f"  - candidates: `{after.get('candidate_values')}`",
            ]
        )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-jsonl", type=Path, required=True)
    parser.add_argument("--after-jsonl", type=Path, required=True)
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    before_rows_list = list(load_jsonl(args.before_jsonl))
    after_rows_list = list(load_jsonl(args.after_jsonl))
    before_rows = {key(row): row for row in before_rows_list}
    after_rows = {key(row): row for row in after_rows_list}
    shared = sorted(set(before_rows) & set(after_rows))

    improved = [
        item_key
        for item_key in shared
        if not before_rows[item_key].get("correct") and after_rows[item_key].get("correct")
    ]
    regressed = [
        item_key
        for item_key in shared
        if before_rows[item_key].get("correct") and not after_rows[item_key].get("correct")
    ]
    remaining = [item_key for item_key in shared if not after_rows[item_key].get("correct")]

    counts = Counter()
    for item_key in improved:
        counts[f"improved:{after_rows[item_key].get('kind')}"] += 1
    for item_key in regressed:
        counts[f"regressed:{after_rows[item_key].get('kind')}"] += 1
    for item_key in remaining:
        counts[f"remaining:{after_rows[item_key].get('kind')}"] += 1

    lines = [
        "# Candidate Ranking Delta",
        "",
        f"Before: `{args.before_jsonl}`",
        "",
        f"After: `{args.after_jsonl}`",
        "",
        "## Accuracy",
        "",
        "| Run | Overall | Tool names | Argument values |",
        "| --- | ---: | ---: | ---: |",
        table_row(args.before_label, before_rows_list),
        table_row(args.after_label, after_rows_list),
        "",
        "## Delta Counts",
        "",
        f"- shared examples: `{len(shared)}`",
        f"- improved examples: `{len(improved)}`",
        f"- regressed examples: `{len(regressed)}`",
        f"- remaining after-run failures: `{len(remaining)}`",
        f"- by kind: `{dict(sorted(counts.items()))}`",
        "",
    ]
    lines.extend(md_list("Improved", improved, before_rows, after_rows))
    lines.extend(md_list("Regressed", regressed, before_rows, after_rows))
    lines.extend(md_list("Remaining Failures", remaining, before_rows, after_rows))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    out_json = args.out_json or args.out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "before_jsonl": str(args.before_jsonl),
                "after_jsonl": str(args.after_jsonl),
                "shared": len(shared),
                "improved": [brief(after_rows[item_key]) for item_key in improved],
                "regressed": [brief(after_rows[item_key]) for item_key in regressed],
                "remaining_failures": [brief(after_rows[item_key]) for item_key in remaining],
                "counts": dict(sorted(counts.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
