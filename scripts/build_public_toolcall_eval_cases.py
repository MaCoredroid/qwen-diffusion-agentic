#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from build_fastdllm_toolcall_data import make_eval_case
from eval_toolcall_jsonl import extract_tool_calls


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_INPUT = ROOT / "data/toolcall_seed/qwen_toolcall_seed.jsonl"
DEFAULT_OUT = ROOT / "data/toolcall_eval/public_onecall_hermes_smoke.jsonl"


def source_allowed(source, sources):
    return not sources or source in sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-gold-calls", type=int, default=1)
    parser.add_argument("--sources", nargs="*", default=["hermes"])
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        "records_seen": 0,
        "records_written": 0,
        "skipped_no_eval_case": 0,
        "skipped_source": 0,
        "skipped_call_count": 0,
        "sources": {},
    }

    with args.input.open("r", encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            totals["records_seen"] += 1
            record = json.loads(line)
            source = record.get("source") or "unknown"
            if not source_allowed(source, set(args.sources)):
                totals["skipped_source"] += 1
                continue
            case = make_eval_case(record)
            if case is None:
                totals["skipped_no_eval_case"] += 1
                continue
            gold_calls, invalid = extract_tool_calls(case.get("gold_assistant") or "")
            if invalid or not gold_calls or len(gold_calls) > args.max_gold_calls:
                totals["skipped_call_count"] += 1
                continue
            case["gold_tool_calls"] = gold_calls
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
            totals["records_written"] += 1
            totals["sources"][source] = totals["sources"].get(source, 0) + 1
            if totals["records_written"] >= args.limit:
                break

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest = {
        "input": str(args.input),
        "output": str(args.out),
        "limit": args.limit,
        "max_gold_calls": args.max_gold_calls,
        "sources": args.sources,
        "totals": totals,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
