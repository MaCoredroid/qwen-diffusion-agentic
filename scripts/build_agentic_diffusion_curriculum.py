#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT_DIR = ROOT / "data/qwen35_9b_diffusion_curriculum"
DEFAULT_SYSTEM = "You are a helpful assistant."


def load_conversation_instances(path):
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    instances = obj.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"{path} does not contain an instances list")
    return instances


def clean_instance(instance):
    messages = []
    for message in instance.get("messages") or []:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant", "tool"} and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return None

    cleaned = {"messages": messages}
    system = str(instance.get("system") or "").strip()
    if system:
        cleaned["system"] = system
    tools = instance.get("tools") or []
    if tools:
        cleaned["tools"] = tools
    return cleaned


def load_jsonl(path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_repo_prompt(task):
    files = task.get("files") or {}
    file_blocks = []
    for name, content in sorted(files.items()):
        file_blocks.append(f"### {name}\n```text\n{str(content).rstrip()}\n```")
    return "\n\n".join(
        [
            "Fix this repository task. Return a unified diff only.",
            f"Task: {task.get('prompt') or task.get('task')}",
            f"Test command: {task.get('test_command')}",
            "Expected editable files: " + ", ".join(task.get("expected_files") or []),
            "Repository files:",
            "\n\n".join(file_blocks),
        ]
    ).strip()


def repo_edit_instances(tasks_path, results_path, cap):
    if cap <= 0:
        return []
    tasks = {record.get("id"): record for record in load_jsonl(tasks_path)}
    instances = []
    for result in load_jsonl(results_path):
        if len(instances) >= cap:
            break
        task = tasks.get(result.get("id"))
        diff = str(result.get("diff") or "").strip()
        if not task or not diff:
            continue
        if not result.get("final_tests_passed"):
            continue
        if result.get("changed_unexpected_files"):
            continue
        instances.append(
            {
                "system": "You are a coding assistant.",
                "messages": [
                    {"role": "user", "content": format_repo_prompt(task)},
                    {"role": "assistant", "content": diff},
                ],
            }
        )
    return instances


def add_source(records, source, instances, cap, rng):
    cleaned = []
    for instance in instances:
        item = clean_instance(instance)
        if item is not None:
            cleaned.append(item)
    rng.shuffle(cleaned)
    if cap is not None and cap >= 0:
        cleaned = cleaned[:cap]
    for item in cleaned:
        records.append({"source": source, "instance": item})


def dedupe(records):
    seen = set()
    out = []
    for record in records:
        key = json.dumps(record["instance"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--public-toolcall", type=Path, default=ROOT / "data/fastdllm_toolcall_train/train_toolcall.json")
    parser.add_argument("--synthetic-onecall", type=Path, default=ROOT / "data/synthetic_onecall_train/train_synthetic_onecall.json")
    parser.add_argument(
        "--synthetic-toolresult",
        type=Path,
        default=ROOT / "data/synthetic_toolresult_train/train_synthetic_toolresult.json",
    )
    parser.add_argument("--repo-edit-tasks", type=Path, default=ROOT / "data/repo_edit_eval/tiny_repo_edit_5.jsonl")
    parser.add_argument(
        "--repo-edit-results",
        type=Path,
        default=ROOT / "data/repo_edit_eval/tiny_repo_edit_qwen_code_q36_8k_requiredall_512_tools12_5.jsonl",
    )
    parser.add_argument("--public-toolcall-cap", type=int, default=96)
    parser.add_argument("--synthetic-onecall-cap", type=int, default=192)
    parser.add_argument("--synthetic-toolresult-cap", type=int, default=10)
    parser.add_argument("--repo-edit-cap", type=int, default=5)
    parser.add_argument("--total-cap", type=int, default=0, help="0 means no total cap after source caps.")
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = []

    add_source(
        records,
        "public_toolcall",
        load_conversation_instances(args.public_toolcall),
        args.public_toolcall_cap,
        rng,
    )
    add_source(
        records,
        "synthetic_onecall",
        load_conversation_instances(args.synthetic_onecall),
        args.synthetic_onecall_cap,
        rng,
    )
    add_source(
        records,
        "synthetic_toolresult",
        load_conversation_instances(args.synthetic_toolresult),
        args.synthetic_toolresult_cap,
        rng,
    )
    add_source(
        records,
        "repo_edit_qwen36_diff",
        repo_edit_instances(args.repo_edit_tasks, args.repo_edit_results, args.repo_edit_cap),
        args.repo_edit_cap,
        rng,
    )

    records = dedupe(records)
    rng.shuffle(records)
    if args.total_cap and args.total_cap > 0:
        records = records[: args.total_cap]

    instances = [record["instance"] for record in records]
    source_counts = Counter(record["source"] for record in records)
    if not instances:
        raise SystemExit("No curriculum instances were produced.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_agentic_mix.json"
    manifest_path = args.out_dir / "train_agentic_mix.manifest"

    train_path.write_text(
        json.dumps({"type": "conversation", "instances": instances}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "train_path": str(train_path),
        "count": len(instances),
        "source_counts": dict(sorted(source_counts.items())),
        "seed": args.seed,
        "source_caps": {
            "public_toolcall": args.public_toolcall_cap,
            "synthetic_onecall": args.synthetic_onecall_cap,
            "synthetic_toolresult": args.synthetic_toolresult_cap,
            "repo_edit_qwen36_diff": args.repo_edit_cap,
            "total": args.total_cap,
        },
        "eval_slices": {
            "synthetic_onecall": str(ROOT / "data/toolcall_eval/synthetic_onecall_smoke.jsonl"),
            "public_onecall": str(ROOT / "data/toolcall_eval/public_onecall_hermes_smoke.jsonl"),
            "public_multicall": str(ROOT / "data/toolcall_eval/public_multicall_hermes_smoke.jsonl"),
            "synthetic_toolresult": str(ROOT / "data/toolcall_eval/synthetic_toolresult_smoke.jsonl"),
            "repo_edit": str(ROOT / "data/repo_edit_eval/tiny_repo_edit_5.jsonl"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
