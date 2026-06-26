#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_TRAIN_DIR = ROOT / "data/synthetic_onecall_train"
DEFAULT_EVAL = ROOT / "data/toolcall_eval/synthetic_onecall_smoke.jsonl"
DEFAULT_TRAIN_EVAL = ROOT / "data/toolcall_eval/synthetic_onecall_train_smoke.jsonl"
DEFAULT_SYSTEM = "You are a helpful assistant."


TOOL_SPECS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "city": ("string", ["Seattle", "Austin", "Boston", "Denver", "Miami"]),
            "unit": ("string", ["fahrenheit", "celsius"]),
        },
        "prompt": "Check the weather in {city}. Use {unit} units.",
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency into another.",
        "parameters": {
            "amount": ("number", [25, 70, 125, 240, 999]),
            "from_currency": ("string", ["USD", "EUR", "GBP", "JPY"]),
            "to_currency": ("string", ["EUR", "USD", "CAD", "AUD"]),
        },
        "prompt": "Convert {amount} {from_currency} into {to_currency}.",
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "title": ("string", ["demo review", "budget sync", "launch check", "design critique"]),
            "date": ("string", ["2026-07-01", "2026-07-08", "2026-07-15", "2026-07-22"]),
            "time": ("string", ["09:00", "11:30", "14:00", "16:15"]),
        },
        "prompt": "Create a calendar event called {title} on {date} at {time}.",
    },
    {
        "name": "send_email",
        "description": "Send a short email.",
        "parameters": {
            "recipient": ("string", ["alex@example.com", "sam@example.com", "lee@example.com"]),
            "subject": ("string", ["status update", "meeting notes", "quick question"]),
            "body": ("string", ["I will send the draft today.", "The meeting moved to Friday.", "Can you review this?"]),
        },
        "prompt": "Email {recipient} with subject '{subject}' and body '{body}'",
    },
    {
        "name": "search_docs",
        "description": "Search internal documents.",
        "parameters": {
            "query": ("string", ["refund policy", "deployment checklist", "API rate limit", "onboarding guide"]),
            "max_results": ("integer", [3, 5, 8, 10]),
        },
        "prompt": "Search docs for '{query}' and return {max_results} results.",
    },
    {
        "name": "lookup_order",
        "description": "Look up an order by id.",
        "parameters": {
            "order_id": ("string", ["ORD-1007", "ORD-2042", "ORD-3310", "ORD-7788"]),
        },
        "prompt": "Look up order {order_id}.",
    },
    {
        "name": "create_support_ticket",
        "description": "Create a support ticket.",
        "parameters": {
            "customer_id": ("string", ["CUST-17", "CUST-42", "CUST-88", "CUST-105"]),
            "issue": ("string", ["login failure", "billing mismatch", "slow export", "missing invoice"]),
            "priority": ("string", ["low", "medium", "high"]),
        },
        "prompt": "Open a {priority} priority support ticket for {customer_id}: {issue}.",
    },
    {
        "name": "translate_text",
        "description": "Translate text into another language.",
        "parameters": {
            "text": ("string", ["good morning", "thank you", "shipment delayed", "see you tomorrow"]),
            "target_language": ("string", ["Spanish", "French", "German", "Japanese"]),
        },
        "prompt": "Translate '{text}' into {target_language}.",
    },
    {
        "name": "set_timer",
        "description": "Set a timer.",
        "parameters": {
            "duration_minutes": ("integer", [5, 12, 20, 45]),
            "label": ("string", ["tea", "focus block", "laundry", "stretch break"]),
        },
        "prompt": "Set a {duration_minutes} minute timer named {label}.",
    },
    {
        "name": "calculate_tip",
        "description": "Calculate a restaurant tip.",
        "parameters": {
            "bill_amount": ("number", [32.5, 48.0, 76.25, 119.9]),
            "tip_percent": ("integer", [15, 18, 20, 22]),
        },
        "prompt": "Calculate a {tip_percent}% tip for a bill of {bill_amount}.",
    },
    {
        "name": "track_package",
        "description": "Track a shipping package.",
        "parameters": {
            "tracking_number": ("string", ["1Z999AA101", "9400111200", "TBA123456", "LX123456789US"]),
        },
        "prompt": "Track package {tracking_number}.",
    },
    {
        "name": "reserve_table",
        "description": "Reserve a restaurant table.",
        "parameters": {
            "restaurant": ("string", ["Northstar", "Cafe Luma", "Sushi Park", "The Corner"]),
            "party_size": ("integer", [2, 3, 4, 6]),
            "date": ("string", ["2026-07-03", "2026-07-10", "2026-07-17"]),
            "time": ("string", ["18:00", "19:30", "20:00"]),
        },
        "prompt": "Reserve a table at {restaurant} for {party_size} people on {date} at {time}.",
    },
]


def parameter_schema(spec):
    properties = {}
    required = []
    for name, (kind, values) in spec["parameters"].items():
        entry = {"type": "number" if kind == "number" else kind}
        if kind == "string" and len(values) <= 6:
            entry["enum"] = values
        properties[name] = entry
        required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def tool_def(spec):
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": parameter_schema(spec),
        },
    }


def sample_args(spec, rng, idx):
    args = {}
    for name, (_, values) in spec["parameters"].items():
        args[name] = values[(idx + rng.randrange(len(values))) % len(values)]
    return args


def assistant_text(name, args):
    payload = {"name": name, "arguments": args}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def make_example(idx, rng, distractor_count):
    spec = TOOL_SPECS[idx % len(TOOL_SPECS)]
    args = sample_args(spec, rng, idx)
    prompt = spec["prompt"].format(**args)

    distractors = [candidate for candidate in TOOL_SPECS if candidate["name"] != spec["name"]]
    rng.shuffle(distractors)
    tools = [tool_def(spec), *[tool_def(item) for item in distractors[:distractor_count]]]
    rng.shuffle(tools)

    instance = {
        "system": DEFAULT_SYSTEM,
        "tools": tools,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text(spec["name"], args)},
        ],
    }
    eval_case = {
        "source": "synthetic_onecall",
        "id": f"synthetic-onecall-{idx:05d}",
        "task": spec["name"],
        "category": "synthetic_tool_call",
        "tools": tools,
        "prompt_messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "gold_assistant": instance["messages"][-1]["content"],
        "gold_tool_names": [spec["name"]],
        "available_tool_names": sorted(tool["function"]["name"] for tool in tools),
    }
    return instance, eval_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-eval-out", type=Path, default=DEFAULT_TRAIN_EVAL)
    parser.add_argument("--num-train", type=int, default=192)
    parser.add_argument("--num-eval", type=int, default=48)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_eval_out.parent.mkdir(parents=True, exist_ok=True)

    train_instances = []
    train_eval_cases = []
    eval_cases = []
    for idx in range(args.num_train + args.num_eval):
        instance, eval_case = make_example(idx, rng, args.distractors)
        if idx < args.num_train:
            train_instances.append(instance)
            train_eval_cases.append(eval_case)
        else:
            eval_cases.append(eval_case)

    train_path = args.train_dir / "train_synthetic_onecall.json"
    train_path.write_text(
        json.dumps({"type": "conversation", "instances": train_instances}, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.eval_out.open("w", encoding="utf-8") as f:
        for case in eval_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    with args.train_eval_out.open("w", encoding="utf-8") as f:
        for case in train_eval_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    manifest = {
        "train_path": str(train_path),
        "train_eval_path": str(args.train_eval_out),
        "eval_path": str(args.eval_out),
        "num_train": len(train_instances),
        "num_train_eval": len(train_eval_cases),
        "num_eval": len(eval_cases),
        "distractors": args.distractors,
        "seed": args.seed,
        "tool_names": [spec["name"] for spec in TOOL_SPECS],
    }
    manifest_path = args.train_dir / "synthetic_onecall.manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
