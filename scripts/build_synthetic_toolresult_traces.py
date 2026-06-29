#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT = ROOT / "data/toolcall_eval/synthetic_toolresult_smoke.jsonl"
DEFAULT_OPENAI_OUT = ROOT / "data/toolcall_eval/synthetic_toolresult_openai_smoke.jsonl"
DEFAULT_TRAIN_DIR = ROOT / "data/synthetic_toolresult_train"
DEFAULT_SYSTEM = "You are a helpful assistant."


def tool_def(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = {
    "lookup_order": tool_def(
        "lookup_order",
        "Look up order status and customer metadata.",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
    "create_support_ticket": tool_def(
        "create_support_ticket",
        "Create a support escalation ticket.",
        {
            "customer_id": {"type": "string"},
            "issue": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        ["customer_id", "issue", "priority"],
    ),
    "send_email": tool_def(
        "send_email",
        "Send a customer email.",
        {
            "recipient": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        ["recipient", "subject", "body"],
    ),
    "issue_refund": tool_def(
        "issue_refund",
        "Issue a refund for an order.",
        {
            "order_id": {"type": "string"},
            "amount": {"type": "number"},
            "reason": {"type": "string"},
        },
        ["order_id", "amount", "reason"],
    ),
    "search_docs": tool_def(
        "search_docs",
        "Search internal policy or runbook documents.",
        {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        ["query", "max_results"],
    ),
    "apply_coupon": tool_def(
        "apply_coupon",
        "Apply a coupon to a cart.",
        {
            "cart_id": {"type": "string"},
            "coupon_code": {"type": "string"},
        },
        ["cart_id", "coupon_code"],
    ),
    "get_inventory": tool_def(
        "get_inventory",
        "Check inventory for a SKU.",
        {
            "sku": {"type": "string"},
            "warehouse": {"type": "string"},
        },
        ["sku", "warehouse"],
    ),
    "reserve_inventory": tool_def(
        "reserve_inventory",
        "Reserve available inventory.",
        {
            "sku": {"type": "string"},
            "quantity": {"type": "integer"},
            "warehouse": {"type": "string"},
        },
        ["sku", "quantity", "warehouse"],
    ),
    "route_incident": tool_def(
        "route_incident",
        "Route an incident to the right response queue.",
        {
            "service": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        ["service", "severity"],
    ),
    "page_oncall": tool_def(
        "page_oncall",
        "Page the on-call engineer.",
        {
            "service": {"type": "string"},
            "urgency": {"type": "string", "enum": ["normal", "urgent"]},
        },
        ["service", "urgency"],
    ),
    "check_account": tool_def(
        "check_account",
        "Check account status and plan eligibility.",
        {"account_id": {"type": "string"}},
        ["account_id"],
    ),
    "update_subscription": tool_def(
        "update_subscription",
        "Update a subscription plan.",
        {
            "account_id": {"type": "string"},
            "plan": {"type": "string", "enum": ["basic", "pro", "enterprise"]},
        },
        ["account_id", "plan"],
    ),
    "schedule_callback": tool_def(
        "schedule_callback",
        "Schedule a customer callback.",
        {
            "customer_id": {"type": "string"},
            "date": {"type": "string"},
            "time": {"type": "string"},
        },
        ["customer_id", "date", "time"],
    ),
}


CASES = [
    {
        "id": "synthetic-toolresult-00001",
        "task": "delayed_order_escalation",
        "user": "Order ORD-2042 is late. Look it up, then escalate if it is more than five days late.",
        "first": ("lookup_order", {"order_id": "ORD-2042"}),
        "observation": {
            "order_id": "ORD-2042",
            "customer_id": "CUST-42",
            "days_late": 8,
            "status": "in_transit",
            "ticket_issue": "order ORD-2042 is 8 days late",
        },
        "gold": ("create_support_ticket", {"customer_id": "CUST-42", "issue": "order ORD-2042 is 8 days late", "priority": "high"}),
        "tools": ["lookup_order", "create_support_ticket", "send_email", "issue_refund"],
    },
    {
        "id": "synthetic-toolresult-00002",
        "task": "delivered_order_update",
        "user": "Check order ORD-7788. If it already arrived, email the customer a short delivery confirmation.",
        "first": ("lookup_order", {"order_id": "ORD-7788"}),
        "observation": {
            "order_id": "ORD-7788",
            "customer_id": "CUST-105",
            "email": "lee@example.com",
            "status": "delivered",
            "email_subject": "order ORD-7788 delivered",
            "email_body": "Order ORD-7788 has been delivered.",
        },
        "gold": ("send_email", {"recipient": "lee@example.com", "subject": "order ORD-7788 delivered", "body": "Order ORD-7788 has been delivered."}),
        "tools": ["lookup_order", "send_email", "create_support_ticket", "issue_refund"],
    },
    {
        "id": "synthetic-toolresult-00003",
        "task": "refund_after_policy_search",
        "user": "Search the refund policy for damaged deliveries, then issue a refund if the policy allows it.",
        "first": ("search_docs", {"query": "damaged delivery refund policy", "max_results": 3}),
        "observation": {
            "top_result": "Damaged deliveries are refundable within 30 days.",
            "order_id": "ORD-3310",
            "refund_amount": 48.5,
            "refund_reason": "damaged delivery within 30 days",
        },
        "gold": ("issue_refund", {"order_id": "ORD-3310", "amount": 48.5, "reason": "damaged delivery within 30 days"}),
        "tools": ["search_docs", "issue_refund", "send_email", "create_support_ticket"],
    },
    {
        "id": "synthetic-toolresult-00004",
        "task": "coupon_after_policy_search",
        "user": "Search the promo policy for CART-77. If the customer is eligible, apply the best coupon.",
        "first": ("search_docs", {"query": "CART-77 promo eligibility", "max_results": 3}),
        "observation": {"cart_id": "CART-77", "eligible": True, "best_coupon": "SAVE15", "expires": "2026-07-31"},
        "gold": ("apply_coupon", {"cart_id": "CART-77", "coupon_code": "SAVE15"}),
        "tools": ["search_docs", "apply_coupon", "send_email", "issue_refund"],
    },
    {
        "id": "synthetic-toolresult-00005",
        "task": "inventory_reservation",
        "user": "Check SKU-9 in warehouse west. If at least three units are available, reserve three.",
        "first": ("get_inventory", {"sku": "SKU-9", "warehouse": "west"}),
        "observation": {"sku": "SKU-9", "warehouse": "west", "available": 12},
        "gold": ("reserve_inventory", {"sku": "SKU-9", "quantity": 3, "warehouse": "west"}),
        "tools": ["get_inventory", "reserve_inventory", "create_support_ticket", "send_email"],
    },
    {
        "id": "synthetic-toolresult-00006",
        "task": "inventory_shortage_ticket",
        "user": "Check SKU-22 in warehouse east. If fewer than five are available, open a medium-priority supply ticket.",
        "first": ("get_inventory", {"sku": "SKU-22", "warehouse": "east"}),
        "observation": {
            "sku": "SKU-22",
            "warehouse": "east",
            "available": 1,
            "customer_id": "CUST-88",
            "ticket_issue": "SKU-22 inventory shortage in east warehouse",
        },
        "gold": ("create_support_ticket", {"customer_id": "CUST-88", "issue": "SKU-22 inventory shortage in east warehouse", "priority": "medium"}),
        "tools": ["get_inventory", "reserve_inventory", "create_support_ticket", "send_email"],
    },
    {
        "id": "synthetic-toolresult-00007",
        "task": "critical_incident_page",
        "user": "Search the payments runbook for the current error rate. If it is critical, page on-call.",
        "first": ("search_docs", {"query": "payments current error rate", "max_results": 2}),
        "observation": {"service": "payments", "error_rate_percent": 9.2, "threshold": "critical"},
        "gold": ("page_oncall", {"service": "payments", "urgency": "urgent"}),
        "tools": ["search_docs", "page_oncall", "route_incident", "send_email"],
    },
    {
        "id": "synthetic-toolresult-00008",
        "task": "noncritical_incident_route",
        "user": "Search the checkout runbook for the current error rate. If it is not critical, route the incident.",
        "first": ("search_docs", {"query": "checkout current error rate", "max_results": 2}),
        "observation": {
            "service": "checkout",
            "error_rate_percent": 1.1,
            "threshold": "warning",
            "route_severity": "medium",
        },
        "gold": ("route_incident", {"service": "checkout", "severity": "medium"}),
        "tools": ["search_docs", "page_oncall", "route_incident", "send_email"],
    },
    {
        "id": "synthetic-toolresult-00009",
        "task": "eligible_subscription_upgrade",
        "user": "Check account ACCT-500. If it is eligible for a paid upgrade, move it to pro.",
        "first": ("check_account", {"account_id": "ACCT-500"}),
        "observation": {"account_id": "ACCT-500", "customer_id": "CUST-17", "eligible_for_upgrade": True},
        "gold": ("update_subscription", {"account_id": "ACCT-500", "plan": "pro"}),
        "tools": ["check_account", "update_subscription", "schedule_callback", "send_email"],
    },
    {
        "id": "synthetic-toolresult-00010",
        "task": "ineligible_subscription_callback",
        "user": "Check account ACCT-920. If it cannot upgrade automatically, schedule a customer callback.",
        "first": ("check_account", {"account_id": "ACCT-920"}),
        "observation": {
            "account_id": "ACCT-920",
            "customer_id": "CUST-99",
            "eligible_for_upgrade": False,
            "reason": "payment verification needed",
            "callback_date": "2026-07-02",
            "callback_time": "10:00",
        },
        "gold": ("schedule_callback", {"customer_id": "CUST-99", "date": "2026-07-02", "time": "10:00"}),
        "tools": ["check_account", "update_subscription", "schedule_callback", "send_email"],
    },
]


def tool_call_text(name, arguments):
    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ": ")) + "\n</tool_call>"


def openai_tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def make_case(spec):
    first_name, first_args = spec["first"]
    gold_name, gold_args = spec["gold"]
    observation = json.dumps(spec["observation"], ensure_ascii=False, sort_keys=True)
    first_call_id = "call_" + spec["id"].replace("-", "_") + "_first"
    gold_call_id = "call_" + spec["id"].replace("-", "_") + "_next"
    tools = [TOOLS[name] for name in spec["tools"]]
    first_call = tool_call_text(first_name, first_args)
    gold_call = tool_call_text(gold_name, gold_args)
    first_openai_call = openai_tool_call(first_call_id, first_name, first_args)
    gold_openai_call = openai_tool_call(gold_call_id, gold_name, gold_args)
    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": spec["user"]},
        {"role": "assistant", "content": first_call},
        {
            "role": "user",
            "content": (
                f"Tool result for {first_name}: {observation}\n"
                "Continue from this observation. Use exact values from the tool result "
                "when present. Do not repeat completed tool calls."
            ),
        },
    ]
    train_messages = [
        {"role": "user", "content": spec["user"]},
        {"role": "assistant", "content": first_call},
        prompt_messages[-1],
        {"role": "assistant", "content": gold_call},
    ]
    prompt_messages_openai = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": spec["user"]},
        {"role": "assistant", "content": None, "tool_calls": [first_openai_call]},
        {
            "role": "tool",
            "tool_call_id": first_call_id,
            "name": first_name,
            "content": observation,
        },
    ]
    gold_assistant_openai = {"role": "assistant", "content": None, "tool_calls": [gold_openai_call]}
    completed_tool_calls = [
        {
            "name": first_name,
            "arguments": first_args,
            "tool_call_id": first_call_id,
            "result": spec["observation"],
        }
    ]
    eval_case = {
        "source": "synthetic_toolresult",
        "id": spec["id"],
        "task": spec["task"],
        "category": "synthetic_tool_result_trace",
        "tools": tools,
        "prompt_messages": prompt_messages,
        "prompt_messages_openai": prompt_messages_openai,
        "gold_assistant": gold_call,
        "gold_assistant_openai": gold_assistant_openai,
        "gold_tool_calls": [gold_openai_call],
        "gold_tool_names": [gold_name],
        "available_tool_names": sorted(spec["tools"]),
        "completed_tool_names": [first_name],
        "completed_tool_calls": completed_tool_calls,
        "teacher_instruction": (
            "Using the tool result above, return exactly the next required tool call. "
            "Use exact values from the tool result when present. Do not repeat earlier "
            "calls. Use only this format and no prose:\n"
            "<tool_call>\n"
            "{\"name\": \"tool_name\", \"arguments\": {}}\n"
            "</tool_call>"
        ),
    }
    openai_eval_case = {
        **eval_case,
        "prompt_messages_text": prompt_messages,
        "prompt_messages": prompt_messages_openai,
        "history_style": "openai_tool_calls_role_tool",
        "teacher_instruction": (
            "Continue from the tool result by returning exactly the next required native tool call. "
            "Use exact values from the tool result when present. Do not repeat earlier calls."
        ),
    }
    return {
        "train_instance": {
            "system": DEFAULT_SYSTEM,
            "tools": tools,
            "messages": train_messages,
        },
        "eval_case": eval_case,
        "openai_eval_case": openai_eval_case,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--openai-out", type=Path, default=DEFAULT_OPENAI_OUT)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.openai_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_dir.mkdir(parents=True, exist_ok=True)

    built = [make_case(spec) for spec in CASES]
    with args.out.open("w", encoding="utf-8") as f:
        for item in built:
            f.write(json.dumps(item["eval_case"], ensure_ascii=False) + "\n")
    with args.openai_out.open("w", encoding="utf-8") as f:
        for item in built:
            f.write(json.dumps(item["openai_eval_case"], ensure_ascii=False) + "\n")

    train_path = args.train_dir / "train_synthetic_toolresult.json"
    train_payload = {"type": "conversation", "instances": [item["train_instance"] for item in built]}
    train_path.write_text(json.dumps(train_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "eval_path": str(args.out),
        "openai_eval_path": str(args.openai_out),
        "train_path": str(train_path),
        "num_examples": len(built),
        "history_styles": [
            "assistant_tool_call_text_plus_user_tool_result",
            "openai_tool_calls_role_tool",
        ],
        "ids": [item["eval_case"]["id"] for item in built],
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    openai_manifest_path = args.openai_out.with_suffix(".manifest.json")
    openai_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
