#!/usr/bin/env python3
"""Build the W-1c (b) throughput corpus and (c) A6 matched-turn corpus.

A6 turns: real banked C46-iter2 gate-OFF agentic requests (dumps), preferring
turns with tool-result copy sources (>=4 messages) and bounded context so the
byte-diff replay stays cheap. Forced to temp-0 greedy + non-stream so gate-ON
and gate-OFF replays are byte-comparable to EACH OTHER (online==offline).

Throughput corpus: copy-heavy write_file turns (long verbatim `content` arg =
the arg-value copy mass the W-1 drafter accelerates) + short path turns
(reject-prone) so the blended tok/fwd AND the reject-tax are both measured.
"""
import glob
import json
import os

ROOT = "/home/mark/qwen_diffusion"
OUTDIR = f"{ROOT}/runs/w1c_livecert"
MODEL = "qwen3.5-9b-flare-hybrid-clean"


def build_a6(n=5, max_ctx_chars=60000):
    dumps = sorted(glob.glob(f"{ROOT}/runs/k_gate_c46_iter2/diffusion/dumps_shard_*/chat_*.json"))
    picks = []
    for f in dumps:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        msgs = d.get("messages") or []
        # need a copy source: a tool result / assistant turn in context
        roles = [m.get("role") for m in msgs]
        if len(msgs) < 4:
            continue
        if not any(r in ("tool", "assistant") for r in roles):
            continue
        approx = len(json.dumps(msgs))
        if approx > max_ctx_chars:
            continue
        picks.append((approx, f, d))
    # spread across context sizes for representativeness
    picks.sort()
    chosen = []
    if picks:
        idxs = [int(i * (len(picks) - 1) / max(1, n - 1)) for i in range(n)]
        seen = set()
        for i in idxs:
            if i in seen:
                i = next((j for j in range(len(picks)) if j not in seen), i)
            seen.add(i)
            chosen.append(picks[i])
    out = f"{OUTDIR}/a6_turns.jsonl"
    with open(out, "w") as w:
        for approx, f, d in chosen:
            body = {
                "model": MODEL,
                "messages": d["messages"],
                "tools": d.get("tools"),
                "temperature": 0.0,
                "max_tokens": min(d.get("max_tokens") or 2048, 2048),
                "stream": False,
                "_src": os.path.relpath(f, ROOT),
                "_ctx_chars": approx,
            }
            w.write(json.dumps(body) + "\n")
    print(f"[a6] wrote {len(chosen)} turns -> {out} (ctx_chars {[c for c,_,_ in chosen]})")


COPY_SNIPPETS = [
    ("utils/retry.py",
     "def retry_with_backoff(fn, attempts=5, base_delay=0.25, max_delay=8.0):\n"
     "    for i in range(attempts):\n"
     "        try:\n"
     "            return fn()\n"
     "        except TransientError as exc:\n"
     "            delay = min(max_delay, base_delay * (2 ** i))\n"
     "            time.sleep(delay + random.uniform(0, delay))\n"
     "    raise RetryBudgetExceeded(attempts)"),
    ("models/user.py",
     "class UserProfile(BaseModel):\n"
     "    user_id: UUID\n"
     "    display_name: str\n"
     "    email_verified: bool = False\n"
     "    preferences: dict[str, Any] = Field(default_factory=dict)\n"
     "    created_at: datetime\n"
     "    last_login_at: Optional[datetime] = None"),
    ("handlers/webhook.py",
     "async def handle_stripe_webhook(request, signing_secret):\n"
     "    payload = await request.body()\n"
     "    sig = request.headers.get('Stripe-Signature')\n"
     "    event = stripe.Webhook.construct_event(payload, sig, signing_secret)\n"
     "    if event.type == 'invoice.payment_failed':\n"
     "        await mark_subscription_past_due(event.data.object.customer)"),
    ("db/migrations/0042_add_index.sql",
     "CREATE INDEX CONCURRENTLY idx_orders_tenant_created\n"
     "    ON orders (tenant_id, created_at DESC)\n"
     "    WHERE status IN ('pending', 'processing');\n"
     "ANALYZE orders;"),
    ("k8s/deployment.yaml",
     "apiVersion: apps/v1\n"
     "kind: Deployment\n"
     "metadata:\n"
     "  name: billing-worker\n"
     "  namespace: payments\n"
     "spec:\n"
     "  replicas: 4\n"
     "  selector:\n"
     "    matchLabels:\n"
     "      app: billing-worker"),
    ("lib/parse.py",
     "def parse_iso8601(value: str) -> datetime:\n"
     "    if value.endswith('Z'):\n"
     "        value = value[:-1] + '+00:00'\n"
     "    return datetime.fromisoformat(value)"),
]
PATHS = [
    "src/backend/services/authentication/token_refresh_handler.py",
    "packages/frontend/components/dashboard/widgets/RevenueForecastChart.tsx",
    "tests/integration/api/v2/subscriptions/test_prorated_billing_cycles.py",
    "config/environments/staging/feature_flags_overrides.yaml",
]

WRITE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write exact content to a file path.",
        "parameters": {"type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]},
    },
}]
READ_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file at an exact repository path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    },
}]


def build_throughput():
    out = f"{OUTDIR}/thru_corpus.jsonl"
    rows = []
    for path, content in COPY_SNIPPETS:
        user = (f"Create the file {path} with exactly this content, using write_file:\n\n"
                f"```\n{content}\n```\n")
        rows.append({"model": MODEL, "messages": [{"role": "user", "content": user}],
                     "tools": WRITE_TOOLS, "temperature": 0.0, "max_tokens": 400,
                     "stream": False, "_class": "copy_heavy", "_path": path,
                     "_want_content": content})
    for p in PATHS:
        user = (f"Read the file at {p} and summarize it. Call read_file with that exact path.")
        rows.append({"model": MODEL, "messages": [{"role": "user", "content": user}],
                     "tools": READ_TOOLS, "temperature": 0.0, "max_tokens": 120,
                     "stream": False, "_class": "path", "_path": p})
    with open(out, "w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    print(f"[thru] wrote {len(rows)} turns ({sum(1 for r in rows if r['_class']=='copy_heavy')} copy_heavy, "
          f"{sum(1 for r in rows if r['_class']=='path')} path) -> {out}")


if __name__ == "__main__":
    build_a6()
    build_throughput()
