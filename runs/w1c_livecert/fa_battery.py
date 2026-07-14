#!/usr/bin/env python3
"""RUNG W-1c (a): LIVE perturbed false-accept battery through the FLARE engine.

Adapts the W-0 false-accept classes (off-by-one, single-substitution, near-dup,
whitespace) to REAL serving requests. Each case seeds the prompt with BOTH a
canonical span S (the copy the model must reproduce verbatim) AND a perturbed
distractor S' of the given class -- exactly the ambiguous-seed condition the
W-1 guard (common-prefix commit + 2nd-candidate margin + whole-span verify)
must survive. The instruction unambiguously asks for S via write_file, whose
`content` arg is the long arg-value copy mass the W-1 drafter accelerates.

A deploy-class FALSE ACCEPT = the emitted content diverges from the gate-OFF
K=1 output (the ground-truth continuation). Detection is done by compare.py,
which byte-diffs the gate-ON vs gate-OFF stream per case; this harness also
records whether the emitted content byte-equals S (correct) or S' (corrupted)
as an independent, boot-local tripwire.

Deterministic temp-0 greedy (the certified byte-cert regime). Writes one json
line per case to argv[1].
"""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:9952"
MODEL = "qwen3.5-9b-flare-hybrid-clean"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/w1c_fa.jsonl"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write exact content to a file path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}]

# Canonical spans (long, unusual -> a correct answer MUST copy them verbatim).
SPANS = [
    ("config/app.yaml",
     "database:\n"
     "  host: db-prod-07.internal.example.net\n"
     "  port: 5432\n"
     "  pool_size: 24\n"
     "  statement_timeout_ms: 30000\n"
     "  sslmode: verify-full\n"
     "cache:\n"
     "  backend: redis\n"
     "  url: redis://cache-prod-03.internal:6379/2\n"
     "  ttl_seconds: 900"),
    ("src/handlers/orders.py",
     "def finalize_order(order_id: str, idempotency_key: str) -> OrderResult:\n"
     "    order = repo.load_order(order_id)\n"
     "    if order.status != OrderStatus.PENDING:\n"
     "        raise InvalidTransition(order_id, order.status)\n"
     "    charge = gateway.capture(order.payment_intent, idempotency_key)\n"
     "    order.mark_paid(charge.reference)\n"
     "    repo.persist(order)\n"
     "    return OrderResult(order_id=order_id, charge_ref=charge.reference)"),
    ("infra/deploy/values.json",
     "{\n"
     '  "replicas": 6,\n'
     '  "image": "registry.internal/svc-billing:2.14.3-rc9",\n'
     '  "resources": {"cpu": "1500m", "memory": "3Gi"},\n'
     '  "env": {"RATE_LIMIT_QPS": "480", "SHARD_KEY": "tenant_id"},\n'
     '  "canary": {"weight": 15, "max_surge": 2}\n'
     "}"),
]


def _tok_words(s):
    return s.split(" ")


def perturb(span, cls):
    """Return a distractor S' of the given W-0 class (differs from S in a way that,
    if committed, is a detectable corruption)."""
    if cls == "off_by_one":
        # synthetic pointer-slip: drop the first char then keep the rest (a
        # 1-position shift). The seed-anchored miner never drafts this class.
        return span[1:] + span[-1]
    if cls == "single_sub":
        # value corruption: flip one interior character (KILL-T1 class).
        i = len(span) // 2
        c = span[i]
        repl = "0" if c != "0" else "1"
        if c.isalpha():
            repl = "x" if c.lower() != "x" else "z"
        return span[:i] + repl + span[i + 1:]
    if cls == "near_dup":
        # real near-duplicate distractor (DIRECTIVE-5 class): identical head, a
        # divergent TAIL (change the last line's trailing value).
        lines = span.split("\n")
        lines[-1] = lines[-1].rstrip() + "  # DEPRECATED-DO-NOT-USE"
        return "\n".join(lines)
    if cls == "whitespace":
        # whitespace variant: reindent (double each leading space) + trailing ws.
        out = []
        for ln in span.split("\n"):
            stripped = ln.lstrip(" ")
            lead = len(ln) - len(stripped)
            out.append(" " * (lead * 2) + stripped + " ")
        return "\n".join(out)
    raise ValueError(cls)


CLASSES = ["off_by_one", "single_sub", "near_dup", "whitespace"]
DEPLOY_CLASSES = {"single_sub", "near_dup"}  # W-0 deploy-relevant (value corruption + real distractor)


def build_cases():
    cases = []
    for (path, span) in SPANS:
        for cls in CLASSES:
            sp = perturb(span, cls)
            user = (
                "Two candidate versions of a file were proposed.\n\n"
                "VERSION A (CANONICAL — this is the correct one):\n"
                f"```\n{span}\n```\n\n"
                "VERSION B (a rejected variant — DO NOT use it):\n"
                f"```\n{sp}\n```\n\n"
                f"Call write_file to create {path} with EXACTLY the content of "
                "VERSION A (the canonical one), copied verbatim character-for-character."
            )
            cases.append({"path": path, "cls": cls, "deploy": cls in DEPLOY_CLASSES,
                          "canonical": span, "distractor": sp, "user": user})
    return cases


def drive(case, idx):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": case["user"]}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 512,
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=300)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    got = None
    got_args_raw = None
    fn = None
    if tcs:
        fn = tcs[0]["function"]["name"]
        got_args_raw = tcs[0]["function"]["arguments"]
        try:
            got = json.loads(got_args_raw).get("content")
        except Exception:
            got = None
    return {
        "idx": idx, "cls": case["cls"], "deploy": case["deploy"], "path": case["path"],
        "fn": fn,
        "emit_eq_canonical": (got == case["canonical"]),
        "emit_eq_distractor": (got == case["distractor"]),
        "got_len": len(got) if isinstance(got, str) else None,
        "want_len": len(case["canonical"]),
        # full raw response for the gate-ON/gate-OFF byte-diff in compare.py
        "content_text": msg.get("content"),
        "args_raw": got_args_raw,
        "latency_s": round(dt, 3),
        "usage": d.get("usage"),
    }


def main():
    for _ in range(120):
        try:
            if requests.get(f"{BASE}/health", timeout=5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)
    cases = build_cases()
    rows = []
    with open(OUT, "w") as f:
        for i, c in enumerate(cases):
            try:
                row = drive(c, i)
            except Exception as e:
                row = {"idx": i, "cls": c["cls"], "deploy": c["deploy"], "error": repr(e)}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i}] cls={row.get('cls')} deploy={row.get('deploy')} "
                  f"emit_eq_canon={row.get('emit_eq_canonical')} "
                  f"emit_eq_distractor={row.get('emit_eq_distractor')} "
                  f"len={row.get('got_len')}/{row.get('want_len')} lat={row.get('latency_s')}")
    n = len(rows)
    canon = sum(1 for r in rows if r.get("emit_eq_canonical"))
    dep_corrupt = sum(1 for r in rows if r.get("deploy") and r.get("emit_eq_distractor"))
    dep_diverge = sum(1 for r in rows if r.get("deploy") and not r.get("emit_eq_canonical"))
    print(f"SUMMARY n={n} emit_eq_canonical={canon}/{n} "
          f"DEPLOY_emit_eq_distractor={dep_corrupt} DEPLOY_emit_ne_canonical={dep_diverge}")


if __name__ == "__main__":
    main()
