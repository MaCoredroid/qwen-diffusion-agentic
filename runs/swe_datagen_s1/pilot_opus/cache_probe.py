#!/usr/bin/env python3
"""Direct caching probe against the local Opus adapter. Confirms the Claude Code
OAuth path honors prompt caching: two requests sharing a large stable system+tools
prefix; the SECOND must report cache_read_input_tokens > 0. No secrets touched (the
adapter holds the token; this client only talks to 127.0.0.1)."""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30050
BASE = f"http://127.0.0.1:{PORT}/v1"

# poll readiness (python sleep is allowed)
for _ in range(120):
    try:
        urllib.request.urlopen(BASE + "/models", timeout=2)
        break
    except Exception:
        time.sleep(1)
else:
    print("ADAPTER NOT READY"); sys.exit(4)

# ~6k tokens of stable filler so the prefix clears the 4096-token Opus cache minimum
FILLER = ("You are assisting with a software engineering task. " * 700)
SYS = "You are a careful coding assistant.\n\nREFERENCE MATERIAL (stable):\n" + FILLER
TOOLS = [{"type": "function", "function": {
    "name": "run_shell_command",
    "description": "Run a shell command in the repo.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def call(messages):
    body = json.dumps({"model": "claude-opus-adapter", "messages": messages,
                       "tools": TOOLS, "max_tokens": 64}).encode()
    r = urllib.request.Request(BASE + "/chat/completions", data=body,
                               headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())


m1 = [{"role": "system", "content": SYS},
      {"role": "user", "content": "Reply with the single word: hello"}]
r1 = call(m1)
u1 = r1["choices"][0]["message"]
m2 = m1 + [{"role": "assistant", "content": r1["choices"][0]["message"].get("content") or "hello"},
           {"role": "user", "content": "Reply with the single word: goodbye"}]
r2 = call(m2)

out = {
    "req1_usage": r1.get("usage"),
    "req2_usage": r2.get("usage"),
    "caching_works": (r2.get("usage", {}).get("cache_read_input_tokens", 0) or 0) > 0,
}
print(json.dumps(out, indent=1))
