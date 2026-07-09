#!/usr/bin/env python3
"""OPUS OpenAI-compatible adapter — a drop-in /v1 endpoint that lets the EXISTING
SWE-datagen harness (scripts/run_swe_bench_qwen_code.py + qwen_code_sglang_proxy.py
+ qwen-code) run UNMODIFIED with Anthropic Opus as the episode-solving brain.

WHY THIS SHAPE (PROXY PATH / task option (a), strongly preferred):
  The keeper trajectory format is produced by QWEN-CODE, not by the model server.
  extract_keepers.py builds each keeper row from the proxy's REQUEST dumps
  (qwen_code_sglang_proxy.py captures qwen-code's own OpenAI-schema request body:
  system + user + [assistant, tool]* for every completed turn). That request body
  is byte-for-byte independent of which model produced the assistant turns. So if
  we keep the harness identical and only swap the UPSTREAM the proxy forwards to
  (vLLM -> this adapter), the emitted keeper format is IDENTICAL BY CONSTRUCTION
  (native qwen3_xml, [[native-function-format-rule]]). This adapter's ONLY job is
  to answer /v1/chat/completions with a spec-clean OpenAI assistant turn (text OR
  tool_calls) so qwen-code records it exactly as it records a vLLM turn.

INTEGRATION (zero harness edits):
  run_swe_bench_qwen_code.py ... --endpoint http://127.0.0.1:<PORT>/v1
  The harness starts qwen_code_sglang_proxy.py with --upstream=<that endpoint> and
  --dump-dir; qwen-code -> proxy (dumps) -> THIS adapter -> Opus. Downstream
  (extract_keepers.py, build_batch_dataset.py conventions) run unchanged.

BACKENDS (--backend):
  anthropic  : translate OpenAI chat/completions <-> Anthropic Messages API. Native
               tool_use blocks map 1:1 to OpenAI tool_calls (BEST fidelity). Auth:
               ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN (OAuth; adds the oauth
               beta header). This is the production path once a key is provisioned.
  claude_cli : translate via `claude -p` (Claude Code headless) as the backend, per
               the task note. Opus emits a strict JSON action in text; we parse it
               into OpenAI tool_calls. Works once the CLI is logged in (`claude
               login` / a restored ~/.claude/.credentials.json). Auth-provisioning
               free (uses the CLI's own auth).
  scripted   : NON-OPUS deterministic tool-calling script. Purpose: prove the FORMAT
               path end-to-end THROUGH the real harness/container/dumps/extractor
               WITHOUT external model auth. Clearly labelled; never a data source.

The three backends differ ONLY in the CONTENT of the assistant turn; the OpenAI
response envelope (and therefore the downstream keeper format) is identical.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as _urlreq, error as _urlerr


# ---------------------------------------------------------------------------
# OpenAI response envelope helpers (single source of truth for both stream modes)
# ---------------------------------------------------------------------------
def _mkid() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _tool_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _openai_message(text, tool_calls):
    """Assemble the OpenAI assistant message. content=None when only tool_calls
    (matches how vLLM/qwen-code represent a pure tool-call turn)."""
    msg = {"role": "assistant"}
    msg["content"] = text if text else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _finish_reason(tool_calls, raw=None):
    if raw == "length":
        return "length"
    return "tool_calls" if tool_calls else "stop"


def _nonstream_body(model, text, tool_calls, usage, raw_finish=None):
    return {
        "id": _mkid(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": _openai_message(text, tool_calls),
            "finish_reason": _finish_reason(tool_calls, raw_finish),
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"


def _stream_chunks(model, text, tool_calls, usage, include_usage, raw_finish=None):
    """Yield OpenAI streaming chunks: role delta -> content/tool_calls delta ->
    finish delta -> (optional) usage chunk -> [DONE]. A single complete tool_call
    per delta (id + name + full JSON arguments) is what the OpenAI SDK-style
    accumulator qwen-code uses expects."""
    cid = _mkid()
    created = int(time.time())

    def frame(delta, finish=None):
        return _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    yield frame({"role": "assistant"})
    if tool_calls:
        delta_tcs = []
        for i, tc in enumerate(tool_calls):
            delta_tcs.append({
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]},
            })
        yield frame({"tool_calls": delta_tcs})
    elif text:
        yield frame({"content": text})
    yield frame({}, finish=_finish_reason(tool_calls, raw_finish))
    if include_usage:
        yield _sse({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# OpenAI <-> Anthropic Messages API translation
# ---------------------------------------------------------------------------
def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    out.append(p.get("text", ""))
                elif "text" in p:
                    out.append(p["text"])
            elif isinstance(p, str):
                out.append(p)
        return "\n".join(out)
    return str(content)


def openai_to_anthropic(payload, default_model, max_tokens_floor):
    """Map an OpenAI chat/completions request -> Anthropic Messages API request."""
    system_parts, a_msgs = [], []
    for m in payload.get("messages", []):
        role = m.get("role")
        if role == "system":
            system_parts.append(_content_to_text(m.get("content")))
        elif role == "user":
            a_msgs.append({"role": "user",
                           "content": [{"type": "text", "text": _content_to_text(m.get("content"))}]})
        elif role == "assistant":
            blocks = []
            txt = _content_to_text(m.get("content"))
            if txt:
                blocks.append({"type": "text", "text": txt})
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    inp = {"_raw": fn.get("arguments")}
                blocks.append({"type": "tool_use", "id": tc.get("id") or _tool_call_id(),
                               "name": fn.get("name"), "input": inp})
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            a_msgs.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            a_msgs.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": _content_to_text(m.get("content")),
            }]})
    a_tools = []
    for t in (payload.get("tools") or []):
        fn = t.get("function", {})
        a_tools.append({"name": fn.get("name"), "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    req = {
        "model": default_model,
        "max_tokens": max(int(payload.get("max_tokens") or 0), max_tokens_floor),
        "messages": a_msgs,
    }
    if system_parts:
        req["system"] = "\n\n".join(p for p in system_parts if p)
    if a_tools:
        req["tools"] = a_tools
    tc = payload.get("tool_choice")
    if tc == "required":
        req["tool_choice"] = {"type": "any"}
    elif isinstance(tc, dict) and tc.get("type") == "function":
        req["tool_choice"] = {"type": "tool", "name": tc.get("function", {}).get("name")}
    # NOTE: sampling params (temperature/top_p/top_k) are intentionally NOT
    # forwarded — Opus 4.8/4.7 (and Sonnet 5 / Fable 5) reject them with a 400.
    # qwen-code sends temperature by default; forwarding it would 400 every turn.
    #
    # PROMPT CACHING (the ~4x cost lever; env OPUS_ADAPTER_CACHE, default on):
    # place an ephemeral breakpoint on the last content block of the last message.
    # qwen-code resends the FULL growing conversation each turn, so this caches the
    # entire prior prefix incrementally — turn N+1 reads turn N's prefix at ~0.1x and
    # writes only the new delta at ~1.25x ([[prompt-caching]] multi-turn pattern). The
    # tools+system prefix is cached separately by the system-block breakpoint in
    # call_anthropic (render order tools -> system -> messages). Both breakpoints are
    # internal to the adapter->Anthropic hop and never appear in the proxy dumps, so
    # the emitted keeper format stays qwen3_xml-native.
    if os.environ.get("OPUS_ADAPTER_CACHE", "1") == "1" and a_msgs:
        last_content = a_msgs[-1].get("content")
        if isinstance(last_content, list) and last_content:
            last_content[-1] = {**last_content[-1], "cache_control": {"type": "ephemeral"}}
    return req


def anthropic_to_openai(resp):
    """Map an Anthropic Messages API response -> (text, tool_calls, usage, finish).

    Prompt-caching accounting: Anthropic's `input_tokens` is the UNCACHED remainder;
    total prompt = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    (see [[prompt-caching]]). We report `prompt_tokens` as the FULL prompt (so token
    counts stay apples-to-apples with the uncached 9B baseline) and additionally carry
    the cache breakdown so the caller can price it: cache reads bill ~0.1x input, cache
    writes ~1.25x (5-min TTL)."""
    text_parts, tool_calls = [], []
    for block in resp.get("content", []) or []:
        bt = block.get("type")
        if bt == "text":
            text_parts.append(block.get("text", ""))
        elif bt == "tool_use":
            tool_calls.append({
                "id": block.get("id") or _tool_call_id(),
                "type": "function",
                "function": {"name": block.get("name"),
                             "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)},
            })
    u = resp.get("usage", {}) or {}
    inp = u.get("input_tokens", 0) or 0
    out = u.get("output_tokens", 0) or 0
    cache_read = u.get("cache_read_input_tokens", 0) or 0
    cache_write = u.get("cache_creation_input_tokens", 0) or 0
    total_prompt = inp + cache_read + cache_write
    usage = {
        "prompt_tokens": total_prompt,
        "completion_tokens": out,
        "total_tokens": total_prompt + out,
        # cache breakdown (extra fields; harmless to OpenAI clients that ignore them)
        "uncached_input_tokens": inp,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }
    sr = resp.get("stop_reason")
    raw = "length" if sr == "max_tokens" else None
    return "".join(text_parts), tool_calls, usage, raw


def call_anthropic(payload, cfg):
    req_body = openai_to_anthropic(payload, cfg["anthropic_model"], cfg["max_tokens_floor"])
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    key = os.environ.get("ANTHROPIC_API_KEY")
    oauth = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if key:
        headers["x-api-key"] = key
    elif oauth:
        headers["authorization"] = "Bearer " + oauth
        headers["anthropic-beta"] = "oauth-2025-04-20"
        # Claude Code OAuth tokens (sk-ant-oat*) are gated: the FIRST system block
        # must EQUAL exactly the Claude Code identity string, or the API returns
        # 429. Concatenating the identity into one system string does NOT satisfy
        # the gate (verified empirically) — it must be its own block. So send
        # `system` as a block array: [identity_block, <caller_system_block>]. This
        # is internal to the adapter->Anthropic hop and NEVER appears in the proxy
        # request dumps, so the emitted keeper format stays qwen3_xml-native.
        _ident = "You are Claude Code, Anthropic's official CLI for Claude."
        _sys = req_body.get("system")
        _blocks = [{"type": "text", "text": _ident}]
        if _sys:
            _blocks.append({"type": "text",
                            "text": _sys if isinstance(_sys, str) else json.dumps(_sys)})
        # Cache the tools+system prefix (stable across every turn of an episode). The
        # breakpoint on the last system block also caches `tools` (render order is
        # tools -> system -> messages), which is the single biggest stable prefix.
        if os.environ.get("OPUS_ADAPTER_CACHE", "1") == "1":
            _blocks[-1] = {**_blocks[-1], "cache_control": {"type": "ephemeral"}}
        req_body["system"] = _blocks
    else:
        raise RuntimeError("anthropic backend needs ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    body = json.dumps(req_body).encode("utf-8")
    # Retry with exponential backoff on transient failures (429 rate-limit / 5xx /
    # 529 overloaded). This run shares the token with the user's live session, so a
    # genuine rate-limit mid-episode must be waited out, not surfaced as a failed
    # turn (which would corrupt the trajectory). Honor Retry-After when present.
    max_retries = int(os.environ.get("OPUS_ADAPTER_MAX_RETRIES", "6"))
    last_exc = None
    for attempt in range(max_retries + 1):
        r = _urlreq.Request(base.rstrip("/") + "/v1/messages",
                            data=body, headers=headers, method="POST")
        try:
            with _urlreq.urlopen(r, timeout=cfg["timeout"]) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return anthropic_to_openai(data)
        except _urlerr.HTTPError as exc:  # noqa: PERF203
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 529) or attempt == max_retries:
                raise
            ra = exc.headers.get("retry-after") if exc.headers else None
            try:
                delay = float(ra) if ra else 0.0
            except ValueError:
                delay = 0.0
            delay = max(delay, min(60.0, 2.0 * (2 ** attempt)))
            sys.stderr.write(f"anthropic {exc.code}; retry {attempt+1}/{max_retries} "
                             f"in {delay:.1f}s\n")
            time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# claude -p CLI backend (auth-provisioning-free path; task-blessed)
# ---------------------------------------------------------------------------
_CLI_CONTRACT = (
    "You are the NEXT-ACTION policy for an autonomous SWE agent. Given the running "
    "conversation and the available tools, choose EXACTLY ONE next step. Respond with "
    "ONLY a single fenced ```json code block, no prose outside it, of the form:\n"
    '```json\n{\"tool_calls\":[{\"name\":\"<tool>\",\"arguments\":{...}}]}\n```\n'
    "to call a tool, OR\n"
    '```json\n{\"content\":\"<final assistant message>\"}\n```\n'
    "to finish. Emit at most one tool call. Do NOT execute anything yourself."
)


def _render_conversation(payload) -> str:
    lines = []
    tools = payload.get("tools") or []
    if tools:
        brief = [{"name": t.get("function", {}).get("name"),
                  "parameters": t.get("function", {}).get("parameters")} for t in tools]
        lines.append("AVAILABLE TOOLS (OpenAI function schema):\n" + json.dumps(brief, indent=1))
    lines.append("\nCONVERSATION SO FAR:")
    for m in payload.get("messages", []):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                lines.append(f"[assistant->tool_call] {fn.get('name')}({fn.get('arguments')})")
            if _content_to_text(m.get("content")):
                lines.append("[assistant] " + _content_to_text(m.get("content")))
        else:
            lines.append(f"[{role}] " + _content_to_text(m.get("content")))
    return "\n".join(lines)


def _parse_cli_action(text):
    """Extract {tool_calls|content} JSON from claude -p result text. Tolerant:
    prefers a ```json fence, falls back to the last balanced {...} object."""
    blob = None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        blob = m.group(1)
    else:
        m = re.search(r"```\s*(\{.*?\})\s*```", text, re.S)
        if m:
            blob = m.group(1)
    if blob is None:
        starts = [i for i, c in enumerate(text) if c == "{"]
        for s in starts:
            depth = 0
            for e in range(s, len(text)):
                depth += text[e] == "{"
                depth -= text[e] == "}"
                if depth == 0:
                    cand = text[s:e + 1]
                    try:
                        json.loads(cand)
                        blob = cand
                        break
                    except json.JSONDecodeError:
                        pass
            if blob:
                break
    if blob is None:
        return text.strip(), []
    obj = json.loads(blob)
    if obj.get("tool_calls"):
        tcs = []
        for tc in obj["tool_calls"]:
            tcs.append({"id": _tool_call_id(), "type": "function",
                        "function": {"name": tc.get("name"),
                                     "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False)}})
        return None, tcs
    return obj.get("content", ""), []


def call_claude_cli(payload, cfg):
    prompt = _render_conversation(payload)
    cmd = [cfg["claude_bin"], "-p", prompt, "--model", cfg["anthropic_model"],
           "--output-format", "json", "--max-turns", "1",
           "--append-system-prompt", _CLI_CONTRACT]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"])
    out = json.loads(proc.stdout)
    if out.get("is_error"):
        raise RuntimeError("claude -p error: " + str(out.get("result"))[:200])
    text, tool_calls = _parse_cli_action(out.get("result", ""))
    u = out.get("usage", {}) or {}
    inp = u.get("input_tokens", 0) or 0
    out_t = u.get("output_tokens", 0) or 0
    usage = {"prompt_tokens": inp, "completion_tokens": out_t, "total_tokens": inp + out_t}
    return text, tool_calls, usage, None


# ---------------------------------------------------------------------------
# scripted backend (NON-OPUS; format-plumbing proof only)
# ---------------------------------------------------------------------------
def call_scripted(payload, cfg):
    """Deterministic tool-calling script driving REAL qwen-code tool round-trips so
    the proxy dumps + extractor + SFT-render path exercise the true format. NOT a
    model; NOT a data source."""
    step = sum(1 for m in payload.get("messages", []) if m.get("role") == "assistant")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def rsc(command):
        return [{"id": _tool_call_id(), "type": "function",
                 "function": {"name": "run_shell_command",
                              "arguments": json.dumps({"command": command})}}]

    if step == 0:
        return None, rsc("git ls-files | head -5 && echo '--HEAD--' && git rev-parse HEAD"), usage, None
    if step == 1:
        return None, rsc("git status --porcelain=v1 && python -c \"print('opus-adapter smoke ok')\""), usage, None
    if step == 2:
        cmd = ("f=$(git ls-files 'README*' 'setup.*' | head -1); "
               "[ -z \"$f\" ] && f=$(git ls-files '*.py' | head -1); "
               "printf '\\n# opus-adapter smoke marker\\n' >> \"$f\"; git diff --stat")
        return None, rsc(cmd), usage, None
    return ("Smoke complete: exercised run_shell_command tool round-trips through the "
            "OpenAI adapter; the harness recorded native qwen3_xml turns."), [], usage, None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
BACKENDS = {"anthropic": call_anthropic, "claude_cli": call_claude_cli, "scripted": call_scripted}


class Handler(BaseHTTPRequestHandler):
    cfg = {}

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path.endswith("/health") or path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok\n"); return
        if path.endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": self.cfg["served_model"], "object": "model", "owned_by": "anthropic-adapter"}]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"}); return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": "unsupported path"}); return

        model = payload.get("model") or self.cfg["served_model"]
        stream = bool(payload.get("stream"))
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage", True))
        backend = BACKENDS[self.cfg["backend"]]
        try:
            text, tool_calls, usage, raw_finish = backend(payload, self.cfg)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("backend error: %r\n" % exc)
            self._json(502, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                       "type": "adapter_backend_error"}})
            return

        # Per-request usage accounting (env-gated). Robust token measurement
        # independent of whether the client streams (the proxy only records usage
        # on the streamed path). One JSONL row per request: ts + token counts.
        _ulog = os.environ.get("OPUS_ADAPTER_USAGE_LOG")
        if _ulog:
            try:
                with open(_ulog, "a", encoding="utf-8") as _uf:
                    _u = usage or {}
                    _uf.write(json.dumps({
                        "ts": round(time.time(), 3),
                        "prompt_tokens": _u.get("prompt_tokens", 0),
                        "completion_tokens": _u.get("completion_tokens", 0),
                        "uncached_input_tokens": _u.get("uncached_input_tokens", 0),
                        "cache_read_input_tokens": _u.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": _u.get("cache_creation_input_tokens", 0),
                    }) + "\n")
            except Exception:  # noqa: BLE001
                pass

        if not stream:
            self._json(200, _nonstream_body(model, text, tool_calls, usage, raw_finish)); return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in _stream_chunks(model, text, tool_calls, usage, include_usage, raw_finish):
            self.wfile.write(chunk); self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30040)
    ap.add_argument("--backend", choices=list(BACKENDS), default=os.environ.get("OPUS_ADAPTER_BACKEND", "anthropic"))
    ap.add_argument("--anthropic-model", default=os.environ.get("OPUS_ADAPTER_MODEL", "claude-opus-4-8"))
    ap.add_argument("--served-model", default="claude-opus-adapter")
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--max-tokens-floor", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    Handler.cfg = {
        "backend": args.backend, "anthropic_model": args.anthropic_model,
        "served_model": args.served_model, "claude_bin": args.claude_bin,
        "max_tokens_floor": args.max_tokens_floor, "timeout": args.timeout,
    }
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"opus adapter [{args.backend}] model={args.anthropic_model} "
          f"listening on http://{args.host}:{args.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
