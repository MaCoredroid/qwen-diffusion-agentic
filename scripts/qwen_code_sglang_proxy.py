#!/usr/bin/env python3
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Monotonic per-request index for the forced per-request seed. itertools.count()
# .__next__ is atomic under CPython, so it is safe to advance from the
# ThreadingHTTPServer worker threads without a lock.
_SEED_COUNTER = itertools.count()

# Env vars -> chat/completions sampling keys for the reference-envelope pins.
# (name, payload_key, caster) — mirrors the flywheel inference_proxy.py
# LUMO_PROXY_FORCE_* pattern, ported onto the OpenAI chat schema's top level.
_ENVELOPE_PINS = (
    ("LUMO_PROXY_FORCE_TEMPERATURE", "temperature", float),
    ("LUMO_PROXY_FORCE_TOP_P", "top_p", float),
    ("LUMO_PROXY_FORCE_TOP_K", "top_k", int),
    ("LUMO_PROXY_FORCE_MIN_P", "min_p", float),
    ("LUMO_PROXY_FORCE_PRESENCE_PENALTY", "presence_penalty", float),
)


def apply_reference_envelope(payload, *, env=None, seed_index=None):
    """Env-gated reference-envelope sampling pins (flywheel LUMO_PROXY_FORCE_*
    pattern, ported to the OpenAI /v1/chat/completions schema).

    The reference envelope (banked in runs/stage_c_n5v2/report.md) is the
    Qwen-official anti-degenerate regime the flywheel SWE campaigns run under:
    temperature 0.6 / top_p 0.95 / top_k 20, seeded per request. qwen-code
    itself sends greedy-ish bodies, so the pins OVERWRITE whatever the client
    sent — that is the whole point (force the envelope server-side without a
    client change).

    Default-OFF contract: with NONE of the LUMO_PROXY_FORCE_* vars set this
    returns the payload UNCHANGED (byte-identical passthrough), which preserves
    the greedy A/B baseline the corrected-ladder premise is measured against.

    Per-request seed: a client-supplied `seed` is left untouched (passthrough).
    Otherwise, if LUMO_PROXY_FORCE_SEED is set, stamps base_seed + seed_index so
    every request is distinct AND reproducible (temp>0 seeded contract).
    """
    env = os.environ if env is None else env
    out = dict(payload)
    for name, key, cast in _ENVELOPE_PINS:
        raw = env.get(name)
        if raw in (None, ""):
            continue
        try:
            out[key] = cast(raw)
        except (TypeError, ValueError):
            pass  # a malformed pin is ignored, never crashes the proxy
    # Per-request seed. Client seed wins (passthrough); else stamp base + index.
    if out.get("seed") is None:
        base = env.get("LUMO_PROXY_FORCE_SEED")
        if base not in (None, ""):
            try:
                out["seed"] = int(base) + int(seed_index or 0)
            except (TypeError, ValueError):
                pass
    return out


class ProxyHandler(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:30000/v1"
    max_tokens = 512
    dump_dir = None
    tool_choice = ""
    tool_choice_turns = 0
    chat_count = 0
    request_count = 0

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        self.forward(None)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return

        if self.path.rstrip("/") == "/v1/chat/completions":
            type(self).chat_count += 1
            payload["chat_template_kwargs"] = {
                **(payload.get("chat_template_kwargs") or {}),
                "enable_thinking": False,
            }
            if self.max_tokens and int(payload.get("max_tokens") or 0) > self.max_tokens:
                payload["max_tokens"] = self.max_tokens
            within_tool_choice_turns = (
                not self.tool_choice_turns or type(self).chat_count <= self.tool_choice_turns
            )
            if (
                self.tool_choice
                and within_tool_choice_turns
                and payload.get("tools")
                and not payload.get("tool_choice")
            ):
                payload["tool_choice"] = self.tool_choice
            # Reference-envelope sampling pins (LUMO_PROXY_FORCE_*, env-gated,
            # default-OFF -> byte-identical passthrough). Applied LAST so the
            # dumped body below is exactly what is forwarded upstream (proxy-side
            # evidence of the envelope the server received).
            payload = apply_reference_envelope(payload, seed_index=next(_SEED_COUNTER))
            if self.dump_dir:
                type(self).request_count += 1
                self._capture_idx = type(self).request_count
                dump_path = self.dump_dir / f"chat_{type(self).request_count:04d}.json"
                dump_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.forward(json.dumps(payload).encode("utf-8"))

    def _record_usage(self, idx, tail_bytes):
        """Append the streamed response's final usage + finish_reason to
        dump_dir/usage.jsonl. Exact per-request tokens + a timestamp, captured
        server-side so they survive ANY qwen-code exit mode (turn-limit 53 /
        loop-detect 1 / budget 55 write nothing to the CLI's stdout)."""
        import time as _t
        text = tail_bytes.decode("utf-8", errors="replace")
        usage = None
        finish = None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                for ch in (obj.get("choices") or []):
                    if isinstance(ch, dict) and ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        rec = {"idx": idx, "ts": round(_t.time(), 3), "usage": usage, "finish_reason": finish}
        try:
            with (self.dump_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # never let telemetry break the proxy
            pass

    def forward(self, body):
        url = self.upstream.rstrip("/") + self.path.removeprefix("/v1")
        headers = {"Content-Type": "application/json"}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        capture_idx = getattr(self, "_capture_idx", None)
        tail = b""
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                content_type = resp.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", content_type)
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if capture_idx is not None and self.dump_dir is not None:
                        tail = (tail + chunk)[-16384:]
            if capture_idx is not None and self.dump_dir is not None:
                self._record_usage(capture_idx, tail)
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            sys.stderr.write(
                f"upstream HTTP {exc.code} for {self.path}: "
                f"{error_body.decode('utf-8', errors='replace')}\n"
            )
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "text/plain"))
            self.end_headers()
            self.wfile.write(error_body)
        except Exception as exc:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"{type(exc).__name__}: {exc}".encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--upstream", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--tool-choice", default="")
    parser.add_argument("--tool-choice-turns", type=int, default=0)
    args = parser.parse_args()

    ProxyHandler.upstream = args.upstream
    ProxyHandler.max_tokens = args.max_tokens
    ProxyHandler.tool_choice = args.tool_choice
    ProxyHandler.tool_choice_turns = args.tool_choice_turns
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        ProxyHandler.dump_dir = args.dump_dir
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"qwen-code SGLang proxy listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
