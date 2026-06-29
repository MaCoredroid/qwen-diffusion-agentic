#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
            if self.dump_dir:
                type(self).request_count += 1
                dump_path = self.dump_dir / f"chat_{type(self).request_count:04d}.json"
                dump_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.forward(json.dumps(payload).encode("utf-8"))

    def forward(self, body):
        url = self.upstream.rstrip("/") + self.path.removeprefix("/v1")
        headers = {"Content-Type": "application/json"}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
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
