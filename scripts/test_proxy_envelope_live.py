#!/usr/bin/env python3
"""Integration test: drive the qwen_code_sglang_proxy PROCESS with a mock
upstream that echoes the forwarded body. Proves the do_POST wiring + env
inheritance (LUMO_PROXY_FORCE_* set on the parent -> the proxy subprocess
applies them) end-to-end, without a GPU.

    .venv/bin/python scripts/test_proxy_envelope_live.py
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROXY = HERE / "qwen_code_sglang_proxy.py"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Echo(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        # Echo the EXACT body the proxy forwarded upstream.
        self.wfile.write(body)


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _wait(url, timeout=10):
    dl = time.time() + timeout
    while time.time() < dl:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            try:
                # /health returns 200; any other 4xx/5xx still means "up"
                urllib.request.urlopen(url, timeout=1)
            except Exception:
                pass
        time.sleep(0.1)
    return False


def _health(host, port, timeout=10):
    dl = time.time() + timeout
    while time.time() < dl:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def run(env_extra, label):
    up_port = _free_port()
    px_port = _free_port()
    up = ThreadingHTTPServer(("127.0.0.1", up_port), _Echo)
    t = threading.Thread(target=up.serve_forever, daemon=True)
    t.start()
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.Popen(
        [sys.executable, str(PROXY), "--host", "127.0.0.1", "--port", str(px_port),
         "--upstream", f"http://127.0.0.1:{up_port}/v1", "--max-tokens", "2048"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not _health("127.0.0.1", px_port):
            raise RuntimeError("proxy did not become healthy")
        base = f"http://127.0.0.1:{px_port}/v1/chat/completions"
        greedy = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
        r1 = _post(base, dict(greedy))
        r2 = _post(base, dict(greedy))
        return r1, r2
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        up.shutdown()


fail = []


def check(name, cond, detail=""):
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        fail.append(name)


# 1) Envelope ON: the forwarded body carries the pins + a per-request seed.
env_on = {
    "LUMO_PROXY_FORCE_TEMPERATURE": "0.6",
    "LUMO_PROXY_FORCE_TOP_P": "0.95",
    "LUMO_PROXY_FORCE_TOP_K": "20",
    "LUMO_PROXY_FORCE_SEED": "2000",
}
a, b = run(env_on, "on")
check("on.temperature", a.get("temperature") == 0.6, f"{a.get('temperature')!r}")
check("on.top_p", a.get("top_p") == 0.95, f"{a.get('top_p')!r}")
check("on.top_k", a.get("top_k") == 20, f"{a.get('top_k')!r}")
check("on.seed_present", isinstance(a.get("seed"), int), f"{a.get('seed')!r}")
check("on.seed_increments", b.get("seed") == a.get("seed") + 1,
      f"a={a.get('seed')} b={b.get('seed')}")
check("on.enable_thinking_false",
      (a.get("chat_template_kwargs") or {}).get("enable_thinking") is False,
      f"{a.get('chat_template_kwargs')!r}")

# 2) Envelope OFF: passthrough (no pins injected, no seed).
c, _ = run({}, "off")
check("off.temperature_untouched", c.get("temperature") == 0, f"{c.get('temperature')!r}")
check("off.no_top_k", "top_k" not in c, f"{c.get('top_k')!r}")
check("off.no_seed", "seed" not in c, f"{c.get('seed')!r}")

if fail:
    print(f"\n{len(fail)} FAILURE(S): {fail}")
    raise SystemExit(1)
print("\nPROXY LIVE-PROCESS ENVELOPE TEST PASSED")
