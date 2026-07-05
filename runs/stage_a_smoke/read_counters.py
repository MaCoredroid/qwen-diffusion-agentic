#!/usr/bin/env python3
"""Extract FLARE engine audit counters + health signals from a served vLLM log.

Reads the diffusion server.log emitted by runcage_diffusion.sh and surfaces the
Stage-A observability instrumentation (pin commit ef97e1e):

  * boot decode-mode line: "Qwen3_5FlareModelState ready: decode_mode=... "
      -> A-G1 gate: decode_mode MUST be hybrid_clean (not canvas), route_verified.
  * per-request audit: "FLARE hybrid_clean req=... done: model_forwards=...
      forced_token_count=... value_tokens=... projected_value_tokens_exact=...
      generated_tokens=... stop_reason=..."
      -> the zero-value-projection tripwire: projected_value_tokens_exact MUST be 0.
  * prefix-cache hit-rate lines (APC reuse), and any ERROR/Traceback/HTTP-4xx/5xx
      that would signal tool-loop breakage.

Usage: read_counters.py <server.log> [--json]
"""
import json
import re
import sys

BOOT_RE = re.compile(r"Qwen3_5FlareModelState ready: (.*)")
REQ_RE = re.compile(
    r"FLARE hybrid_clean req=(?P<req>\S+) done: "
    r"model_forwards=(?P<model_forwards>\d+) "
    r"forced_token_count=(?P<forced_token_count>\d+) "
    r"value_tokens=(?P<value_tokens>\d+) "
    r"projected_value_tokens_exact=(?P<projected_value_tokens_exact>\d+) "
    r"generated_tokens=(?P<generated_tokens>\d+) "
    r"stop_reason=(?P<stop_reason>\S+)"
)
HIT_RE = re.compile(r"[Pp]refix cache hit rate[^\d]*([\d.]+)%")
GPU_HIT_RE = re.compile(r"GPU prefix cache hit rate[^\d]*([\d.]+)%")
ERR_RE = re.compile(r"\b(ERROR|Traceback|CRITICAL)\b")
HTTP_ERR_RE = re.compile(r'"(POST|GET) /v1/\S+ HTTP/1\.1" (4\d\d|5\d\d)')
UPSTREAM_ERR_RE = re.compile(r"upstream HTTP (4\d\d|5\d\d)")


def main():
    path = sys.argv[1]
    as_json = "--json" in sys.argv[2:]
    text = open(path, encoding="utf-8", errors="replace").read()

    boot = BOOT_RE.search(text)
    reqs = [m.groupdict() for m in REQ_RE.finditer(text)]
    for r in reqs:
        for k in ("model_forwards", "forced_token_count", "value_tokens",
                  "projected_value_tokens_exact", "generated_tokens"):
            r[k] = int(r[k])
    hit_rates = [float(x) for x in HIT_RE.findall(text)]
    gpu_hit_rates = [float(x) for x in GPU_HIT_RE.findall(text)]
    err_lines = [ln for ln in text.splitlines() if ERR_RE.search(ln)]
    http_errs = [f"{a} /v1 {b}" for (a, b) in HTTP_ERR_RE.findall(text)]

    n_req = len(reqs)
    proj_nonzero = [r for r in reqs if r["projected_value_tokens_exact"] != 0]
    total_forwards = sum(r["model_forwards"] for r in reqs)
    total_gen = sum(r["generated_tokens"] for r in reqs)
    stop_reasons = {}
    for r in reqs:
        stop_reasons[r["stop_reason"]] = stop_reasons.get(r["stop_reason"], 0) + 1

    summary = {
        "log": path,
        "boot_decode_line": boot.group(1) if boot else None,
        "decode_mode_hybrid_clean": bool(boot and "decode_mode=hybrid_clean" in boot.group(1)),
        "n_hybrid_clean_requests": n_req,
        "projected_value_tokens_exact_all_zero": (n_req > 0 and not proj_nonzero),
        "projected_value_tokens_exact_violations": len(proj_nonzero),
        "total_model_forwards": total_forwards,
        "total_generated_tokens": total_gen,
        "stop_reasons": stop_reasons,
        "prefix_cache_hit_rates_pct": hit_rates[-5:],
        "gpu_prefix_cache_hit_rates_pct": gpu_hit_rates[-5:],
        "n_error_lines": len(err_lines),
        "n_http_4xx_5xx": len(http_errs),
        "counters_clean": bool(
            boot and "decode_mode=hybrid_clean" in boot.group(1)
            and n_req > 0 and not proj_nonzero and not http_errs
        ),
    }
    if as_json:
        print(json.dumps({"summary": summary, "requests": reqs,
                          "error_lines_sample": err_lines[:10],
                          "http_errors_sample": http_errs[:10]}, indent=2))
    else:
        print(json.dumps(summary, indent=2))
        if err_lines:
            print("\n--- error/traceback lines (sample) ---")
            for ln in err_lines[:10]:
                print(ln)


if __name__ == "__main__":
    main()
