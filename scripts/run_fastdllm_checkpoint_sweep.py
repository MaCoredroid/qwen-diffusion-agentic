#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
PYTHON = ROOT / ".venv-lmeval/bin/python"
EVAL = ROOT / "fast-dllm/v2/eval.py"
BASE = ROOT / "models/qwen2.5-1.5b-fastdllm-init"
LORA_ROOT = ROOT / "runs/fastdllm_qwen25_1p5b_alpaca_lora_full"
OUT_ROOT = ROOT / "runs/lmeval_checkpoint_sweep"


def checkpoint_specs():
    specs = [
        {
            "name": "local_base",
            "model_path": str(BASE),
            "adapter_path": None,
            "step": 0,
            "kind": "local_base",
        }
    ]
    for step in [8000, 8500, 9000, 9500, 9575]:
        specs.append(
            {
                "name": f"lora_{step}",
                "model_path": str(BASE),
                "adapter_path": str(LORA_ROOT / f"checkpoint-{step}/adapter_model"),
                "step": step,
                "kind": "lora_checkpoint",
            }
        )
    specs.extend(
        [
            {
                "name": "lora_final",
                "model_path": str(BASE),
                "adapter_path": str(LORA_ROOT),
                "step": 9575,
                "kind": "lora_final",
            },
            {
                "name": "released_fastdllm_v2_1p5b",
                "model_path": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
                "adapter_path": None,
                "step": None,
                "kind": "reference",
            },
        ]
    )
    return specs


def task_args(task):
    args = ["--tasks", task, "--limit", "10", "--batch_size", "4"]
    if task == "gsm8k":
        args.extend(["--num_fewshot", "0"])
    return args


def model_args(spec):
    parts = [
        f"model_path={spec['model_path']}",
        "max_new_tokens=256",
        "threshold=0.9",
        "show_speed=True",
    ]
    if spec["adapter_path"]:
        parts.append(f"adapter_path={spec['adapter_path']}")
        parts.append(f"tokenizer_path={spec['model_path']}")
    return ",".join(parts)


def latest_result_json(run_dir):
    files = sorted(run_dir.glob("**/results_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    tps = None
    toks = None
    seconds = None
    for match in re.finditer(r"Tokens per second:\s*([0-9.]+)", text):
        tps = float(match.group(1))
    for match in re.finditer(r"Total number of tokens generated:\s*([0-9.]+)", text):
        toks = float(match.group(1))
    for match in re.finditer(r"Total time taken:\s*([0-9.]+) seconds", text):
        seconds = float(match.group(1))
    return {"tokens_per_second": tps, "generated_tokens": toks, "generation_seconds": seconds}


def extract_metrics(result_path):
    data = json.loads(result_path.read_text(encoding="utf-8"))
    task, metrics = next(iter(data["results"].items()))
    row = {"task": task, "result_path": str(result_path)}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            row[key] = value
    samples = data.get("n-samples") or data.get("n_samples") or {}
    if task in samples:
        row["effective_samples"] = samples[task].get("effective")
        row["original_samples"] = samples[task].get("original")
    return row


def write_summaries(rows):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "checkpoint_sweep_summary.json"
    csv_path = OUT_ROOT / "checkpoint_sweep_summary.csv"
    md_path = OUT_ROOT / "checkpoint_sweep_summary.md"

    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    keys = [
        "name",
        "kind",
        "step",
        "task",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
        "tokens_per_second",
        "generated_tokens",
        "generation_seconds",
        "status",
        "result_path",
        "log_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = [
        "# Fast-dLLM Checkpoint Sweep",
        "",
        "Limited smoke evals: `--limit 10`, `batch_size=4`, `max_new_tokens=256`, `threshold=0.9`.",
        "",
        "| Model | Task | GSM8K flex | GSM8K strict | IFEval prompt strict | IFEval inst strict | tok/s | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {task} | {gsm_flex} | {gsm_strict} | {ifeval_prompt} | {ifeval_inst} | {tps} | {status} |".format(
                name=row["name"],
                task=row.get("task", ""),
                gsm_flex=fmt(row.get("exact_match,flexible-extract")),
                gsm_strict=fmt(row.get("exact_match,strict-match")),
                ifeval_prompt=fmt(row.get("prompt_level_strict_acc,none")),
                ifeval_inst=fmt(row.get("inst_level_strict_acc,none")),
                tps=fmt(row.get("tokens_per_second")),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "Full machine-readable outputs:",
            "",
            f"- JSON: `{json_path}`",
            f"- CSV: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def run_one(spec, task, resume):
    run_dir = OUT_ROOT / spec["name"] / task
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    existing = latest_result_json(run_dir)
    if resume and existing:
        row = extract_metrics(existing)
        row.update(parse_log(log_path))
        row.update({**spec, "task": task, "status": "cached", "log_path": str(log_path)})
        return row

    cmd = [
        str(PYTHON),
        str(EVAL),
        *task_args(task),
        "--model",
        "fast_dllm_v2",
        "--model_args",
        model_args(spec),
        "--confirm_run_unsafe_code",
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--output_path",
        str(run_dir),
        "--log_samples",
    ]

    env = os.environ.copy()
    env.update({"HF_ALLOW_CODE_EVAL": "1", "HF_DATASETS_TRUST_REMOTE_CODE": "true"})

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    result_path = latest_result_json(run_dir)
    log_metrics = parse_log(log_path)
    if proc.returncode != 0 or result_path is None:
        return {
            **spec,
            "task": task,
            "status": f"failed:{proc.returncode}",
            "elapsed_seconds": elapsed,
            "log_path": str(log_path),
        }
    row = extract_metrics(result_path)
    row.update(log_metrics)
    row.update({**spec, "task": task, "status": "ok", "elapsed_seconds": elapsed, "log_path": str(log_path)})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "ifeval"])
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not PYTHON.exists():
        raise SystemExit(f"Missing eval venv: {PYTHON}")

    specs = checkpoint_specs()
    if args.models:
        selected = set(args.models)
        specs = [spec for spec in specs if spec["name"] in selected]

    rows = []
    for spec in specs:
        for task in args.tasks:
            print(f"==> {spec['name']} {task}", flush=True)
            row = run_one(spec, task, resume=args.resume)
            rows.append(row)
            write_summaries(rows)
            print(f"    {row.get('status')} {row.get('result_path', row.get('log_path'))}", flush=True)

    json_path, csv_path, md_path = write_summaries(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    failures = [row for row in rows if not str(row.get("status", "")).startswith(("ok", "cached"))]
    if failures:
        print(json.dumps(failures, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
