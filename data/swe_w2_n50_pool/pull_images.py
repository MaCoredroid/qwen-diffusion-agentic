#!/usr/bin/env python3
"""Pull the W2 N=50 official swebench images with disk-floor guard + accounting.

Records per-image {rc, wall_s, on_disk_bytes} incrementally (JSONL) and a final
summary {n_ok, n_fail, total_wall_s, disk_used_delta_gb, docker_images_size_gb}.
Docker is invoked via SWE_DOCKER_CMD (this box: 'sudo -A docker', SUDO_ASKPASS set).
"""
from __future__ import annotations
import concurrent.futures as cf
import json, os, shlex, shutil, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = (HERE / "images.txt").read_text().split()
JSONL = HERE / "pull_progress.jsonl"
SUMMARY = HERE / "pull_summary.json"
DOCKER = shlex.split(os.environ.get("SWE_DOCKER_CMD", "sudo -A docker"))
WORKERS = int(os.environ.get("PULL_WORKERS", "4"))
RETRIES = int(os.environ.get("PULL_RETRIES", "1"))
DISK_FLOOR_GB = float(os.environ.get("PULL_DISK_FLOOR_GB", "150"))  # abort below


def _df_avail_gb(path="/") -> float:
    t, u, f = shutil.disk_usage(path)
    return f / 1e9


def _df_used_gb(path="/") -> float:
    t, u, f = shutil.disk_usage(path)
    return u / 1e9


def _img_size_bytes(img: str) -> int:
    p = subprocess.run(DOCKER + ["image", "inspect", "--format", "{{.Size}}", img],
                       capture_output=True, text=True, timeout=120)
    try:
        return int(p.stdout.strip())
    except Exception:
        return -1


def _docker_images_size_gb() -> float:
    p = subprocess.run(DOCKER + ["system", "df", "--format", "{{.Type}}\t{{.Size}}"],
                       capture_output=True, text=True, timeout=120)
    for line in p.stdout.splitlines():
        if line.startswith("Images"):
            raw = line.split("\t", 1)[1].strip()
            # e.g. "93.93GB" / "512MB" / "1.2kB"
            for suf, mul in (("GB", 1), ("MB", 1e-3), ("kB", 1e-6), ("B", 1e-9)):
                if raw.endswith(suf):
                    try:
                        return float(raw[:-len(suf)]) * mul
                    except Exception:
                        return -1.0
    return -1.0


def pull_one(img: str) -> dict:
    last = None
    for attempt in range(RETRIES + 1):
        t0 = time.time()
        p = subprocess.run(DOCKER + ["pull", img], capture_output=True, text=True,
                           timeout=3600)
        dt = time.time() - t0
        last = {"image": img, "attempt": attempt, "rc": p.returncode,
                "wall_s": round(dt, 1)}
        if p.returncode == 0:
            last["on_disk_bytes"] = _img_size_bytes(img)
            last["ok"] = True
            return last
        last["stderr_tail"] = p.stderr.strip()[-300:]
    last["ok"] = False
    return last


def main() -> None:
    started = time.time()
    disk_used_start = _df_used_gb()
    imgsize_start = _docker_images_size_gb()
    JSONL.write_text("")  # fresh
    results: dict[str, dict] = {}
    aborted = False

    # Guard: bail immediately if we're already under the floor.
    if _df_avail_gb() < DISK_FLOOR_GB:
        SUMMARY.write_text(json.dumps({
            "status": "ABORTED_PREFLIGHT_DISK",
            "df_avail_gb": round(_df_avail_gb(), 1),
            "floor_gb": DISK_FLOOR_GB}, indent=1))
        return

    lock = __import__("threading").Lock()

    def _log(rec: dict):
        with lock:
            with JSONL.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for img in IMAGES:
            # Disk-floor guard before scheduling each pull.
            if _df_avail_gb() < DISK_FLOOR_GB:
                aborted = True
                _log({"image": img, "skipped": "DISK_FLOOR",
                      "df_avail_gb": round(_df_avail_gb(), 1)})
                continue
            futs[ex.submit(pull_one, img)] = img
        for fut in cf.as_completed(futs):
            rec = fut.result()
            results[rec["image"]] = rec
            _log(rec)

    disk_used_end = _df_used_gb()
    imgsize_end = _docker_images_size_gb()
    ok = [r for r in results.values() if r.get("ok")]
    fail = [r for r in results.values() if not r.get("ok")]
    virt_sum = sum(r.get("on_disk_bytes", 0) for r in ok if r.get("on_disk_bytes", 0) > 0)
    summary = {
        "status": "ABORTED_DISK_FLOOR" if aborted else "DONE",
        "n_requested": len(IMAGES),
        "n_scheduled": len(results),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "failed_images": [r["image"] for r in fail],
        "total_wall_s": round(time.time() - started, 1),
        "sum_per_image_wall_s": round(sum(r.get("wall_s", 0) for r in results.values()), 1),
        "workers": WORKERS,
        "disk_used_delta_gb": round(disk_used_end - disk_used_start, 2),
        "docker_images_size_start_gb": round(imgsize_start, 2),
        "docker_images_size_end_gb": round(imgsize_end, 2),
        "docker_images_size_delta_gb": round(imgsize_end - imgsize_start, 2),
        "sum_per_image_virtual_gb": round(virt_sum / 1e9, 2),
        "df_avail_gb_end": round(_df_avail_gb(), 1),
        "disk_floor_gb": DISK_FLOOR_GB,
    }
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
