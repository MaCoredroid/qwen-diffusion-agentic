"""Shared GPU-util / mem sampler + host-RAM peak helpers for the batched-rollout
throughput bench. Background thread polls `nvidia-smi` at ~4 Hz so utilization is
sampled ACROSS each timed batch's whole wall (not one mid-wave snapshot). Host-RAM
peak is read from getrusage (ru_maxrss, KB on Linux) -- the true process peak,
which is what the RAM-cage budget cares about.
"""
import resource
import subprocess
import threading
import time


class GpuSampler:
    def __init__(self, interval=0.25):
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thr = None
        self.util = []
        self.mem = []

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
                u, m = out.split(",")
                self.util.append(int(u))
                self.mem.append(int(m))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self.util = []
        self.mem = []
        self._stop.clear()
        self._thr = threading.Thread(target=self._poll, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)

    def summary(self):
        def pct(a, p):
            if not a:
                return None
            s = sorted(a)
            i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
            return s[i]
        return {
            "gpu_util_mean_pct": round(sum(self.util) / len(self.util), 1) if self.util else None,
            "gpu_util_p50_pct": pct(self.util, 50),
            "gpu_util_p90_pct": pct(self.util, 90),
            "gpu_util_max_pct": max(self.util) if self.util else None,
            "gpu_util_n_samples": len(self.util),
            "gpu_mem_peak_mb": max(self.mem) if self.mem else None,
        }


def host_ram_peak_gb():
    # ru_maxrss is in KiB on Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 3)


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
        u, m = out.split(",")
        return int(u), int(m)
    except Exception:
        return None, None
