"""GPU telemetry via nvidia-smi sampling."""
from __future__ import annotations
import subprocess
import threading
import time
import json
import hashlib
from pathlib import Path
from typing import Optional

def _run(cmd: list[str], timeout: float = 2) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, timeout=timeout).decode().strip()
        return out
    except Exception:
        return None

def query_nvidia_smi():
    try:
        out = _run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,name,driver_version",
             "--format=csv,noheader,nounits"],
            timeout=2,
        )
        if not out:
            raise RuntimeError("no output")
        # name and driver contain commas? last two fields may contain commas in name, but we requested name last -> split from left
        # format: 1242, 24576, 0, 121.00, 53, NVIDIA GeForce RTX 3090, 590.57
        # split by comma: first 5 numeric, rest is name+driver
        parts = [p.strip() for p in out.split(",")]
        # first 5 are numeric, remaining are name parts + driver
        if len(parts) < 7:
            raise RuntimeError(f"unexpected parts {parts}")
        return {
            "vram_used_mb": float(parts[0]),
            "vram_total_mb": float(parts[1]),
            "gpu_util": float(parts[2]),
            "power_w": float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else None,
            "temp_c": float(parts[4]),
            "gpu_name": ",".join(parts[5:-1]).strip(),
            "driver_version": parts[-1].strip(),
            "ts": time.monotonic(),
        }
    except Exception as e:
        return {"error": str(e), "ts": time.monotonic()}

def get_cuda_version() -> Optional[str]:
    out = _run(["nvcc", "--version"], timeout=2)
    if out:
        for line in out.splitlines():
            if "release" in line:
                return line.strip()
    out = _run(["nvidia-smi"], timeout=2)
    if out and "CUDA Version" in out:
        for line in out.splitlines():
            if "CUDA Version" in line:
                return line.strip()
    return None

def get_git_revision() -> Optional[str]:
    out = _run(["git", "rev-parse", "HEAD"], timeout=2)
    return out.strip() if out else None

def file_hash(path: str | Path, max_bytes: int = 10_000_000) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            # hash first max_bytes to avoid hashing 17GB fully
            chunk = f.read(max_bytes)
            h.update(chunk)
            # also include file size
            h.update(str(p.stat().st_size).encode())
        return h.hexdigest()[:16]
    except Exception:
        return None

class GPUTelemetry:
    """Samples nvidia-smi at interval_ms and tracks peak VRAM."""
    def __init__(self, interval_ms: int = 200, log_path: Optional[Path] = None):
        self.interval = interval_ms / 1000.0
        self.log_path = Path(log_path) if log_path else None
        self.samples: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peak_vram = 0.0
        self._start_monotonic: Optional[float] = None

    def start(self):
        self._running = True
        self.samples = []
        self._peak_vram = 0.0
        self._start_monotonic = time.monotonic()
        # truncate log
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                open(self.log_path, "w").close()
            except: pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            s = query_nvidia_smi()
            # add elapsed since start
            if self._start_monotonic is not None:
                s["elapsed_s"] = round(s["ts"] - self._start_monotonic, 3)
            if "vram_used_mb" in s:
                self._peak_vram = max(self._peak_vram, float(s["vram_used_mb"]))
            self.samples.append(s)
            if self.log_path:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(s) + "\n")
                except: pass
            time.sleep(self.interval)

    def peak_vram_mb(self) -> float:
        return float(self._peak_vram)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0, "peak_vram_mb": self._peak_vram}
        vals = [s for s in self.samples if "vram_used_mb" in s]
        if not vals:
            return {"samples": len(self.samples), "peak_vram_mb": self._peak_vram, "error": self.samples[0].get("error")}
        avg_util = sum(float(s["gpu_util"]) for s in vals) / len(vals) if vals else 0
        avg_power = sum(float(s["power_w"]) for s in vals if s.get("power_w") is not None) / max(1, len([s for s in vals if s.get("power_w") is not None]))
        avg_temp = sum(float(s["temp_c"]) for s in vals) / len(vals) if vals else 0
        return {
            "samples": len(vals),
            "peak_vram_mb": self._peak_vram,
            "avg_gpu_util": round(avg_util, 1),
            "avg_power_w": round(avg_power, 1) if avg_power else None,
            "avg_temp_c": round(avg_temp, 1),
            "last": vals[-1] if vals else None,
        }

def snapshot_vram() -> Optional[float]:
    s = query_nvidia_smi()
    return s.get("vram_used_mb")
