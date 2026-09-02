"""GPU telemetry via nvidia-smi sampling."""
from __future__ import annotations
import subprocess
import threading
import time
import json
from pathlib import Path
from typing import Optional

def query_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "vram_used_mb": float(parts[0]),
            "vram_total_mb": float(parts[1]),
            "gpu_util": float(parts[2]),
            "power_w": float(parts[3]) if parts[3] != "[N/A]" else None,
            "temp_c": float(parts[4]),
            "ts": time.monotonic(),
        }
    except Exception as e:
        return {"error": str(e), "ts": time.monotonic()}

class GPUTelemetry:
    def __init__(self, interval_ms: int = 200, log_path: Optional[Path] = None):
        self.interval = interval_ms / 1000.0
        self.log_path = Path(log_path) if log_path else None
        self.samples = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peak_vram = 0

    def start(self):
        self._running = True
        self.samples = []
        self._peak_vram = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            s = query_nvidia_smi()
            if "vram_used_mb" in s:
                self._peak_vram = max(self._peak_vram, s["vram_used_mb"])
            self.samples.append(s)
            if self.log_path:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(s) + "\n")
                except:
                    pass
            time.sleep(self.interval)

    def peak_vram_mb(self) -> float:
        return self._peak_vram

    def summary(self) -> dict:
        if not self.samples:
            return {}
        vals = [s for s in self.samples if "vram_used_mb" in s]
        if not vals:
            return {"samples": len(self.samples)}
        avg_util = sum(s["gpu_util"] for s in vals) / len(vals)
        return {
            "samples": len(vals),
            "peak_vram_mb": self._peak_vram,
            "avg_gpu_util": avg_util,
            "last": vals[-1] if vals else None,
        }

def snapshot_vram() -> Optional[float]:
    s = query_nvidia_smi()
    return s.get("vram_used_mb")
