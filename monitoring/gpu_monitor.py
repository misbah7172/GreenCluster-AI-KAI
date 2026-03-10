"""
GPU monitoring using NVIDIA NVML.

Samples GPU power draw, utilization, memory usage, and temperature at a
configurable interval (default 1 second) in a background thread.
"""

import csv
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import pynvml; if unavailable, the monitor will operate in
# fallback mode and return empty samples.
try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False
    logger.warning("pynvml not available — GPU monitoring disabled")


class GPUMonitor:
    """Background-thread GPU metrics sampler using NVIDIA NVML.

    Parameters
    ----------
    gpu_index : int
        Index of the GPU device to monitor (default 0).
    interval : float
        Sampling interval in seconds (default 1.0).
    """

    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.gpu_index = gpu_index
        self.interval = interval

        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._nvml_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize NVML and begin sampling in a background thread."""
        if not _NVML_AVAILABLE:
            logger.warning("NVML unavailable — start() is a no-op")
            return

        pynvml.nvmlInit()
        self._nvml_initialized = True
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

        device_name = pynvml.nvmlDeviceGetName(self._handle)
        logger.info("GPU monitor started: device %d (%s), interval %.1fs",
                     self.gpu_index, device_name, self.interval)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the sampling thread and shut down NVML."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3)
            self._thread = None

        if self._nvml_initialized:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False
            logger.info("GPU monitor stopped (%d samples collected)", len(self._samples))

    def get_samples(self) -> List[Dict[str, Any]]:
        """Return a copy of all collected samples."""
        with self._lock:
            return list(self._samples)

    def clear(self) -> None:
        """Discard all collected samples."""
        with self._lock:
            self._samples.clear()

    def export_csv(self, path: str) -> str:
        """Write samples to a CSV file.

        Returns the absolute path of the written file.
        """
        samples = self.get_samples()
        if not samples:
            logger.warning("No GPU samples to export")
            return ""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(samples[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        logger.info("Exported %d GPU samples to %s", len(samples), out)
        return str(out.resolve())

    def export_json(self, path: str) -> str:
        """Write samples to a JSON file.

        Returns the absolute path of the written file.
        """
        samples = self.get_samples()
        if not samples:
            logger.warning("No GPU samples to export")
            return ""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(samples, f, indent=2)

        logger.info("Exported %d GPU samples to %s", len(samples), out)
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Continuously sample GPU metrics until stopped."""
        while not self._stop_event.is_set():
            try:
                sample = self._read_metrics()
                with self._lock:
                    self._samples.append(sample)
            except Exception:
                logger.exception("Error reading GPU metrics")
            self._stop_event.wait(self.interval)

    def _read_metrics(self) -> Dict[str, Any]:
        """Read a single set of GPU metrics from NVML."""
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)       # milliwatts
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            self._handle, pynvml.NVML_TEMPERATURE_GPU
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_index": self.gpu_index,
            "power_w": round(power_mw / 1000.0, 2),
            "gpu_utilization_pct": util.gpu,
            "memory_used_mb": round(mem_info.used / (1024 * 1024), 2),
            "memory_total_mb": round(mem_info.total / (1024 * 1024), 2),
            "temperature_c": temp,
        }
