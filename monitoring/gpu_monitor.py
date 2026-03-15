"""
GPU monitoring using NVIDIA NVML.

Samples GPU power draw, utilization, memory usage, and temperature at a
configurable interval (default 1 second) in a background thread.

Supports high-frequency sampling (e.g. 100ms) with a bounded ring buffer
for transient spike capture, TDP-relative threshold evaluation, and an
optional spike callback.
"""

import collections
import csv
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
        Sampling interval in seconds (default 1.0).  Sub-second values
        (e.g. 0.1 for 100 ms) are supported for high-frequency mode.
    buffer_size : int
        Maximum number of samples kept in the ring buffer (default 600,
        which equals 60 s at 100 ms interval).
    tdp_watts : float
        Thermal Design Power of the GPU in watts.  Set to 0 (default) to
        auto-detect from NVML at :meth:`start` time.
    spike_callback : callable, optional
        Called with the sample dict whenever instantaneous power reaches
        or exceeds 80 % of TDP.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        interval: float = 1.0,
        buffer_size: int = 600,
        tdp_watts: float = 0.0,
        spike_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.gpu_index = gpu_index
        self.interval = interval

        self._samples: List[Dict[str, Any]] = []
        self._ring_buffer: collections.deque = collections.deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._nvml_initialized = False

        self._tdp_watts = tdp_watts
        self._spike_callback = spike_callback
        self._spike_threshold_pct = 80.0  # % of TDP that triggers spike

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

        # Auto-detect TDP if not provided
        if self._tdp_watts <= 0.0:
            self._tdp_watts = self._detect_tdp()

        logger.info(
            "GPU monitor started: device %d (%s), interval %.2fs, TDP %.1fW",
            self.gpu_index, device_name, self.interval, self._tdp_watts,
        )

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

    def get_recent_samples(self, n: int = 0) -> List[Dict[str, Any]]:
        """Return the last *n* samples from the ring buffer.

        Parameters
        ----------
        n : int
            Number of recent samples to return.  ``0`` (default) returns
            all samples currently in the ring buffer.
        """
        with self._lock:
            if n <= 0 or n >= len(self._ring_buffer):
                return list(self._ring_buffer)
            return list(self._ring_buffer)[-n:]

    def get_tdp_watts(self) -> float:
        """Return TDP in watts (auto-detected or user-configured)."""
        return self._tdp_watts

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
                    self._ring_buffer.append(sample)

                # Fire spike callback when power >= spike threshold
                if (
                    self._spike_callback is not None
                    and self._tdp_watts > 0
                    and sample["power_w"] >= self._tdp_watts * (self._spike_threshold_pct / 100.0)
                ):
                    try:
                        self._spike_callback(sample)
                    except Exception:
                        logger.exception("Error in spike callback")
            except Exception:
                logger.exception("Error reading GPU metrics")
            self._stop_event.wait(self.interval)

    def _detect_tdp(self) -> float:
        """Auto-detect TDP from NVML power management limit.

        Returns TDP in watts, or 0.0 if detection fails.
        """
        try:
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)
            tdp_w = limit_mw / 1000.0
            logger.info("Auto-detected TDP: %.1fW", tdp_w)
            return tdp_w
        except Exception:
            logger.warning("Could not auto-detect TDP; threshold features disabled")
            return 0.0

    def _read_metrics(self) -> Dict[str, Any]:
        """Read a single set of GPU metrics from NVML."""
        power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)       # milliwatts
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            self._handle, pynvml.NVML_TEMPERATURE_GPU
        )

        power_w = round(power_mw / 1000.0, 2)

        # TDP-relative percentage (None when TDP is unknown)
        tdp_pct = None
        if self._tdp_watts > 0:
            tdp_pct = round(power_w / self._tdp_watts * 100.0, 2)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_index": self.gpu_index,
            "power_w": power_w,
            "gpu_utilization_pct": util.gpu,
            "memory_used_mb": round(mem_info.used / (1024 * 1024), 2),
            "memory_total_mb": round(mem_info.total / (1024 * 1024), 2),
            "temperature_c": temp,
            "tdp_pct": tdp_pct,
        }
