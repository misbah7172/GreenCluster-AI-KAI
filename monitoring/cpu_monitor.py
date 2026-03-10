"""
CPU monitoring using psutil.

Samples overall and per-core CPU utilization at a configurable interval
(default 1 second) in a background thread.
"""

import csv
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class CPUMonitor:
    """Background-thread CPU utilization sampler.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 1.0).
    per_core : bool
        If True, record per-core utilization in addition to overall (default True).
    """

    def __init__(self, interval: float = 1.0, per_core: bool = True):
        self.interval = interval
        self.per_core = per_core

        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin sampling CPU utilization in a background thread."""
        # Prime psutil's internal counter so the first real sample is meaningful.
        psutil.cpu_percent(percpu=self.per_core)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        core_count = psutil.cpu_count(logical=True)
        logger.info("CPU monitor started: %d logical cores, interval %.1fs",
                     core_count, self.interval)

    def stop(self) -> None:
        """Stop the sampling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3)
            self._thread = None
        logger.info("CPU monitor stopped (%d samples collected)", len(self._samples))

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
            logger.warning("No CPU samples to export")
            return ""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(samples[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        logger.info("Exported %d CPU samples to %s", len(samples), out)
        return str(out.resolve())

    def export_json(self, path: str) -> str:
        """Write samples to a JSON file.

        Returns the absolute path of the written file.
        """
        samples = self.get_samples()
        if not samples:
            logger.warning("No CPU samples to export")
            return ""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(samples, f, indent=2)

        logger.info("Exported %d CPU samples to %s", len(samples), out)
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Continuously sample CPU utilization until stopped."""
        while not self._stop_event.is_set():
            try:
                sample = self._read_metrics()
                with self._lock:
                    self._samples.append(sample)
            except Exception:
                logger.exception("Error reading CPU metrics")
            self._stop_event.wait(self.interval)

    def _read_metrics(self) -> Dict[str, Any]:
        """Read a single CPU utilization snapshot."""
        overall = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_utilization_pct": overall,
            "memory_used_mb": round(mem.used / (1024 * 1024), 2),
            "memory_total_mb": round(mem.total / (1024 * 1024), 2),
            "memory_pct": mem.percent,
        }

        if self.per_core:
            per_core = psutil.cpu_percent(percpu=True)
            for i, pct in enumerate(per_core):
                result[f"core_{i}_pct"] = pct

        return result
