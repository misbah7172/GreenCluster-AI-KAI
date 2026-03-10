"""
Unified metrics collector.

Combines GPU and CPU monitors with inference-level metrics (latency,
throughput) into a single ``MetricsCollector`` that can be started /
stopped around an experiment run and exports structured results.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from monitoring.gpu_monitor import GPUMonitor
from monitoring.cpu_monitor import CPUMonitor

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Orchestrates GPU + CPU monitoring and records inference events.

    Parameters
    ----------
    gpu_index : int
        GPU device index to monitor.
    interval : float
        Sampling interval in seconds for both GPU and CPU monitors.
    enable_gpu : bool
        Set to False to skip GPU monitoring (e.g. when no NVIDIA GPU).
    enable_cpu : bool
        Set to False to skip CPU monitoring.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        interval: float = 1.0,
        enable_gpu: bool = True,
        enable_cpu: bool = True,
    ):
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu

        self._gpu_monitor: Optional[GPUMonitor] = None
        self._cpu_monitor: Optional[CPUMonitor] = None

        if enable_gpu:
            self._gpu_monitor = GPUMonitor(gpu_index=gpu_index, interval=interval)
        if enable_cpu:
            self._cpu_monitor = CPUMonitor(interval=interval)

        # Inference-level measurements
        self._inference_latencies: List[float] = []   # milliseconds
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all monitors and record the experiment start time."""
        self._start_time = time.perf_counter()
        if self._gpu_monitor:
            self._gpu_monitor.start()
        if self._cpu_monitor:
            self._cpu_monitor.start()
        logger.info("MetricsCollector started")

    def stop(self) -> None:
        """Stop all monitors and record the experiment end time."""
        self._end_time = time.perf_counter()
        if self._gpu_monitor:
            self._gpu_monitor.stop()
        if self._cpu_monitor:
            self._cpu_monitor.stop()
        logger.info("MetricsCollector stopped")

    # ------------------------------------------------------------------
    # Inference event recording
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float) -> None:
        """Record a single inference latency measurement (ms)."""
        self._inference_latencies.append(latency_ms)

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_gpu_samples(self) -> List[Dict[str, Any]]:
        if self._gpu_monitor:
            return self._gpu_monitor.get_samples()
        return []

    def get_cpu_samples(self) -> List[Dict[str, Any]]:
        if self._cpu_monitor:
            return self._cpu_monitor.get_samples()
        return []

    def get_inference_latencies(self) -> List[float]:
        return list(self._inference_latencies)

    # ------------------------------------------------------------------
    # Computed summary
    # ------------------------------------------------------------------

    def compute_summary(self, execution_mode: str = "local") -> Dict[str, Any]:
        """Compute aggregate metrics from the collected data.

        Parameters
        ----------
        execution_mode : str
            ``"local"`` or ``"kubernetes"`` — stored in the result.

        Returns
        -------
        dict
            Summary containing averages, totals, and raw sample lists.
        """
        total_runtime_s = 0.0
        if self._start_time is not None and self._end_time is not None:
            total_runtime_s = self._end_time - self._start_time

        # --- GPU aggregates ---
        gpu_samples = self.get_gpu_samples()
        avg_power_w = 0.0
        avg_gpu_util = 0.0
        avg_mem_used_mb = 0.0
        avg_temp_c = 0.0
        if gpu_samples:
            avg_power_w = sum(s["power_w"] for s in gpu_samples) / len(gpu_samples)
            avg_gpu_util = sum(s["gpu_utilization_pct"] for s in gpu_samples) / len(gpu_samples)
            avg_mem_used_mb = sum(s["memory_used_mb"] for s in gpu_samples) / len(gpu_samples)
            avg_temp_c = sum(s["temperature_c"] for s in gpu_samples) / len(gpu_samples)

        # Total energy: average_power_W * runtime_s / 3600 = Wh
        total_energy_wh = avg_power_w * total_runtime_s / 3600.0

        # --- CPU aggregates ---
        cpu_samples = self.get_cpu_samples()
        avg_cpu_util = 0.0
        if cpu_samples:
            avg_cpu_util = sum(s["cpu_utilization_pct"] for s in cpu_samples) / len(cpu_samples)

        # --- Inference aggregates ---
        latencies = self.get_inference_latencies()
        num_inferences = len(latencies)
        avg_latency_ms = 0.0
        throughput = 0.0
        energy_per_inference_wh = 0.0
        if latencies:
            avg_latency_ms = sum(latencies) / len(latencies)
        if total_runtime_s > 0 and num_inferences > 0:
            throughput = num_inferences / total_runtime_s
        if num_inferences > 0:
            energy_per_inference_wh = total_energy_wh / num_inferences

        summary: Dict[str, Any] = {
            "execution_mode": execution_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_runtime_s": round(total_runtime_s, 4),
            "num_inferences": num_inferences,

            # GPU
            "avg_power_w": round(avg_power_w, 2),
            "avg_gpu_utilization_pct": round(avg_gpu_util, 2),
            "avg_gpu_memory_used_mb": round(avg_mem_used_mb, 2),
            "avg_gpu_temperature_c": round(avg_temp_c, 2),
            "total_energy_wh": round(total_energy_wh, 6),
            "energy_per_inference_wh": round(energy_per_inference_wh, 8),

            # CPU
            "avg_cpu_utilization_pct": round(avg_cpu_util, 2),

            # Inference
            "avg_latency_ms": round(avg_latency_ms, 4),
            "throughput_inferences_per_sec": round(throughput, 4),

            # Raw data for downstream analysis
            "gpu_samples": gpu_samples,
            "cpu_samples": cpu_samples,
            "inference_latencies_ms": latencies,
        }

        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(
        self,
        output_dir: str,
        filename: Optional[str] = None,
        execution_mode: str = "local",
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute summary and write it to a JSON file.

        Parameters
        ----------
        output_dir : str
            Directory to write the result file into.
        filename : str, optional
            Custom filename.  Defaults to ``<mode>_results_<timestamp>.json``.
        execution_mode : str
            ``"local"`` or ``"kubernetes"``.
        extra_data : dict, optional
            Additional key-value pairs to merge into the summary before
            writing (e.g. experiment configuration).

        Returns
        -------
        str
            Absolute path of the written JSON file.
        """
        summary = self.compute_summary(execution_mode=execution_mode)
        if extra_data:
            summary.update(extra_data)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{execution_mode}_results_{ts}.json"

        filepath = out / filename
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Exported results to %s", filepath)
        return str(filepath.resolve())
