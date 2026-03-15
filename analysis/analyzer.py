"""
Post-experiment analysis module.

Loads experiment result JSON files produced by the local and Kubernetes
runners, computes derived metrics, builds a side-by-side comparison
table, and exports a CSV summary.

Usage::

    python -m analysis.analyzer \\
        --local  logs/local_results_20260311_143000.json \\
        --k8s    logs/k8s_results_20260311_143500.json \\
        --output analysis/output

Or programmatically::

    from analysis.analyzer import ExperimentAnalyzer
    analyzer = ExperimentAnalyzer()
    analyzer.load_local("logs/local_results.json")
    analyzer.load_k8s("logs/k8s_results.json")
    table = analyzer.summary_table()
    analyzer.export_csv("analysis/output/summary.csv")
"""

import argparse
import csv
import json
import logging
import statistics
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _percentiles(values: List[float]) -> Dict[str, float]:
    """Compute p50, p90, p95, p99 for a list of values."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)

    def _pct(p: float) -> float:
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        return s[lo] + frac * (s[hi] - s[lo])

    return {
        "p50": round(_pct(0.50), 4),
        "p90": round(_pct(0.90), 4),
        "p95": round(_pct(0.95), 4),
        "p99": round(_pct(0.99), 4),
    }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ExperimentAnalyzer:
    """Analyse and compare local vs Kubernetes experiment results."""

    def __init__(self):
        self._local: Optional[Dict[str, Any]] = None
        self._k8s: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_local(self, path: str) -> Dict[str, Any]:
        """Load a local-mode result JSON file."""
        self._local = _load_json(path)
        logger.info("Loaded local results from %s", path)
        return self._local

    def load_k8s(self, path: str) -> Dict[str, Any]:
        """Load a Kubernetes-mode result JSON file."""
        self._k8s = _load_json(path)
        logger.info("Loaded K8s results from %s", path)
        return self._k8s

    def load_combined(self, path: str) -> None:
        """Load a combined experiment JSON (has 'local' and 'kubernetes' keys)."""
        data = _load_json(path)
        if "local" in data:
            self._local = data["local"]
        if "kubernetes" in data:
            self._k8s = data["kubernetes"]
        logger.info("Loaded combined results from %s", path)

    # ------------------------------------------------------------------
    # Per-mode derived metrics
    # ------------------------------------------------------------------

    def analyse_local(self) -> Dict[str, Any]:
        """Compute derived metrics for the local experiment.

        Returns
        -------
        dict
            Derived metric dict with latency percentiles, energy
            efficiency, performance-per-watt, etc.
        """
        if self._local is None:
            raise ValueError("No local results loaded")

        d = self._local
        latencies = d.get("inference_latencies_ms", [])
        pcts = _percentiles(latencies)

        avg_latency = d.get("avg_latency_ms", 0.0)
        avg_power = d.get("avg_power_w", 0.0)
        total_energy = d.get("total_energy_wh", 0.0)
        throughput = d.get("throughput_inferences_per_sec", 0.0)
        epi = d.get("energy_per_inference_wh", 0.0)
        num_inferences = d.get("num_inferences", len(latencies))
        runtime = d.get("total_runtime_s", 0.0)

        # Performance-per-watt: inferences/s per watt
        perf_per_watt = throughput / avg_power if avg_power > 0 else 0.0

        # Latency stddev
        lat_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        return {
            "mode": "local",
            "num_inferences": num_inferences,
            "total_runtime_s": round(runtime, 4),

            "avg_latency_ms": round(avg_latency, 4),
            "latency_std_ms": round(lat_std, 4),
            "latency_p50_ms": pcts["p50"],
            "latency_p90_ms": pcts["p90"],
            "latency_p95_ms": pcts["p95"],
            "latency_p99_ms": pcts["p99"],

            "throughput_inf_per_sec": round(throughput, 4),
            "avg_power_w": round(avg_power, 2),
            "total_energy_wh": round(total_energy, 6),
            "energy_per_inference_wh": round(epi, 8),
            "performance_per_watt": round(perf_per_watt, 6),

            "avg_gpu_utilization_pct": d.get("avg_gpu_utilization_pct", 0.0),
            "avg_gpu_memory_used_mb": d.get("avg_gpu_memory_used_mb", 0.0),
            "avg_gpu_temperature_c": d.get("avg_gpu_temperature_c", 0.0),
            "avg_cpu_utilization_pct": d.get("avg_cpu_utilization_pct", 0.0),
        }

    def analyse_k8s(self) -> Dict[str, Any]:
        """Compute derived metrics for the Kubernetes experiment.

        Returns
        -------
        dict
            Derived metric dict with latency percentiles, energy
            efficiency, performance-per-watt, network overhead, etc.
        """
        if self._k8s is None:
            raise ValueError("No K8s results loaded")

        d = self._k8s
        latencies = d.get("e2e_latencies_ms", [])
        pcts = _percentiles(latencies)

        cfg = d.get("experiment_config", {})
        avg_latency = d.get("avg_e2e_latency_ms", 0.0)
        avg_gw_latency = d.get("avg_gateway_latency_ms", 0.0)
        avg_net_overhead = d.get("avg_network_overhead_ms", 0.0)
        avg_power = d.get("total_avg_power_w", 0.0)
        total_energy = d.get("total_energy_wh", 0.0)
        throughput = d.get("throughput_inferences_per_sec", 0.0)
        epi = d.get("energy_per_inference_wh", 0.0)
        runtime = d.get("total_runtime_s", 0.0)
        num_inferences = cfg.get("successful_iterations", len(latencies))

        perf_per_watt = throughput / avg_power if avg_power > 0 else 0.0
        lat_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        return {
            "mode": "kubernetes",
            "num_chunks": cfg.get("num_chunks", 0),
            "num_inferences": num_inferences,
            "total_runtime_s": round(runtime, 4),

            "avg_latency_ms": round(avg_latency, 4),
            "avg_gateway_latency_ms": round(avg_gw_latency, 4),
            "avg_network_overhead_ms": round(avg_net_overhead, 4),
            "latency_std_ms": round(lat_std, 4),
            "latency_p50_ms": pcts["p50"],
            "latency_p90_ms": pcts["p90"],
            "latency_p95_ms": pcts["p95"],
            "latency_p99_ms": pcts["p99"],

            "throughput_inf_per_sec": round(throughput, 4),
            "avg_power_w": round(avg_power, 2),
            "total_energy_wh": round(total_energy, 6),
            "energy_per_inference_wh": round(epi, 8),
            "performance_per_watt": round(perf_per_watt, 6),

            "per_chunk_avg_latency_ms": d.get("per_chunk_avg_latency_ms", []),

            "migration_count": len(d.get("migration_events", [])),
            "cluster_eer": d.get("energy_efficiency_ratio", 0.0),
        }

    # ------------------------------------------------------------------
    # DEAS / migration helpers
    # ------------------------------------------------------------------

    def compute_migration_energy(
        self,
        power_samples: List[Dict[str, Any]],
        migration_start_ts: str,
        migration_end_ts: str,
    ) -> Dict[str, float]:
        """Compute energy consumed during a single migration event.

        Uses trapezoidal integration over GPU power samples that fall
        within the *[migration_start_ts, migration_end_ts]* window.

        Parameters
        ----------
        power_samples : list[dict]
            Each element must contain ``"timestamp"`` (ISO-8601 string)
            and ``"power_w"`` (float, watts).
        migration_start_ts, migration_end_ts : str
            ISO-8601 timestamps delimiting the migration window.

        Returns
        -------
        dict
            ``migration_energy_wh`` -- energy in watt-hours,
            ``migration_duration_s`` -- wall-clock seconds,
            ``avg_migration_power_w`` -- mean power during migration.
        """
        start = _dt.fromisoformat(migration_start_ts)
        end = _dt.fromisoformat(migration_end_ts)

        # Filter and sort samples within the migration window
        window = sorted(
            (
                (_dt.fromisoformat(s["timestamp"]), s["power_w"])
                for s in power_samples
                if start <= _dt.fromisoformat(s["timestamp"]) <= end
            ),
            key=lambda x: x[0],
        )

        migration_duration_s = (end - start).total_seconds()

        if len(window) < 2:
            # Not enough samples for integration
            avg_power = window[0][1] if window else 0.0
            energy_wh = avg_power * (migration_duration_s / 3600.0)
            return {
                "migration_energy_wh": round(energy_wh, 8),
                "migration_duration_s": round(migration_duration_s, 4),
                "avg_migration_power_w": round(avg_power, 4),
            }

        # Trapezoidal integration  (energy in watt-seconds, then -> Wh)
        energy_ws = 0.0
        for i in range(1, len(window)):
            dt_s = (window[i][0] - window[i - 1][0]).total_seconds()
            avg_p = (window[i][1] + window[i - 1][1]) / 2.0
            energy_ws += avg_p * dt_s

        energy_wh = energy_ws / 3600.0
        avg_power = energy_ws / migration_duration_s if migration_duration_s > 0 else 0.0

        return {
            "migration_energy_wh": round(energy_wh, 8),
            "migration_duration_s": round(migration_duration_s, 4),
            "avg_migration_power_w": round(avg_power, 4),
        }

    def compute_offloading_overhead(
        self,
        baseline_latencies: List[float],
        offloaded_latencies: List[float],
    ) -> Dict[str, float]:
        """Compare baseline vs offloaded inference latencies.

        Parameters
        ----------
        baseline_latencies : list[float]
            Latencies (ms) measured without offloading.
        offloaded_latencies : list[float]
            Latencies (ms) measured with offloading enabled.

        Returns
        -------
        dict
            ``baseline_avg_ms``, ``offloaded_avg_ms``,
            ``overhead_delta_ms`` (offloaded - baseline),
            ``overhead_pct`` (delta / baseline * 100).
        """
        baseline_avg = (
            statistics.mean(baseline_latencies) if baseline_latencies else 0.0
        )
        offloaded_avg = (
            statistics.mean(offloaded_latencies) if offloaded_latencies else 0.0
        )
        delta = offloaded_avg - baseline_avg
        pct = (delta / baseline_avg * 100.0) if baseline_avg > 0 else 0.0

        return {
            "baseline_avg_ms": round(baseline_avg, 4),
            "offloaded_avg_ms": round(offloaded_avg, 4),
            "overhead_delta_ms": round(delta, 4),
            "overhead_pct": round(pct, 4),
        }

    def analyse_deas(self) -> Dict[str, Any]:
        """Analyse Dynamic Energy-Aware Scheduling (DEAS) metrics.

        Reads ``migration_events`` from the loaded Kubernetes data and
        computes per-migration energy via :meth:`compute_migration_energy`.

        Returns
        -------
        dict
            ``migration_count`` -- number of migration events,
            ``total_migration_energy_wh`` -- summed migration energy,
            ``avg_energy_recovery_s`` -- mean recovery time after
            migrations, ``cluster_eer`` -- cluster energy-efficiency
            ratio from the K8s data.

        Raises
        ------
        ValueError
            If no K8s results have been loaded.
        """
        if self._k8s is None:
            raise ValueError("No K8s results loaded")

        migration_events = self._k8s.get("migration_events", [])
        cluster_eer = self._k8s.get("energy_efficiency_ratio", 0.0)

        total_energy_wh = 0.0
        recovery_times: List[float] = []

        for event in migration_events:
            start_ts = event.get("start_ts")
            end_ts = event.get("end_ts")
            gpu_samples = event.get("gpu_power_samples", [])

            if start_ts and end_ts and gpu_samples:
                result = self.compute_migration_energy(
                    gpu_samples, start_ts, end_ts,
                )
                total_energy_wh += result["migration_energy_wh"]

            recovery_s = event.get("energy_recovery_s")
            if recovery_s is not None:
                recovery_times.append(float(recovery_s))

        avg_recovery = (
            statistics.mean(recovery_times) if recovery_times else 0.0
        )

        return {
            "migration_count": len(migration_events),
            "total_migration_energy_wh": round(total_energy_wh, 8),
            "avg_energy_recovery_s": round(avg_recovery, 4),
            "cluster_eer": round(cluster_eer, 6),
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self) -> Dict[str, Any]:
        """Build a side-by-side comparison of local and K8s metrics.

        Returns
        -------
        dict
            Keyed by metric name, each value has ``local``, ``kubernetes``,
            and ``ratio_k8s_over_local``.

        Raises
        ------
        ValueError
            If either result set has not been loaded.
        """
        la = self.analyse_local()
        ka = self.analyse_k8s()

        def _ratio(k: float, l: float) -> Optional[float]:
            if l == 0:
                return None
            return round(k / l, 4)

        metrics = [
            ("avg_latency_ms",           la["avg_latency_ms"],           ka["avg_latency_ms"]),
            ("latency_p50_ms",           la["latency_p50_ms"],           ka["latency_p50_ms"]),
            ("latency_p95_ms",           la["latency_p95_ms"],           ka["latency_p95_ms"]),
            ("latency_p99_ms",           la["latency_p99_ms"],           ka["latency_p99_ms"]),
            ("throughput_inf_per_sec",   la["throughput_inf_per_sec"],   ka["throughput_inf_per_sec"]),
            ("avg_power_w",              la["avg_power_w"],              ka["avg_power_w"]),
            ("total_energy_wh",          la["total_energy_wh"],          ka["total_energy_wh"]),
            ("energy_per_inference_wh",  la["energy_per_inference_wh"],  ka["energy_per_inference_wh"]),
            ("performance_per_watt",     la["performance_per_watt"],     ka["performance_per_watt"]),
        ]

        comparison = {}
        for name, lv, kv in metrics:
            comparison[name] = {
                "local": lv,
                "kubernetes": kv,
                "ratio_k8s_over_local": _ratio(kv, lv),
            }

        # K8s-specific extras
        comparison["network_overhead_ms"] = {
            "local": 0.0,
            "kubernetes": ka["avg_network_overhead_ms"],
        }

        return comparison

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self) -> List[Tuple[str, str, str, str]]:
        """Return a list of (metric, local, k8s, ratio) string tuples.

        Suitable for printing or writing to CSV.
        """
        comp = self.compare()
        rows: List[Tuple[str, str, str, str]] = [
            ("Metric", "Local", "Kubernetes", "Ratio (K8s/Local)"),
        ]

        display = [
            ("Avg Latency (ms)",            "avg_latency_ms"),
            ("Latency P50 (ms)",            "latency_p50_ms"),
            ("Latency P95 (ms)",            "latency_p95_ms"),
            ("Latency P99 (ms)",            "latency_p99_ms"),
            ("Throughput (inf/s)",           "throughput_inf_per_sec"),
            ("Avg GPU Power (W)",           "avg_power_w"),
            ("Total Energy (Wh)",           "total_energy_wh"),
            ("Energy/Inference (Wh)",       "energy_per_inference_wh"),
            ("Performance/Watt (inf/s/W)",  "performance_per_watt"),
            ("Network Overhead (ms)",       "network_overhead_ms"),
        ]

        for label, key in display:
            entry = comp[key]
            local_val = entry["local"]
            k8s_val = entry["kubernetes"]
            ratio = entry.get("ratio_k8s_over_local")
            ratio_str = f"{ratio:.4f}" if ratio is not None else "N/A"
            rows.append((label, f"{local_val}", f"{k8s_val}", ratio_str))

        return rows

    def print_summary(self) -> None:
        """Print the summary table to the logger."""
        rows = self.summary_table()
        col_widths = [
            max(len(r[i]) for r in rows) for i in range(4)
        ]
        fmt = "  {:<{w0}}  {>{w1}}  {>{w2}}  {>{w3}}"

        logger.info("=" * (sum(col_widths) + 10))
        for row in rows:
            logger.info(
                fmt.format(*row, w0=col_widths[0], w1=col_widths[1],
                           w2=col_widths[2], w3=col_widths[3]),
            )
            if row == rows[0]:
                logger.info("  " + "-" * (sum(col_widths) + 6))
        logger.info("=" * (sum(col_widths) + 10))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: str) -> str:
        """Export the summary table to a CSV file.

        Parameters
        ----------
        path : str
            Output CSV file path.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        rows = self.summary_table()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        logger.info("Summary CSV written to %s", out)
        return str(out.resolve())

    def export_json(self, path: str) -> str:
        """Export the full comparison dict to a JSON file.

        Parameters
        ----------
        path : str
            Output JSON file path.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        data = {
            "local_analysis": self.analyse_local(),
            "k8s_analysis": self.analyse_k8s(),
            "comparison": self.compare(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Analysis JSON written to %s", out)
        return str(out.resolve())

    # ------------------------------------------------------------------
    # Accessors for raw data (used by plots.py)
    # ------------------------------------------------------------------

    def get_local_gpu_samples(self) -> List[Dict[str, Any]]:
        if self._local is None:
            return []
        return self._local.get("gpu_samples", [])

    def get_local_cpu_samples(self) -> List[Dict[str, Any]]:
        if self._local is None:
            return []
        return self._local.get("cpu_samples", [])

    def get_local_latencies(self) -> List[float]:
        if self._local is None:
            return []
        return self._local.get("inference_latencies_ms", [])

    def get_k8s_latencies(self) -> List[float]:
        if self._k8s is None:
            return []
        return self._k8s.get("e2e_latencies_ms", [])

    def get_k8s_node_metrics(self) -> List[Dict[str, Any]]:
        if self._k8s is None:
            return []
        return self._k8s.get("node_metrics", [])

    def get_k8s_inference_results(self) -> List[Dict[str, Any]]:
        if self._k8s is None:
            return []
        return self._k8s.get("inference_results", [])

    @property
    def local_data(self) -> Optional[Dict[str, Any]]:
        return self._local

    @property
    def k8s_data(self) -> Optional[Dict[str, Any]]:
        return self._k8s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse and compare experiment results",
    )
    parser.add_argument(
        "--local", type=str, default=None,
        help="Path to local results JSON",
    )
    parser.add_argument(
        "--k8s", type=str, default=None,
        help="Path to K8s results JSON",
    )
    parser.add_argument(
        "--combined", type=str, default=None,
        help="Path to combined experiment JSON (has 'local' and 'kubernetes' keys)",
    )
    parser.add_argument(
        "--output", type=str, default="analysis/output",
        help="Output directory for CSV/JSON exports (default: analysis/output)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyzer = ExperimentAnalyzer()

    if args.combined:
        analyzer.load_combined(args.combined)
    else:
        if args.local:
            analyzer.load_local(args.local)
        if args.k8s:
            analyzer.load_k8s(args.k8s)

    out_dir = Path(args.output)

    # Single-mode analysis
    if analyzer.local_data is not None and analyzer.k8s_data is None:
        la = analyzer.analyse_local()
        logger.info("Local analysis: %s", json.dumps(la, indent=2))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "local_analysis.json", "w") as f:
            json.dump(la, f, indent=2)
        return

    if analyzer.k8s_data is not None and analyzer.local_data is None:
        ka = analyzer.analyse_k8s()
        logger.info("K8s analysis: %s", json.dumps(ka, indent=2))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "k8s_analysis.json", "w") as f:
            json.dump(ka, f, indent=2)
        return

    # Both modes — full comparison
    analyzer.print_summary()
    analyzer.export_csv(str(out_dir / "summary.csv"))
    analyzer.export_json(str(out_dir / "analysis.json"))


if __name__ == "__main__":
    main()
