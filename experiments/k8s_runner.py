"""
Kubernetes inference runner.

Deploys a chunked AI model across a Kubernetes cluster, runs N inference
iterations through the gateway service, collects per-node GPU metrics from
monitor pods, computes aggregated performance/energy statistics, and saves
structured results to ``logs/``.

Usage::

    python -m experiments.k8s_runner \
        --model transformer \
        --num-chunks 3 \
        --iterations 50 \
        --batch-size 8 \
        --output-dir logs

Or programmatically::

    from experiments.k8s_runner import run_k8s_experiment
    results = run_k8s_experiment(
        model_type="transformer",
        num_chunks=3,
        iterations=50,
        batch_size=8,
    )
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Import KAIController — handle the kubernetes/ directory name collision
# with the ``kubernetes`` pip package exactly like controller.py itself does.
# ---------------------------------------------------------------------------

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# We need to import controller.py from the local kubernetes/ directory.
# importlib is used so the local folder name does not clash with pip's
# ``kubernetes`` package (controller.py handles that internally).
import importlib.util as _ilu

_controller_path = os.path.join(_project_root, "kubernetes", "controller.py")
_spec = _ilu.spec_from_file_location("kai_controller", _controller_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KAIController = _mod.KAIController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes using ``torch.save``."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _build_dummy_input(
    model_type: str,
    batch_size: int,
) -> torch.Tensor:
    """Generate a dummy input tensor on CPU for the given model type."""
    if model_type == "transformer":
        from model.transformer import build_model, get_dummy_input
        model = build_model()
        dummy = get_dummy_input(
            batch_size=batch_size,
            seq_len=128,
            vocab_size=model.config["vocab_size"],
        )
        return dummy
    elif model_type == "cnn":
        from model.cnn import build_model, get_dummy_input
        model = build_model()
        dummy = get_dummy_input(
            batch_size=batch_size,
            channels=model.config["in_channels"],
            height=224,
            width=224,
        )
        return dummy
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_k8s_experiment(
    model_type: str = "transformer",
    num_chunks: int = 3,
    iterations: int = 50,
    batch_size: int = 8,
    output_dir: str = "logs",
    gateway_url: Optional[str] = None,
    wait_timeout: float = 300.0,
    warmup_iterations: int = 5,
    auto_teardown: bool = True,
    enable_deas: bool = False,
    deas_cooldown: float = 30.0,
) -> Dict[str, Any]:
    """Run a Kubernetes-based inference experiment.

    Parameters
    ----------
    model_type : str
        ``"transformer"`` or ``"cnn"``.
    num_chunks : int
        Number of model chunks to deploy (2-8).
    iterations : int
        Number of measured inference iterations.
    batch_size : int
        Input batch size.
    output_dir : str
        Directory for result JSON output.
    gateway_url : str, optional
        Override the auto-detected gateway URL (e.g. for port-forward).
    wait_timeout : float
        Maximum seconds to wait for pods to become ready.
    warmup_iterations : int
        Number of untimed warmup iterations.
    auto_teardown : bool
        If True, teardown K8s resources after the experiment.
    enable_deas : bool
        Enable Dynamic Energy-Aware Scheduling during the experiment.
    deas_cooldown : float
        Minimum seconds between DEAS migration attempts.

    Returns
    -------
    dict
        The full experiment summary (also saved as JSON).
    """
    logger.info(
        "K8s experiment: model=%s, chunks=%d, iterations=%d, batch=%d",
        model_type, num_chunks, iterations, batch_size,
    )

    # --- Initialise controller ---
    ctrl = KAIController()

    # --- Deploy full pipeline ---
    logger.info("Deploying pipeline (%d chunks, model=%s)...", num_chunks, model_type)
    ctrl.deploy_all(num_chunks=num_chunks, model_type=model_type)

    # --- Wait for readiness ---
    ready = ctrl.wait_for_ready(timeout=wait_timeout)
    if not ready:
        logger.error("Pods did not become ready within %ds — aborting", int(wait_timeout))
        if auto_teardown:
            ctrl.teardown()
        raise RuntimeError("K8s pods failed to reach Ready state")

    # --- Resolve gateway URL ---
    gw_url = gateway_url or ctrl.get_gateway_url()
    logger.info("Gateway URL: %s", gw_url)

    # --- Health check ---
    health = ctrl.check_gateway_health(gateway_url=gw_url)
    logger.info("Gateway health: %s", json.dumps(health, indent=2))

    # --- Build dummy input ---
    dummy_input = _build_dummy_input(model_type, batch_size)
    input_bytes = _serialize_tensor(dummy_input)
    logger.info("Input tensor shape: %s  (%d bytes serialized)", list(dummy_input.shape), len(input_bytes))

    # --- Reset and start monitoring ---
    ctrl.reset_monitoring()
    ctrl.start_monitoring()

    # --- Start DEAS if enabled ---
    deas_active = False
    if enable_deas:
        try:
            from monitoring.event_bus import EventBus
            from model.auto_partitioner import AutoPartitioner
            event_bus = EventBus()
            event_bus.start()
            partitioner = AutoPartitioner()
            ctrl.start_deas(event_bus, partitioner, cooldown_s=deas_cooldown)
            deas_active = True
            logger.info("DEAS enabled (cooldown=%.1fs)", deas_cooldown)
        except Exception as e:
            logger.warning("Failed to start DEAS: %s", e)

    # --- Warmup iterations ---
    logger.info("Running %d warmup iterations...", warmup_iterations)
    for _ in range(warmup_iterations):
        try:
            ctrl.send_inference_request(input_bytes, gateway_url=gw_url)
        except Exception as e:
            logger.warning("Warmup request failed: %s", e)

    # --- Measured inference loop ---
    logger.info("Running %d measured iterations...", iterations)
    inference_results: List[Dict[str, Any]] = []
    total_latencies: List[float] = []
    chunk_latencies_all: List[List[float]] = []
    network_overheads: List[float] = []

    for i in range(iterations):
        request_id = f"iter-{i}"
        t_start = time.perf_counter()

        try:
            resp = ctrl.send_inference_request(
                input_bytes, gateway_url=gw_url, request_id=request_id,
            )
        except Exception as e:
            logger.error("Inference request %d failed: %s", i, e)
            continue

        e2e_latency_ms = (time.perf_counter() - t_start) * 1000.0

        total_latencies.append(e2e_latency_ms)
        chunk_latencies_all.append(resp.get("chunk_times_ms", []))
        network_overheads.append(resp.get("network_overhead_ms", 0.0))

        inference_results.append({
            "iteration": i,
            "e2e_latency_ms": round(e2e_latency_ms, 4),
            "gateway_total_ms": resp.get("total_time_ms", 0.0),
            "chunk_times_ms": resp.get("chunk_times_ms", []),
            "network_overhead_ms": resp.get("network_overhead_ms", 0.0),
            "output_shape": resp.get("output_shape"),
            "request_id": request_id,
        })

        if (i + 1) % max(1, iterations // 5) == 0:
            logger.info(
                "  Iteration %d/%d — e2e %.2f ms, gateway %.2f ms",
                i + 1, iterations, e2e_latency_ms,
                resp.get("total_time_ms", 0.0),
            )

    # --- Stop monitoring and collect metrics ---
    ctrl.stop_monitoring()
    node_metrics = ctrl.collect_metrics()

    # --- Stop DEAS and collect migration data ---
    migration_events = []
    threshold_events = []
    if deas_active:
        try:
            deas = getattr(ctrl, "_deas", None)
            if deas is not None:
                migration_events = deas.get_migration_history()
            threshold_events = ctrl.collect_threshold_events()
            ctrl.stop_deas()
            event_bus.stop()
            logger.info(
                "DEAS stopped: %d migrations, %d threshold events",
                len(migration_events), len(threshold_events),
            )
        except Exception as e:
            logger.warning("Error stopping DEAS: %s", e)

    # --- Compute aggregated statistics ---
    successful = len(total_latencies)
    failed = iterations - successful

    avg_e2e_latency_ms = 0.0
    avg_gateway_latency_ms = 0.0
    avg_network_overhead_ms = 0.0
    throughput = 0.0
    total_runtime_s = 0.0

    if total_latencies:
        avg_e2e_latency_ms = sum(total_latencies) / len(total_latencies)
        total_runtime_s = sum(total_latencies) / 1000.0
        throughput = successful / total_runtime_s if total_runtime_s > 0 else 0.0

    gateway_totals = [r["gateway_total_ms"] for r in inference_results]
    if gateway_totals:
        avg_gateway_latency_ms = sum(gateway_totals) / len(gateway_totals)

    if network_overheads:
        avg_network_overhead_ms = sum(network_overheads) / len(network_overheads)

    # Per-chunk average latency
    num_chunk_slots = num_chunks
    per_chunk_avg_ms = [0.0] * num_chunk_slots
    if chunk_latencies_all:
        for chunk_idx in range(num_chunk_slots):
            vals = [
                cl[chunk_idx] for cl in chunk_latencies_all
                if chunk_idx < len(cl)
            ]
            if vals:
                per_chunk_avg_ms[chunk_idx] = sum(vals) / len(vals)

    # --- Aggregate GPU power/energy across all nodes ---
    total_avg_power_w = 0.0
    total_energy_wh = 0.0
    node_power_details = []

    for nm in node_metrics:
        if "error" in nm:
            node_power_details.append(nm)
            continue

        node_avg_power = nm.get("avg_power_w", 0.0)
        node_runtime = nm.get("total_runtime_s", total_runtime_s)
        node_energy = node_avg_power * node_runtime / 3600.0

        total_avg_power_w += node_avg_power
        total_energy_wh += node_energy

        node_power_details.append({
            "pod_name": nm.get("pod_name", "unknown"),
            "node_name": nm.get("node_name", "unknown"),
            "avg_power_w": round(node_avg_power, 2),
            "avg_gpu_utilization_pct": nm.get("avg_gpu_utilization_pct", 0.0),
            "avg_gpu_memory_used_mb": nm.get("avg_gpu_memory_used_mb", 0.0),
            "avg_gpu_temperature_c": nm.get("avg_gpu_temperature_c", 0.0),
            "node_energy_wh": round(node_energy, 6),
            "node_runtime_s": round(node_runtime, 4),
        })

    energy_per_inference_wh = total_energy_wh / successful if successful > 0 else 0.0

    # --- Build summary ---
    experiment_config = {
        "model_type": model_type,
        "num_chunks": num_chunks,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "gateway_url": gw_url,
        "wait_timeout_s": wait_timeout,
        "successful_iterations": successful,
        "failed_iterations": failed,
    }

    summary: Dict[str, Any] = {
        "execution_mode": "kubernetes",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_config": experiment_config,

        # Latency
        "avg_e2e_latency_ms": round(avg_e2e_latency_ms, 4),
        "avg_gateway_latency_ms": round(avg_gateway_latency_ms, 4),
        "avg_network_overhead_ms": round(avg_network_overhead_ms, 4),
        "per_chunk_avg_latency_ms": [round(v, 4) for v in per_chunk_avg_ms],

        # Throughput
        "throughput_inferences_per_sec": round(throughput, 4),
        "total_runtime_s": round(total_runtime_s, 4),

        # Energy (aggregated across all GPU nodes)
        "total_avg_power_w": round(total_avg_power_w, 2),
        "total_energy_wh": round(total_energy_wh, 6),
        "energy_per_inference_wh": round(energy_per_inference_wh, 8),

        # Per-node details
        "node_metrics": node_power_details,

        # Raw per-iteration data
        "inference_results": inference_results,
        "e2e_latencies_ms": [round(v, 4) for v in total_latencies],
    }

    # Add DEAS data if enabled
    if deas_active:
        summary["deas_enabled"] = True
        summary["deas_cooldown_s"] = deas_cooldown
        summary["migration_events"] = migration_events
        summary["threshold_events"] = threshold_events
        summary["migration_count"] = len(migration_events)

    # --- Save results ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"k8s_results_{ts}.json"
    result_file = out_path / filename

    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", result_file)

    # --- Print summary ---
    logger.info("===== Kubernetes Experiment Results =====")
    logger.info("  Model:                %s", model_type)
    logger.info("  Chunks:               %d", num_chunks)
    logger.info("  Iterations:           %d (%d succeeded, %d failed)", iterations, successful, failed)
    logger.info("  Avg E2E latency:      %.2f ms", avg_e2e_latency_ms)
    logger.info("  Avg gateway latency:  %.2f ms", avg_gateway_latency_ms)
    logger.info("  Avg network overhead: %.2f ms", avg_network_overhead_ms)
    logger.info("  Throughput:           %.2f inf/s", throughput)
    logger.info("  Total GPU power:      %.2f W (summed across %d node(s))", total_avg_power_w, len(node_power_details))
    logger.info("  Total energy:         %.6f Wh", total_energy_wh)
    logger.info("  Energy/inference:     %.8f Wh", energy_per_inference_wh)
    for ci, cl in enumerate(per_chunk_avg_ms):
        logger.info("  Chunk %d avg latency:  %.2f ms", ci, cl)
    logger.info("  Results saved to:     %s", result_file)

    # --- Teardown ---
    if auto_teardown:
        logger.info("Tearing down K8s resources...")
        ctrl.teardown()
        logger.info("Teardown complete")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Kubernetes inference experiment",
    )
    parser.add_argument(
        "--model", type=str, default="transformer",
        choices=["transformer", "cnn"],
        help="Model architecture to benchmark (default: transformer)",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=3,
        help="Number of model chunks (default: 3)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of inference iterations (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Input batch size (default: 8)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs",
        help="Directory for result files (default: logs)",
    )
    parser.add_argument(
        "--gateway-url", type=str, default=None,
        help="Override gateway URL (e.g. http://localhost:30080)",
    )
    parser.add_argument(
        "--wait-timeout", type=float, default=300.0,
        help="Pod readiness timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--no-teardown", action="store_true",
        help="Do not teardown K8s resources after the experiment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_k8s_experiment(
        model_type=args.model,
        num_chunks=args.num_chunks,
        iterations=args.iterations,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        gateway_url=args.gateway_url,
        wait_timeout=args.wait_timeout,
        warmup_iterations=args.warmup,
        auto_teardown=not args.no_teardown,
    )


if __name__ == "__main__":
    main()
