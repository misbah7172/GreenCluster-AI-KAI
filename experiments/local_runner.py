"""
Local inference runner.

Executes a full AI model on a single GPU (no Kubernetes), collects GPU
power, CPU usage, and inference latency metrics, then exports structured
results to ``logs/``.

Usage::

    python -m experiments.local_runner \
        --model transformer \
        --iterations 50 \
        --batch-size 8 \
        --output-dir logs

Or programmatically::

    from experiments.local_runner import run_local_experiment
    results = run_local_experiment(
        model_type="transformer",
        iterations=50,
        batch_size=8,
    )
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch

from monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


def _build_model_and_input(
    model_type: str,
    batch_size: int,
    device: torch.device,
):
    """Instantiate the model and generate a dummy input tensor.

    Returns
    -------
    model : nn.Module
        The model moved to ``device`` and set to eval mode.
    dummy_input : torch.Tensor
        A single batch of random input on ``device``.
    config : dict
        Model configuration dict for logging.
    """
    if model_type == "transformer":
        from model.transformer import build_model, get_dummy_input
        model = build_model()
        dummy_input = get_dummy_input(
            batch_size=batch_size,
            seq_len=128,
            vocab_size=model.config["vocab_size"],
        )
        config = model.config
    elif model_type == "cnn":
        from model.cnn import build_model, get_dummy_input
        model = build_model()
        dummy_input = get_dummy_input(
            batch_size=batch_size,
            channels=model.config["in_channels"],
            height=224,
            width=224,
        )
        config = model.config
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    return model, dummy_input, config


def run_local_experiment(
    model_type: str = "transformer",
    iterations: int = 50,
    batch_size: int = 8,
    device_str: str = "auto",
    output_dir: str = "logs",
    monitor_interval: float = 1.0,
    warmup_iterations: int = 5,
) -> Dict[str, Any]:
    """Run a local (single-GPU) inference experiment.

    Parameters
    ----------
    model_type : str
        ``"transformer"`` or ``"cnn"``.
    iterations : int
        Number of inference iterations to run.
    batch_size : int
        Batch size for the dummy input.
    device_str : str
        Torch device string.  ``"auto"`` will pick CUDA if available,
        else CPU.
    output_dir : str
        Directory to write the result JSON into.
    monitor_interval : float
        GPU/CPU sampling interval in seconds.
    warmup_iterations : int
        Number of untimed warmup iterations before the measured run.

    Returns
    -------
    dict
        The full experiment summary (also saved as JSON).
    """
    # --- Resolve device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    has_gpu = device.type == "cuda"

    logger.info(
        "Local experiment: model=%s, iterations=%d, batch_size=%d, device=%s",
        model_type, iterations, batch_size, device,
    )

    # --- Build model and input ---
    model, dummy_input, model_config = _build_model_and_input(
        model_type, batch_size, device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s (%d parameters)", model_type, total_params)

    # --- Warmup ---
    logger.info("Running %d warmup iterations...", warmup_iterations)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    if has_gpu:
        torch.cuda.synchronize(device)

    # --- Start monitoring ---
    collector = MetricsCollector(
        gpu_index=device.index or 0 if has_gpu else 0,
        interval=monitor_interval,
        enable_gpu=has_gpu,
        enable_cpu=True,
    )
    collector.start()

    # --- Measured inference loop ---
    logger.info("Running %d measured iterations...", iterations)
    with torch.no_grad():
        for i in range(iterations):
            if has_gpu:
                torch.cuda.synchronize(device)

            t_start = time.perf_counter()
            _ = model(dummy_input)

            if has_gpu:
                torch.cuda.synchronize(device)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            collector.record_latency(latency_ms)

            if (i + 1) % max(1, iterations // 5) == 0:
                logger.info(
                    "  Iteration %d/%d — latency %.2f ms",
                    i + 1, iterations, latency_ms,
                )

    # --- Stop monitoring ---
    collector.stop()

    # --- Experiment configuration ---
    experiment_config = {
        "model_type": model_type,
        "model_config": model_config,
        "total_parameters": total_params,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "device": str(device),
        "monitor_interval_s": monitor_interval,
    }

    # --- Export (includes experiment_config in the JSON) ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"local_results_{ts}.json"
    result_path = collector.export_results(
        output_dir=output_dir,
        filename=filename,
        execution_mode="local",
        extra_data={"experiment_config": experiment_config},
    )

    # --- Read back the full summary for return / logging ---
    summary = collector.compute_summary(execution_mode="local")
    summary["experiment_config"] = experiment_config

    # --- Print summary ---
    logger.info("===== Local Experiment Results =====")
    logger.info("  Model:              %s", model_type)
    logger.info("  Device:             %s", device)
    logger.info("  Iterations:         %d", iterations)
    logger.info("  Avg latency:        %.2f ms", summary["avg_latency_ms"])
    logger.info("  Throughput:         %.2f inf/s", summary["throughput_inferences_per_sec"])
    logger.info("  Avg GPU power:      %.2f W", summary["avg_power_w"])
    logger.info("  Total energy:       %.6f Wh", summary["total_energy_wh"])
    logger.info("  Energy/inference:   %.8f Wh", summary["energy_per_inference_wh"])
    logger.info("  Avg CPU util:       %.1f%%", summary["avg_cpu_utilization_pct"])
    logger.info("  Results saved to:   %s", result_path)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run local (single-GPU) inference experiment",
    )
    parser.add_argument(
        "--model", type=str, default="transformer",
        choices=["transformer", "cnn"],
        help="Model architecture to benchmark (default: transformer)",
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
        "--device", type=str, default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. (default: auto)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs",
        help="Directory for result files (default: logs)",
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=1.0,
        help="GPU/CPU sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    run_local_experiment(
        model_type=args.model,
        iterations=args.iterations,
        batch_size=args.batch_size,
        device_str=args.device,
        output_dir=args.output_dir,
        monitor_interval=args.monitor_interval,
        warmup_iterations=args.warmup,
    )


if __name__ == "__main__":
    main()
