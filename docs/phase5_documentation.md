# Phase 5: Local Execution Mode -- Documentation

## What Was Done

Implemented `experiments/local_runner.py`, the local (single-GPU) inference runner that executes a full AI model, collects performance and power metrics, and exports structured results.

Also patched `monitoring/metrics.py` to support an `extra_data` parameter in `export_results()` so that experiment configuration metadata is written into the output JSON alongside the computed metrics.

---

## Files Created

| File | Purpose |
|------|---------|
| `experiments/local_runner.py` | Local inference experiment runner with CLI and programmatic API |

## Files Modified

| File | Change |
|------|--------|
| `monitoring/metrics.py` | Added `extra_data` parameter to `export_results()` so callers can inject experiment config into the exported JSON |

---

## How It Works

### Execution Flow

```
1. Resolve device (auto -> CUDA if available, else CPU)
2. Build model (Transformer or CNN) and move to device
3. Generate dummy input tensor on the same device
4. Run warmup iterations (untimed, to stabilize GPU clocks / JIT)
5. Start MetricsCollector (GPU + CPU background sampling)
6. Run N measured inference iterations:
   - torch.cuda.synchronize() before/after each run (if GPU)
   - Record per-iteration latency via collector.record_latency()
   - Log progress every ~20% of iterations
7. Stop MetricsCollector
8. Compute summary: avg latency, throughput, avg power, total energy, energy/inference
9. Export JSON to logs/ with full summary + experiment_config
10. Print results to log
```

### CLI Interface

```bash
python -m experiments.local_runner \
    --model transformer \       # or cnn
    --iterations 50 \           # number of timed runs
    --batch-size 8 \            # input batch size
    --device auto \             # auto, cpu, cuda, cuda:0
    --output-dir logs \         # where to save results
    --monitor-interval 1.0 \   # GPU/CPU sampling interval (seconds)
    --warmup 5                  # untimed warmup iterations
```

### Programmatic Interface

```python
from experiments.local_runner import run_local_experiment

summary = run_local_experiment(
    model_type="transformer",
    iterations=50,
    batch_size=8,
    device_str="auto",
    output_dir="logs",
)
# summary is a dict with all metrics + experiment_config
```

### Output JSON Structure

Each run produces a file `logs/local_results_<YYYYMMDD_HHMMSS>.json` containing:

```
{
  "execution_mode": "local",
  "timestamp": "...",
  "total_runtime_s": ...,
  "num_inferences": ...,
  "avg_power_w": ...,
  "avg_gpu_utilization_pct": ...,
  "avg_gpu_memory_used_mb": ...,
  "avg_gpu_temperature_c": ...,
  "total_energy_wh": ...,
  "energy_per_inference_wh": ...,
  "avg_cpu_utilization_pct": ...,
  "avg_latency_ms": ...,
  "throughput_inferences_per_sec": ...,
  "gpu_samples": [ ... ],          // time-series power/util/mem/temp
  "cpu_samples": [ ... ],          // time-series cpu util/mem
  "inference_latencies_ms": [ ... ], // per-iteration latency
  "experiment_config": {
    "model_type": "transformer",
    "model_config": { ... },        // model hyperparameters
    "total_parameters": 12555274,
    "batch_size": 8,
    "iterations": 50,
    "warmup_iterations": 5,
    "device": "cuda",
    "monitor_interval_s": 1.0
  }
}
```

---

## Design Decisions

1. **Warmup iterations**: The first few forward passes on a GPU can be slower due to CUDA kernel JIT compilation and clock ramp-up. Running untimed warmup iterations before the measured loop ensures stable, representative latency numbers.

2. **torch.cuda.synchronize**: GPU operations are asynchronous by default. Calling `synchronize()` before starting and after finishing each inference ensures the measured wall-clock time reflects actual GPU compute time, not just the time to enqueue the operation.

3. **Device auto-detection**: The `"auto"` device option picks CUDA if available, otherwise CPU. This lets the same script run on machines with and without GPUs, with GPU monitoring automatically disabled when running on CPU.

4. **MetricsCollector integration**: Rather than managing GPU and CPU monitors separately, the runner uses the unified `MetricsCollector` from Phase 4. This guarantees synchronized start/stop of all monitors and consistent result structure.

5. **extra_data in export_results**: The `MetricsCollector.export_results()` method was extended with an `extra_data` parameter so the runner can inject experiment configuration into the JSON without modifying the metrics module's core data model.

---

## Verification

Tested with both model types on CPU:

**Transformer** (20 iterations, batch=4):
- Avg latency: 27.67 ms
- Throughput: 35.73 inf/s
- CPU utilization: 31.5%

**CNN** (10 iterations, batch=2):
- Avg latency: 68.30 ms
- Throughput: 14.53 inf/s
- CPU utilization: 42.6%

JSON output verified to contain all expected keys including `experiment_config` with full model configuration and run parameters.
