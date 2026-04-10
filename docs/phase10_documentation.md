# Phase 10: Unified Experiment Runner -- Documentation

## Objective

Create a single CLI entry-point (`experiments/experiment_runner.py`) that can run local, Kubernetes, or both experiments back-to-back and produce comparable, side-by-side results with ratios.

---

## What Was Created

### `experiments/experiment_runner.py`

A single module containing:

- **`run_experiment()`** -- main programmatic entry point that dispatches to local and/or K8s runners
- **`_build_comparison()`** -- builds a side-by-side metrics comparison dict
- **`_print_comparison()`** -- logs a formatted comparison table
- **`main()`** -- CLI wrapper with argparse

---

## How It Works

### 1. Mode Dispatch

The runner accepts `--mode` with three options:

| Mode         | What Runs                                    |
|--------------|----------------------------------------------|
| `local`      | `experiments.local_runner.run_local_experiment()` |
| `kubernetes` | `experiments.k8s_runner.run_k8s_experiment()`     |
| `both`       | Local first, then Kubernetes, then comparison |

Imports are lazy -- the local runner is imported only when local mode runs, and the K8s runner only when Kubernetes mode runs. This allows the module to be imported even if only one mode's dependencies are available.

### 2. Comparison (both mode)

When `--mode both` is used, after both experiments complete, the runner computes a comparison across six key metrics:

| Metric                  | Local Source Key                  | K8s Source Key                   |
|-------------------------|-----------------------------------|----------------------------------|
| Avg Latency (ms)        | `avg_latency_ms`                  | `avg_e2e_latency_ms`             |
| Throughput (inf/s)       | `throughput_inferences_per_sec`   | `throughput_inferences_per_sec`  |
| Avg GPU Power (W)       | `avg_power_w`                     | `total_avg_power_w`              |
| Total Energy (Wh)       | `total_energy_wh`                 | `total_energy_wh`                |
| Energy/Inference (Wh)   | `energy_per_inference_wh`         | `energy_per_inference_wh`        |
| Network Overhead (ms)   | 0.0 (no network in local)        | `avg_network_overhead_ms`        |

Each metric includes a `ratio_k8s_over_local` value. A ratio > 1 means K8s uses more (higher latency, more power, etc.); < 1 means K8s is better.

### 3. Output Files

Each mode saves its own per-mode result file (via the underlying runner):
- `logs/local_results_<timestamp>.json`
- `logs/k8s_results_<timestamp>.json`

The unified runner also saves a combined file:
- `logs/experiment_<timestamp>.json`

The combined file contains:
```json
{
  "mode": "both",
  "model_type": "transformer",
  "timestamp": "...",
  "local": { ... full local summary ... },
  "kubernetes": { ... full k8s summary ... },
  "comparison": {
    "latency_ms": { "local": 27.67, "kubernetes": 185.0, "ratio_k8s_over_local": 6.69 },
    "throughput_inf_per_sec": { ... },
    "avg_power_w": { ... },
    "total_energy_wh": { ... },
    "energy_per_inference_wh": { ... },
    "network_overhead_ms": { ... }
  }
}
```

### 4. CLI Arguments

| Argument             | Default     | Description                                   |
|----------------------|-------------|-----------------------------------------------|
| `--mode`             | both        | local, kubernetes, or both                    |
| `--model`            | transformer | transformer or cnn                            |
| `--num-chunks`       | 3           | Model chunks for K8s mode                     |
| `--iterations`       | 50          | Measured inference iterations                 |
| `--batch-size`       | 8           | Input batch size                              |
| `--output-dir`       | logs        | Output directory                              |
| `--device`           | auto        | Torch device for local mode                   |
| `--monitor-interval` | 1.0         | GPU/CPU sampling interval (seconds)           |
| `--warmup`           | 5           | Warmup iterations                             |
| `--gateway-url`      | auto-detect | Override gateway URL for K8s                  |
| `--wait-timeout`     | 300         | Pod readiness timeout (seconds)               |
| `--no-teardown`      | false       | Keep K8s resources after experiment           |

---

## Verification

1. **Import test** -- `run_experiment` and `_build_comparison` imported successfully with all 12 parameters confirmed.
2. **CLI test** -- `python -m experiments.experiment_runner --help` displayed all 12 arguments with correct defaults.
3. **Comparison test** -- `_build_comparison()` tested with mock data:
   - Latency ratio: 6.69x (K8s slower due to network + multi-hop)
   - Throughput ratio: 0.15x (K8s lower throughput)
   - Power ratio: 2.67x (K8s uses multiple GPUs)
   - Energy ratio: 3.0x
   - Network overhead: 13.6 ms (K8s only)

---

## Design Decisions

1. **Lazy imports:** Each runner is imported only when its mode is selected. This avoids import failures when only one mode's dependencies are installed (e.g., no `kubernetes` pip package for local-only use).

2. **K8s runner loaded via importlib:** The `k8s_runner.py` itself uses `importlib.util` to load the controller (due to the `kubernetes/` directory name collision). The experiment runner loads `k8s_runner.py` the same way to maintain consistency.

3. **Comparison ratios:** All ratios are K8s/Local, making it intuitive -- ratio > 1 means K8s used more of that resource, ratio < 1 means K8s used less. Division-by-zero returns `None`.

4. **Three output files:** Each mode saves its own standalone result file (for independent analysis). The combined file aggregates both plus the comparison (for the dashboard and unified analysis).

---

## Dependencies on Previous Phases

- **Phase 5 (Local Runner):** `run_local_experiment()` function
- **Phase 9 (K8s Runner):** `run_k8s_experiment()` function

---

## Implementation Update - 2026-04-11

### Scope Completed
- Comprehensive dashboard upgraded and stabilized for production-style workflow.
- Live inference flow upgraded to asynchronous execution with responsive stop handling.
- GPU-first execution enforced for efficiency-focused workloads.
- Live GPU telemetry panel implemented with real sampling and trend visualization.
- Low-level KV cache counters implemented and wired to dashboard analytics.
- Prompt run history implemented for repeatable multi-prompt validation.

### Dashboard Implementation Status
- Main dashboard file: dashboard/comprehensive_dashboard.py
- Pages available:
  - Home
  - Live Inference
  - Performance Monitor
  - KV Cache Analytics
  - Routing Telemetry
  - Comparisons and Benchmarks
  - System Config
- Live Inference now supports:
  - Repeated prompts in a single session
  - Run history with per-run metrics
  - KV runtime mode and low-level counter visibility

### GPU Runtime Notes
- The project currently has two Python environments with different torch builds.
- CPU-only environment:
  - .venv (torch CPU build)
- CUDA-enabled environment:
  - .venv310 (torch CUDA build)
- For GPU experiments and power-efficiency analysis, use the CUDA environment.

### Recommended Run Commands
- Launch dashboard with CUDA runtime:
  - ./.venv310/Scripts/python -m streamlit run dashboard/comprehensive_dashboard.py
- Optional launcher command:
  - python kai_cli_dashboard.py --port 8511 dashboard-pro

### Live GPU Telemetry (Implemented)
- Real-time panel now shows:
  - GPU utilization percent
  - VRAM used and total
  - Temperature
  - Power draw
  - CUDA allocated and reserved memory
  - Rolling energy estimate in Wh
- Data source priority:
  - NVML (pynvml)
  - nvidia-smi fallback

### KV Cache Upgrade (Implemented)
- Low-level token-prefix reuse counters implemented in runtime path.
- Counters include:
  - kv_cache_hit
  - kv_cache_miss
  - kv_reused_prefix_tokens
  - kv_new_prefill_tokens
  - kv_prompt_tokens
  - kv_runtime_mode
- KV analytics page now updates from measured session counters.
- KV controls added:
  - Reset low-level KV context
  - Clear KV session history
  - Export KV telemetry JSON

### Validation and Behavior
- Python compile checks for dashboard pass.
- Prompt output no longer overwrites silently; session history captures each run.
- KV panel now reflects runtime/session data instead of static placeholders.
- If low-level reuse fails for a model path, fallback to standard generation is applied with runtime notes.

### Change Artifacts
- Updated implementation files include:
  - dashboard/comprehensive_dashboard.py
  - kai_cli_dashboard.py
  - README.md
- Documentation synchronized with current implemented state as of this update.
