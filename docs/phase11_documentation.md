# Phase 11: Analysis Module -- Documentation

## Objective

Build post-experiment analysis tools to compute derived metrics, produce a summary comparison table, and generate publication-quality visualizations comparing local vs Kubernetes inference benchmarks.

---

## What Was Created

### `analysis/analyzer.py`

The `ExperimentAnalyzer` class provides:

- **Loading:** `load_local()`, `load_k8s()`, `load_combined()` -- read result JSON files
- **Per-mode analysis:** `analyse_local()`, `analyse_k8s()` -- compute derived metrics (percentiles, stddev, performance-per-watt)
- **Comparison:** `compare()` -- side-by-side dict with ratios across 10 metrics
- **Summary table:** `summary_table()`, `print_summary()` -- formatted tabular output
- **Export:** `export_csv()`, `export_json()` -- write results to files
- **Data accessors:** `get_local_gpu_samples()`, `get_local_latencies()`, `get_k8s_latencies()`, `get_k8s_node_metrics()`, etc. -- used by plots.py

CLI: `python -m analysis.analyzer --local <file> --k8s <file> --output <dir>`

### `analysis/plots.py`

Eight visualization functions plus a `plot_all()` master:

1. **`plot_power_over_time()`** -- Line plot of GPU power draw over time (local + K8s nodes overlaid)
2. **`plot_gpu_utilization_over_time()`** -- Line plot of GPU utilization % over time
3. **`plot_latency_comparison()`** -- Bar chart: avg latency local vs K8s
4. **`plot_energy_comparison()`** -- Bar chart: total energy consumed
5. **`plot_energy_efficiency()`** -- Bar chart: energy per inference
6. **`plot_throughput_comparison()`** -- Bar chart: inferences/sec
7. **`plot_latency_distribution()`** -- Box plot: latency distributions
8. **`plot_per_chunk_latency()`** -- Bar chart: per-chunk avg latency (K8s only)

CLI: `python -m analysis.plots --local <file> --k8s <file> --output <dir>`

---

## How It Works

### Analyzer

The analyzer loads raw JSON from the local and K8s runners, which use different key names:

| Metric        | Local Key                        | K8s Key                        |
|---------------|----------------------------------|--------------------------------|
| Avg latency   | `avg_latency_ms`                 | `avg_e2e_latency_ms`           |
| Raw latencies | `inference_latencies_ms`         | `e2e_latencies_ms`             |
| Avg power     | `avg_power_w`                    | `total_avg_power_w`            |
| GPU samples   | `gpu_samples` (list of dicts)    | `node_metrics[].gpu_samples`   |

The `analyse_*()` methods compute derived metrics not present in the raw data:
- **Latency percentiles** (p50, p90, p95, p99) via linear interpolation
- **Latency standard deviation**
- **Performance-per-watt** = throughput / avg_power (inferences/s/W)

The `compare()` method produces ratios (K8s/Local) for each metric. Ratio > 1 means K8s uses more; < 1 means K8s is more efficient. Division by zero returns `None`.

### Plots

All plots use the `Agg` backend (non-interactive) and save PNG at 150 DPI. The `plot_all()` function auto-detects which data is available:
- Time-series plots require raw `gpu_samples` (may not be present in K8s node metrics)
- Comparison bar charts require both local and K8s data
- The latency distribution and per-chunk plots work with one mode

---

## Verification

### Import Test
Both modules import cleanly: `ExperimentAnalyzer` exposes 18 public methods/properties.

### Analyzer Test (mock data)
- Local analysis: latency p50=27.45 ms, p95=29.66 ms, perf/watt=0.794
- K8s analysis: latency p50=184.9 ms, p95=189.31 ms, perf/watt=0.045
- Comparison: latency ratio 6.69x, energy ratio 3.0x
- CSV export: 11 rows written successfully
- Summary table: 10 metric rows + header

### Plot Test (mock data)
All 8 plots generated successfully:
- `power_over_time.png`
- `gpu_utilization_over_time.png`
- `latency_comparison.png`
- `energy_comparison.png`
- `energy_efficiency.png`
- `throughput_comparison.png`
- `latency_distribution.png`
- `per_chunk_latency.png`

### CLI Test
Both `python -m analysis.analyzer --help` and `python -m analysis.plots --help` display correct arguments.

---

## Metrics Computed

| Metric                     | Formula / Source                                  |
|----------------------------|---------------------------------------------------|
| Avg Latency (ms)           | mean(latencies)                                   |
| Latency Std (ms)           | statistics.stdev(latencies)                       |
| Latency P50/P90/P95/P99    | Linear interpolation on sorted latencies          |
| Throughput (inf/s)         | num_inferences / total_runtime_s                  |
| Avg GPU Power (W)          | Local: mean(samples.power_w), K8s: sum(nodes)     |
| Total Energy (Wh)          | avg_power_w * runtime_s / 3600                    |
| Energy/Inference (Wh)      | total_energy / num_inferences                     |
| Performance/Watt (inf/s/W) | throughput / avg_power                            |
| Network Overhead (ms)      | K8s only: gateway_total - sum(chunk_times)        |

---

## Dependencies on Previous Phases

- **Phase 5 (Local Runner):** JSON output format with `gpu_samples`, `cpu_samples`, `inference_latencies_ms`
- **Phase 9 (K8s Runner):** JSON output format with `e2e_latencies_ms`, `node_metrics`, `per_chunk_avg_latency_ms`
- **Phase 10 (Unified Runner):** Combined JSON with `local` and `kubernetes` keys (loadable via `load_combined()`)

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
