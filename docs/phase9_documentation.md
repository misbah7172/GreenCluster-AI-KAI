# Phase 9: Kubernetes Execution Mode -- Documentation

## Objective

Create a Kubernetes inference runner (`experiments/k8s_runner.py`) that mirrors the local runner from Phase 5 but executes inference through a distributed chunk pipeline on Kubernetes, collecting multi-node metrics and computing aggregated energy/performance statistics.

---

## What Was Created

### `experiments/k8s_runner.py`

A single-file module containing:

- **`run_k8s_experiment()`** -- the main programmatic entry point
- **`main()`** -- CLI wrapper with argparse
- Helper functions `_serialize_tensor()` and `_build_dummy_input()`

---

## How It Works

### 1. Controller Import

The `KAIController` class from `kubernetes/controller.py` is imported via `importlib.util.spec_from_file_location` to avoid the name collision between the local `kubernetes/` directory and the `kubernetes` pip package. This is the same strategy used by the controller itself.

### 2. Experiment Flow

The experiment proceeds through these stages:

```
Deploy -> Wait -> Health Check -> Warmup -> Measured Loop -> Collect Metrics -> Save -> Teardown
```

**Deploy:**
- Calls `ctrl.deploy_all(num_chunks, model_type)` which creates chunk Deployments+Services, gateway Deployment+Service, and monitor DaemonSet+Service in the `kai` namespace.

**Wait for Readiness:**
- Calls `ctrl.wait_for_ready(timeout)` which polls all `app=kai` pods until they reach Running/Ready status.
- If pods do not become ready within the timeout, the experiment aborts (with teardown if `auto_teardown` is True).

**Health Check:**
- Calls `ctrl.check_gateway_health()` to verify the gateway can reach all chunk servers.

**Warmup:**
- Sends `warmup_iterations` inference requests through the gateway. Failures are logged but do not abort the experiment.

**Measured Loop:**
- For each of `iterations` iterations:
  - Serializes a dummy input tensor via `torch.save` into bytes.
  - Sends it to the gateway via `ctrl.send_inference_request()`.
  - Records end-to-end latency (measured client-side with `time.perf_counter()`).
  - Records per-chunk latencies and network overhead from the gateway response.
  - Progress is logged every ~20% of iterations.

**Collect Metrics:**
- Calls `ctrl.stop_monitoring()` then `ctrl.collect_metrics()` to gather GPU power, utilization, memory, and temperature from every monitor pod (one per GPU node).

**Aggregation:**
- End-to-end latency: averaged across all successful iterations
- Gateway latency: from the gateway's own timing
- Network overhead: gateway total minus sum of chunk compute times
- Per-chunk latency: averaged per chunk index
- GPU power: summed across all nodes
- Energy: (node_avg_power * node_runtime) / 3600 for each node, then summed
- Energy per inference: total_energy / successful_iterations
- Throughput: successful_iterations / total_runtime

**Save:**
- Results are written to `logs/k8s_results_<YYYYMMDD_HHMMSS>.json`.

**Teardown:**
- If `auto_teardown` is True (default), calls `ctrl.teardown()` to remove all KAI resources.

### 3. Output JSON Structure

```json
{
  "execution_mode": "kubernetes",
  "timestamp": "2026-03-11T14:30:00",
  "experiment_config": {
    "model_type": "transformer",
    "num_chunks": 3,
    "batch_size": 8,
    "iterations": 50,
    "warmup_iterations": 5,
    "gateway_url": "http://10.0.0.5:30080",
    "wait_timeout_s": 300.0,
    "successful_iterations": 50,
    "failed_iterations": 0
  },
  "avg_e2e_latency_ms": 185.42,
  "avg_gateway_latency_ms": 144.30,
  "avg_network_overhead_ms": 13.60,
  "per_chunk_avg_latency_ms": [42.10, 45.30, 43.30],
  "throughput_inferences_per_sec": 5.39,
  "total_runtime_s": 9.271,
  "total_avg_power_w": 120.50,
  "total_energy_wh": 0.000310,
  "energy_per_inference_wh": 0.00000620,
  "node_metrics": [...],
  "inference_results": [...],
  "e2e_latencies_ms": [...]
}
```

### 4. CLI Arguments

| Argument         | Default       | Description                                 |
|------------------|---------------|---------------------------------------------|
| `--model`        | transformer   | Model architecture (transformer or cnn)     |
| `--num-chunks`   | 3             | Number of model chunks to deploy            |
| `--iterations`   | 50            | Number of measured inference iterations     |
| `--batch-size`   | 8             | Input batch size                            |
| `--output-dir`   | logs          | Output directory for results JSON           |
| `--gateway-url`  | auto-detect   | Override gateway URL                        |
| `--wait-timeout` | 300           | Pod readiness timeout (seconds)             |
| `--warmup`       | 5             | Warmup iterations (not measured)            |
| `--no-teardown`  | false         | Keep K8s resources after experiment         |

---

## Verification

1. **Import test** -- Module loaded successfully via `importlib.util`, exposing `run_k8s_experiment` and `main`.
2. **CLI test** -- `python -m experiments.k8s_runner --help` displayed all 9 arguments correctly.

Full end-to-end verification requires a running Kubernetes cluster with GPU nodes and the NVIDIA device plugin, which is beyond local testing scope.

---

## Key Design Decisions

1. **Client-side E2E timing:** The runner measures wall-clock latency from the client's perspective (including HTTP round-trip), while also recording the gateway's internal timing for comparison. The difference reveals network overhead between the runner and the gateway.

2. **Multi-node energy aggregation:** Energy is computed per-node (from each monitor pod's power readings and runtime), then summed across all nodes. This captures the total GPU energy cost of the distributed pipeline.

3. **Graceful failure handling:** Individual inference requests that fail are logged and skipped. The experiment continues and reports both successful and failed iteration counts.

4. **Monitoring lifecycle:** The runner calls `reset_monitoring()` before starting to clear any stale data, `start_monitoring()` to begin fresh recording, and `stop_monitoring()` after the measured loop completes.

5. **Auto-teardown:** By default, all K8s resources are cleaned up after the experiment. The `--no-teardown` flag allows keeping resources alive for debugging or follow-up experiments.

---

## Dependencies on Previous Phases

- **Phase 2 (AI Models):** `build_model()` and `get_dummy_input()` for generating input tensors
- **Phase 6 (Docker):** Container images that the K8s deployments reference
- **Phase 7 (K8s YAML):** Resource templates (the controller builds equivalent objects programmatically)
- **Phase 8 (K8s Controller):** `KAIController` class providing deploy, wait, infer, monitor, and teardown methods

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
