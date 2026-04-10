# Phase 8: Kubernetes Controller -- Documentation

## Objective

Build a Python-based Kubernetes controller that automates deployment, readiness polling, inference requests, metrics collection, and teardown of the distributed model-chunk inference pipeline on a Kubernetes cluster.

## What Was Built

### `kubernetes/controller.py`

A comprehensive orchestration script using the `kubernetes` Python client library, providing both a programmatic API (`KAIController` class) and a CLI interface.

#### KAIController Class

**Deployment methods:**

- `deploy_chunks(num_chunks, model_type, image)` -- Creates N chunk Deployments and ClusterIP Services. Each chunk gets a unique name (`kai-chunk-0`, `kai-chunk-1`, ...), its own service for DNS-based discovery, GPU resource requests, pod anti-affinity for node spreading, and GPU-taint tolerations. Existing resources with the same name are deleted first for idempotency.

- `deploy_gateway(num_chunks, image)` -- Creates the gateway Deployment and NodePort Service. Dynamically builds the `CHUNK_HOSTS` env var from chunk count (e.g., `kai-chunk-0:50051,kai-chunk-1:50051,...`). Exposes the gateway externally on NodePort 30080.

- `deploy_monitor(image)` -- Creates the monitor DaemonSet and ClusterIP Service. Uses a raw dict for the DaemonSet spec since the kubernetes client's DaemonSet support varies by version. Schedules on GPU-equipped nodes via nodeSelector.

- `deploy_all(num_chunks, model_type)` -- Convenience method that calls all three deploy methods in sequence.

**Readiness and status:**

- `wait_for_ready(timeout, poll_interval)` -- Polls all KAI pods every `poll_interval` seconds until all are Running with Ready condition True, or until timeout. Logs progress showing ready/total counts and names of pending pods. Returns bool.

- `get_status()` -- Returns a dict with namespace and a list of all KAI pods, each showing name, phase, ready state, node, IP, and component label.

**Inference:**

- `get_gateway_url()` -- Auto-detects a node IP from the cluster and constructs the gateway URL using the NodePort (30080).

- `send_inference_request(input_data, gateway_url, request_id)` -- Sends serialized tensor bytes to `POST /infer` on the gateway. Returns the JSON response with `output_shape`, `total_time_ms`, `chunk_times_ms`, `network_overhead_ms`.

- `check_gateway_health(gateway_url)` -- Queries `GET /health` on the gateway.

**Metrics collection:**

- `collect_metrics()` -- Discovers all monitor pods by label selector, queries each pod's `/metrics/summary` endpoint directly via pod IP. Returns a list of per-node metric summaries including `avg_power_w`, GPU utilization, energy, etc.

- `start_monitoring()` -- Sends `POST /start` to all monitor pods.
- `stop_monitoring()` -- Sends `POST /stop` to all monitor pods and returns summaries.
- `reset_monitoring()` -- Sends `POST /reset` to all monitor pods (clear and restart).

**Teardown:**

- `teardown()` -- Discovers all chunk deployments/services by label selector and deletes them, then deletes the gateway deployment/service, monitor DaemonSet/service, and GPU resource quota. Uses `_delete_if_exists()` to ignore 404 errors gracefully. Does not delete the namespace itself.

#### CLI Interface

```
python kubernetes/controller.py <command> [options]

Commands:
    deploy     Deploy the full pipeline
               --num-chunks N   Number of chunks (default: 3)
               --model TYPE     transformer or cnn (default: transformer)
               --wait           Wait for pods to be ready
               --timeout SECS   Readiness timeout (default: 300)

    status     Show pod status (JSON output)

    health     Check gateway health
               --gateway-url URL   Override auto-detected URL

    metrics    Collect metrics from all monitor pods (JSON output)

    teardown   Remove all KAI resources
```

#### Import Handling

The controller lives inside the `kubernetes/` directory, which conflicts with the `kubernetes` pip package name. The import logic temporarily removes the project root from `sys.path` and clears cached module references before importing the pip package, then restores the path. This avoids the name shadowing issue.

## How It Was Done

1. **Studied the K8s manifests** -- Read all YAML files from Phase 7 to understand the exact labels, selectors, ports, resource specifications, and environment variables. The controller's Python builder functions replicate these specs programmatically.

2. **Used the kubernetes Python client** -- `client.AppsV1Api` for Deployments and DaemonSets, `client.CoreV1Api` for Services, Namespaces, Pods, and Nodes. The DaemonSet is built as a raw dict because the client's typed model varies across versions.

3. **Idempotent deployments** -- Every deploy method deletes existing resources with the same name before creating new ones. The `_delete_if_exists` helper swallows 404 errors so teardown/redeploy is safe to run repeatedly.

4. **Dynamic chunk count** -- The controller supports any number of chunks (2-8). `deploy_chunks()` loops from 0 to N-1, and `deploy_gateway()` constructs the `CHUNK_HOSTS` env var dynamically from the chunk count.

5. **Direct pod access for metrics** -- Metrics are collected by querying each monitor pod's IP directly rather than going through the ClusterIP service. This gives per-node granularity since the DaemonSet runs one pod per GPU node.

6. **Installed kubernetes pip package** -- The `kubernetes>=28.1.0` dependency was listed in requirements.txt but not installed. Installed v35.0.0.

## Verification

- Controller module loaded successfully via `importlib.util.spec_from_file_location()`.
- All 14 public methods confirmed on the `KAIController` class:
  `check_gateway_health`, `collect_metrics`, `deploy_all`, `deploy_chunks`,
  `deploy_gateway`, `deploy_monitor`, `get_gateway_url`, `get_status`,
  `reset_monitoring`, `send_inference_request`, `start_monitoring`,
  `stop_monitoring`, `teardown`, `wait_for_ready`.
- CLI `--help` output verified for both the top-level command and all subcommands.
- No kubeconfig was available locally (no active cluster), so live cluster operations were not tested. The controller will be exercised end-to-end in Phase 9 (k8s_runner.py).

## Files Created

| File | Purpose |
|------|---------|
| `kubernetes/controller.py` | K8s orchestration controller (KAIController class + CLI) |

## Architecture: Controller in the Pipeline

```
User / k8s_runner.py
         |
         v
  [KAIController]
         |
         +-- deploy_chunks()  --> K8s API: Deployments + Services
         +-- deploy_gateway() --> K8s API: Deployment + NodePort Service
         +-- deploy_monitor() --> K8s API: DaemonSet + Service
         |
         +-- wait_for_ready() --> K8s API: poll pod status
         |
         +-- send_inference_request() --> HTTP POST to gateway NodePort
         |
         +-- collect_metrics() --> HTTP GET from each monitor pod IP
         |
         +-- teardown()       --> K8s API: delete all resources
```

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
