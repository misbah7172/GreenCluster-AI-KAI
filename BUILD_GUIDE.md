# KAI — Kubernetes AI Inference Energy Benchmarking Platform

## Build Guide: Development Process

---

## Overview

This platform measures energy consumption and performance of AI inference workloads in two scenarios:

1. **Local Mode** — Running an AI model on a single GPU (no Kubernetes)
2. **Kubernetes Mode** — Running the same AI model split into chunks across multiple nodes in a Kubernetes cluster

The system is written primarily in Python with supporting Docker and Kubernetes configuration.

---

## Final Project Structure

```
project/
├── model/                    # AI model definition and chunking logic
│   ├── __init__.py
│   ├── transformer.py        # Transformer model definition
│   ├── cnn.py                # CNN model definition
│   ├── chunker.py            # Model splitting/chunking utility
│   └── chunk_server.py       # gRPC server for serving a model chunk
│
├── monitoring/               # Power and performance monitoring
│   ├── __init__.py
│   ├── gpu_monitor.py        # NVIDIA NVML-based GPU power monitoring
│   ├── cpu_monitor.py        # CPU usage monitoring
│   └── metrics.py            # Unified metrics collection and export
│
├── kubernetes/               # Kubernetes configuration and controller
│   ├── deployments/
│   │   ├── chunk-deployment.yaml
│   │   ├── gateway-deployment.yaml
│   │   └── monitor-deployment.yaml
│   ├── services/
│   │   ├── chunk-service.yaml
│   │   ├── gateway-service.yaml
│   │   └── monitor-service.yaml
│   ├── gpu-resource-quota.yaml
│   └── controller.py         # Python K8s controller script
│
├── docker/                   # Dockerfiles
│   ├── Dockerfile.chunk      # Model chunk microservice
│   ├── Dockerfile.gateway    # Inference gateway
│   └── Dockerfile.monitor    # Monitoring service
│
├── experiments/              # Experiment runner scripts
│   ├── __init__.py
│   ├── local_runner.py       # Run inference locally on single GPU
│   ├── k8s_runner.py         # Run inference on Kubernetes cluster
│   └── experiment_runner.py  # Unified experiment orchestrator
│
├── dashboard/                # Streamlit visualization dashboard
│   └── app.py
│
├── logs/                     # Experiment output logs (CSV/JSON)
│   └── .gitkeep
│
├── analysis/                 # Post-experiment analysis and plotting
│   ├── __init__.py
│   ├── analyzer.py           # Compute energy, latency, efficiency
│   └── plots.py              # Matplotlib visualization
│
├── proto/                    # gRPC protocol buffer definitions
│   └── inference.proto
│
├── requirements.txt
├── BUILD_GUIDE.md
└── README.md
```

---

## Development Phases

---

### Phase 1: Project Scaffolding & Dependencies

**Goal:** Set up the project skeleton, virtual environment, and all dependency declarations.

**Tasks:**

1. Create the full directory structure as shown above.
2. Create `requirements.txt` with all dependencies:
   - `torch` and `torchvision` (PyTorch for model inference)
   - `pynvml` (NVIDIA NVML Python bindings for GPU monitoring)
   - `psutil` (CPU usage monitoring)
   - `grpcio` and `grpcio-tools` (gRPC for inter-chunk communication)
   - `protobuf` (protocol buffer serialization)
   - `kubernetes` (Python Kubernetes client)
   - `streamlit` (dashboard)
   - `matplotlib` (analysis plots)
   - `pandas` (data handling)
   - `numpy`
3. Create all `__init__.py` files for Python packages.
4. Create `.gitkeep` in `logs/` to track the empty directory.

**Deliverables:**
- Complete folder tree
- `requirements.txt`
- Empty `__init__.py` files

---

### Phase 2: AI Model Module

**Goal:** Define the AI models and implement the chunking mechanism that allows splitting a model into independent layer groups.

**Tasks:**

1. **`model/transformer.py`** — Implement a small Transformer model (e.g., a few-layer encoder) using PyTorch.
   - Define the model class with configurable number of layers.
   - Implement a `forward()` method.
   - Include a helper to generate dummy input tensors for benchmarking.

2. **`model/cnn.py`** — Implement a simple CNN model as an alternative workload.
   - Define convolution + fully connected layers.
   - Implement `forward()`.
   - Include dummy input generator.

3. **`model/chunker.py`** — Model chunking utility.
   - Accept a PyTorch model and a number of chunks.
   - Split the model's layers into `N` sequential groups.
   - Each chunk is a `nn.Sequential` sub-model that can run independently.
   - Provide `serialize_chunk()` and `load_chunk()` for saving/loading chunks.
   - Provide `get_chunk_info()` returning metadata (layer names, input/output shapes).

**Deliverables:**
- Working Transformer and CNN models
- Chunker that splits any sequential model into N independent parts
- Unit-testable: each chunk's output feeds into the next chunk's input

---

### Phase 3: gRPC Communication Layer

**Goal:** Define the gRPC protocol and implement the chunk server so model chunks can communicate intermediate tensors over the network.

**Tasks:**

1. **`proto/inference.proto`** — Define the protobuf service:
   - `InferenceService` with an `Infer` RPC method.
   - Request message: serialized tensor (bytes), chunk ID, metadata.
   - Response message: serialized output tensor (bytes), timing info.

2. **Generate Python gRPC stubs** — Use `grpcio-tools` to compile the proto file.
   - Generated files: `inference_pb2.py`, `inference_pb2_grpc.py`.

3. **`model/chunk_server.py`** — gRPC server for a single model chunk.
   - Load a specific chunk by ID.
   - Receive input tensor via gRPC, run inference on the chunk, return output tensor.
   - Log per-chunk inference latency.
   - Configurable port and chunk ID via environment variables.

**Deliverables:**
- `.proto` definition
- Generated gRPC stubs
- Chunk server that can serve any model chunk independently

---

### Phase 4: Power & Performance Monitoring Module

**Goal:** Build the monitoring subsystem that records GPU power, GPU utilization, GPU memory, CPU usage, and timestamps.

**Tasks:**

1. **`monitoring/gpu_monitor.py`** — GPU monitoring using NVML.
   - Initialize NVML on start, shutdown on exit.
   - Sample GPU metrics every 1 second (configurable):
     - Power draw (Watts)
     - GPU utilization (%)
     - GPU memory used (MB)
     - GPU temperature (°C)
     - Timestamp (ISO 8601)
   - Run sampling in a background thread.
   - Provide `start()`, `stop()`, and `get_samples()` methods.
   - Export samples to CSV or JSON.

2. **`monitoring/cpu_monitor.py`** — CPU monitoring using `psutil`.
   - Sample CPU utilization (%) per core and overall.
   - Sample every 1 second (configurable).
   - Run in a background thread alongside GPU monitor.
   - Provide `start()`, `stop()`, `get_samples()`.

3. **`monitoring/metrics.py`** — Unified metrics collector.
   - Combine GPU and CPU monitors.
   - Add inference-level metrics: latency per request, throughput.
   - Provide a single `MetricsCollector` class that starts/stops all monitors.
   - Export combined results to a structured JSON or CSV file in `logs/`.

**Deliverables:**
- GPU monitor with 1-second sampling via NVML
- CPU monitor with 1-second sampling via psutil
- Unified metrics collector that writes to `logs/`

---

### Phase 5: Local Execution Mode

**Goal:** Implement the local inference runner that executes the full model on a single GPU and collects all metrics.

**Tasks:**

1. **`experiments/local_runner.py`** — Local inference script.
   - Load the selected model (Transformer or CNN) onto a single GPU.
   - Generate dummy input data (configurable batch size).
   - Run inference N times (configurable iterations).
   - For each run:
     - Record start/end timestamps.
     - Compute inference latency.
   - Use `MetricsCollector` to capture GPU/CPU metrics during all runs.
   - After all runs, compute:
     - Average inference latency
     - Average GPU power consumption
     - Total energy consumption (Wh) = average power × total time / 3600
     - Throughput (inferences/second)
   - Save results to `logs/local_results_<timestamp>.json`.

**Deliverables:**
- Complete local inference pipeline
- Automated metrics collection and result logging

---

### Phase 6: Docker Setup

**Goal:** Create Docker images for the model chunk services, inference gateway, and monitoring service.

**Tasks:**

1. **`docker/Dockerfile.chunk`** — Model chunk microservice container.
   - Base image: `nvidia/cuda` with Python 3.
   - Install PyTorch, gRPC dependencies.
   - Copy model code and gRPC stubs.
   - Entrypoint: `chunk_server.py` with configurable `CHUNK_ID` and `PORT` env vars.

2. **`docker/Dockerfile.gateway`** — Inference gateway container.
   - Receives inference requests from the experiment runner.
   - Forwards input through the chain of chunk services via gRPC.
   - Returns final output and timing data.
   - Acts as the entry point for Kubernetes-mode inference.

3. **`docker/Dockerfile.monitor`** — Monitoring sidecar container.
   - Runs the GPU/CPU monitor.
   - Exposes metrics via a simple HTTP endpoint or writes to a shared volume.
   - Can be deployed as a sidecar or DaemonSet.

**Deliverables:**
- Three Dockerfiles, each buildable and runnable
- Docker Compose file (optional, for local multi-container testing)

---

### Phase 7: Kubernetes Configuration

**Goal:** Create all Kubernetes manifests to deploy the chunked model as a distributed inference pipeline.

**Tasks:**

1. **`kubernetes/deployments/chunk-deployment.yaml`** — One Deployment template per chunk.
   - Parameterized by chunk ID.
   - GPU resource requests/limits (e.g., `nvidia.com/gpu: 1`).
   - Environment variables: `CHUNK_ID`, `PORT`, `NEXT_CHUNK_HOST`.

2. **`kubernetes/deployments/gateway-deployment.yaml`** — Gateway Deployment.
   - Single pod that orchestrates the inference pipeline.
   - Knows the addresses of all chunk services.

3. **`kubernetes/deployments/monitor-deployment.yaml`** — Monitor DaemonSet or sidecar.
   - Runs GPU monitoring on each node with a GPU.

4. **`kubernetes/services/chunk-service.yaml`** — ClusterIP Service per chunk.
   - Exposes each chunk pod internally for gRPC communication.

5. **`kubernetes/services/gateway-service.yaml`** — NodePort or LoadBalancer service.
   - Exposes the gateway for external inference requests.

6. **`kubernetes/services/monitor-service.yaml`** — Monitor service.
   - Exposes metrics endpoint.

7. **`kubernetes/gpu-resource-quota.yaml`** — ResourceQuota for GPU allocation.

**Deliverables:**
- Full set of Kubernetes YAML manifests
- Configurable for different numbers of chunks/nodes

---

### Phase 8: Kubernetes Controller

**Goal:** Build the Python-based Kubernetes controller that automates deployment, monitoring, and data collection.

**Tasks:**

1. **`kubernetes/controller.py`** — Kubernetes orchestration script.
   - Use the `kubernetes` Python client library.
   - Functions:
     - `deploy_chunks(num_chunks)` — Create deployments and services for N chunks.
     - `wait_for_ready()` — Poll pod status until all are Running.
     - `send_inference_request(input_data)` — Call the gateway service.
     - `collect_metrics()` — Gather metrics from monitor pods.
     - `teardown()` — Delete all created resources.
   - Handle errors and retries for pod scheduling.
   - Log all operations.

**Deliverables:**
- Controller script that can deploy, run, collect, and teardown a full experiment

---

### Phase 9: Kubernetes Execution Mode

**Goal:** Implement the Kubernetes inference runner analogous to the local runner.

**Tasks:**

1. **`experiments/k8s_runner.py`** — Kubernetes inference script.
   - Use the controller to deploy the chunked model.
   - Wait for all pods to be ready.
   - Run inference N times through the gateway.
   - For each run:
     - Record end-to-end latency (including network overhead).
     - Record per-chunk latency from gateway responses.
   - Collect GPU metrics from all nodes via the monitor service.
   - After all runs, compute:
     - Average inference latency
     - Average GPU power across all nodes
     - Total energy consumption (sum across all GPUs)
     - Throughput
     - Network overhead (total latency − sum of chunk compute times)
   - Save results to `logs/k8s_results_<timestamp>.json`.
   - Teardown Kubernetes resources.

**Deliverables:**
- Complete Kubernetes inference pipeline
- Multi-node metrics aggregation
- Result logging

---

### Phase 10: Unified Experiment Runner

**Goal:** Create a single entry-point script that runs both local and Kubernetes experiments back-to-back and produces comparable results.

**Tasks:**

1. **`experiments/experiment_runner.py`** — Unified orchestrator.
   - CLI arguments:
     - `--mode`: `local`, `kubernetes`, or `both`
     - `--model`: `transformer` or `cnn`
     - `--num-chunks`: number of chunks for K8s mode
     - `--iterations`: number of inference runs
     - `--batch-size`: input batch size
     - `--output-dir`: directory for results (default: `logs/`)
   - Runs the selected experiment(s).
   - Produces a comparison summary if both modes are run.
   - Saves a combined results file: `logs/experiment_<timestamp>.json`.

**Deliverables:**
- Single CLI-based experiment runner
- Comparison output when both modes are executed

---

### Phase 11: Analysis Module

**Goal:** Build post-experiment analysis tools to compute derived metrics and generate visualizations.

**Tasks:**

1. **`analysis/analyzer.py`** — Metrics computation.
   - Load experiment result JSON files.
   - Compute:
     - Total energy usage (Wh) per experiment
     - Average GPU power (W) per experiment
     - Average inference latency (ms)
     - Throughput (inferences/sec)
     - Energy per inference (Wh/inference)
     - Performance-per-watt ratio
   - Compare local vs. Kubernetes results side-by-side.
   - Output a summary table (printed and saved as CSV).

2. **`analysis/plots.py`** — Matplotlib visualizations.
   - **Power over time:** Line plot of GPU power draw over the experiment duration (both modes overlaid).
   - **GPU utilization over time:** Line plot comparing utilization.
   - **Latency comparison:** Bar chart of average latency (local vs. K8s).
   - **Energy comparison:** Bar chart of total energy consumed.
   - **Energy efficiency:** Bar chart of energy-per-inference.
   - **Throughput comparison:** Bar chart.
   - Save all plots as PNG files in `analysis/output/`.

**Deliverables:**
- Analysis script with summary table
- 6+ publication-quality plots
- Saved outputs

---

### Phase 12: Streamlit Dashboard

**Goal:** Create an interactive web dashboard for real-time and post-experiment visualization.

**Tasks:**

1. **`dashboard/app.py`** — Streamlit application.
   - **Sidebar:** Select experiment results file, filter by mode.
   - **Power Consumption Panel:** Interactive line chart of GPU power over time.
   - **GPU Utilization Panel:** Interactive line chart.
   - **Latency Comparison Panel:** Bar chart comparing local vs. K8s.
   - **Energy Comparison Panel:** Bar chart.
   - **Summary Table:** Key metrics side-by-side.
   - Auto-refresh capability for live monitoring (optional).
   - Load data from `logs/` directory.

**Deliverables:**
- Fully functional Streamlit dashboard
- Runnable with `streamlit run dashboard/app.py`

---

### Phase 13: Documentation & Final Integration

**Goal:** Write documentation, ensure all components integrate cleanly, and verify the end-to-end workflow.

**Tasks:**

1. **`README.md`** — Project documentation.
   - Project overview and goals.
   - Prerequisites (NVIDIA GPU, CUDA, Docker, Kubernetes cluster).
   - Installation instructions.
   - Quick-start guide for local mode.
   - Quick-start guide for Kubernetes mode.
   - Dashboard usage.
   - Analysis instructions.
   - Architecture diagram (text-based).

2. **Integration testing checklist:**
   - [ ] Local mode runs end-to-end and produces logs.
   - [ ] GPU monitoring records data correctly.
   - [ ] Model chunking produces correct outputs (chunk chain output == full model output).
   - [ ] Docker images build successfully.
   - [ ] Kubernetes manifests deploy without errors.
   - [ ] gRPC communication between chunks works.
   - [ ] Kubernetes mode runs end-to-end and produces logs.
   - [ ] Analysis script processes both result sets.
   - [ ] Plots are generated.
   - [ ] Dashboard loads and displays data.

3. **Code quality:**
   - Ensure all modules have docstrings.
   - Consistent logging throughout.
   - Error handling at system boundaries.

**Deliverables:**
- Complete README
- Verified end-to-end pipeline

---

## Phase Dependency Graph

```
Phase 1  (Scaffolding)
   │
   ├── Phase 2  (AI Models)
   │      │
   │      ├── Phase 3  (gRPC Layer)
   │      │      │
   │      │      ├── Phase 6  (Docker)
   │      │      │      │
   │      │      │      └── Phase 7  (K8s YAML)
   │      │      │             │
   │      │      │             ├── Phase 8  (K8s Controller)
   │      │      │             │      │
   │      │      │             │      └── Phase 9  (K8s Runner)
   │      │      │             │             │
   │      │      │             │             └─┐
   │      │      │             │               │
   │      ├── Phase 4  (Monitoring)            │
   │      │      │                             │
   │      │      └── Phase 5  (Local Runner)   │
   │      │             │                      │
   │      │             └──────────────────────┤
   │      │                                    │
   │      │                    Phase 10  (Unified Runner)
   │      │                        │
   │      │                        ├── Phase 11  (Analysis)
   │      │                        │
   │      │                        └── Phase 12  (Dashboard)
   │      │
   │      └── Phase 13  (Documentation & Integration)
```

---

## Technology Stack Summary

| Component          | Technology                        |
|--------------------|-----------------------------------|
| Language           | Python 3.10+                      |
| Deep Learning      | PyTorch                           |
| GPU Monitoring     | pynvml (NVIDIA NVML)              |
| CPU Monitoring     | psutil                            |
| Inter-chunk Comm   | gRPC + Protocol Buffers           |
| Containerization   | Docker (NVIDIA CUDA base images)  |
| Orchestration      | Kubernetes                        |
| K8s Client         | `kubernetes` Python package       |
| Dashboard          | Streamlit                         |
| Plotting           | Matplotlib                        |
| Data Handling      | Pandas, NumPy                     |
| Serialization      | JSON, CSV                         |

---

## Key Design Decisions

1. **gRPC for tensor passing:** Chosen over REST because gRPC supports efficient binary serialization, which is critical for large intermediate tensors between model chunks.

2. **NVML over nvidia-smi:** Direct NVML API calls via `pynvml` are faster and more programmatic than parsing `nvidia-smi` CLI output. They allow precise 1-second sampling without subprocess overhead.

3. **Sequential chunk pipeline:** Model chunks are arranged in a linear pipeline (chunk 1 → chunk 2 → ... → chunk N). This matches the sequential nature of Transformer and CNN architectures and simplifies the communication pattern.

4. **Separate monitoring containers:** Running GPU monitoring as a sidecar or DaemonSet ensures that monitoring does not interfere with inference performance and can capture node-level metrics.

5. **JSON for result storage:** JSON is human-readable, easily loadable in Python, and compatible with both the analysis module and the Streamlit dashboard.

---

## Notes for Development

- **GPU Requirement:** An NVIDIA GPU with CUDA support is required for local mode. Kubernetes mode requires a cluster with GPU-equipped nodes and the NVIDIA device plugin installed.
- **Chunk Count Flexibility:** The system should support 2–8 chunks, configurable at experiment time.
- **Idempotent Kubernetes Deployments:** The controller should handle re-deployments gracefully (delete existing resources before creating new ones).
- **Reproducibility:** All experiments log their configuration (model type, batch size, iterations, chunk count) alongside results for reproducibility.

---

*This guide will be updated as development progresses. Await instruction to begin implementation.*
