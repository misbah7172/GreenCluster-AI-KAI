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
│   ├── chunk_server.py       # gRPC server for serving a model chunk (+ migration RPCs)
│   ├── gateway.py            # HTTP gateway for chaining chunks (+ relinking)
│   ├── hf_loader.py          # HuggingFace model loader
│   ├── layer_chunker.py      # Layer-wise model splitting
│   ├── weight_utils.py       # Partial weight loading from HF checkpoints
│   ├── quantizer.py          # 4-bit/8-bit quantization
│   ├── generation.py         # Autoregressive text generation (+ offloaded prefetch)
│   ├── resource_detector.py  # GPU/CPU/RAM detection
│   ├── auto_partitioner.py   # Smart layer-to-node assignment
│   ├── deas_scheduler.py     # Dynamic Energy-Aware Scheduler (Phase 21)
│   ├── tiered_weight_manager.py  # GPU/RAM/Disk tiered weight placement (Phase 22)
│   ├── prefetch_engine.py    # Double-buffered async weight prefetching (Phase 22)
│   │
│   │   # === NEXT-GEN FEATURES (Phase 24) ===
│   ├── plugin_architecture.py    # Modular plugin system with registry
│   ├── adaptive_precision.py     # Dynamic precision based on layer criticality
│   ├── kv_cache_optimizer.py     # Mixed-precision KV cache with reuse
│   ├── intelligent_placement.py  # Multi-objective placement optimization
│   ├── network_aware_scheduler.py # Enhanced DEAS with network awareness
│   ├── hybrid_parallelism.py     # Pipeline + tensor parallelism engine
│   ├── energy_feedback_loop.py   # PID-based energy optimization
│   ├── speculative_decoder.py    # Draft model speculation with verification
│   ├── fault_tolerant_pipeline.py # Failure detection and recovery
│   ├── auto_tuner.py             # Auto-tuning benchmark system
│   │
│   │   # === ADVANCED ALGORITHMS (Phase 25) ===
│   ├── fcim_worker_selector.py   # FCIM - Fair Cost-Efficient Worker Selection
│   ├── adsa_scheduler.py         # ADSA - Adaptive Dynamic Scheduling Algorithm
│   ├── active_inference.py       # Active Inference (Non-DRL) controller
│   ├── batch_processor.py        # Dynamic batch processing
│   ├── dfs_scheduler.py          # DFS scheduler with pruning
│   ├── ilp_scheduler.py          # ILP/Heuristic scheduler
│   ├── onnx_converter.py         # PyTorch to ONNX conversion
│   └── simulation_optimizer.py   # Simulation optimization tricks
│
├── monitoring/               # Power and performance monitoring
│   ├── __init__.py
│   ├── gpu_monitor.py        # NVIDIA NVML-based GPU power monitoring (+ ring buffer, TDP)
│   ├── cpu_monitor.py        # CPU usage monitoring
│   ├── metrics.py            # Unified metrics collection (+ trapezoidal energy, EER)
│   ├── monitor_service.py    # HTTP service wrapping MetricsCollector (+ threshold endpoints)
│   ├── event_bus.py          # Async pub/sub for threshold events (Phase 20)
│   └── threshold_service.py  # TDP-relative power threshold evaluation (Phase 20)
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
│   └── controller.py         # Python K8s controller (+ DEAS, threshold propagation)
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
│   └── app.py                # (+ migration/offloading panels)
│
├── logs/                     # Experiment output logs (CSV/JSON)
│   └── .gitkeep
│
├── analysis/                 # Post-experiment analysis and plotting
│   ├── __init__.py
│   ├── analyzer.py           # Compute energy, latency, efficiency (+ migration, offloading)
│   └── plots.py              # Matplotlib visualization (10 plot types)
│
├── proto/                    # gRPC protocol buffer definitions
│   └── inference.proto       # InferenceService (+ Pause/Checkpoint/Resume RPCs)
│
├── tests/                    # Test suites
│   ├── test_integration.py   # 25 tests (Phases 1-13)
│   ├── test_distributed.py   # 30 tests (Phases 14-18)
│   ├── test_phase19.py       # 27 tests (Phase 19)
│   ├── test_phase20.py       # ~15 tests (Phase 20)
│   ├── test_phase21.py       # ~19 tests (Phase 21)
│   ├── test_phase22.py       # ~14 tests (Phase 22)
│   ├── test_phase23.py       # ~14 tests (Phase 23)
│   └── test_nextgen_features.py  # ~80 tests (Phases 24-25: next-gen features + algorithms)
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

## Pivot: Distributed Inference for Large Models on Low-End Hardware

> **Phases 14–18** extend KAI from an energy-benchmarking tool into a
> **distributed inference platform** that can run large HuggingFace models
> (e.g., LLaMA 7B, Mistral 7B, Phi-2) on clusters of low-end PCs using
> Kubernetes. Each node loads only the layers it is responsible for, so no
> single machine needs enough VRAM/RAM for the full model.

---

### Phase 14: HuggingFace Large Model Support

**Goal:** Load real HuggingFace models (LLMs) and integrate them into the KAI pipeline alongside the existing benchmark models.

**Tasks:**

1. **`model/hf_loader.py`** — HuggingFace model loader.
   - Accept a HuggingFace model name or local path (e.g., `microsoft/phi-2`, `meta-llama/Llama-2-7b-hf`).
   - Load the tokenizer via `transformers.AutoTokenizer`.
   - Load the model configuration and layer structure via `transformers.AutoModelForCausalLM` or `AutoConfig`.
   - Support loading with `torch_dtype=torch.float16` / `bfloat16` to reduce memory.
   - Provide `get_layer_list()` returning an ordered list of `(name, module)` for all model layers (embeddings, transformer blocks, output head).
   - Provide `get_tokenizer()`, `get_config()` helper methods.
   - **Memory-safe loading:** Support `device_map="meta"` or `low_cpu_mem_usage=True` to avoid loading full weights into RAM during layer enumeration.
   - Validate that the model is a causal LM or encoder-decoder; refuse unsupported architectures.

2. **Update `model/__init__.py`** — Expose the new module.

3. **Update `requirements.txt`** — Add:
   - `transformers>=4.36.0`
   - `accelerate>=0.25.0`
   - `sentencepiece>=0.1.99`
   - `safetensors>=0.4.0`

**Deliverables:**
- `model/hf_loader.py` with `HFModelLoader` class
- Able to enumerate layers of any HuggingFace causal LM without loading full weights
- Tokenizer integration for encoding text prompts → token IDs

---

### Phase 15: Layer-Wise Distributed Chunking

**Goal:** Split a large HuggingFace model layer-by-layer across nodes so each node holds only its assigned layers in memory.

**Tasks:**

1. **`model/layer_chunker.py`** — Layer-aware chunking for HuggingFace models.
   - Accept the output of `HFModelLoader.get_layer_list()`.
   - Split layers into N groups (configurable).
   - Each chunk loads only its assigned layers' weights from disk/hub — NOT the full model.
   - Use `safetensors` shard loading or `accelerate`'s `load_checkpoint_in_model` to load only specific layers.
   - Each chunk is a `LayerChunk(nn.Module)` that:
     - Holds its assigned layers.
     - Accepts hidden states as input, returns hidden states as output.
     - Handles embedding layer (chunk 0 only) and LM head (last chunk only).
   - Provide `estimate_chunk_memory(chunk)` → estimated VRAM in MB.
   - Provide `save_chunk_weights(chunk, path)` and `load_chunk_weights(chunk, path)`.

2. **`model/weight_utils.py`** — Utility for partial weight loading.
   - Map HuggingFace checkpoint shards to specific layer indices.
   - Load only needed shard files for a given chunk.
   - Support both `.safetensors` and `.bin` weight formats.
   - Avoid loading the entire model into memory at any point.

**Deliverables:**
- Layer-wise chunking that loads only needed weights per chunk
- Memory estimation per chunk
- Chunk serialization for deployment to nodes

---

### Phase 16: Text Generation Inference Pipeline

**Goal:** Implement a text generation pipeline that chains chunk services to perform autoregressive token generation across distributed nodes.

**Tasks:**

1. **`model/generation.py`** — Distributed text generation engine.
   - Implement autoregressive generation loop:
     1. Tokenize input prompt.
     2. Forward through all chunks (chunk 0 → chunk 1 → ... → chunk N).
     3. Chunk N returns logits; sample next token.
     4. Append token to sequence; repeat until stop condition.
   - Support generation parameters:
     - `max_new_tokens`: maximum tokens to generate.
     - `temperature`: sampling temperature.
     - `top_k`: top-k sampling.
     - `top_p`: nucleus sampling.
     - `repetition_penalty`: penalize repeated tokens.
   - Implement KV-cache forwarding:
     - Each chunk maintains a key-value cache for its layers.
     - On subsequent tokens, only the new token is forwarded (not the full sequence).
     - Cache is stored on the chunk's device, passed by reference (not over network).
   - Streaming support: yield tokens as they are generated.

2. **Update `model/chunk_server.py`** — Extend the gRPC servicer:
   - Add `InferCausal` RPC method for causal LM chunks.
   - Handle KV-cache state across calls within a session.
   - Accept `session_id` to maintain per-request caches.
   - Add `ClearCache` RPC to free memory when generation is done.

3. **Update `proto/inference.proto`** — Add new messages:
   - `CausalInferRequest`: token IDs, session_id, generation step.
   - `CausalInferResponse`: output logits/hidden states, cache status.
   - `ClearCacheRequest` / `ClearCacheResponse`.

4. **Update `model/gateway.py`** — Add generation endpoint:
   - `POST /generate` accepts `{"prompt": "...", "max_tokens": 100, ...}`.
   - Orchestrates the generation loop across chunk services.
   - Returns generated text (and optionally streams tokens via SSE).

**Deliverables:**
- End-to-end text generation across distributed chunks
- KV-cache support for efficient autoregressive decoding
- Streaming token output
- Generation parameter support (temperature, top_k, top_p)

---

### Phase 17: Smart Resource Detection & Auto-Partitioning

**Goal:** Automatically detect available resources on each Kubernetes node and partition the model proportionally.

**Tasks:**

1. **`model/resource_detector.py`** — Node resource scanner.
   - Query each Kubernetes node for:
     - Available GPU VRAM (via NVML if GPU present, else 0).
     - Available system RAM.
     - CPU core count.
   - Also detect GPU type (e.g., RTX 3050 Ti 4GB vs. GTX 1060 6GB).
   - Return a sorted list of nodes with their capabilities.
   - Support CPU-only nodes (layers can run on CPU with reduced speed).

2. **`model/auto_partitioner.py`** — Intelligent model partitioner.
   - Given the model's layer list and each node's capability:
     - Estimate memory per layer (parameters × dtype size × ~1.2 overhead).
     - Assign layers to nodes proportional to their available memory.
     - Nodes with more VRAM get more layers.
     - Ensure every layer is assigned to exactly one node.
   - Handle edge cases:
     - Model too large for the cluster → clear error message with requirements.
     - Only one node available → all layers on that node.
     - Mixed GPU/CPU nodes → GPU nodes get more layers.
   - Output a partition plan: `{node_id: [layer_start, layer_end], ...}`.

3. **Update `kubernetes/controller.py`** — Enhanced deployment:
   - Use the partition plan to create per-node deployments.
   - Pass `LAYER_START` and `LAYER_END` environment variables to chunk containers.
   - Deploy chunk containers with appropriate resource limits matching node capabilities.

**Deliverables:**
- Automatic node capability detection
- Proportional model partitioning based on real hardware
- Partition plan generation and validation

---

### Phase 18: End-to-End CLI & Integration

**Goal:** Provide a unified CLI that lets users run large models on their K8s cluster with a single command, plus update all documentation.

**Tasks:**

1. **`kai_cli.py`** — Main entry point (root of project).
   - Subcommands:
     - `kai run --model <hf_model_name> --prompt "Hello" --max-tokens 100`
       → Downloads model, partitions across cluster, generates text.
     - `kai scan` → Detect cluster resources and show capabilities.
     - `kai partition --model <name> --preview`
       → Show how the model would be split without deploying.
     - `kai benchmark --model <name> --mode both`
       → Run the original energy benchmarking workflow.
     - `kai dashboard` → Launch the Streamlit dashboard.
   - Progress indicators for model download, partitioning, deployment.
   - Clean error messages for common failures (no cluster, OOM, etc.).

2. **Update `docker/Dockerfile.chunk`** — Support HuggingFace models:
   - Install `transformers`, `accelerate`, `safetensors`.
   - Accept `HF_MODEL_NAME`, `LAYER_START`, `LAYER_END` env vars.
   - Load only assigned layers on startup.

3. **Update `README.md`** — Complete documentation rewrite:
   - New project description: distributed inference platform for large AI models.
   - Quick-start: "Run LLaMA 7B on 3 budget PCs."
   - Architecture diagram showing layer distribution.
   - Resource requirements table.
   - Comparison with Petals, AirLLM, Ollama.

4. **Update integration tests** — Add tests for new modules:
   - Test HFModelLoader with a small model (e.g., `sshleifer/tiny-gpt2`).
   - Test layer chunking produces valid chunks.
   - Test generation pipeline end-to-end with tiny model.
   - Test resource detector with mock data.
   - Test auto-partitioner with various node configurations.

**Deliverables:**
- Single-command CLI for distributed LLM inference
- Updated Docker images for HuggingFace model serving
- Complete documentation and tests

---

### Phase 19: Gap Coverage & Production Readiness

**Goal:** Close 5 remaining gaps to make KAI fully operational end-to-end without manual steps, support real HuggingFace models in benchmarks, handle 70B+ parameter models, and add quantization support.

**Tasks:**

1. **`kai_cli.py build`** — Auto-build Docker images.
   - Add `build` subcommand that runs `docker build` for all 3 images (chunk, gateway, monitor).
   - Accepts `--tag` for custom image tags.
   - Optionally pushes to a container registry with `--push`.
   - Validates Docker is installed and accessible.

2. **`kai_cli.py prepare`** — Save chunk weights for K8s deployment.
   - Add `prepare` subcommand that downloads a model, splits it into chunks, and saves each chunk's weights to disk.
   - Accepts `--model`, `--num-chunks`, `--output-dir`, `--dtype`.
   - Output directory structure: `<output_dir>/chunk_0.pt`, `chunk_1.pt`, etc.
   - These files are what K8s chunk pods mount from a shared volume (NFS/PVC).
   - Uses shard-based loading (`WeightMapper`) for large models to avoid OOM.

3. **Extend `benchmark` to support HuggingFace models.**
   - Add `--hf-model` argument to `kai_cli.py benchmark`.
   - When `--hf-model` is provided, use the HuggingFace model for both local and K8s benchmarks instead of the dummy transformer/CNN.
   - Local mode: loads model on single GPU, runs generation, records energy.
   - K8s mode: deploys chunks via controller, runs generation through gateway, records energy.
   - Comparison output includes energy/power ratios proving distributed = lower power.

4. **Integrate shard-based weight loading into CLI for 70B+ models.**
   - Update `_load_real_weights()` in `kai_cli.py` to use `WeightMapper.load_state_dict_for_layers()` when the full model would exceed available RAM.
   - Detect available system memory and choose between full-model load (fast, for small models) vs. shard-by-shard load (memory-safe, for large models).
   - Each chunk loads only its own weight shards — never the full model.

5. **Add quantization support via `bitsandbytes`.**
   - Create `model/quantizer.py` — Quantization utility.
     - Support 4-bit (NF4) and 8-bit (INT8) quantization via `bitsandbytes`.
     - `quantize_chunk(chunk, mode="4bit"|"8bit")` quantizes a LayerChunk in-place.
     - `estimate_quantized_memory(chunk, mode)` returns expected memory after quantization.
   - Add `--quantize` argument to `kai_cli.py run` and `kai_cli.py prepare`.
   - When enabled, each chunk is quantized after weight loading.
   - Reduces memory per chunk by 4-8x while maintaining near-full-precision output.
   - Update `requirements.txt` with `bitsandbytes>=0.41.0`.

6. **Update Docker images for HuggingFace support.**
   - Update `Dockerfile.chunk` to install `transformers`, `accelerate`, `safetensors`, `bitsandbytes`.
   - Add `HF_MODEL_NAME`, `LAYER_START`, `LAYER_END`, `QUANTIZE` env vars.
   - Chunk container loads only its assigned layers on startup using `WeightMapper`.

7. **Add integration tests for all new functionality.**
   - Test `build` command (validates Docker command construction).
   - Test `prepare` command (saves and reloads chunk weights).
   - Test shard-based loading with a small model.
   - Test quantization (4-bit/8-bit) on a small model.
   - Test HF benchmark mode.

**Deliverables:**
- `kai_cli.py build` and `kai_cli.py prepare` commands
- HuggingFace model support in benchmark mode
- Shard-based weight loading for 70B+ models
- `model/quantizer.py` with 4-bit and 8-bit quantization
- Updated Docker images
- Integration tests for all new features

---

### Phase 20: Real-Time Energy Instrumentation & Trigger Framework

**Goal:** Replace coarse 1-second GPU sampling with sub-100ms monitoring, add bounded ring buffers, auto-detect GPU TDP, compute energy via trapezoidal integration, and create an async event bus for power threshold alerts.

**Tasks:**

1. **`monitoring/gpu_monitor.py`** (modified) — Add ring buffer (`collections.deque(maxlen=buffer_size)`), TDP auto-detection via NVML, spike callback when power >= 80% TDP, `tdp_pct` key in samples, `get_recent_samples(n)` and `get_tdp_watts()` methods.

2. **`monitoring/metrics.py`** (modified) — Replace `avg_power * time / 3600` with trapezoidal integration via `_compute_energy_trapezoidal()`. Add `enable_threshold`, `tdp_watts`, `node_name` params. Wire EventBus and PowerThresholdService lifecycle.

3. **`monitoring/event_bus.py`** (new) — Thread-safe async pub/sub with `ThresholdLevel` enum (OPTIMAL/WARNING/CRITICAL), `ThresholdEvent` dataclass, bounded queue, background dispatcher, subscriber level filtering, event history deque.

4. **`monitoring/threshold_service.py`** (new) — Evaluates GPU power against TDP thresholds (< 70% = OPTIMAL, 70-80% = WARNING, >= 80% = CRITICAL), publishes to EventBus on state change.

5. **`monitoring/monitor_service.py`** (modified) — New endpoints: `GET /metrics/threshold`, `GET /metrics/events?n=50`. New CLI args: `--sampling-rate`, `--tdp-watts`, `--enable-threshold`.

6. **`kubernetes/controller.py`** (modified) — Propagate sampling_rate, tdp_watts, enable_threshold as container env vars. New `collect_threshold_events()` method.

7. **`kai_cli.py`** (modified) — Add `--sampling-rate` to benchmark subparser.

8. **`tests/test_phase20.py`** — ~15 tests covering sampling, ring buffer, trapezoidal energy, event bus, threshold classification.

---

### Phase 21: Dynamic Energy-Aware Scheduling (DEAS)

**Goal:** Enable runtime chunk migration between nodes based on energy metrics, with automatic rebalancing when GPUs exceed power thresholds.

**Tasks:**

1. **`monitoring/metrics.py`** (modified) — Add `energy_efficiency_ratio` (EER = throughput / avg_power) to `compute_summary()` output.

2. **`proto/inference.proto`** (modified) — Add `Pause`, `Checkpoint`, `Resume` RPCs with corresponding request/response messages.

3. **`model/chunk_server.py`** (modified) — Implement `Pause()` (sets `_paused`), `Checkpoint()` (saves weights + hidden state), `Resume()` (restores + unpauses). Guard `Infer()` against paused state.

4. **`model/gateway.py`** (modified) — Add `_chain_lock` for thread-safe inference, `update_chunk_host()` for live stub replacement, `get_chain_topology()`. New HTTP endpoints: `POST /relink`, `GET /topology`.

5. **`model/deas_scheduler.py`** (new) — `DEASScheduler` subscribes to CRITICAL events, enforces cooldown, plans migrations (diff current vs optimal partition), executes 5-step migration workflow (Pause → Checkpoint → Migrate → Relink → Resume).

6. **`kubernetes/controller.py`** (modified) — `start_deas()`, `stop_deas()`, `get_node_energy_profiles()`, `trigger_rebalance()` methods.

7. **`tests/test_phase21.py`** — ~19 tests covering EER, DEAS scheduler, migration RPCs, gateway relinking.

---

### Phase 22: FlexGen-Style CPU Offloading

**Goal:** Enable models exceeding total cluster VRAM to run by spilling weights to System RAM and disk, with double-buffered prefetching to hide transfer latency.

**Tasks:**

1. **`model/tiered_weight_manager.py`** (new) — `StorageTier` enum (GPU_VRAM/SYSTEM_RAM/DISK_SWAP), `TieredWeightManager` with `plan_placement()` (fill GPU first, then RAM, then disk), `store_layer_to_disk()` (safetensors format), `load_layer_from_disk()`, `promote_layer()`, `demote_layer()`.

2. **`model/prefetch_engine.py`** (new) — Double-buffered prefetcher with `prefetch_layer()` (non-blocking background thread), `wait_and_swap()` (blocks + swaps buffers), timing stats (prefetch_ms, swap_ms, gpu_idle_ms, overlap_pct).

3. **`model/layer_chunker.py`** (modified) — Add `create_offloaded_chunks()` method returning `(chunks, weight_manager, prefetch_engine)`.

4. **`model/generation.py`** (modified) — Add `prefetch_engine` and `weight_manager` params to `DistributedGenerator`. New `_forward_all_chunks_offloaded()` method that prefetches chunk N+1 while computing chunk N.

5. **`kai_cli.py`** (modified) — Add `--offload`, `--gpu-budget-mb`, `--disk-swap-dir` to `run` subparser.

6. **`tests/test_phase22.py`** — ~14 tests covering tiered weights, prefetch engine, offloaded generation, CLI flags.

---

### Phase 23: Integrated Validation & Energy-Efficiency Analysis

**Goal:** Add migration energy analysis, offloading overhead comparison, new visualization plots, and dashboard panels.

**Tasks:**

1. **`analysis/analyzer.py`** (modified) — Add `compute_migration_energy()` (trapezoidal integration over migration window), `compute_offloading_overhead()` (baseline vs offloaded latency comparison), `analyse_deas()` (migration history, cluster EER). Update `analyse_k8s()` with `migration_count` and `cluster_eer` keys.

2. **`analysis/plots.py`** (modified) — Add `plot_migration_energy_impact()` (power timeline with migration windows shaded) and `plot_vram_ram_tradeoff()` (grouped bar chart of GPU vs CPU execution time). Add both to `plot_all()`.

3. **`dashboard/app.py`** (modified) — Add "Migration Energy Impact" panel (power timeline + event summary table) and "VRAM vs RAM Execution Trade-off" panel (grouped bar chart with memory saved annotations).

4. **`tests/test_phase23.py`** — ~14 tests covering trigger latency, migration integrity, tiered transitions, energy recovery, analyzer methods, plot imports.

---

### Phase 24: Next-Generation Features

**Goal:** Implement 10 advanced features to improve efficiency, scalability, and intelligence without reducing model output quality.

**Tasks:**

1. **`model/plugin_architecture.py`** (new) — Modular plugin system with `PluginRegistry` for registering and retrieving plugins by category (scheduler, optimizer, executor, cache, placement, parallelism). Base plugin interfaces: `SchedulerPlugin`, `OptimizerPlugin`, `ExecutorPlugin`, `CachePlugin`, `PlacementPlugin`, `ParallelismPlugin`. Default implementations for each.

2. **`model/adaptive_precision.py`** (new) — `AdaptivePrecisionController` with `analyze_model()` for layer criticality analysis, `PrecisionLevel` enum (FP32, FP16, INT8, INT4), `get_precision_for_layer()` based on criticality and pressure, `apply_precision()` to dynamically apply precision per layer.

3. **`model/kv_cache_optimizer.py`** (new) — `MixedPrecisionKVCache` (FP16 recent tokens, INT8 older tokens), `QuantizedTensor` class with scale/zero_point, `CacheReuser` for prompt overlap detection, `KVCacheOptimizer` high-level interface with memory-aware eviction.

4. **`model/intelligent_placement.py`** (new) — `IntelligentPlacementEngine` with multi-objective optimization (`_optimize_latency`, `_optimize_energy`, `_optimize_memory`, `_optimize_balanced`), `NetworkTopology` for inter-node latency/bandwidth, `LayerInfo`, `PlacementAssignment`, `PlacementPlan` dataclasses.

5. **`model/network_aware_scheduler.py`** (new) — `NetworkAwareScheduler` extending `DEASScheduler`, `NetworkMonitor` for continuous latency/bandwidth measurement, `compute_network_aware_plan()` that groups layers to minimize inter-node transfers.

6. **`model/hybrid_parallelism.py`** (new) — `HybridParallelismEngine` with `ParallelismMode` enum (PIPELINE_ONLY, TENSOR_ONLY, HYBRID), `AttentionParallel` and `FeedForwardParallel` wrappers for tensor parallelism, `WorkloadAnalyzer.recommend_mode()` for automatic strategy selection.

7. **`model/energy_feedback_loop.py`** (new) — `EnergyFeedbackController` with PID controllers for power and latency targets, `ControlConfig` dataclass (batch_size, power_limit, precision, offload_threshold), `PowerLimitManager` for NVML power limit control.

8. **`model/speculative_decoder.py`** (new) — `SpeculativeDecoder` with `DraftModelWrapper`, `VerificationMode` enum (STRICT, THRESHOLD, SAMPLING), rejection sampling for mathematically exact output, `AdaptiveSpeculativeDecoder` that adjusts speculation length.

9. **`model/fault_tolerant_pipeline.py`** (new) — `FaultTolerantPipeline` wrapping `InferenceGateway`, `FailureDetector` with health monitoring, `CheckpointManager` for recovery, `LayerReassigner` for failover.

10. **`model/auto_tuner.py`** (new) — `AutoTuner` with `ConfigurationSpace`, `BenchmarkRunner`, `TrialResult`, `TuningResult`. Search strategies: `RandomSearch`, `GridSearch`, `BayesianSearch`. `TuningObjective` enum (LATENCY, THROUGHPUT, ENERGY_EFFICIENCY, MEMORY, BALANCED).

11. **`kai_cli.py`** (modified) — Add 7 new subcommands: `autotune`, `speculative`, `hybrid`, `placement`, `energy-loop`, `fault-tolerant`, `plugins`.

12. **`tests/test_nextgen_features.py`** (new) — ~40 tests covering all 10 new modules.

**Deliverables:**
- 10 new modules implementing next-generation features
- 7 new CLI commands
- ~40 new tests
- Updated documentation

---

### Phase 25: Advanced Algorithms & Optimization

**Goal:** Implement advanced scheduling, optimization, and simulation algorithms for maximum efficiency and scalability.

**Tasks:**

1. **`model/fcim_worker_selector.py`** — Fair Cost-Efficient Worker Selection
   - Multi-criteria worker scoring: cost, efficiency, fairness
   - Jain's Fairness Index for balanced allocation
   - Dynamic fairness tracking across allocations

2. **`model/adsa_scheduler.py`** — Adaptive Dynamic Scheduling Algorithm
   - Multiple scheduling policies: FIFO, SJF, SRPT, WEIGHTED, ADAPTIVE
   - Task aging to prevent starvation
   - Dynamic policy switching based on workload

3. **`model/active_inference.py`** — Active Inference Controller (Non-DRL)
   - Bayesian belief updating from environment observations
   - Expected Free Energy (EFE) minimization
   - Real-time decision adjustment under uncertainty
   - State inference without deep reinforcement learning

4. **`model/batch_processor.py`** — Dynamic Batch Processing
   - Multiple batching strategies: FIXED_SIZE, FIXED_TIME, ADAPTIVE, CONTINUOUS
   - Continuous batching for streaming workloads
   - Memory-aware batch formation
   - Request prioritization

5. **`model/dfs_scheduler.py`** — DFS Scheduler with Pruning
   - Depth-first search for task scheduling
   - Multiple pruning strategies: ALPHA_BETA, BOUND, BEAM, HEURISTIC
   - Branch-and-bound optimization
   - Configurable search depth and beam width

6. **`model/ilp_scheduler.py`** — ILP/Heuristic Scheduler
   - Integer Linear Programming for small systems (uses PuLP)
   - Genetic algorithm for large-scale optimization
   - Simulated annealing fallback
   - Automatic algorithm selection based on problem size

7. **`model/onnx_converter.py`** — PyTorch to ONNX Conversion
   - Model export with configurable opset version
   - ONNX Runtime optimization
   - Dynamic quantization (INT8)
   - Output validation between PyTorch and ONNX

8. **`model/simulation_optimizer.py`** — Simulation Optimization Tricks
   - Layer simplification (repeated layer grouping)
   - Decode phase approximation (sampling)
   - Attention approximation
   - Quantization during simulation
   - Multi-level optimization control

**New CLI Commands:**
- `kai_cli.py fcim --report` — FCIM worker selection analysis
- `kai_cli.py adsa --policy adaptive` — ADSA scheduling
- `kai_cli.py batch --strategy adaptive` — Batch processing
- `kai_cli.py active-inference --show-beliefs` — Active inference
- `kai_cli.py dfs-scheduler --pruning bound` — DFS scheduling
- `kai_cli.py ilp-scheduler --algorithm auto` — ILP/heuristic scheduling
- `kai_cli.py onnx --model <name> --optimize` — ONNX conversion
- `kai_cli.py simulate --optimization-level 2` — Optimized simulation

**Deliverables:**
- 8 new advanced algorithm modules
- 9 new CLI commands
- ~40 new tests for Phase 25 features
- Updated documentation

---

## Updated Phase Dependency Graph

```
Phase 1-13  (Original KAI -- Energy Benchmarking)
   |         (All completed)
   |
   +-- Phase 14  (HuggingFace Model Support)
          |
          +-- Phase 15  (Layer-Wise Chunking)
          |      |
          |      +-- Phase 16  (Generation Pipeline)
          |      |
          |      +-- Phase 17  (Smart Resource Detection)
          |             |
          |             +-- Phase 18  (E2E CLI & Integration)
          |                    |
          |                    +-- Phase 19  (Gap Coverage & Production Readiness)
          |                           |
          |                           +-- Phase 20  (Real-Time Instrumentation & Event Bus)
          |                                  |
          |                                  +-- Phase 21  (Dynamic Energy-Aware Scheduling)
          |                                         |
          |                                         +-- Phase 22  (CPU/Disk Offloading)
          |                                                |
          |                                                +-- Phase 23  (Validation & Analysis)
          |                                                       |
          |                                                       +-- Phase 24  (Next-Gen Features)
          |                                                              |
          |                                                              +-- Phase 25  (Advanced Algorithms)
```

---

## Updated Technology Stack

| Component            | Technology                                |
|----------------------|-------------------------------------------|
| Language             | Python 3.10+                              |
| Deep Learning        | PyTorch                                   |
| LLM Models           | HuggingFace Transformers + Accelerate     |
| Weight Format        | SafeTensors                               |
| Tokenization         | HuggingFace Tokenizers / SentencePiece    |
| Quantization         | bitsandbytes (4-bit NF4, 8-bit INT8)      |
| GPU Monitoring       | pynvml (NVIDIA NVML)                      |
| CPU Monitoring       | psutil                                    |
| Inter-chunk Comm     | gRPC + Protocol Buffers                   |
| Containerization     | Docker (NVIDIA CUDA base images)          |
| Orchestration        | Kubernetes                                |
| K8s Client           | `kubernetes` Python package               |
| Dashboard            | Streamlit                                 |
| Plotting             | Matplotlib                                |
| Data Handling        | Pandas, NumPy                             |
| ILP Optimization     | PuLP (optional)                           |
| ONNX Export          | ONNX, ONNXRuntime (optional)              |
| Bayesian Inference   | NumPy/SciPy                               |

---

*This guide will be updated as development progresses. Await instruction to begin implementation.*

---

## Implementation Status - 2026-04-11

### Build-Guide Context
- The phased build workflow remains accurate as a development narrative and structure reference.
- Current implementation adds a stabilized comprehensive dashboard workflow as the primary runtime validation surface.
- Telemetry and KV instrumentation described across phases are now exposed through live session-driven UI panels.

### Current Practical Build/Run Reality
- For GPU-backed execution, use .venv310 (CUDA torch).
- .venv is CPU-only and should be treated as a fallback environment for non-GPU checks.
- Dashboard-driven validation is now a first-class part of post-build verification.

### Suggested Verification Commands
```bash
# CUDA verification
./.venv310/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Dashboard verification
./.venv310/Scripts/python -m streamlit run dashboard/comprehensive_dashboard.py
```

### Reader Note
- This build guide status note reflects the implementation state as of 2026-04-11.