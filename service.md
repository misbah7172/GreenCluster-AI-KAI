# KAI — Service Overview

## What KAI Can Do

KAI is a **distributed AI inference platform** that lets you run large language models across a cluster of low-end PCs using Kubernetes. Here's what it can do:

---

## Subsystem-to-Role Map (Quick)

Use this format: `role (module)`.

### Core Inference Pipeline

- **conductor** (`kai_cli.py`) - Main command router for run, benchmark, deploy, tune, and analysis flows.
- **launcher** (`kai_cli_dashboard.py`) - Starts the recommended dashboard runtime profile.
- **loader** (`model/hf_loader.py`) - Reads HuggingFace model structure and metadata.
- **scanner** (`model/resource_detector.py`) - Detects GPU VRAM, RAM, CPU, and node capabilities.
- **splitter** (`model/chunker.py`) - Splits baseline models into sequential chunk modules.
- **layer-splitter** (`model/layer_chunker.py`) - Splits transformer blocks layer-wise for distributed serving.
- **assigner** (`model/auto_partitioner.py`) - Maps layers/chunks to nodes based on capacity.
- **server** (`model/chunk_server.py`) - gRPC chunk worker that executes assigned layers.
- **router** (`model/gateway.py`) - HTTP/gRPC gateway that chains chunk services and routes requests.
- **generator** (`model/generation.py`) - Autoregressive decoding pipeline across distributed chunks.
- **weights-reader** (`model/weight_utils.py`) - Loads partial checkpoint weights for chunk-specific execution.
- **quantizer** (`model/quantizer.py`) - Applies 4-bit/8-bit quantization paths.
- **model-core** (`model/transformer.py`, `model/cnn.py`) - Built-in reference model definitions.

### Scheduling, Control, and Optimization

- **planner** (`model/ilp_scheduler.py`) - ILP and hybrid assignment optimization engine.
- **tuner** (`model/energy_feedback_loop.py`) - Closed-loop energy-performance controller.
- **fixer** (`model/deas_scheduler.py`) - Predictive, cost-aware rebalance and migration planner.
- **network-fixer** (`model/network_aware_scheduler.py`) - DEAS extension that includes network penalties.
- **placer** (`model/intelligent_placement.py`) - Multi-objective placement optimizer.
- **parallelizer** (`model/hybrid_parallelism.py`) - Pipeline plus tensor parallel strategy manager.
- **accelerator** (`model/speculative_decoder.py`) - Draft-and-verify speculative decoding engine.
- **healer** (`model/fault_tolerant_pipeline.py`) - Failure detection, checkpoint recovery, and reassignment.
- **precision-tuner** (`model/adaptive_precision.py`) - Dynamic FP16/INT8/INT4 precision control.
- **runtime-precision-manager** (`model/runtime_precision_manager.py`) - Runtime precision plan switching under memory pressure.
- **cache-manager** (`model/kv_cache_optimizer.py`) - KV cache reuse, compression, and eviction policies.
- **tier-manager** (`model/tiered_weight_manager.py`) - GPU/RAM/disk weight placement manager.
- **prefetcher** (`model/prefetch_engine.py`) - Async prefetch to hide offload transfer latency.
- **layer-streamer** (`model/layer_streamer.py`) - Streams layer batches through GPU with load/compute/evict flow.
- **memory-pool** (`model/gpu_memory_pool.py`) - Reuses preallocated GPU buffers to reduce fragmentation.
- **oom-guardian** (`model/oom_guardian.py`) - Classifies memory pressure and triggers preemptive safeguards.
- **batch-governor** (`model/adaptive_batch_controller.py`) - Adapts batch size based on pressure and latency outcomes.
- **auto-tuner** (`model/auto_tuner.py`) - Searches configs for best objective (energy/latency/throughput).
- **plugin-hub** (`model/plugin_architecture.py`) - Strategy registry and extension interface.
- **latency-prober** (`model/latency_probe.py`) - Real endpoint RTT probing with cache.

### Advanced Algorithm Modules (Phase 25)

- **fairness-selector** (`model/fcim_worker_selector.py`) - Cost-performance-fairness worker selection.
- **task-reorderer** (`model/adsa_scheduler.py`) - Adaptive task scheduling with aging and policy switching.
- **belief-controller** (`model/active_inference.py`) - Bayesian decision control without DRL.
- **batch-scheduler** (`model/batch_processor.py`) - Dynamic/continuous batching control.
- **search-scheduler** (`model/dfs_scheduler.py`) - DFS-based scheduling with pruning strategies.
- **exporter** (`model/onnx_converter.py`) - PyTorch-to-ONNX conversion and optimization.
- **sim-optimizer** (`model/simulation_optimizer.py`) - Fast simulation with approximation controls.

### Monitoring and Telemetry

- **gpu-sampler** (`monitoring/gpu_monitor.py`) - NVML GPU power/util/temp/memory sampling.
- **cpu-sampler** (`monitoring/cpu_monitor.py`) - CPU usage sampling.
- **metrics-aggregator** (`monitoring/metrics.py`) - Unified energy, latency, throughput, and EER metrics.
- **threshold-evaluator** (`monitoring/threshold_service.py`) - TDP-based OPTIMAL/WARNING/CRITICAL detection.
- **event-bus** (`monitoring/event_bus.py`) - Async pub/sub for threshold and control signals.
- **monitor-api** (`monitoring/monitor_service.py`) - HTTP service exposing monitor and threshold endpoints.
- **telemetry-recorder** (`monitoring/telemetry.py`) - Routing/inference telemetry persistence and export.

### Kubernetes and Runtime Infrastructure

- **orchestrator** (`kubernetes/controller.py`) - Deploys services, gathers profiles, and triggers rebalance.
- **deploy-specs** (`kubernetes/deployments/`) - Deployment manifests for chunk/gateway/monitor services.
- **service-specs** (`kubernetes/services/`) - Kubernetes service manifests for traffic routing.
- **quota-guard** (`kubernetes/gpu-resource-quota.yaml`) - Namespace GPU quota policy.
- **image-chunk** (`docker/Dockerfile.chunk`) - Container build recipe for chunk workers.
- **image-gateway** (`docker/Dockerfile.gateway`) - Container build recipe for gateway.
- **image-monitor** (`docker/Dockerfile.monitor`) - Container build recipe for monitor daemon.
- **local-stack** (`docker/docker-compose.yml`) - Local multi-service compose profile.

### Experiments, Analysis, and Dashboard

- **experiment-conductor** (`experiments/experiment_runner.py`) - Runs local, k8s, or both benchmark modes.
- **local-runner** (`experiments/local_runner.py`) - Single-node benchmark execution.
- **cluster-runner** (`experiments/k8s_runner.py`) - Kubernetes benchmark execution.
- **analyzer** (`analysis/analyzer.py`) - Computes summary metrics and comparisons.
- **plotter** (`analysis/plots.py`) - Generates publication-style figures.
- **main-dashboard** (`dashboard/unified_app.py`, `dashboard/comprehensive_dashboard.py`) - Unified operational UI.
- **legacy-dashboard** (`dashboard/app.py`) - Older analysis dashboard path.
- **telemetry-dashboard** (`dashboard/telemetry_dashboard.py`) - Live telemetry-focused dashboard.

### API Contract Layer

- **contract** (`proto/inference.proto`) - gRPC service definition for infer/health/migration RPCs.
- **python-stubs** (`proto/inference_pb2.py`, `proto/inference_pb2_grpc.py`) - Generated protocol bindings.

---

## Distributed LLM Inference

- **Run models too large for one machine** — splits HuggingFace models (GPT-2, LLaMA, Mistral, Phi-2, Falcon, Qwen, etc.) layer-by-layer across multiple nodes. Each node loads only its assigned layers.
- **Autoregressive text generation** with temperature, top-k, top-p sampling, repetition penalty, and streaming output.
- **Single command**: `python kai_cli.py run --model microsoft/phi-2 --prompt "Hello" --max-tokens 100 --stream`

---

## Next-Generation Features (NEW)

### Hybrid Parallelism Engine
- **Combines pipeline and tensor parallelism** — splits attention layers across multiple GPUs (tensor parallel) while keeping feed-forward layers in sequential pipeline mode.
- **Dynamic mode switching** — automatically selects pipeline-only, tensor-only, or hybrid mode based on workload characteristics and cluster resources.
- **CLI**: `python kai_cli.py hybrid --model <model> --prompt "text" --mode auto`

### Intelligent Model Placement
- **Multi-objective optimization** — considers GPU VRAM, CPU RAM, network latency, and energy efficiency (EER) when assigning layers to nodes.
- **Avoids network bottlenecks** — ensures consecutive layers with high activation transfer are placed on well-connected nodes.
- **CLI**: `python kai_cli.py placement --model <model> --objective balanced`

### KV Cache Optimization (TurboQuant-style)
- **Mixed-precision cache** — stores recent tokens in FP16 for fast access, older tokens in INT8/compressed format to save memory.
- **Cache reuse** — detects overlapping prompts across requests and reuses cached KV states.
- **Memory-aware eviction** — intelligently evicts entries based on memory pressure and access patterns.

### Network-Aware Scheduling
- **Tracks inter-node latency and bandwidth** — continuously monitors network performance between nodes.
- **Groups dependent layers** — places layers with high activation transfer on nearby nodes to minimize network overhead.
- **Extends existing DEAS** — enhances the Dynamic Energy-Aware Scheduler with network awareness.

### Energy Feedback Control Loop
- **Hybrid control strategy** — combines PID feedback with short-horizon prediction to proactively prevent power and latency violations.
- **Risk-aware decisioning** — scores actions by expected benefit, operational cost, and risk, then selects the safest high-value action.
- **Adaptive adjustment** — tunes batch size, GPU power limits, precision strategy, and offloading thresholds with stability-aware escalation.
- **Safety guardrails** — enforces latency/throughput/memory constraints, applies cooldown and rollback behavior, and avoids repeating harmful actions.
- **Scheduler integration** — emits overloaded-worker and inefficient-node signals for predictive DEAS rebalancing.
- **CLI**: `python kai_cli.py energy-loop --power-target 100 --latency-target 50 --daemon`

### Speculative Decoding
- **Draft model speculation** — uses a smaller, faster model to generate candidate tokens ahead of the main model.
- **Verification with rejection sampling** — main model verifies candidates, accepting correct ones and rejecting incorrect ones.
- **Mathematically identical output** — no change in final output quality, just faster inference.
- **CLI**: `python kai_cli.py speculative --model <model> --prompt "text" --speculation-length 5`

### Fault-Tolerant Pipeline
- **Automatic failure detection** — monitors node health and detects failures mid-inference.
- **Checkpoint-based recovery** — resumes from the last checkpoint without output corruption.
- **Layer reassignment** — dynamically moves layers from failed nodes to healthy ones.
- **CLI**: `python kai_cli.py fault-tolerant --model <model> --prompt "text"`

### Adaptive Precision Controller
- **Layer criticality analysis** — identifies which layers are most sensitive to quantization.
- **Dynamic precision** — applies FP16 to critical layers, INT8/INT4 to non-critical layers.
- **Pressure-aware** — adjusts precision based on memory pressure and power usage.

### Auto-Tuning Benchmark System
- **Multi-strategy search** — supports random, grid, and Bayesian optimization.
- **Configuration space** — tests partition strategies, precision modes, batch sizes, offloading settings.
- **Optimal config selection** — outputs the best configuration for energy efficiency, latency, or throughput.
- **CLI**: `python kai_cli.py autotune --model <model> --objective energy --max-trials 20`

### Modular Plugin Architecture
- **Pluggable subsystems** — scheduler, optimizer, executor, cache, placement, and parallelism are all replaceable.
- **Registry-based** — register custom implementations via `@PluginRegistry.register` decorator.
- **Easy strategy swapping** — switch between algorithms without code changes.
- **CLI**: `python kai_cli.py plugins --action list`

---

## Advanced Algorithms (Phase 25)

### FCIM (Fair Cost-Efficient Analysis Mechanism)
- **Multi-criteria worker selection** — balances cost efficiency, performance, and fairness using weighted scoring.
- **Jain's Fairness Index** — ensures equitable workload distribution across workers over time.
- **Dynamic tracking** — continuously monitors allocation history to maintain fairness.
- **CLI**: `python kai_cli.py fcim --report`

### ADSA (Adaptive Dynamic Scheduling Algorithm)
- **Multiple scheduling policies** — supports FIFO, SJF (Shortest Job First), SRPT (Shortest Remaining Processing Time), WEIGHTED, and ADAPTIVE modes.
- **Task aging** — prevents starvation by gradually increasing priority of waiting tasks.
- **Deadline support** — tasks can specify deadlines for priority scheduling.
- **CLI**: `python kai_cli.py adsa --policy adaptive --num-tasks 20 --show-metrics`

### TPI (Tensor Parallel Interface) - Enhanced
- **Model chunking across nodes** — breaks model tensors into shards distributed across multiple workers.
- **Reduce-scatter optimization** — efficient gradient aggregation for large-scale distributed inference.
- **Integrated with hybrid parallelism** — seamlessly combines with pipeline parallelism for optimal performance.

### Active Inference Controller (Non-DRL)
- **Bayesian belief updating** — learns from environment observations without deep reinforcement learning.
- **Expected Free Energy minimization** — selects actions that minimize uncertainty and achieve goals.
- **Real-time adaptation** — adjusts decisions dynamically as conditions change.
- **Uncertainty handling** — explicitly models and handles uncertainty in observations and actions.
- **CLI**: `python kai_cli.py active-inference --show-beliefs`

### Batch Processing
- **Multiple batching strategies** — FIXED_SIZE, FIXED_TIME, ADAPTIVE, and CONTINUOUS modes.
- **Continuous batching** — adds new requests to running batches for maximized throughput.
- **Memory-aware** — limits batch size based on available GPU memory.
- **Priority support** — high-priority requests can preempt lower priority work.
- **CLI**: `python kai_cli.py batch --max-batch-size 8 --strategy adaptive --show-status`

### DFS Scheduler with Pruning
- **Depth-first search** — explores scheduling/allocation space efficiently.
- **Multiple pruning strategies** — ALPHA_BETA, BOUND (branch-and-bound), BEAM, and HEURISTIC pruning.
- **Optimal resource allocation** — finds near-optimal task-to-worker assignments.
- **Configurable depth** — balance between solution quality and computation time.
- **CLI**: `python kai_cli.py dfs-scheduler --pruning bound --num-tasks 10 --num-workers 5`

### ILP/Heuristic Scheduler
- **Integer Linear Programming** — finds provably optimal solutions for small systems using PuLP.
- **Genetic algorithm** — evolutionary optimization for large-scale problems.
- **Simulated annealing** — temperature-based search for escaping local optima.
- **Auto-selection** — automatically chooses ILP vs. heuristic based on problem size.
- **CLI**: `python kai_cli.py ilp-scheduler --algorithm auto --num-tasks 20`

### PyTorch to ONNX Conversion
- **Model export** — converts PyTorch models to ONNX format for cross-platform deployment.
- **ONNX Runtime optimization** — applies graph optimizations for faster inference.
- **Dynamic quantization** — INT8 quantization during export for smaller model size.
- **Output validation** — verifies ONNX output matches PyTorch output within tolerance.
- **CLI**: `python kai_cli.py onnx --model <model> --output model.onnx --optimize`

### Simulation Optimization
- **Layer simplification** — groups repeated layers to reduce simulation complexity.
- **Decode approximation** — samples decode steps instead of simulating all tokens.
- **Attention approximation** — simplified attention computation for faster simulation.
- **Multi-level optimization** — configurable optimization levels (0-3) trading accuracy for speed.
- **CLI**: `python kai_cli.py simulate --model <model> --optimization-level 2 --approximate-decode`

---

## Energy-Efficient Inference (Core Value)

A key advantage of KAI is **reducing power consumption while producing identical output**.

Running a massive model like LLaMA 7B (14GB in fp16) on a single high-end GPU consumes significant power:

| Setup | Hardware | Power Draw | Output |
|-------|----------|------------|--------|
| Single GPU | 1x RTX 4090 (24GB) | ~350-450W | "The capital of France is Paris..." |
| KAI Cluster | 3x RTX 3050 Ti (4GB each) | ~60W x 3 = **180W** | "The capital of France is Paris..." |
| KAI Cluster | 4x GTX 1060 (6GB each) | ~40W x 4 = **160W** | "The capital of France is Paris..." |

**Same model. Same weights. Same output. Less than half the power.**

Why this works:
- High-end GPUs (RTX 4090, A100) are built for maximum throughput and draw enormous power even when most of their compute units are idle during autoregressive generation (which is sequential by nature — one token at a time).
- Low-end GPUs draw far less power per card. Each card only loads a fraction of the model layers, so it does less work per token but at a fraction of the power cost.
- The total cluster power (sum of all small GPUs) is often **less than a single big GPU**, because each small GPU operates within its efficient power envelope.
- The model weights are identical — no quantization, no approximation. Layer 0-10 on Node A produce the exact same hidden states as they would on a single machine. The math is the same. The output is the same.

This is what makes KAI unique: **it's not just about making large models accessible to low-end hardware — it's about making inference more energy-efficient by design.**

> **Important:** KAI's strongest energy-saving advantage still comes from 2+ distributed nodes. However, KAI now also supports single-machine oversized-model execution using layer streaming, adaptive precision, and OOM-guarded degradation when VRAM is insufficient.

KAI's built-in energy benchmarking proves this. Run `python kai_cli.py benchmark --mode both` and KAI will:
1. Run the model on a single GPU and measure total energy (Wh).
2. Run the same model distributed across your cluster and measure total energy (Wh).
3. Produce a side-by-side comparison showing the energy savings.

---

## Smart Hardware Detection & Auto-Partitioning

- **Scans your hardware** — detects GPU model, VRAM, system RAM, and CPU cores on each node (local or Kubernetes).
- **Proportional layer assignment** — nodes with more VRAM/RAM get more layers automatically. Supports mixed GPU/CPU-only nodes.
- **Preview splits** before deploying: `python kai_cli.py partition --model microsoft/phi-2 --num-nodes 3`

---

## Energy Benchmarking

- **Measures real energy costs** of AI inference — GPU power draw (W), total energy (Wh), CPU utilization, inference latency, and throughput.
- **Compares local vs. Kubernetes** — runs the same model in both modes and produces side-by-side metrics with efficiency ratios.
- **Quantifies distribution overhead** — isolates gRPC serialization, network, and orchestration costs.

---

## Real-Time Energy Instrumentation (Phase 20)

- **High-frequency GPU sampling** — configurable down to 100ms intervals for capturing transient power spikes.
- **Ring buffer** — bounded in-memory buffer (default 600 samples) for efficient recent-sample access without unbounded memory growth.
- **TDP auto-detection** — automatically reads GPU Thermal Design Power via NVML; computes real-time `tdp_pct` for each sample.
- **Trapezoidal energy integration** — more accurate energy calculation than simple `avg_power * time`, especially with variable workloads.
- **Event bus** — async pub/sub system for threshold events. Subscribers receive real-time notifications when GPU power crosses configurable thresholds.
- **Power threshold service** — classifies GPU power draw as OPTIMAL (<70% TDP), WARNING (70-80%), or CRITICAL (>=80%) and publishes state change events.

---

## Dynamic Energy-Aware Scheduling — DEAS (Phase 21)

- **Energy-Efficiency Ratio (EER)** — `throughput / avg_power` metric computed per node and cluster-wide.
- **Reactive + predictive triggering** — rebalances on CRITICAL events, warning-level degradation trends, and scheduler signals from the energy feedback loop.
- **Cost-aware migration scoring** — ranks candidates using expected EER gain minus migration cost and latency penalty.
- **Top-k multi-chunk planning** — evaluates multiple chunks and targets per cycle, selects best migrations, and supports explicit no-action decisions when no beneficial move exists.
- **Optional localized ILP refinement** — uses a small binary optimization pass for candidate selection, with greedy fallback when ILP is unavailable or unnecessary.
- **Adaptive cooldown control** — dynamically tightens or relaxes cooldown based on migration success, failure, or no-action outcomes.
- **History-informed decisions** — incorporates migration success/downtime history to prefer safer, higher-yield move patterns over time.
- **Batch migration execution** — supports executing a bounded set of migrations in one rebalance cycle.
- **Live chunk migration** — Pause/Checkpoint/Resume gRPC RPCs allow moving model chunks between nodes without restarting the pipeline.
- **Gateway relinking** — thread-safe hot-swapping of chunk endpoints during migration (`POST /relink`).
- **Topology inspection** — `GET /topology` returns the current chunk-to-host mapping.

---

## CPU/Disk Offloading (Phase 22)

- **Tiered weight management** — places model weights across GPU VRAM, System RAM, and Disk (safetensors format) based on configurable budgets.
- **Double-buffered prefetching** — while the GPU processes Layer N, a background thread loads Layer N+1 from RAM/disk into a second buffer, hiding transfer latency behind computation.
- **CLI integration** — `--offload` flag enables FlexGen-style offloading; `--gpu-budget-mb` and `--disk-swap-dir` control placement.
- **Models exceeding VRAM** — KAI can now run models that don't fit in total cluster GPU memory by spilling to RAM and disk.

---

## Monitoring & Visualization

- **Real-time GPU/CPU monitoring** at configurable resolution (down to 100ms) via NVML and psutil.
- **10 publication-quality plots** — power over time, latency distribution, energy comparison, per-chunk latency, migration energy impact, VRAM vs RAM tradeoff, etc.
- **Interactive Streamlit dashboard** for exploring results, comparing experiments, viewing migration events, offloading trade-offs, and auto-refreshing live data.
- **Threshold event endpoints** — `GET /metrics/threshold` and `GET /metrics/events` for real-time power status.

---

## Infrastructure

- **Docker images** for chunk servers, gateway, and monitoring — ready for Kubernetes deployment.
- **One-command Docker build**: `python kai_cli.py build` builds all images (chunk, gateway, monitor).
- **One-command weight preparation**: `python kai_cli.py prepare --model <name> --num-chunks 3` downloads model, splits weights, and saves per-chunk files.
- **Kubernetes controller** — automated deploy, health check, metrics collection, and teardown.
- **gRPC pipeline** — binary tensor serialization between chunks with 256 MB max message size.

---

## Quantization Support

KAI supports optional **4-bit (NF4) and 8-bit (INT8) quantization** via bitsandbytes to further reduce memory usage per chunk:

```bash
# Run with 4-bit quantization
python kai_cli.py run --model microsoft/phi-2 --prompt "Hello" --quantize 4bit

# Prepare quantized chunk weights for K8s deployment
python kai_cli.py prepare --model microsoft/phi-2 --num-chunks 3 --quantize 8bit
```

Memory savings:

| Mode | Memory vs fp16 | Compression |
|------|---------------|-------------|
| 8-bit (INT8) | 50% | 2x |
| 4-bit (NF4) | 25% | 4x |

This means a LLaMA 7B model (~14 GB in fp16) can be reduced to ~3.5 GB in 4-bit — making it fit on a cluster of even smaller GPUs.

---

## CLI Commands

| Command | What It Does |
|---------|-------------|
| `kai_cli.py run` | Generate text with a distributed HuggingFace model |
| `kai_cli.py run --offload` | Run with CPU/disk offloading for oversized models |
| `kai_cli.py scan` | Show available GPU/CPU/RAM on your machine or cluster |
| `kai_cli.py partition` | Preview how a model would be split across N nodes |
| `kai_cli.py benchmark` | Run energy benchmarking (local, K8s, or both) |
| `kai_cli.py benchmark --sampling-rate 0.1` | Benchmark with 100ms GPU sampling |
| `kai_cli.py benchmark --hf-model <name>` | Benchmark a HuggingFace model with energy monitoring |
| `kai_cli.py dashboard` | Launch the Streamlit visualization dashboard |
| `kai_cli.py build` | Build Docker images for chunk/gateway/monitor |
| `kai_cli.py prepare` | Download model, chunk weights, save for K8s deployment |

### Next-Generation Commands

| Command | What It Does |
|---------|-------------|
| `kai_cli.py autotune` | Auto-tune configuration for optimal performance |
| `kai_cli.py speculative` | Run with speculative decoding (faster inference) |
| `kai_cli.py hybrid` | Run with hybrid parallelism (tensor + pipeline) |
| `kai_cli.py placement` | Generate intelligent placement plan |
| `kai_cli.py energy-loop` | Start adaptive predictive energy control loop with safety guardrails |
| `kai_cli.py fault-tolerant` | Run with fault-tolerant pipeline |
| `kai_cli.py plugins` | List and manage plugins |

### Phase 25 Advanced Algorithm Commands

| Command | What It Does |
|---------|-------------|
| `kai_cli.py fcim` | FCIM worker selection analysis and reports |
| `kai_cli.py adsa` | ADSA adaptive scheduling with multiple policies |
| `kai_cli.py batch` | Dynamic batch processing configuration |
| `kai_cli.py active-inference` | Active inference controller (non-DRL) |
| `kai_cli.py dfs-scheduler` | DFS scheduler with various pruning strategies |
| `kai_cli.py ilp-scheduler` | ILP/heuristic scheduling optimization |
| `kai_cli.py onnx` | PyTorch to ONNX model conversion |
| `kai_cli.py simulate` | Optimized simulation with approximations |

---

## Supported Model Families

| Model Family | Examples |
|-------------|----------|
| LLaMA | LLaMA-2-7B, LLaMA-2-13B, LLaMA-3-8B |
| Mistral | Mistral-7B, Mixtral-8x7B |
| Phi | Phi-2, Phi-3-mini |
| GPT-2 | GPT-2, GPT-2-Medium, GPT-2-Large, GPT-2-XL |
| GPT-NeoX | GPT-NeoX-20B, Pythia |
| GPT-J | GPT-J-6B |
| Falcon | Falcon-7B, Falcon-40B |
| Gemma | Gemma-2B, Gemma-7B |
| Qwen | Qwen-7B, Qwen-14B |
| OPT | OPT-1.3B, OPT-6.7B, OPT-13B |
| BLOOM | BLOOM-560M, BLOOM-7B |

Any HuggingFace `AutoModelForCausalLM` architecture is supported.

---

## Test Coverage

- **~220 integration tests** — all passing
  - 25 tests for energy benchmarking (Phases 1-13)
  - 30 tests for distributed inference (Phases 14-18)
  - 27 tests for gap coverage & production readiness (Phase 19)
  - ~15 tests for real-time instrumentation & event bus (Phase 20)
  - ~19 tests for dynamic scheduling & migration (Phase 21)
  - ~14 tests for CPU/disk offloading & prefetching (Phase 22)
  - ~14 tests for validation & energy analysis (Phase 23)
  - ~40 tests for next-gen features (Phase 24) — plugin architecture, adaptive precision, KV cache, intelligent placement, network-aware scheduling, hybrid parallelism, energy feedback loop, speculative decoding, fault tolerance, auto-tuning
  - **~40 tests for advanced algorithms (Phase 25)** — FCIM, ADSA, active inference, batch processing, DFS scheduler, ILP/heuristic scheduler, ONNX conversion, simulation optimization

---

## Where KAI is NOT the Best Tool

KAI solves a specific problem. Here are scenarios where other tools are a better fit:

### 1. You Prioritize Speed Over Energy Efficiency

If your only goal is **minimum latency per token** and you already have a powerful GPU, a single-GPU setup will be faster because there's no network overhead between chunks.

- **Ollama / vLLM** on an RTX 4090 will generate tokens faster than a KAI cluster of 3x RTX 3050 Ti.

However, KAI still has an advantage here: the single RTX 4090 draws ~350-450W while the KAI cluster draws ~180W total — **same output, lower energy bill**. So the trade-off is:

| Priority | Best Choice |
|----------|-------------|
| Fastest tokens per second | Single powerful GPU (Ollama/vLLM) |
| Lowest energy per inference | KAI distributed cluster |
| Same output quality | Both (identical — same model, same weights) |

### 2. You Need Production-Grade Throughput

KAI is designed for **low-end hardware clusters**, not high-throughput production serving. If you need to serve hundreds of concurrent users with low latency:

- **vLLM** — Optimized with PagedAttention, continuous batching, and tensor parallelism on high-end GPUs.
- **TGI (Text Generation Inference)** — HuggingFace's production server with batching and quantization.
- **TensorRT-LLM** — NVIDIA's maximum-performance inference engine.

KAI's sequential chunk pipeline (Node A -> Node B -> Node C) adds network round-trips per token. Production systems use tensor parallelism (all GPUs compute simultaneously), which is fundamentally faster.

### 3. You Want Quick Local Chat with No Setup

If you just want to chat with a model on your own PC without configuring Kubernetes, Docker, or Python environments:

- **Ollama** — Single binary, `ollama run llama3`, done. No Python, no K8s, no Docker.
- **LM Studio** — GUI app, download a model, click play.
- **GPT4All** — Desktop app with offline models.

KAI requires Python, PyTorch, Kubernetes, Docker, and multiple dependencies. It's an infrastructure tool, not a consumer app.

### 4. You Don't Have Multiple Machines

KAI's core value is **distributing a model across multiple nodes**. If you only have one PC, KAI's distributed features don't help — **but Phase 22's CPU/disk offloading allows running models that exceed your GPU's VRAM on a single machine** by spilling weights to RAM and disk. For other single-machine alternatives:

- **AirLLM** — Streams layers from disk on a single machine (slow but works with just disk space).
- **Ollama with quantization** — Runs 4-bit quantized models that fit in less VRAM.
- **llama.cpp** — Runs quantized models on CPU with no GPU required.

### 5. You Need Volunteer/Public Network Distribution

KAI runs on your **private LAN cluster**. If you don't own the hardware and want to borrow compute from strangers over the internet:

- **Petals** — Peer-to-peer swarm over the public internet. No hardware ownership needed.

KAI assumes you control all nodes, configure Kubernetes, and have a local network. It doesn't work across the internet.

### 6. You Need Fine-Tuning, Not Just Inference

KAI is **inference-only**. It generates text from pre-trained models but cannot train or fine-tune them. For training distributed across multiple machines:

- **DeepSpeed** — Distributed training with ZeRO optimization.
- **FSDP (Fully Sharded Data Parallel)** — PyTorch's native distributed training.
- **Megatron-LM** — NVIDIA's large-scale training framework.

### 7. You Need Mobile or Edge Deployment

KAI targets server/desktop hardware with NVIDIA GPUs. For phones, Raspberry Pi, or embedded devices:

- **llama.cpp** — Runs on ARM, x86, Apple Silicon, even Android.
- **MLC LLM** — Compiles models for mobile GPUs.
- **ONNX Runtime** — Cross-platform inference on any hardware.

### 8. You Need Maximum Accuracy with Quantization

KAI now supports **4-bit (NF4) and 8-bit (INT8) quantization** via bitsandbytes. You can run with `--quantize 4bit` or `--quantize 8bit` to reduce memory per chunk. However, if your priority is maximum quantization flexibility with many format options:

- **Ollama / llama.cpp** — Supports GGUF 2/3/4/5/6/8-bit quantization with many quant variants.
- **GPTQ / AWQ** — GPU-optimized quantized inference with near-full-precision quality.

KAI's quantization is simpler (NF4 or INT8 only), but it combines with distribution — you can quantize AND split across nodes, which none of the above tools do.

---

## Honest Summary

| KAI's Strength | KAI's Weakness |
|----------------|----------------|
| Pool multiple cheap GPUs to run large models | Slower than single powerful GPU (network overhead) |
| **Lower power consumption** — same output at less than half the watts | Complex setup vs. single-binary tools like Ollama |
| Energy benchmarking — proves the savings with real data | Not designed for production throughput |
| Kubernetes-native with health checks and scaling | Inference only — no training or fine-tuning |
| Smart auto-partitioning based on real hardware | Requires multiple machines for full energy savings |
| 4-bit/8-bit quantization to reduce memory per chunk | Fewer quant formats than llama.cpp/GGUF |
| Private and secure (your own cluster) | Overkill for casual local use |
| Full monitoring + dashboard | |
| Single-GPU layer streaming for models larger than VRAM | |
| CPU/disk offloading for models exceeding VRAM | |
| Dynamic energy-aware scheduling with live migration | |
| Real-time power threshold alerts via event bus | |
| **Hybrid parallelism** (pipeline + tensor) | |
| **Speculative decoding** for faster inference | |
| **Auto-tuning** finds optimal configuration | |
| **Fault-tolerant** recovery from node failures | |
| **Intelligent placement** optimizes layer-to-node mapping | |
| **Energy feedback loop** minimizes power consumption | |

**KAI is best when:** You have 2+ low-end PCs with small GPUs, you want to run a model that doesn't fit on any single one, you want **lower power consumption** than a single high-end GPU, and you want provable energy metrics. Or: you have a single machine with limited VRAM and want layer streaming plus offloading with graceful degradation. Or: you need advanced features like speculative decoding, hybrid parallelism, or fault tolerance.

**KAI is NOT best when:** You need maximum speed above all else, need production throughput, want zero-setup local chat, or only have one machine where the model already fits comfortably and you do not need streaming/offloading controls.

---

---

## Implementation Status - 2026-04-28

### Service-Level State
- Core dashboard-driven operations are implemented and stable for interactive inference and telemetry visibility.
- Runtime service behavior now includes asynchronous generation control and session-based run tracking.
- KV cache reporting is based on observed counters and runtime mode, including fallback awareness.
- DEAS runtime now includes predictive and cost-aware scheduling with top-k migration planning, adaptive cooldown tuning, and optional localized ILP candidate refinement.
- Single-GPU oversized-model execution stack is implemented with layer streaming, GPU memory pooling, OOMGuard action hooks, adaptive batch control, and runtime precision management.

### Operational Highlights
- GPU telemetry stream includes utilization, VRAM, temperature, power, and rolling energy estimation.
- Routing/performance views reflect live measured values rather than static summaries.
- Export controls are available for telemetry and KV diagnostics workflows.
- Controller-to-scheduler integration supports explicit no-beneficial-plan outcomes and bounded batch migration execution.

### Environment Guidance
- For service-level efficiency validation, use the CUDA-capable .venv310 runtime.
- CPU-only .venv remains suitable for basic non-GPU checks.

### Reader Note
- This service overview is synchronized with current implementation behavior as of 2026-04-28.