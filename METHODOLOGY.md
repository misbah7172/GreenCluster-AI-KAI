# KAI: Kubernetes AI Inference Platform — Methodology

## Abstract

KAI (Kubernetes AI Inference) is a distributed AI inference platform designed to run large language models (LLMs) across clusters of commodity hardware while optimizing for energy efficiency, latency, and throughput. This document presents the comprehensive research methodology behind KAI's design, implementation, and evaluation using a structured 13-section research framework.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [System Model / Architecture](#2-system-model--architecture)
3. [Data Description](#3-data-description)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Proposed Method / Algorithm](#5-proposed-method--algorithm)
6. [Baseline Methods](#6-baseline-methods)
7. [Implementation Details](#7-implementation-details)
8. [Experimental Setup](#8-experimental-setup)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Validation Strategy](#10-validation-strategy)
11. [Workflow / Pipeline](#11-workflow--pipeline)
12. [Assumptions](#12-assumptions)
13. [Limitations](#13-limitations)

---

## 1. Problem Formulation

### 1.1 Clear Definition of the Problem

Large Language Models (LLMs) such as LLaMA, Mistral, and GPT variants require significant computational resources, often exceeding the memory and compute capacity of single consumer-grade GPUs. This creates three critical challenges:

1. **Hardware Cost**: Single high-performance GPUs (A100, H100) cost $10,000-$40,000 with high power consumption (300-700W)
2. **Quality Degradation**: Aggressive quantization to fit on small GPUs compromises model output quality and responsiveness
3. **Accessibility**: Cloud service dependency creates operational latency, recurring costs, and privacy concerns

### 1.2 Mathematical Formulation

#### Primary Optimization Objective

We aim to minimize total cost while maintaining inference quality:

$$\text{Minimize: } C = \alpha \cdot E_{total} + \beta \cdot H_{cost} + \gamma \cdot L_{latency}$$

**Subject to:**
- Memory constraint: $\sum_{i} M_i \leq \sum_{j} C_j$ (layer memory ≤ node capacity)
- Power constraint: $P_j(t) \leq TDP_j$ (power ≤ thermal design power)
- Quality constraint: $Q \geq Q_{min}$ (inference quality threshold)
- Network constraint: $\tau_{network} < \tau_{target}$ (communication latency)

**Where:**
- $E_{total}$ = total energy consumption (Wh)
- $H_{cost}$ = hardware cost (USD)
- $L_{latency}$ = inference latency (ms)
- $\alpha, \beta, \gamma$ = weighting parameters (typically 0.4, 0.4, 0.2)
- $M_i$ = memory of layer $i$
- $C_j$ = capacity of node $j$
- $Q$ = quality metric (BLEU score, perplexity)

#### Energy Efficiency Ratio (EER)

$$\text{EER} = \frac{\text{Throughput (tokens/sec)}}{\text{Power (Watts)}} \quad \text{[higher is better]}$$

Calculated continuously and logged every 100ms for real-time dashboard display.

#### Fairness Index (Jain's Index)

$$J(x_1, x_2, ..., x_n) = \frac{(\sum_{i=1}^{n} x_i)^2}{n \cdot \sum_{i=1}^{n} x_i^2}$$

Where $x_i$ = utilization of worker $i$, Range: $[\frac{1}{n}, 1]$, with 1.0 = perfect fairness.

### 1.3 Objectives of the Study

**Primary Objectives:**
1. Enable inference of 7B-13B parameter models on heterogeneous commodity hardware clusters
2. Reduce energy consumption by 40-60% compared to single high-end GPU deployment
3. Maintain inference quality (< 5% output variance from single-GPU baseline)
4. Achieve sub-100ms per-token generation latency (P99 percentile)

**Secondary Objectives:**
1. Support dynamic workload scheduling with provable fairness guarantees
2. Enable automatic fault recovery without user intervention (target: <30 sec)
3. Provide real-time observability via dashboards and telemetry (sub-1 sec latency)
4. Support multiple optimization modes (energy, latency, throughput, balanced)
5. Scale horizontally from 2 to 64+ nodes with minimal overhead

---

## 2. System Model / Architecture

### 2.1 Overall Framework

KAI is composed of two logical planes:

**Control Plane (Centralized):**
- **Gateway**: HTTP/gRPC entry point for inference requests with tokenization
- **Scheduler**: Task assignment and resource allocation (supports DEAS, FCIM, ADSA, ILP modes)
- **Monitor**: Real-time power and performance telemetry collection via NVML
- **Auto-Tuner**: Bayesian optimization for dynamic configuration adjustment

**Data Plane (Distributed, Kubernetes-native):**
- **Chunk Servers**: Stateful pods hosting model layer partitions with gRPC inference RPCs
- **Network**: gRPC inter-chunk communication with Protocol Buffers tensor encoding
- **Storage**: Distributed KV cache management at each chunk with tiered memory hierarchy
- **Event Bus**: Async pub/sub for threshold notifications and migration triggers

### 2.2 Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KAI Control Plane                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Gateway   │  │  Scheduler  │  │   Monitor   │  │  Auto-Tuner │ │
│  │  (HTTP/gRPC)│  │   (DEAS)    │  │   (NVML)    │  │  (Bayesian) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │ Event Bus                        │
├───────────────────────────────────┼──────────────────────────────────┤
│                         Data Plane (Kubernetes)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Node 0     │  │   Node 1     │  │   Node 2     │  ...          │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │               │
│  │ │ Chunk 0  │ │  │ │ Chunk 1  │ │  │ │ Chunk 2  │ │               │
│  │ │Layer 0-10│ │  │ │Layer11-20│ │  │ │Layer21-31│ │               │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │               │
│  │   GPU: 4GB   │  │   GPU: 6GB   │  │   GPU: 8GB   │               │
│  │   gRPC Server│  │   gRPC Server│  │   gRPC Server│               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Description of Components/Modules

| Component | Purpose | Key Technology | Responsibility |
|-----------|---------|-----------------|-----------------|
| **Gateway** | HTTP/gRPC entry point | Flask/FastAPI, gRPC | Request tokenization, routing, output aggregation |
| **Chunk Server** | Model layer execution | PyTorch, gRPC Server | Layer forward pass, KV cache ops, state management |
| **Scheduler** | Task assignment | DEAS, FCIM, ADSA, ILP | Worker selection, load balancing, policy switching |
| **Monitor** | Telemetry collection | NVML, psutil, pynvml | Power sampling, latency probing, health checks |
| **Auto-Tuner** | Configuration optimization | Bayesian Optimization, Scipy | Hyperparameter search, online tuning |
| **Event Bus** | Async communication | Python Queue, threading | Threshold notifications, migration triggers |

### 2.4 Data Flow Pipeline

```
1. Request Ingestion (Gateway)
   Input: HTTP POST {model: "llama-2-7b", prompt: "...", max_tokens: 128}
   ↓
2. Tokenization (Gateway + HF Tokenizer)
   Output: token_ids = [29871, 13, 5618, ...]
   ↓
3. Embedding Layer (Chunk 0)
   Input: token_ids, Output: embedding vectors
   ↓
4. Sequential Layer Processing (Chunk 0 → 1 → 2)
   For each chunk: activations = forward_pass(activations, layers)
   ↓
5. Token Prediction  
   Input: logits from final layer
   Output: next_token via sampling (temperature, top-k, top-p)
   ↓
6. KV Cache Management (Distributed)
   Update: cache[token_pos] = (key_tensor, value_tensor)
   ↓
7. Autoregressive Loop (Loop 4-6 until max_tokens or </END>)
   ↓
8. Output Aggregation (Gateway)
   Detokenize token_ids back to text string
   ↓
9. Client Response (HTTP/REST)
   Output: {text: "AI is a field...", latency_ms: 1234, tokens: 42}
```

---

## 3. Data Description

### 3.1 Dataset Source

KAI is an inference platform; evaluation datasets come from multiple sources:

**Public Datasets (Pre-trained Models):**
- **HuggingFace Model Hub**: Llama-2-7b, Mistral-7b, Phi-2, OpenHermes
- **Benchmarks**: MMLU (multi-task language understanding), HellaSwag, GSM8K (grade school math)
- **Text Corpus**: Wikipedia, Common Crawl for language modeling validation

**Proprietary Telemetry & Traces:**
- Real inference traces from production deployments (anonymized)
- GPU power consumption logs (NVML sampling every 100ms)
- Network latency measurements (gRPC inter-chunk communication)
- KV cache utilization patterns and reuse statistics

### 3.2 Data Size and Characteristics

| Dataset | Size | Type | Update Freq | Used For |
|---------|------|------|-------------|----------|
| Model Weights (Llama-2-7b) | 13-14 GB | Float16 | Static | Inference execution |
| Tokenized Prompts | 1-2 KB each | Int32 tokens | Per-request | Input sequences |
| GPU Power Telemetry | 100-200 samples/sec | Float32 W | Real-time | Energy monitoring, alerts |
| KV Cache State | 50-300 MB per request | Mixed-precision | Per-token | Session memory, optimization |
| Network Traces | 10-100 KB per transfer | Binary protobuf | Per-activation | Communication analysis |
| Experiment Logs | 1-10 MB per run | JSON | Per-experiment | Analytics, reproducibility |

### 3.3 Features/Attributes Used

**Model Architecture Features:**
- Number of layers: 32 (Llama-2-7b)
- Hidden dimension: 4096
- Attention heads: 32
- FFN dimension: 11008 (expansion ratio 2.67x)
- Sequence length support: 1-8192 tokens (context window)
- Vocabulary size: 32,000 tokens

**Hardware Features:**
- GPU memory: 4-24 GB per node (heterogeneous support)
- GPU compute: 2-5 TFLOPS per GPU (FP16 operations)
- Network bandwidth: 1-10 Gbps inter-node
- CPU core count: 4-16 cores per node
- System RAM: 16-64 GB per node
- Storage: NVMe SSD (optional, for weight caching)

**Workload Features (Inference):**
- Prompt length: 1-512 tokens (input context)
- Generation length: 1-1024 tokens (tokens to generate)
- Batch size: 1-16 concurrent requests
- Request arrival rate: 0.1-100 requests/sec (workload intensity)
- Temperature/top-k/top-p: configurable sampling parameters

---

## 4. Data Preprocessing

### 4.1 Data Cleaning

**Model Weight Cleaning:**
- Load PyTorch checkpoints with error recovery and corrupt file handling
- Validate tensor shapes match expected architecture specification
- Detect and handle mixed precision (FP32, FP16, INT8conversion)
- Skip deprecated layers or redundant components

**Telemetry Cleaning:**
- Remove GPU sampling failures (zero power readings, NVML errors)
- Interpolate missing data points (max 1 sec gaps only)
- Detect and filter anomalies (> 3σ deviation from rolling mean)
- Align timestamps across distributed components using NTP

**Power Measurement Anomalies:**
- Idle power baseline (measured during warm-up phase)
- Peak power cleanup (remove spikes > 1.1x historical max)
- Consecutive zero readings → mark chunk as offline

### 4.2 Normalization / Scaling

**Power Normalization:**
$$P_{normalized} = \frac{P_{sampled} - P_{idle}}{TDP - P_{idle}} \quad \in [0, 1]$$

Where $P_{idle}$ = idle power (e.g., 20W), $TDP$ = thermal design power (e.g., 150W)

**Latency Scaling (Per-request):**
$$L_{scaled} = \frac{L_{measured} - L_{min}}{L_{max} - L_{min}} \quad \in [0, 1]$$

Computed over sliding 1-hour window to adapt to workload patterns.

**Throughput Normalization:**
$$T_{normalized} = \frac{\text{tokens/sec}}{\text{model\_max\_throughput}} \quad \in [0, 1]$$

Baseline: 10 tokens/sec (conservative estimate for 7B model on RTX 3050).

### 4.3 Feature Engineering / Selection

**Derived Features (Computed):**
- **Energy Efficiency Ratio (EER)** = Throughput / Power (tokens/Wh)
- **Per-layer latency** = forward_pass_time / num_layers
- **Network overhead ratio** = inter_chunk_latency / total_latency
- **KV cache savings ratio** = (uncompressed_size - compressed_size) / uncompressed_size
- **Worker fairness index** = Jain's index of utilization across nodes
- **Queue depth** = pending_tasks / running_tasks

**Feature Selection Strategy:**
1. Remove highly correlated features (Pearson > 0.95)
2. Drop zero-variance features (e.g., unused modes)
3. Keep interpretable features for operator understanding
4. Prioritize features with known causal relationships to objectives
5. Final feature set: ~12-15 features for scheduler decision making

### 4.4 Data Splitting

**Train/Validation/Test Split (Historical Data):**
- **Training**: 60% of historical runs (random selection, stratified by workload)
- **Validation**: 20% (for hyperparameter tuning and online learning)
- **Testing**: 20% (final evaluation, held-out completely)

**Stratification Dimensions:**
- Model type: tiny-gpt2, Phi-2, Llama-2-7b
- Cluster configuration: homogeneous (all RTX 3050 Ti) vs heterogeneous mix
- Workload type: latency-critical (< 10 batch) vs batch-optimized (large batch)
- Time period: separate train/val/test by date to avoid temporal leakage

**Time-Series Splitting (Live Telemetry):**
- No future data leakage: training data strictly before validation temporal window
- Rolling validation: validate on next 1-hour window after training on past 24 hours
- Daily retraining: coefficients Kp, Ki, Kd retrained every 24 hours

---

## 5. Proposed Method / Algorithm

### 5.1 Detailed Explanation of Methods

#### Layer-Wise Model Partitioning

KAI partitions transformer models at layer boundaries, enabling distribution across nodes without modifying model internals.

**Algorithm 1: Proportional Layer Assignment**

```
Input: model M with L layers, nodes N with capacities C[j]
Output: partition P = {(layer_start, layer_end) → node_id}

1. Initialize:
   C_total = Σ C[j] for all nodes j
   
2. For each layer l in M:
   Estimate memory: mem[l] = num_params[l] × dtype_size × overhead_factor
   (overhead_factor = 1.2 for activations/buffers)

3. For each node n in N:
   Calculate target layers: layers_target[n] = L × (C[n] / C_total)
   
4. Assign layers sequentially:
   layer_idx = 0
   for each node n:
      num_layers = round(layers_target[n])
      P[n] = (layer_idx, layer_idx + num_layers - 1)
      layer_idx += num_layers

5. Validation & rebalancing:
   if max_load > 1.1 × avg_load:
      re-partition with load balancing algorithm
      
6. Return partition P with node assignments
```

#### Energy-Aware Scheduling (DEAS)

DEAS continuously monitors power consumption and triggers live migrations to prevent TDP violations.

**Algorithm 2: Dynamic Energy-Aware Scheduler**

```
1. Initialization phase:
   For each GPU:
   - Setup NVML handles
   - Record idle_power (no active inference)
   - Set sample_interval = 100ms
   - Initialize ring_buffer for 600 samples (~1 minute history)

2. Background monitoring (async):
   Every sample_interval milliseconds:
   a. For each GPU:
      power_draw[gpu] = nvmlDeviceGetPowerUsage()
      tdp_pct = (power_draw - idle_power) / (TDP - idle_power)
      ring_buffer.append(tdp_pct)
      
   b. Compute running average of last 10 samples
      avg_tdp = mean(ring_buffer[-10:])
      
   c. Check thresholds:
      if avg_tdp > optimal_threshold (70%):
          log WARNING event to event_bus
      
      if avg_tdp > critical_threshold (90%):
          trigger_migration(source_gpu, target_gpu)

3. Migration protocol:
   a. PAUSE: Send Pause RPC to source chunk (stop forward pass)
   b. CHECKPOINT: Serialize layer weights + KV cache → disk
   c. TRANSFER: Copy checkpoint file to target node NFS
   d. LOAD: Target chunk loads checkpoint into GPU memory
   e. RELINK: Update gateway routing table (layer → new chunk mapping)
   f. RESUME: Send Resume RPC to target chunk (resume inference)
   g. CLEANUP: Delete source checkpoint file
```

#### Advanced Scheduling Algorithms

**FCIM (Fair Cost-Efficient Analysis Mechanism):**

Multi-criteria worker selection balancing cost, efficiency, and fairness.

$$Score(w, t) = \alpha \cdot \frac{\text{CostEff}(w, t)}{\max_w \text{CostEff}} + \beta \cdot \frac{\text{PerfEff}(w, t)}{\max_w \text{PerfEff}} + \gamma \cdot (1 - \text{Util}(w))$$

Where: $\alpha + \beta + \gamma = 1.0$ (tunable weights, default: 0.35, 0.35, 0.30)

**ADSA (Adaptive Dynamic Scheduling Algorithm):**

Dynamic task queue reordering based on workload characteristics via Policy Switching.

| Policy | Formula | Selection Trigger |
|--------|---------|-------------------|
| **FIFO** | Sort by arrival_time ascending | Low queue depth |
| **SJF** | Sort by estimated_size ascending | Known job sizes |
| **SRPT** | Sort by remaining_time ascending | Variable-sized jobs |
| **WEIGHTED** | $w_p \times priority + w_a \times age - w_s \times size$ | Balanced workload |
| **ADAPTIVE** | Dynamic policy switching | Variable patterns detected |

**ILP Scheduler:**

For optimal solutions on smaller systems (< 32 nodes):

$$\text{Minimize: } \sum_{i,j} c_{ij} \times x_{ij}$$

Subject to:
- $\sum_j x_{ij} = 1, \forall i$ (each task assigned exactly once)
- $\sum_i m_i \times x_{ij} \leq M_j, \forall j$ (memory capacity constraints)
- $\sum_i f_i \times x_{ij} \leq F_j, \forall j$ (compute capacity constraints)
- $x_{ij} \in \{0, 1\}$ (binary assignment variables)

Solved using PuLP with CBC solver, 60-second timeout for production.

### 5.2 Mathematical Models

#### Hybrid Parallelism Decision Model

Automatically selects parallelism strategy based on cluster characteristics:

```
Algorithm: Parallelism Mode Selection

Input: model M, cluster topology T, workload W
Output: mode ∈ {PIPELINE, TENSOR, HYBRID}

1. Feature extraction:
   attention_ratio = attention_params / total_params
   ffn_ratio = ffn_params / total_params
   gpus_per_node = count(GPUs on each physical node)
   inter_node_bandwidth = measure_network_speed()  # Mbps
   intra_node_bandwidth = theoretical_nvlink_speed  # Gbps

2. Decision logic:
   if gpus_per_node > 1 AND attention_ratio > 0.3:
       if inter_node_bandwidth < 5000 Mbps (5 Gbps):
           return HYBRID    # Tensor parallelism within node
                             # Pipeline across nodes
       else:
           return TENSOR    # Full tensor parallelism possible
   else:
       return PIPELINE      # Standard sequential layer distribution
```

#### Active Inference Controller (Bayesian)

Expected Free Energy minimization for adaptive configuration:

$$EFE(a) = E_Q[H(o|s,a)] + D_{KL}[Q(s|a) \| P(s)]$$
$$= \text{Expected Uncertainty} + \text{Goal Divergence}$$

Where:
- $H(o|s,a)$ = Shannon entropy (uncertainty reduction)
- $D_{KL}$ = Kullback-Leibler divergence (distance to goal state)
- $Q(s|a)$ = predicted state distribution after action
- $P(s)$ = goal/prior state distribution

Used for online adaptation of batch_size, quantization precision, offload decisions.

### 5.3 Pseudocode / Flowchart

**Main Inference Pipeline (Python-style):**

```python
def kai_inference(model_name: str, prompt: str, config: InferenceConfig) -> str:
    """
    Main inference loop combining partitioning, scheduling, and optimization
    """
    # Stage 1: Preparation
    tokenizer = load_tokenizer(model_name)
    tokens = tokenizer.encode(prompt)
    partition = load_partition_config(model_name)
    
    # Stage 2: Initialization
    gateway = GatewayServer(partition, scheduler_config=config.scheduler)
    chunk_servers = [
        ChunkServer(layer_range) 
        for layer_range in partition
    ]
    monitor = PowerMonitor(
        sample_interval_ms=100,
        ring_buffer_size=600
    )
    
    # Stage 3: Generation Loop
    generated_tokens = []
    kv_cache = {}  # Distributed across chunks
    
    for token_position in range(config.max_tokens):
        # Get next token logits
        activations = tokens if token_position == 0 else last_activations
        
        for chunk_idx, chunk in enumerate(chunk_servers):
            activations = chunk.forward(
                activations=activations,
                layer_range=partition[chunk_idx],
                kv_cache=kv_cache.get(chunk_idx),
                token_position=token_position
            )
            
            # Update cache
            if chunk_idx not in kv_cache:
                kv_cache[chunk_idx] = {}
            kv_cache[chunk_idx][token_position] = activations
        
        # Sample next token
        logits = activations[-1, :]  # Last position, all vocab
        next_token = sample_token(
            logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )
        generated_tokens.append(next_token)
        tokens = [*tokens, next_token]
        
        # Monitoring and adaptation
        power_samples = monitor.get_recent_samples()
        if should_migrate(power_samples):
            trigger_migration(gateway, monitor)
        
        if next_token == tokenizer.eos_token_id:
            break  # Early exit on EOS
    
    # Stage 4: Output
    return tokenizer.decode(generated_tokens)
```

**Scheduling Decision Flowchart:**

```
New Request arrives
        ↓
  ┌─────────────────┐
  │ Extract features│
  │ (model, size,   │
  │  arrival_time)  │
  └────────┬────────┘
           ↓
  ┌─────────────────────────┐
  │ ADSA: Select policy     │
  │ (FIFO/SJF/WEIGHTED)     │
  │ based on queue_stats    │
  └────────┬────────────────┘
           ↓
  ┌─────────────────────────┐
  │ Reorder task queue      │
  │ using selected policy   │
  └────────┬────────────────┘
           ↓
  ┌─────────────────────────┐
  │ Next task in queue      │
  └────────┬────────────────┘
           ↓
  ┌─────────────────────────┐
  │ FCIM: Score all workers │
  │ (cost/perf/fairness)    │
  └────────┬────────────────┘
           ↓
  ┌─────────────────────────────────┐
  │ Select best-scored worker       │
  │ (multiple criteria optimization)│
  └────────┬────────────────────────┘
           ↓
  ┌────────────────────────────────────────┐
  │ Assign task to selected worker         │
  │ Update worker utilization counters     │
  └────────┬───────────────────────────────┘
           ↓
  ┌────────────────────────────────────────┐
  │ Layer-wise forward pass                │
  │ (Chunk 0 → 1 → 2 → ... → Output)      │
  │ gRPC inter-chunk communication         │
  └────────┬───────────────────────────────┘
           ↓
  ┌────────────────────────────────────────┐
  │ Token sampling                         │
  │ (temperature, top_k, top_p)            │
  └────────┬───────────────────────────────┘
           ↓
  ┌────────────────────────────────────────┐
  │ Power monitoring & anomaly detection   │
  │ (Compare to thresholds)                │
  └────────┬───────────────────────────────┘
           ↓
       HIGH POWER?
      /      \
    YES      NO → Continue token loop (or exit if max_tokens)
    │
    ↓
  ┌────────────────────────────────┐
  │ Trigger layer migration        │
  │ (PAUSE→CHECKPOINT→TRANSFER)    │
  └────────┬───────────────────────┘
           ↓
  Continue inference (with new chunk mapping)
```

---

## 6. Baseline Methods

### 6.1 Existing Methods Used for Comparison

| Baseline | Description | Technology | Pros | Cons |
|----------|-------------|-----------|------|------|
| **Single GPU (Baseline 1)** | Full model on one A100/H100 | CUDA + PyTorch | Simplest, lowest latency, optimal output quality | High cost ($10K+), TDP 400W+, no distribution |
| **Ollama (Baseline 2)** | Optimized single-machine inference | C/Go, quantization | User-friendly, good UX, quantized models | Limited to 1 machine, no distributed setup |
| **Petals (Baseline 3)** | Distributed inference (BitTorrent P2P) | PyTorch + DHT | Decentralized, peer-to-peer, democratic | High latency, unreliable peers, no SLAs |
| **vLLM (Baseline 4)** | Paged attention optimization | PyTorch + CUDA Graphs | High single-GPU throughput, production-grade | Single machine only, fixed optimization |
| **Megatron-LM (Baseline 5)** | Model parallelism (FAIR training) | PyTorch + NCCL | Proven tensor parallelism, well-documented | Training-focused, heavy infrastructure, not inference |

### 6.2 Justification for Selection

**Selection Criteria Applied:**
1. Publicly available and reproducible (no proprietary/closed systems)
2. Support same models (Llama family, Phi, Mistral)
3. Provide comparable metrics (latency, throughput, power)
4. Representative of different architectural approaches
5. Actively maintained (last update < 1 year ago)

**Why Each Baseline:**
- **Single GPU**: Industry standard reference point
- **Ollama**: Commercial reference implementation, widely used
- **Petals**: Represents collaborative/P2P distributed inference
- **vLLM**: State-of-the-art single-machine optimization (paged attention)
- **Megatron-LM**: Established distributed parallelism reference (NVIDIA standard)

---

## 7. Implementation Details

### 7.1 Programming Language

**Primary: Python 3.10+**
- Native deep learning ecosystem (PyTorch, HuggingFace)
- Rapid iteration for research and experimentation
- Strong type hints for production reliability (mypy checked)

**Secondary Languages:**
- **Protocol Buffers**: gRPC message definition and serialization (.proto files)
- **Bash**: Kubernetes deployment and cluster initialization scripts
- **YAML**: Configuration and Kubernetes manifests (ConfigMaps, Deployments)

### 7.2 Tools / Frameworks

| Component | Framework | Version | Purpose |
|-----------|-----------|---------|---------|
| **Core Inference** | PyTorch | 2.0+ | Tensor operations, model loading |
| **Model Loading** | HuggingFace Transformers | 4.36+ | LLM tokenization and inference |
| **GPU Monitoring** | pynvml | 11.5+ | NVIDIA GPU power/temperature telemetry |
| **System Monitoring** | psutil | 5.9+ | CPU, memory, network statistics |
| **IPC** | gRPC + Protocol Buffers | 1.60+ | Inter-process communication |
| **Orchestration** | Kubernetes Client | 1.28+ | Cluster management |
| **Dashboard** | Streamlit | 1.30+ | Web-based monitoring UI |
| **Charts** | Plotly | 5.17+ | Interactive visualization |
| **Optimization** | PuLP | 2.7+ | Integer Linear Programming solver (optional) |
| **Conversion** | ONNX Runtime | 1.16+ | Model conversion (optional) |
| **Testing** | pytest | 8.0+ | Unit and integration tests |

### 7.3 Hardware Setup

**Minimum Configuration (Lab/Testing):**
- CPU: 4 cores (Intel i5 or AMD Ryzen 5), 16 GB RAM
- GPU: 1× consumer GPU (RTX 3050 Ti, RTX 4060, A10)
- Network: 1 Gbps Ethernet, <2ms RTT
- Storage: 100 GB SSD for model cache

**Recommended Configuration (Development):**
- CPU: 8 cores (Intel i7/i9 or AMD Ryzen 7), 64 GB RAM
- GPU: 3× RTX 3050 Ti (4 GB each) or heterogeneous mix
- Network: 10 Gbps inter-node, <1ms RTT, dedicated network segment
- Storage: 250 GB NVMe SSD for model weights and temp cache

**Tested Hardware Configurations:**
1. **Homogeneous Cluster**: 3× RTX 3050 Ti (12 GB total VRAM)
2. **Heterogeneous Cluster**: RTX 3050 Ti (4GB) + GTX 1060 (3GB) + A10 (24GB)
3. **Mixed CPU/GPU**: 2× GPU nodes + 1× CPU-only offload node for weights
4. **Multi-node (K8s)**: 4-node cluster with mixed GPUs and 1Gbps networking

---

## 8. Experimental Setup

### 8.1 Parameter Settings

**Model Parameters:**
- **Primary Model**: Llama-2-7b (7,250,000,000 parameters)
- **Sequence Length**: 2048 tokens (context window)
- **Precision**: Float16 for inference (model weights + activations)
- **Quantization Modes**: FP16 (default), INT8 (mixed-precision cache), INT4 (experimental)
- **Batch Size**: 1-16 concurrent requests (tunable)

**Scheduling Parameters:**

| Parameter | Default Value | Tuning Method | Justification |
|-----------|---------------|---------------|---------------|
| FCIM cost_weight | 0.35 | Grid search 0.1-0.5 | Cost-aware deployment |
| FCIM efficiency_weight | 0.35 | Grid search | Performance optimization |
| FCIM fairness_weight | 0.30 | ABC (Active Fairness) | Prevent starvation |
| ADSA aging_rate | 0.1 tokens/sec | Empirical tuning per workload | Adapt to load patterns |
| ILP time_limit | 60 seconds | Wall-clock constrained | Production deadline |
| Migration_threshold (TDP%) | 90% | Safety margin above optimal_threshold (70%) | Balance performance/thermals |
| PID Kp (proportional gain) | 1.0 | Ziegler-Nichols method | Fast response, safe margins |
| PID Ki (integral gain) | 0.1 | Empirical tuning | Steady-state error elimination |
| PID Kd (derivative gain) | 0.05 | Tuning for oscillation dampingDampening | Smooth control |

**Energy Control Loop (PID):**
- Target energy per token: 100W (configurable)
- Control period: 1 token generation cycle
- Setpoint: error = target_energy - actual_energy_per_token
- Action: adjust batch_size, quantization precision, or offload_enabled

### 8.2 Training Procedure (N/A - Inference Only)

**Configuration Tuning Procedure (Alternate):**
1. **Baseline Measurement**: Single model on all 5 baseline systems (3 runs per baseline)
2. **Parameter Sweep**: Vary FCIM weights [0.2, 0.35, 0.5], ADSA policies, batch sizes
3. **Statistical Validation**: 10 runs per configuration (different prompts), report mean ± std
4. **Pareto Frontier**: Identify non-dominated configurations across energy/latency/throughput
5. **Best Configuration**: Selected via weighted sum of objectives (typically 0.4E + 0.4L + 0.2T)

**Warm-up Protocol (Standardization):**
- Pre-load model onto all assigned chunk servers
- Prime KV cache with dummy forward pass (token prediction)
- Stabilize GPU clocks via 5 dummy inference runs
- Record 3 baseline measurements (discard) before logging official runs
- Purpose: Remove cold-start effects and GPU boost variability

### 8.3 Testing Environment

**Hardware (Reproduction Setup):**
- GPU: 3× RTX 3050 Ti (4 GB each, GDDR6 memory)
- CPU: Intel i7-11700K (8 cores, 5.0 GHz boost), 16GB DDR4 RAM
- Network: 1 Gbps Ethernet, <1ms inter-node latency
- Storage: 250 GB NVMe SSD (/scratch for model cache)
- OS: Ubuntu 20.04 LTS or Windows 11

**Software Environment:**
- Python: 3.10.20
- PyTorch: 2.0+ with CUDA 11.8+
- HuggingFace Transformers: 4.36+
- Kubernetes (optional): 1.28+ (can run standalone)
- NVIDIA Driver: 530+ (NVML support)

**Test Dataset (Benchmark):**
- **Model**: meta-llama/Llama-2-7b (HF Hub)
- **Total Prompts**: 100 diverse prompts (1-512 tokens each)
- **Generation**: 1-128 tokens per prompt (varied)
- **Repetitions**: 3 runs per configuration (statistical power)
- **Stratification**: Balanced by prompt length (short/medium/long)

---

## 9. Evaluation Metrics

### 9.1 Metrics Used

| Metric | Formula | Unit | Target | Priority |
|--------|---------|------|--------|----------|
| **Latency (TTFT)** | time_to_first_token = first_token_output - request_arrival | ms | < 500ms | High |
| **Latency (TPS)** | time_per_token = (total_time - TTFT) / num_generated_tokens | ms | < 100ms | High |
| **Throughput** | tokens_per_sec = num_tokens_generated / total_time | tokens/sec | > 5 | High |
| **Energy** | energy_consumed = ∫ power(t) dt (trapezoidal integration) | Wh | < 0.5 Wh/req | High |
| **EER** | energy_efficiency_ratio = throughput / power_avg | tokens/Wh | > 10 | Medium |
| **Quality** | BLEU-4 score vs gold reference | score | ≥ 0.85× baseline | Very High |
| **Fairness** | Jain's Index of worker utilization | [0,1] | > 0.8 | Medium |
| **Availability** | uptime = (total_time - downtime) / total_time | % | > 99% | High |

### 9.2 Justification for Metrics

**Why Each Metric:**

1. **Latency (TTFT)**: First token generation time = perceived responsiveness for interactive users
2. **Latency (TPS)**: Per-token speed = indicates real-time inference capability
3. **Throughput**: Total token production = batch processing efficiency, validates parallelism
4. **Energy**: Direct measure of cost objective function = sustainability metric
5. **EER**: Energy efficiency aggregation = enables carbon footprint reduction claims
6. **Quality**: Ensures output validity; output quality degradation invalidates entire system
7. **Fairness**: Validates scheduling algorithms, prevents resource starvation
8. **Availability**: Production-grade SLA requirement for enterprise deployments

---

## 10. Validation Strategy

### 10.1 Cross-Validation / Hold-Out Method

**For Configuration Validation (Scheduler):**
- 5-fold cross-validation over historical run dataset
- Each fold: hold out 20% of runs, train scheduler on 80%
- Average performance metrics over 5 folds
- Report uncertainty (±1 std) for statistical significance

**For Live Deployment (Online):**
- **Canary deployment**: Route 5% traffic to new config, 95% to current stable config
- **Monitoring SLOs**: error_rate < 1%, latency_p99 within ±5% of baseline
- **Automatic rollback**: If SLOs violated for > 5 mins, revert to previous config
- **Full rollout**: Gradual 5% → 25% → 50% → 100% over 24-48 hours

**For Model Evaluation (Output Quality):**
- Hold-out test set: 20% of total prompts (unseen during tuning)
- Stratified by: prompt_length (short/med/long) and generation_length
- Separate test sets per model: tiny-gpt2, Phi-2, Llama-2-7b
- Manual inspection: Sample 5% of outputs for fluency critique

### 10.2 Hyperparameter Tuning Approach

**Stage 1: Grid Search (Initial)**
- FCIM weights: [0.2, 0.35, 0.5] for each of (cost, efficiency, fairness)
- Constraint: α + β + γ = 1.0 (normalized)
- Batch sizes: [1, 2, 4, 8, 16]
- Policies: FIFO, SJF, SRPT, WEIGHTED
- Compute all combinations: ~5 × 5 × 5 = 125 configurations
- Evaluate on 50-run benchmark per config

**Stage 2: Bayesian Optimization (Refinement)**
- Objective: minimize (latency_p99 + 0.1 × energy_per_token)
- Acquisition function: Expected Improvement (EI)
- Max 50 iterations (typical → convergence in 30 iterations)
- Early stop: if improvement < 1% over last 5 iterations
- Result: typically 5-10% improvement over grid search

**Stage 3: Manual Tuning (Safety Check)**
- Domain expert review of Pareto frontier
- Identify pathological cases (e.g., all traffic to single worker)
- Fairness audit: verify fairness_score > 0.8 across configs
- Final approval gate before canary deployment

---

## 11. Workflow / Pipeline

### 11.1 Step-by-Step Process

**Step 1: Request Arrival**
```
Client sends HTTP POST to /inference endpoint
Request body (JSON):
{
  "model": "llama-2-7b",
  "prompt": "What is artificial intelligence?",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.95
}
```

**Step 2: Tokenization**
```
Gateway process:
1. Load tokenizer for model (cached in memory)
2. tokens = tokenizer.encode(prompt) → [29871, 13, 5618, ...]
3. token_count = len(tokens)
4. Embed into sequence format: [<BOS>, token_0, token_1, ...]
```

**Step 3: Layer-wise Forward Pass**
```
For each chunk_server (sequential):
  1. Receive input activations from previous chunk
  2. Load assigned layers into GPU memory
  3. For each layer in chunk:
     - Execute forward pass: activation = layer(activation_in)
     - Update KV cache: cache[token_pos][layer] = (K, V)
  4. Send activations to next chunk server (gRPC)
  5. Return final activations (logits from last layer)
```

**Step 4: Token Prediction**
```
Final chunk produces output:
1. logits shape: [1, vocab_size]
2. Apply temperature scaling: logits = logits / temperature
3. Sample next token using top-k/top-p filtering
   - top_k: keep only top-50 logits
   - top_p: keep cumulative prob < 0.95
   - Multinomial sample: next_token ~ softmax(filtered_logits)
```

**Step 5: Autoregressive Loop**
```
repeat Step 3-4 (forward pass + sampling) until:
   - generated_tokens >= max_tokens (default 128), OR
   - model outputs <END> token (EOS), OR
   - token_limit_exceeded (safety)
```

**Step 6: Response Assembly**
```
Gateway process:
1. Collect all generated token_ids
2. Detokenize: text = tokenizer.decode(token_ids)
3. Prepare response (JSON):
   {
     "text": "Artificial intelligence is...",
     "tokens_generated": 42,
     "latency_ms": 1234,
     "energy_wh": 0.45
   }
4. Send HTTP response to client
```

### 11.2 Algorithm Steps (Scheduling Loop)

```
1. Task Arrival:
   - Inference request enqueued in scheduler's task_queue
   - Extract features: model_size_gb, prompt_len_tokens, generation_len_tokens
   - Estimate execution duration (linear predictor from historical data)

2. Policy Decision (ADSA):
   - Analyze current queue (depth, sizes, ages)
   - Select scheduling policy: FIFO, SJF, SRPT, or WEIGHTED
   - Reorder task_queue per selected policy

3. Worker Selection (FCIM):
   - Enumerate all available workers (chunk servers)
   - For each worker:
     a. Check memory availability: required_memory < free_gpu_memory
     b. Calculate cost_efficiency_score
     c. Calculate performance_efficiency_score (TFLOPS available)
     d. Calculate fairness_reward (1 - current_utilization)
   - Compute composite score: S = 0.35*C + 0.35*P + 0.3*F
   - Assign task to worker with max(S)

4. Execution:
   - Worker pulls task from its queue
   - Load assigned layers into GPU memory (if not cached)
   - Execute layer-by-layer forward pass
   - Update KV cache incrementally
   - Send intermediate result to next worker (gRPC)

5. Monitoring:
   - Sample GPU power every 100 ms
   - Track latency per layer, per token
   - If power > 90% TDP: flag for migration
   - Update worker utilization counters every token

6. Completion:
   - Aggregate output from final worker
   - Log execution metrics (latency, energy, tokens)
   - Clean up KV cache (optionally persist for replay)
   - Return result to client
```

### 11.3 Flowchart (Operator View)

```
         Client HTTP request
              ↓
    ┌────────────────────────┐
    │ Gateway receives request│
    │ Extract features       │
    └────────┬───────────────┘
             ↓
    ┌────────────────────────────────┐
    │ ADSA: Select policy            │
    │ (FIFO/SJF/WEIGHTED/ADAPTIVE)   │
    └────────┬───────────────────────┘
             ↓
    ┌────────────────────────────────┐
    │ Reorder task queue according   │
    │ to selected policy             │
    └────────┬───────────────────────┘
             ↓
    ┌────────────────────────────────┐
    │ Next task in queue             │
    └────────┬───────────────────────┘
             ↓
    ┌────────────────────────────────┐
    │ FCIM: Score all workers  │
    │ Multi-criteria optimization    │
    └────────┬───────────────────────┘
             ↓
    ┌────────────────────────────────┐
    │ Select best-scored worker      │
    │ (highest composite score)      │
    └────────┬───────────────────────┘
             ↓
    ┌──────────────────────────────────────┐
    │ Layer-wise forward pass              │
    │ Chunk_0 → Chunk_1 → Chunk_2 → ...   │
    │ gRPC inter-chunk communication       │
    └────────┬─────────────────────────────┘
             ↓
    ┌──────────────────────────────────────┐
    │ Token sampling                       │
    │ (temperature, top_k, top_p)          │
    └────────┬─────────────────────────────┘
             ↓
    ┌──────────────────────────────────────┐
    │ Power monitoring                     │
    │ (Compare GPU power to thresholds)    │
    └────────┬─────────────────────────────┘
             ↓
         High power?
        /          \
      YES          NO
       │            └────→ Continue token loop
       │                   (or exit if max_tokens)
       ↓
    ┌─────────────────────────────────┐
    │ Trigger layer migration         │
    │ (PAUSE→CHECKPOINT→TRANSFER)     │
    └────────┬────────────────────────┘
             ↓
    Continue inference (with updated routing)
```

---

## 12. Assumptions

### 12.1 System Assumptions

1. **Stateless Layer Execution**: Individual transformer layers can execute independently; no global synchronization (broadcast) needed
2. **Proportional Memory Assignment**: Layer memory requirements scale linearly with parameter count
3. **Synchronous Execution**: All chunks synchronize at layer boundaries (simplifying model: no pipelining bubbles)
4. **Negligible Serialization Overhead**: Tensor serialization time << compute time (< 1%)
5. **Persistent GPU State**: KV caches remain in GPU memory across all tokens (no swap/disk offload)
6. **Single Failure Model**: At most 1 node fails at a time (no cascading failures)
7. **Deterministic Power Mapping**: Power consumption scales predictably with compute utilization

### 12.2 Workload Assumptions

1. **Homogeneous Requests**: All prompts follow similar length distributions (1-512 tokens, generation 1-128 tokens)
2. **Independent Requests**: No cross-request optimization or multi-turn conversation context
3. **No Dynamic Model Changes**: Model weights immutable during deployment (no LoRA, adapters)
4. **Fixed Tokenizer**: Same tokenizer across all deployments (no dynamic vocabulary)
5. **Gaussian Error Distribution**: Monitoring errors follow normal distribution (for anomaly detection)
6. **No Request Batching**: Each inference request processed independently (no request fusion across clients)

### 12.3 Hardware Assumptions

1. **Reliable Network**: <1ms inter-node latency, <0.1% packet loss
2. **Power Linearity**: Power consumption scales linearly with computation (Watts ∝ utilization %)
3. **Heterogeneous but Similar**: GPUs differ by factor of ≤ 4x in capacity (not 10x+ variance)
4. **Shared Storage**: NFS or equivalent for model weight caching and checkpoints
5. **Kubernetes Availability**: K8s cluster healthy and responsive (<100ms API latency)
6. **Stable Idle Power**: GPU idle power varies < ±10% during monitoring period

---

## 13. Limitations

### 13.1 Method Limitations

1. **Synchronous Bottleneck**: Layers executed sequentially; pipelined execution not modeled
   - Impact: Theoretical speedup from pipelining (2-3x) not captured
   - Mitigation: Implement token-level pipelining (FlashAttention-style) in future work

2. **Quantization Quality Impact**: Degradation from INT8/INT4 not fully characterized
   - Impact: Mixed-precision cache quality may degrade output BLEU by 2-5%
   - Mitigation: Run quality benchmarks (MMLU, HellaSwag) on quantized variants

3. **Network Congestion Modeling**: Assumes static network; no competing traffic
   - Impact: In shared clusters, inter-chunk latency may increase 50-200%
   - Mitigation: Implement network-aware scheduling with traffic prediction

4. **Cold Start Overhead**: First token includes model loading/warm-up
   - Impact: TTFT inflated 500-1000ms on first request
   - Mitigation: Model pre-load, persistent model caches

5. **Single Model Configuration**: Tuning specific to Llama-2-7b
   - Impact: Generalization to Phi-2 (2.7B), Mistral, other architectures uncertain
   - Mitigation: Run configuration sweep on multiple model families

### 13.2 Experimental Limitations

1. **Small Cluster**: 3-4 nodes tested; scalability to 10+ node clusters extrapolated
   - Impact: Scheduling overhead may not scale linearly
   - Mitigation: Benchmark on 8-16 node clusters

2. **Limited Workload Diversity**: Simple prompt/generation; no multi-turn conversations
   - Impact: Results may not apply to chat, RAG, or long-context scenarios
   - Mitigation: Expand evaluation to multi-turn and context-heavy workloads

3. **Lab Environment**: No real production traffic patterns or user think time
   - Impact: Scheduling policies may not adapt well to bursty traffic
   - Mitigation: Use production traces (with anonymization) for simulation

4. **Fixed Hardware**: RTX 3050 Ti focus; results specific to this GPU class
   - Impact: May not transfer to A100/H100 (different power profiles)
   - Mitigation: Benchmark on 2-3 additional GPU classes (A10, V100, RTX 4090)

5. **Short Evaluation**: 100 prompts × 3 runs = limited statistical power  
   - Impact: Confidence intervals wide; effect sizes may be overestimated
   - Mitigation: Expand to 1000+ prompts, 10+ runs per config

### 13.3 Deployment Limitations

1. **Kubernetes Requirement**: Requires K8s cluster (non-trivial setup cost ~$500-1000)
   - Impact: High barrier to entry for small organizations
   - Mitigation: Provide Helm charts, Docker Compose variant for standalone use

2. **Python GIL Contention**: GIL throttles CPU threads (scheduler, monitor threads)
   - Impact: Scheduling decisions delayed during GPU compute
   - Mitigation: Use multiprocessing or move scheduler to Rust/C++

3. **gRPC Overhead**: Inter-chunk communication adds latency (~10-50ms per layer)
   - Impact: 30-40 layers × 20ms = 600-800ms communication overhead
   - Mitigation: Implement layer fusion, reduce chunk granularity

4. **Fault Recovery Time**: 30-60 sec migration time (checkpoint, transfer, load)
   - Impact: Acceptable for batch, unacceptable for real-time applications
   - Mitigation: Implement speculative execution and hot standbys

5. **Monitoring Blind Spots**: No GPU utilization profiling; only power-based triggers
   - Impact: May over/under-migrate in underutilized GPU scenarios
   - Mitigation: Add per-kernel profiling (NVIDIA NSight Compute integration)

### 13.4 Future Mitigation Strategies

- Implement token-level pipelining (reduce TTFT by 2-3x)
- Profile and characterize quantization-quality tradeoffs (cost/quality frontier)
- Add network-aware scheduling with traffic prediction
- Benchmark on larger clusters (10-64 nodes) to validate scalability
- Support multi-turn conversations with state persistence
- Add GPU utilization profiling for comprehensive monitoring

---

## Implementation Status - 2026-04-11

KAI methodology is fully implemented with comprehensive dashboard observability ([dashboard/comprehensive_dashboard.py](dashboard/comprehensive_dashboard.py)), distributed scheduling (DEAS, FCIM, ADSA, ILP modes), advanced memory optimization (mixed-precision KV cache, tiered weight management), and fault tolerance support. 

**Recommended Path for Reproduction:**
- Use `.venv310` environment for CUDA-backed execution to reproduce performance characteristics
- Run benchmark suite: `python -m pytest tests/test_phase*.py`
- Dashboard: `streamlit run dashboard/comprehensive_dashboard.py`

---

*Document Version: 2.0 (Research Methodology Format)*
*Last Updated: April 2026*
*Structure: 13-Section Research Paper Framework*
*Maintainer: KAI Development Team*

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KAI Control Plane                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Gateway   │  │  Scheduler  │  │   Monitor   │  │  Auto-Tuner │ │
│  │  (HTTP/gRPC)│  │   (DEAS)    │  │   (NVML)    │  │  (Bayesian) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │ Event Bus                        │
├───────────────────────────────────┼──────────────────────────────────┤
│                         Data Plane (Kubernetes)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Node 0     │  │   Node 1     │  │   Node 2     │  ...          │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │               │
│  │ │ Chunk 0  │ │  │ │ Chunk 1  │ │  │ │ Chunk 2  │ │               │
│  │ │Layer 0-10│ │  │ │Layer11-20│ │  │ │Layer21-31│ │               │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │               │
│  │   GPU: 4GB   │  │   GPU: 6GB   │  │   GPU: 8GB   │               │
│  │   gRPC Server│  │   gRPC Server│  │   gRPC Server│               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Purpose | Technology | Responsibility |
|-----------|---------|------------|-----------------|
| **Gateway** | HTTP/gRPC entry point | Flask, gRPC | Request tokenization, routing, output aggregation |
| **Chunk Server** | Model layer execution | PyTorch, gRPC | Layer forward pass, KV cache management |
| **Scheduler** | Task assignment | DEAS, FCIM, ADSA, ILP | Worker selection, load balancing |
| **Monitor** | Telemetry collection | NVML, psutil | Power sampling, latency probing, health checks |
| **Auto-Tuner** | Configuration optimization | Bayesian Optimization | Hyperparameter search, dynamic tuning |
| **Event Bus** | Async communication | Python Queue | Threshold notifications, migration triggers |

### 2.3 Data Flow Pipeline

1. **Request Ingestion**: Client sends prompt to Gateway via HTTP POST
2. **Tokenization**: Gateway tokenizes input using model's tokenizer
3. **Forward Pass**: Activations flow through chunk servers via gRPC (Layer 0→1→2...)
4. **Token Generation**: Autoregressive decoding with KV cache updates
5. **Output Aggregation**: Generated tokens collected and detokenized
6. **Response**: Generated text returned to client via HTTP/REST

---

## 3. Data Description

### 3.1 Dataset Source

**Public Models & Benchmarks:**
- HuggingFace Model Hub (Llama-2-7b, Mistral, Phi-2)
- MMLU, HellaSwag, GSM8K benchmarks

**Proprietary Telemetry:**
- GPU power traces, network latency measurements, KV cache stats

### 3.2 Data Size and Characteristics

| Type | Size | Frequency | Format |
|------|------|-----------|--------|
| Model Weights | 13-14 GB | Static | Float16 |
| Power Telemetry | 100-200 samples/sec | Real-time | Float32 W |
| KV Cache State | 50-300 MB/request | Per-token | Mixed-precision |
| Network Traces | 10-100 KB/transfer | Per-activation | Binary Protobuf |

### 3.3 Features/Attributes Used

- Model: 32 layers, 4096 hidden dim, 32 attention heads
- GPU: 4-8 GB VRAM per node, 2-5 TFLOPS  
- Workload: 1-512 token prompts, 1-1024 token generation

---

## 4. Data Preprocessing

### 4.1 Data Cleaning

- Remove GPU sampling failures (zero power readings)
- Interpolate missing data (max 1-sec gaps)
- Filter anomalies (>3σ deviation)
- Align distributed timestamps

### 4.2 Normalization / Scaling

$$P_{norm} = \frac{P_{sampled} - P_{idle}}{TDP - P_{idle}}$$

### 4.3 Feature Engineering / Selection

- Compute EER, latency ratios, resource utilization
- Remove correlated features (>0.95 correlation)
- Keep interpretable features

### 4.4 Data Splitting

- **Train**: 60% of runs (random stratified selection)
- **Validation**: 20% (hyperparameter tuning)
- **Test**: 20% (final evaluation, held-out)

---

## 5. Proposed Method / Algorithm

[See sections 5.1-5.3 above for detailed algorithms, mathematical models, and pseudocode]

---

## 6. Baseline Methods

### 6.1 Comparison Systems

| System | Type | Key Feature |
|--------|------|-------------|
| Single GPU | Monolithic | A100/H100 high-end GPU |
| Ollama | Single-machine | Quantized inference |
| Petals | P2P Distributed | BitTorrent-style |
| vLLM | Single-machine | Paged attention |
| Megatron-LM | Training framework | Tensor parallelism |

### 6.2 Selection Justification

All baselines: publicly available, support same models, comparable metrics, different architectures.

---

## 7. Implementation Details

[See section 7 above for full stack details]

**Technology Stack:** Python 3.10+, PyTorch 2.0+, HuggingFace Transformers, gRPC, Kubernetes, Streamlit

---

## 8. Experimental Setup

[See section 8 above for parameter settings, tuning procedures, and test environments]

**Tested Configuration:** 3× RTX 3050 Ti, 100 prompts, 3 runs each

---

## 9. Evaluation Metrics

[See section 9 above for complete metric definitions and justifications]

**Key Metrics:** Latency (TTFT/TPS), Throughput, Energy, EER, Quality, Fairness, Availability

---

## 10. Validation Strategy

### 10.1 Cross-Validation

- 5-fold cross-validation over historical runs
- Stratified by model type, cluster config, workload

### 10.2 Hyperparameter Tuning

1. Grid search (initial): 125 configurations
2. Bayesian optimization (refinement): 50 iterations max
3. Manual review (safety): Pareto frontier analysis

---

## 11. Workflow / Pipeline

[See section 11 above for complete step-by-step process, algorithm flow, and flowcharts]

**Main Loop:** Tokenize → ADSA Policy Select → FCIM Worker Select → Forward Pass → Token Sample → Monitor Power → Loop or Migrate

---

## 12. Assumptions

### 12.1 System Assumptions

1. Stateless layer execution (no global broadcast)
2. Proportional memory scaling
3. Synchronous execution at layer boundaries
4. Negligible serialization overhead (< 1%)
5. Persistent GPU KV cache
6. Single failure model
7. Deterministic power mapping

### 12.2 Workload Assumptions

1. Homogeneous prompt distributions
2. Independent requests (no batching)
3. Immutable model weights
4. Fixed tokenizer
5. Gaussian-distributed errors

### 12.3 Hardware Assumptions

1. Reliable network (<1 ms latency, <0.1% loss)
2. Power scales linearly with utilization
3. GPUs differ by factor ≤ 4x
4. Shared storage (NFS)
5. Healthy K8s cluster (<100 ms API latency)

---

## 13. Limitations

### 13.1 Method Limitations

1. Synchronous bottleneck (no pipelining bubbles modeled)
2. Quantization quality impact not fully characterized
3. Static network model (no competing traffic)
4. Cold start overhead not optimized
5. Single model configuration (Llama-2-7b specific)

### 13.2 Experimental Limitations

1. Small cluster (3-4 nodes only)
2. Simple workload (no multi-turn conversations)
3. Lab environment (no real production traffic)
4. Fixed hardware class (RTX 3050 Ti)
5. Limited statistical power (100 prompts × 3 runs)

### 13.3 Deployment Limitations

1. Kubernetes requirement (high setup cost)
2. Python GIL contention (scheduler threads)
3. gRPC overhead (~10-50 ms per layer)
4. Fault recovery time (30-60 sec migrations)
5. Monitoring blind spots (no GPU util profiling)

### 13.4 Future Mitigation

- Token-level pipelining (2-3x TTFT reduction)
- Quantization-quality characterization
- Network-aware scheduling with traffic prediction
- Large-scale benchmark (10-64 nodes)
- Multi-turn conversation support
- GPU utilization profiling integration

---

## Implementation Status - 2026-04-11

KAI methodology is fully implemented with comprehensive dashboard observability through `dashboard/comprehensive_dashboard.py`, distributed scheduling (DEAS, FCIM, ADSA, ILP), advanced memory optimization (mixed-precision KV cache, tiered weight management), and fault tolerance support.

**Recommended Path for Reproduction:**
Use `.venv310` environment for CUDA-backed execution to reproduce performance and energy characteristics.

---

*Document Version: 2.0*
*Last Updated: April 2026*
*Structure: 13-Section Research Methodology Format*