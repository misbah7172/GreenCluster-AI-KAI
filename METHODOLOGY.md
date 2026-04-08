# KAI: Kubernetes AI Inference Platform — Methodology

## Abstract

KAI (Kubernetes AI Inference) is a distributed AI inference platform designed to run large language models (LLMs) across clusters of commodity hardware while optimizing for energy efficiency, latency, and throughput. This document presents the comprehensive methodology behind KAI's design, implementation, and evaluation, covering distributed inference architecture, energy-aware scheduling, advanced optimization algorithms, and fault-tolerant execution.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Distributed Inference Methodology](#3-distributed-inference-methodology)
4. [Energy-Aware Scheduling (DEAS)](#4-energy-aware-scheduling-deas)
5. [Advanced Scheduling Algorithms](#5-advanced-scheduling-algorithms)
6. [Parallelism Strategies](#6-parallelism-strategies)
7. [Memory Optimization Techniques](#7-memory-optimization-techniques)
8. [Fault Tolerance and Reliability](#8-fault-tolerance-and-reliability)
9. [Adaptive Optimization](#9-adaptive-optimization)
10. [Model Conversion and Simulation](#10-model-conversion-and-simulation)
11. [Implementation Details](#11-implementation-details)
12. [Evaluation Methodology](#12-evaluation-methodology)
13. [Conclusion](#13-conclusion)

---

## 1. Introduction

### 1.1 Problem Statement

Large Language Models (LLMs) such as LLaMA, Mistral, and GPT variants require significant computational resources, often exceeding the memory and compute capacity of single consumer-grade GPUs. Traditional approaches either:

1. **Use expensive high-end GPUs** (A100, H100) with high power consumption (300-700W)
2. **Apply aggressive quantization** that degrades output quality
3. **Rely on cloud services** with associated costs and latency

### 1.2 Proposed Solution

KAI addresses these challenges through:

- **Layer-wise model partitioning** across multiple low-power GPUs
- **Energy-aware distributed scheduling** to minimize power consumption
- **Intelligent placement algorithms** considering network topology and device capabilities
- **Hybrid parallelism** combining pipeline and tensor parallelism
- **Advanced caching and quantization** without quality degradation

### 1.3 Key Contributions

1. **FCIM (Fair Cost-Efficient Analysis Mechanism)** — Multi-criteria worker selection balancing cost, efficiency, and fairness
2. **ADSA (Adaptive Dynamic Scheduling Algorithm)** — Dynamic task reordering based on workload characteristics
3. **TPI (Tensor Parallel Interface)** — Efficient tensor splitting across distributed nodes
4. **Active Inference Controller** — Bayesian belief-based decision making for real-time optimization
5. **ILP/Heuristic Scheduler** — Optimal resource allocation for varying system scales

---

## 2. System Architecture

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
│                                   │                                  │
├───────────────────────────────────┼──────────────────────────────────┤
│                         Data Plane (Kubernetes)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Node 0     │  │   Node 1     │  │   Node 2     │  ...          │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │               │
│  │ │ Chunk 0  │ │  │ │ Chunk 1  │ │  │ │ Chunk 2  │ │               │
│  │ │Layer 0-10│ │  │ │Layer11-20│ │  │ │Layer21-31│ │               │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │               │
│  │   GPU: 4GB   │  │   GPU: 6GB   │  │   GPU: 8GB   │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| Gateway | HTTP/gRPC entry point, request routing | Flask, gRPC |
| Chunk Server | Serves model layers, processes activations | PyTorch, gRPC |
| Scheduler | Task scheduling, resource allocation | DEAS, ADSA, ILP |
| Monitor | Real-time power/performance monitoring | NVML, psutil |
| Auto-Tuner | Configuration optimization | Bayesian Optimization |
| Controller | Kubernetes orchestration | kubernetes-client |

### 2.3 Data Flow

1. **Request Ingestion**: Client sends prompt to Gateway via HTTP POST
2. **Tokenization**: Gateway tokenizes input using model's tokenizer
3. **Forward Pass**: Activations flow through chunk servers via gRPC
4. **Token Generation**: Autoregressive decoding with KV cache
5. **Response**: Generated text returned to client

---

## 3. Distributed Inference Methodology

### 3.1 Layer-Wise Model Partitioning

KAI partitions transformer models at layer boundaries, enabling distribution across heterogeneous hardware.

#### Algorithm: Auto-Partitioning

```
Algorithm 1: Proportional Layer Assignment
─────────────────────────────────────────
Input: model M with L layers, nodes N with capacities C
Output: partition P mapping layers to nodes

1. Calculate total cluster capacity: C_total = Σ C[i]
2. For each layer l in M:
   a. Estimate memory: mem[l] = params[l] × dtype_size × 1.2
3. For each node n in N:
   a. Assign layers proportional to capacity:
      layers[n] = L × (C[n] / C_total)
4. Adjust for layer boundaries (no splitting within layers)
5. Return partition P
```

#### Memory Estimation Formula

```
Memory_per_layer = (num_params × bytes_per_param × overhead_factor)

Where:
- bytes_per_param: 2 (FP16), 1 (INT8), 0.5 (INT4)
- overhead_factor: 1.2 (activations, gradients, buffers)
```

### 3.2 Activation Transfer Protocol

Inter-node communication uses gRPC with Protocol Buffers for efficient tensor serialization:

```protobuf
message TensorPayload {
  bytes data = 1;           // Serialized tensor
  repeated int64 shape = 2; // Tensor dimensions
  string dtype = 3;         // Data type (float16, float32)
  int32 layer_id = 4;       // Source layer identifier
}
```

### 3.3 KV Cache Management

For autoregressive generation, KV caches are maintained at each chunk:

```
KV_Cache_Size = batch_size × num_heads × seq_len × head_dim × 2 × dtype_size
```

---

## 4. Energy-Aware Scheduling (DEAS)

### 4.1 Dynamic Energy-Aware Scheduler

DEAS monitors real-time power consumption and makes scheduling decisions to minimize energy while maintaining performance.

#### Power Monitoring

```
Algorithm 2: Real-Time Power Monitoring
───────────────────────────────────────
1. Initialize NVML for each GPU
2. Every Δt milliseconds:
   a. Sample power_draw[gpu] via nvmlDeviceGetPowerUsage()
   b. Calculate TDP percentage: tdp_pct = power_draw / TDP
   c. Update ring buffer with sample
   d. If tdp_pct > threshold:
      - Publish CRITICAL event to event bus
3. Calculate energy using trapezoidal integration:
   E = Σ (P[i] + P[i+1]) / 2 × Δt
```

#### Energy Efficiency Ratio (EER)

```
EER = Throughput (tokens/sec) / Power (Watts)

Higher EER indicates better energy efficiency.
```

### 4.2 Threshold-Based Migration

When a node exceeds power thresholds, DEAS triggers migration:

| State | TDP % | Action |
|-------|-------|--------|
| OPTIMAL | < 70% | Continue normal operation |
| WARNING | 70-80% | Log warning, prepare migration |
| CRITICAL | ≥ 80% | Trigger layer migration |

#### Migration Protocol

```
Algorithm 3: Live Layer Migration
─────────────────────────────────
1. PAUSE: Send Pause RPC to source chunk
2. CHECKPOINT: Serialize weights + KV cache to disk
3. TRANSFER: Copy checkpoint to target node
4. LOAD: Target chunk loads checkpoint
5. RELINK: Update gateway routing table
6. RESUME: Send Resume RPC to target chunk
7. CLEANUP: Delete source checkpoint
```

---

## 5. Advanced Scheduling Algorithms

### 5.1 FCIM (Fair Cost-Efficient Analysis Mechanism)

FCIM balances three objectives when selecting workers for task assignment:

#### Multi-Criteria Scoring Function

```
Score(w, t) = α × CostEfficiency(w, t) + 
              β × PerformanceEfficiency(w, t) + 
              γ × FairnessScore(w)

Where: α + β + γ = 1.0
```

#### Cost Efficiency

```
CostEfficiency(w, t) = 1 / (power_consumption × energy_cost × estimated_time)
```

#### Jain's Fairness Index

```
J(x₁, x₂, ..., xₙ) = (Σxᵢ)² / (n × Σxᵢ²)

Where xᵢ = utilization of worker i
Range: [1/n, 1], where 1 = perfect fairness
```

### 5.2 ADSA (Adaptive Dynamic Scheduling Algorithm)

ADSA dynamically reorders the task queue based on multiple policies:

#### Scheduling Policies

| Policy | Formula | Use Case |
|--------|---------|----------|
| FIFO | arrival_time | Default fairness |
| SJF | estimated_size | Minimize avg wait time |
| SRPT | remaining_time | Minimize response time |
| WEIGHTED | w₁×priority + w₂×age + w₃×size | Balanced optimization |
| ADAPTIVE | Dynamic policy switching | Variable workloads |

#### Task Aging Mechanism

To prevent starvation in priority-based scheduling:

```
effective_priority(t) = base_priority + aging_rate × wait_time
```

### 5.3 ILP (Integer Linear Programming) Scheduler

For smaller systems, ILP finds provably optimal solutions:

#### Optimization Problem

```
Minimize: Σᵢⱼ cᵢⱼ × xᵢⱼ  (total cost)

Subject to:
- Σⱼ xᵢⱼ = 1, ∀i          (each task assigned once)
- Σᵢ mᵢ × xᵢⱼ ≤ Mⱼ, ∀j   (memory constraints)
- Σᵢ fᵢ × xᵢⱼ ≤ Fⱼ, ∀j   (compute constraints)
- xᵢⱼ ∈ {0, 1}            (binary assignment)

Where:
- xᵢⱼ = 1 if task i assigned to worker j
- cᵢⱼ = cost of running task i on worker j
- mᵢ = memory requirement of task i
- fᵢ = compute requirement of task i
```

### 5.4 DFS Scheduler with Pruning

For complex scheduling problems, depth-first search with pruning explores the solution space efficiently:

#### Pruning Strategies

| Strategy | Description | Complexity Reduction |
|----------|-------------|---------------------|
| Alpha-Beta | Prune branches worse than known best | ~50% |
| Branch-and-Bound | Use lower bounds to prune | Variable |
| Beam Search | Keep only top-k candidates | O(k × depth) |
| Heuristic | Domain-specific pruning rules | Problem-dependent |

---

## 6. Parallelism Strategies

### 6.1 Pipeline Parallelism (Existing)

Layers are distributed sequentially across nodes:

```
Node 0: Layers 0-10  →  Node 1: Layers 11-20  →  Node 2: Layers 21-31
         ↓                      ↓                        ↓
    [Activations]         [Activations]            [Logits]
```

**Advantages**: Simple implementation, minimal communication
**Disadvantages**: Sequential bottleneck, idle time during forward pass

### 6.2 Tensor Parallelism

Attention layers are split across GPUs within a node:

```
Attention Head Split:
┌─────────────────────────────────────────┐
│          Multi-Head Attention           │
├─────────────┬─────────────┬─────────────┤
│   Heads 0-3 │   Heads 4-7 │  Heads 8-11 │
│   (GPU 0)   │   (GPU 1)   │   (GPU 2)   │
└─────────────┴─────────────┴─────────────┘
              ↓ AllReduce ↓
         [Combined Output]
```

### 6.3 Hybrid Parallelism

KAI combines both strategies based on workload analysis:

```
Algorithm 4: Parallelism Mode Selection
───────────────────────────────────────
Input: model M, cluster topology T, workload W
Output: parallelism mode (PIPELINE, TENSOR, HYBRID)

1. Analyze model architecture:
   - attention_ratio = attention_params / total_params
   - ffn_ratio = ffn_params / total_params

2. Analyze cluster:
   - gpus_per_node = count GPUs on each node
   - inter_node_bandwidth = measure network speed

3. Decision:
   if gpus_per_node > 1 AND attention_ratio > 0.3:
       if inter_node_bandwidth < threshold:
           return HYBRID  # Tensor within node, pipeline across
       else:
           return TENSOR
   else:
       return PIPELINE
```

---

## 7. Memory Optimization Techniques

### 7.1 Mixed-Precision KV Cache

Recent tokens are stored at full precision; older tokens are quantized:

```
┌────────────────────────────────────────────────────┐
│                    KV Cache                         │
├──────────────────────┬─────────────────────────────┤
│   Recent Window      │      Historical             │
│   (FP16, high acc)   │    (INT8, compressed)       │
│   tokens [t-k..t]    │    tokens [0..t-k-1]        │
└──────────────────────┴─────────────────────────────┘
```

#### Compression Ratio

```
Compression = FP16_size / INT8_size = 2x

For 2048 token context with 32 heads × 128 dim:
- FP16: 2048 × 32 × 128 × 2 × 2 = 32 MB
- Mixed (50% INT8): ~24 MB (25% savings)
```

### 7.2 Tiered Weight Management

Weights are placed across GPU VRAM, System RAM, and Disk based on access patterns:

```
Tier Hierarchy:
┌─────────────────┐  Fastest
│   GPU VRAM      │  < 1 μs
├─────────────────┤
│   System RAM    │  ~ 100 μs
├─────────────────┤
│   Disk (SSD)    │  ~ 10 ms
└─────────────────┘  Slowest
```

### 7.3 Prefetch Engine

Double-buffered prefetching hides weight loading latency:

```
Algorithm 5: Double-Buffer Prefetching
──────────────────────────────────────
While inference running:
  Buffer A: Current layer weights (GPU executing)
  Buffer B: Next layer weights (async loading)
  
  On layer completion:
    Swap(A, B)
    Start async load of layer+2 into B
```

### 7.4 Quantization Support

KAI supports multiple quantization levels:

| Level | Bits | Memory | Quality Impact |
|-------|------|--------|----------------|
| FP16 | 16 | 100% | None |
| INT8 | 8 | 50% | Minimal |
| INT4 (NF4) | 4 | 25% | Low |

---

## 8. Fault Tolerance and Reliability

### 8.1 Failure Detection

Continuous health monitoring detects node failures:

```
Algorithm 6: Failure Detection
──────────────────────────────
1. For each registered node:
   a. Send heartbeat every health_check_interval
   b. If no response within timeout:
      - Mark node as SUSPECTED
   c. If 3 consecutive failures:
      - Mark node as FAILED
      - Trigger recovery protocol
```

### 8.2 Checkpoint-Based Recovery

Inference state is periodically checkpointed:

```
Checkpoint Contents:
- Hidden states at layer boundaries
- KV cache state
- Generation position
- Random state for reproducibility
```

### 8.3 Layer Reassignment

On node failure, layers are redistributed:

```
Algorithm 7: Layer Reassignment
───────────────────────────────
Input: failed_node, healthy_nodes, layer_assignment
Output: new_assignment

1. Identify orphaned layers from failed_node
2. For each healthy node:
   a. Calculate available capacity
   b. Sort by: capacity, current_load, network_proximity
3. Assign orphaned layers to nodes with most capacity
4. Update gateway routing table
5. Resume inference from last checkpoint
```

---

## 9. Adaptive Optimization

### 9.1 Active Inference Controller

Non-DRL approach using Bayesian belief updating:

```
Algorithm 8: Active Inference
─────────────────────────────
State Beliefs: P(s | observations)
Action Selection: argmin_a EFE(a)

Expected Free Energy:
EFE(a) = E[H(o|s,a)] + D_KL[Q(s|a) || P(s)]
       = Uncertainty  + Goal Divergence

1. Observe: latency, power, throughput
2. Update beliefs using Bayes rule:
   P(s|o) ∝ P(o|s) × P(s)
3. For each possible action:
   - Predict outcome distribution
   - Calculate EFE
4. Select action with minimum EFE
5. Execute and observe result
```

### 9.2 Energy Feedback Control Loop

PID-based closed-loop optimization:

```
Control Variables:
- batch_size ∈ [1, 32]
- gpu_power_limit ∈ [50%, 100%] of TDP
- precision ∈ {FP16, INT8, INT4}
- offload_threshold ∈ [0, 1]

PID Controller:
error(t) = target_energy_per_token - actual_energy_per_token
u(t) = Kp × e(t) + Ki × ∫e(τ)dτ + Kd × de/dt
```

### 9.3 Auto-Tuning System

Bayesian optimization finds optimal configurations:

```
Configuration Space:
- num_chunks: [2, 8]
- precision: {fp16, int8, int4}
- batch_size: [1, 2, 4, 8, 16]
- offload_enabled: {true, false}

Objective Functions:
- ENERGY: minimize energy_per_token
- LATENCY: minimize time_per_token
- THROUGHPUT: maximize tokens_per_second

Acquisition Function: Expected Improvement (EI)
EI(x) = E[max(f(x) - f(x⁺), 0)]
```

---

## 10. Model Conversion and Simulation

### 10.1 PyTorch to ONNX Conversion

For cross-platform deployment testing:

```
Export Pipeline:
1. Trace model with sample input
2. Export to ONNX with dynamic axes
3. Apply ONNX Runtime optimizations
4. Optional: INT8 quantization
5. Validate output against PyTorch
```

### 10.2 Simulation Optimization

For rapid configuration testing without full inference:

#### Optimization Levels

| Level | Technique | Speedup | Accuracy |
|-------|-----------|---------|----------|
| 0 | No optimization | 1x | 100% |
| 1 | Layer grouping | 2-3x | 99% |
| 2 | + Decode sampling | 5-10x | 95% |
| 3 | + Attention approximation | 10-20x | 90% |

---

## 11. Implementation Details

### 11.1 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | 2.0+ |
| Model Loading | HuggingFace Transformers | 4.36+ |
| GPU Monitoring | pynvml | 11.5+ |
| IPC | gRPC + Protocol Buffers | 1.60+ |
| Orchestration | Kubernetes | 1.28+ |
| Dashboard | Streamlit | 1.30+ |
| ILP Solver | PuLP | 2.7+ (optional) |
| ONNX | onnxruntime | 1.16+ (optional) |

### 11.2 Module Structure

```
model/
├── Core Inference
│   ├── hf_loader.py          # HuggingFace model loading
│   ├── layer_chunker.py      # Layer-wise splitting
│   ├── chunk_server.py       # gRPC inference server
│   ├── gateway.py            # HTTP/gRPC gateway
│   └── generation.py         # Autoregressive decoding
│
├── Scheduling (Phase 21-25)
│   ├── deas_scheduler.py     # Energy-aware scheduling
│   ├── fcim_worker_selector.py   # FCIM algorithm
│   ├── adsa_scheduler.py     # ADSA algorithm
│   ├── ilp_scheduler.py      # ILP/heuristic solver
│   └── dfs_scheduler.py      # DFS with pruning
│
├── Optimization (Phase 24-25)
│   ├── adaptive_precision.py     # Dynamic precision
│   ├── kv_cache_optimizer.py     # Mixed-precision cache
│   ├── energy_feedback_loop.py   # PID controller
│   ├── active_inference.py       # Bayesian controller
│   └── batch_processor.py        # Request batching
│
├── Parallelism
│   ├── hybrid_parallelism.py     # Tensor + pipeline
│   ├── intelligent_placement.py  # Placement optimization
│   └── network_aware_scheduler.py # Network-aware scheduling
│
└── Reliability
    ├── fault_tolerant_pipeline.py # Failure recovery
    ├── speculative_decoder.py     # Speculative execution
    └── auto_tuner.py              # Configuration tuning
```

### 11.3 CLI Interface

```bash
# Core Commands
kai_cli.py run --model <name> --prompt "text"
kai_cli.py benchmark --mode both
kai_cli.py scan

# Scheduling Commands
kai_cli.py fcim --report
kai_cli.py adsa --policy adaptive
kai_cli.py ilp-scheduler --algorithm auto

# Optimization Commands
kai_cli.py autotune --objective energy
kai_cli.py energy-loop --power-target 100
kai_cli.py batch --strategy adaptive

# Parallelism Commands
kai_cli.py hybrid --mode auto
kai_cli.py placement --objective balanced
```

---

## 12. Evaluation Methodology

### 12.1 Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| Latency | time_to_first_token + time_per_token × num_tokens | ms |
| Throughput | tokens_generated / total_time | tokens/s |
| Energy | Σ power_samples × Δt | Wh |
| EER | throughput / power | tokens/Wh |
| Fairness | Jain's Index of worker utilization | [0, 1] |

### 12.2 Experimental Setup

```
Hardware Configurations Tested:
1. Homogeneous: 3× RTX 3050 Ti (4GB each)
2. Heterogeneous: RTX 3050 + GTX 1060 + GTX 1650
3. Mixed CPU/GPU: 2× GPU + 1× CPU-only node

Models Tested:
- sshleifer/tiny-gpt2 (small, validation)
- microsoft/phi-2 (2.7B parameters)
- meta-llama/Llama-2-7b (7B parameters)

Workloads:
- Single request (latency focus)
- Batch processing (throughput focus)
- Continuous streaming (mixed)
```

### 12.3 Baseline Comparisons

| System | Description |
|--------|-------------|
| Single GPU | Full model on one high-end GPU |
| Ollama | Optimized single-machine inference |
| Petals | Distributed inference (BitTorrent-style) |
| vLLM | Paged attention, single machine |

### 12.4 Expected Results

Based on the methodology, expected improvements:

| Metric | vs Single High-End GPU |
|--------|------------------------|
| Energy Consumption | 40-60% reduction |
| Hardware Cost | 50-70% reduction |
| Throughput | Similar (within 20%) |
| Latency | 10-30% increase (network overhead) |

---

## 13. Conclusion

### 13.1 Summary

KAI presents a comprehensive methodology for energy-efficient distributed LLM inference combining:

1. **Intelligent partitioning** — Layer-wise model splitting with proportional assignment
2. **Energy-aware scheduling** — Real-time monitoring with threshold-based migration
3. **Advanced algorithms** — FCIM, ADSA, ILP, DFS for optimal resource allocation
4. **Hybrid parallelism** — Adaptive combination of pipeline and tensor parallelism
5. **Memory optimization** — Mixed-precision caching, tiered storage, prefetching
6. **Fault tolerance** — Checkpoint-based recovery with automatic reassignment
7. **Adaptive control** — Active inference and PID-based feedback loops

### 13.2 Key Innovations

1. **FCIM** — First fairness-aware worker selection for distributed inference
2. **Active Inference Controller** — Non-DRL approach to real-time optimization
3. **Mixed-Precision KV Cache** — Temporal-aware precision allocation
4. **Hybrid Parallelism Engine** — Dynamic mode switching based on workload

### 13.3 Future Work

- Extend to multi-modal models (vision + language)
- Implement speculative execution with draft models
- Add support for model fine-tuning in distributed setting
- Integrate with cloud spot instances for cost optimization

---

## References

1. Shoeybi, M., et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053 (2019).

2. Rajbhandari, S., et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC20 (2020).

3. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP (2023).

4. Sheng, Y., et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." ICML (2023).

5. Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314 (2023).

6. Leviathan, Y., et al. "Fast Inference from Transformers via Speculative Decoding." ICML (2023).

7. Friston, K. "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience (2010).

---

## Appendix A: Algorithm Pseudocode

### A.1 FCIM Worker Selection

```python
def select_worker(task, workers, weights):
    """
    FCIM: Fair Cost-Efficient Analysis Mechanism
    
    Args:
        task: TaskRequirement with memory, compute needs
        workers: List of WorkerProfile
        weights: (cost_weight, efficiency_weight, fairness_weight)
    
    Returns:
        Selected worker with allocation decision
    """
    scores = {}
    for worker in workers:
        if not meets_requirements(worker, task):
            continue
        
        cost_score = 1.0 / (worker.power * worker.energy_cost * est_time)
        perf_score = worker.flops / task.compute_required
        fair_score = 1.0 - worker.utilization  # Prefer underutilized
        
        scores[worker] = (
            weights.cost * normalize(cost_score) +
            weights.efficiency * normalize(perf_score) +
            weights.fairness * fair_score
        )
    
    return max(scores, key=scores.get)
```

### A.2 ADSA Task Scheduling

```python
def schedule_next_task(queue, policy):
    """
    ADSA: Adaptive Dynamic Scheduling Algorithm
    
    Args:
        queue: PriorityQueue of ADSATask
        policy: SchedulingPolicy enum
    
    Returns:
        Next task to execute
    """
    if policy == FIFO:
        return min(queue, key=lambda t: t.arrival_time)
    elif policy == SJF:
        return min(queue, key=lambda t: t.estimated_size)
    elif policy == SRPT:
        return min(queue, key=lambda t: t.remaining_time)
    elif policy == WEIGHTED:
        return max(queue, key=lambda t: weighted_score(t))
    elif policy == ADAPTIVE:
        policy = select_best_policy(queue_stats)
        return schedule_next_task(queue, policy)

def weighted_score(task):
    age = current_time - task.arrival_time
    return (
        PRIORITY_WEIGHT * task.priority +
        AGE_WEIGHT * age -
        SIZE_WEIGHT * task.estimated_size
    )
```

---

## Appendix B: Configuration Parameters

### B.1 Scheduling Configuration

```yaml
scheduler:
  type: deas  # deas, fcim, adsa, ilp
  
  fcim:
    cost_weight: 0.35
    efficiency_weight: 0.35
    fairness_weight: 0.30
    fairness_threshold: 0.8
  
  adsa:
    initial_policy: adaptive
    aging_rate: 0.1
    reorder_interval_ms: 100
  
  ilp:
    time_limit_s: 60
    gap_tolerance: 0.05
    fallback: genetic
```

### B.2 Memory Configuration

```yaml
memory:
  gpu_budget_mb: auto  # or specific value
  ram_budget_mb: auto
  disk_swap_dir: /tmp/kai_swap
  
  kv_cache:
    max_length: 2048
    recent_window: 256
    compression: int8
  
  prefetch:
    enabled: true
    buffer_count: 2
```

### B.3 Energy Configuration

```yaml
energy:
  monitoring:
    sampling_rate_ms: 100
    ring_buffer_size: 600
  
  thresholds:
    optimal_pct: 70
    warning_pct: 80
    critical_pct: 90
  
  feedback_loop:
    enabled: true
    power_target_w: 100
    latency_target_ms: 50
    pid_kp: 1.0
    pid_ki: 0.1
    pid_kd: 0.05
```

---

*Document Version: 1.0*
*Last Updated: April 2026*
*Authors: KAI Development Team*
