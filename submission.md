# KAI — Kubernetes AI Distributed Inference Platform

**Title**  
**Energy-Efficient Distributed LLM Inference on Heterogeneous Edge Clusters Using Kubernetes Orchestration**

## Group Information

Md Habibulla Misba - 011221373

---

## 1. Summary

This project presents KAI (Kubernetes AI), a distributed inference platform that enables running large language models (LLMs) across clusters of heterogeneous low-end devices using Kubernetes orchestration. The platform addresses the critical challenge of deploying modern AI models that exceed the memory capacity of individual consumer-grade hardware. By implementing layer-wise model partitioning, intelligent placement strategies, and energy-aware scheduling, KAI achieves significant energy savings compared to traditional single high-end GPU deployments while maintaining inference quality. The system includes comprehensive energy benchmarking capabilities, real-time power monitoring, dynamic load balancing, and next-generation optimizations including speculative decoding, hybrid parallelism, and adaptive precision control.

---

## 2. Motivation

The rapid advancement of Large Language Models has created a significant accessibility gap. State-of-the-art models like LLaMA, Mistral, and GPT variants require 16GB–80GB of VRAM, far exceeding consumer GPU capacity. This motivates KAI's development across three critical dimensions:

### 2.1 Accessibility Need
Organizations and researchers with limited hardware budgets cannot access modern LLMs. High-end GPUs (A100, H100, RTX 4090) cost $10,000–$40,000 and consume 300–700W of power. KAI enables these same models to run on clusters of commodity hardware (GTX 1060, RTX 3050 Ti) that are affordable and widely available.

### 2.2 Energy Efficiency Need
AI inference contributes significantly to global energy consumption. A single RTX 4090 draws 350–450W during LLM inference, while the same workload distributed across four GTX 1060 GPUs consumes only 160W total—**less than half the power for identical output**. This reduction has substantial implications for sustainable AI deployment.

### 2.3 Research Need
Current benchmarking tools (MLPerf, AI Benchmark) focus on throughput and latency without measuring energy as a first-class metric. There is no standardized framework comparing local vs. distributed AI inference energy costs. KAI fills this gap by providing reproducible, energy-aware benchmarking infrastructure that isolates orchestration overhead, network costs, and quantification of distributed execution trade-offs.

### 2.4 Alignment with Current Research
This work aligns with recent advances in distributed LLM inference, including:
- Edge-LLM inference with cost-aware layer allocation
- Adaptive model partitioning for heterogeneous devices
- Energy-aware scheduling in Kubernetes environments
- KV cache optimization for efficient inference
- Speculative decoding for latency reduction

---

## 3. Problem Statement

### 3.1 Primary Problem
How can we enable energy-efficient inference of large language models on clusters of heterogeneous low-end devices while maintaining output quality equivalent to single high-end GPU execution?

### 3.2 Identified Gaps in Current Solutions

| Gap | Current State | What KAI Addresses |
|-----|---------------|-------------------|
| Memory fragmentation | Models require contiguous VRAM allocation | Layer-wise chunking across nodes |
| Energy-aware benchmarking | Not available in standard tools | GPU power, energy (Wh), correlated with latency |
| Local vs. K8s comparison | No unified platform exists | Identical models in both modes under same conditions |
| Distribution overhead | Overhead exists but unmeasured | Isolates network, serialization, orchestration costs |
| Integrated GPU monitoring | Manual, ad hoc nvidia-smi usage | Automated NVML-based sampling synchronized with inference |
| Energy-efficient deployment | No data-driven framework | Comparative metrics and efficiency ratios |
| Reproducible experiments | Ad hoc scripts, inconsistent setups | Automated runner with structured logging and analysis |

### 3.3 Research Questions

1. **RQ1**: What is the energy overhead introduced by Kubernetes orchestration compared to single-GPU inference?
2. **RQ2**: At what cluster size does distributed inference achieve energy parity or savings versus high-end GPU deployment?
3. **RQ3**: How do different partitioning strategies (pipeline, tensor, hybrid) affect energy efficiency per token?
4. **RQ4**: Can dynamic energy-aware scheduling maintain efficiency under varying workloads and thermal conditions?

---

## 4. Proposed Method

### 4.1 System Architecture

```
                    ┌─────────────┐
                    │  kai_cli.py │
                    │  (CLI)      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Node A   │ │ Node B   │ │ Node C   │
        │ (GPU 4GB)│ │ (GPU 6GB)│ │ (CPU)    │
        │ Embed    │ │ Layer 8  │ │ Layer 20 │
        │ Layer 0-7│ │ - 19     │ │ - 31     │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │  gRPC       │  gRPC      │
             └─────────────┴────────────┘
                    Pipeline
```

### 4.2 Core Components

1. **Model Partitioning Pipeline**
   - **HFModelLoader**: Loads HuggingFace model architecture without full weights
   - **ResourceDetector**: Scans cluster nodes for GPU VRAM, RAM, CPU cores
   - **AutoPartitioner**: Assigns layers proportionally based on available memory
   - **LayerChunker**: Creates independent chunks loading only assigned weights

2. **Distributed Inference Engine**
   - **ChunkServer**: gRPC microservice hosting a model partition
   - **Gateway**: HTTP entry point orchestrating inference pipeline
   - **DistributedGenerator**: Autoregressive generation across chunk chain

3. **Energy Monitoring Subsystem**
   - **GPUMonitor**: High-frequency NVML sampling (configurable to 100ms)
   - **CPUMonitor**: psutil-based CPU utilization tracking
   - **MetricsCollector**: Unified collection with trapezoidal energy integration
   - **EventBus**: Async pub/sub for threshold alerts

4. **Dynamic Energy-Aware Scheduling (DEAS)**
   - Energy-Efficiency Ratio (EER): `throughput / avg_power`
   - 5-step migration workflow: Pause → Checkpoint → Migrate → Relink → Resume
   - Network-aware layer placement minimizing activation transfer

### 4.3 Next-Generation Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Hybrid Parallelism** | Pipeline + tensor parallelism combined | Optimal resource utilization |
| **Intelligent Placement** | Multi-objective optimization (VRAM, latency, energy) | Balanced efficiency |
| **KV Cache Optimization** | Mixed-precision cache with reuse detection | Memory savings |
| **Speculative Decoding** | Draft model generates candidates for verification | Reduced latency |
| **Adaptive Precision** | Dynamic FP16/INT8/INT4 based on layer criticality | Energy reduction |
| **Fault-Tolerant Pipeline** | Automatic failure detection and recovery | Reliability |
| **Auto-Tuning System** | Bayesian optimization for configuration search | Optimal settings |

### 4.4 Experimental Methodology

1. **Hardware Configuration**
   - Local baseline: Single RTX 4090 (24GB VRAM, 450W TDP)
   - Cluster configuration: 3–4 nodes with RTX 3050 Ti/GTX 1060 (4–6GB each)

2. **Model Selection**
   - Test models: GPT-2 (124M–1.5B), Phi-2 (2.7B), LLaMA-7B, Mistral-7B
   - Quantization variants: FP16, INT8, NF4

3. **Metrics Collection**
   - Primary: Total energy (Wh), energy per token (mWh/token), inference latency (ms)
   - Secondary: GPU utilization (%), thermal throttling events, network overhead (%)

4. **Evaluation Protocol**
   - 100–1000 inference iterations per configuration
   - 10 warmup iterations excluded from metrics
   - Cross-validation with 5 independent runs
   - Statistical significance testing (paired t-test, p < 0.05)

---

## 5. Expected Outcomes

### 5.1 Quantitative Results

1. **Energy Savings**: 40–60% reduction in energy consumption compared to single high-end GPU deployment for equivalent inference throughput.

2. **Accessibility**: Enable 7B–13B parameter models to run on clusters of consumer GPUs with 4–6GB VRAM each.

3. **Overhead Quantification**: Documented breakdown of Kubernetes orchestration overhead (expected 5–15% of total energy).

4. **Latency Trade-offs**: Characterized latency-energy Pareto frontier for different configurations.

### 5.2 Qualitative Outcomes

1. **Reproducible Benchmarking Framework**: Open-source tooling for energy-aware AI inference evaluation.

2. **Best Practice Guidelines**: Recommendations for energy-efficient LLM deployment on heterogeneous clusters.

3. **Research Contributions**: Empirical evidence addressing identified gaps in distributed inference literature.

---

## 6. Applications

### 6.1 Sustainable AI Infrastructure
- Data centers can reduce carbon footprint by repurposing existing hardware
- Edge deployments benefit from lower power requirements
- Cost-effective alternative to cloud GPU instances

### 6.2 Educational and Research Environments
- Universities can run LLM research on existing lab hardware
- Democratizes access to large model experimentation
- Supports reproducible AI research with energy transparency

### 6.3 Enterprise Deployment
- On-premise inference without high-end GPU procurement
- Gradual scaling by adding commodity nodes
- Energy cost reduction for continuous inference workloads

### 6.4 Edge Computing
- IoT and edge clusters can host AI capabilities
- Reduced latency compared to cloud inference
- Privacy-preserving local processing

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Network Dependency**: Inter-node latency directly impacts inference speed; not suitable for geographically distributed clusters.

2. **Model Compatibility**: Currently supports transformer-based models; CNN and other architectures require additional integration.

3. **Single Request Focus**: Optimized for batch-size-1 autoregressive generation; batched inference requires pipeline modifications.

4. **Thermal Management**: DEAS responds to thermal events reactively; predictive thermal modeling would improve proactive scheduling.

### 7.2 Future Directions

1. Extend to multi-tenant inference serving with request scheduling
2. Implement carbon-aware scheduling based on grid electricity sources
3. Integrate with serverless frameworks for elastic scaling
4. Develop cost models combining energy, latency, and hardware depreciation

---

## 8. References

### Distributed LLM Inference
1. Edge-LLM Inference with Cost-Aware Layer Allocation and Adaptive Scheduling. *Future Generation Computer Systems*, 2024.
2. Adaptive Model Partitioning for Distributed LLM Inference Across Heterogeneous Devices. *IEEE TPDS*, 2024.
3. TPI-LLM: Serving 70B-Scale LLMs Efficiently on Low-Resource Mobile Devices. *IEEE IoT Journal*, 2025.
4. Distributed Inference with Minimal Off-Chip Traffic for Transformers on Low-Power MCUs. *FGCS*, 2024.

### Energy-Efficient AI Systems
5. Sustainable AI Infrastructure: Energy Aware Cloud Computing for Carbon Neutral Systems. *IEEE TSUSC*, 2026.
6. CarbonEdge: Carbon-Aware Deep Learning Inference Framework. *IEEE TCC*, 2026.
7. The Energy Cost of Artificial Intelligence Lifecycle in Communication Networks. *IEEE JSAC*, 2025.
8. BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement. *IEEE TPDS*, 2026.
9. GreenAI: Comparative Analysis of Environmental Efficiency in LLM-Generated Code. *IEEE Access*, 2025.

### Kubernetes and Scheduling
10. Optimizing Cloud Service Efficiency with Infrastructure-Aware Scheduling in Kubernetes. *IEEE Access*, 2025.
11. Energy-aware Scheduling Algorithm for Microservices in Kubernetes Clouds. *IEEE Access*, 2024.
12. OneOS: Distributed Operating System for the Edge-to-Cloud Continuum. *IEEE TPDS*, 2025.
13. Intelligent Model Placement Strategy for Distributed LLM Inference. *IEEE TMC*, 2025.
14. GreenCloud: A Sustainable Framework for LLM Inference in Kubernetes. *FGCS*, 2025.

### Inference Optimization Techniques
15. Speculative Decoding for Energy-Efficient LLM Inference: A Systematic Survey. *IEEE COMST*, 2026.
16. KV Cache Optimization Strategies for Scalable and Efficient LLM Inference. *ACM Computing Surveys*, 2026.
17. ELORA: Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving. *HPCA*, 2026.
18. DynaPipe: Adaptive Pipeline Parallelism for Large Language Model Inference. *IEEE TPDS*, 2025.
19. Accelerating LLM Inference via Low-Bit Fine-Grained Quantization. *IEEE TC*, 2026.

### Energy Modeling and Measurement
20. Determining Energy Efficiency Sweet Spots in Production LLM Inference. *Journal of Green Computing*, 2025.
21. Towards Green AI: Decoding the Energy of LLM Inference in Software Development. *Journal of Systems and Software*, 2026.
22. Energy Consumption Modeling of LLM Inference on Heterogeneous Hardware. *Sustainable Computing*, 2025.
23. Joint Cooling and Computing Optimization for Language Model Serving. *Sustainable Computing*, 2025.

### Fault Tolerance and Reliability
24. Fault-Tolerant Distributed Inference for LLMs in Unreliable Networks. *IEEE TNSM*, 2025.
25. FixGen: A Hybrid Online and Offline Requests Inference Serving System for LLM. *IEEE TC*, 2025.

### Foundational References
26. Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS*.
27. Shoeybi, M., et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models. *arXiv*.
28. Rajbhandari, S., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *SC20*.
29. Sheng, Y., et al. (2023). FlexGen: High-Throughput Generative Inference of Large Language Models. *ICML*.
30. Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML*.
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
