# KAI -- Kubernetes AI Inference Energy Benchmarking Platform

## Project Overview

KAI is an experimental platform designed to measure and compare the energy consumption and performance characteristics of AI inference workloads under two distinct deployment scenarios:

1. **Local Execution** -- Running an AI model on a single GPU without any orchestration layer.
2. **Kubernetes Execution** -- Running the same AI model split into chunks, distributed across multiple nodes in a Kubernetes cluster, with inter-chunk communication via gRPC.

The platform collects GPU power draw, CPU usage, inference latency, throughput, and total energy consumption, then produces comparative analysis and visualizations.

---

## Gap Analysis

### Problem Statement

The adoption of Kubernetes for AI/ML inference workloads is increasing rapidly, driven by scalability, fault tolerance, and resource management benefits. However, there is a significant lack of empirical data on the energy overhead introduced by Kubernetes orchestration compared to running inference directly on bare metal or a single GPU. Organizations making deployment decisions have no standardized way to quantify this trade-off.

### Identified Gaps

#### 1. Lack of Energy-Aware Benchmarking Tools for AI Inference

Most existing AI benchmarking frameworks (MLPerf, AI Benchmark) focus on throughput and latency. They do not measure or report energy consumption as a first-class metric. There is no widely available tool that correlates GPU power draw with inference performance across different deployment topologies.

#### 2. No Standardized Comparison Between Local and Orchestrated Execution

Current literature and tooling treat local GPU inference and Kubernetes-based inference as separate domains. There is no unified platform that runs the exact same model in both environments under identical conditions and produces directly comparable energy and performance metrics.

#### 3. Hidden Costs of Model Distribution Are Unquantified

When a model is split across multiple nodes in Kubernetes:
- Intermediate tensors must be serialized and transmitted over the network (gRPC overhead).
- Each chunk runs in a separate container with its own runtime overhead.
- Kubernetes scheduling, service discovery, and pod lifecycle management consume additional resources.

These costs exist but are rarely measured in isolation or attributed to specific components.

#### 4. GPU Power Monitoring Is Not Integrated Into Inference Pipelines

GPU power consumption data (via NVIDIA NVML) is available but is typically collected ad hoc using command-line tools like `nvidia-smi`. It is not systematically recorded alongside inference metrics (latency, throughput) in a way that enables time-correlated analysis.

#### 5. No Decision Framework for Energy-Efficient Deployment

Organizations deploying AI models lack data to answer questions such as:
- How much additional energy does Kubernetes orchestration consume for a given model?
- At what scale does the energy overhead of distribution become acceptable?
- What is the energy cost per inference in each deployment mode?
- Is the operational flexibility of Kubernetes worth the energy premium?

#### 6. Absence of Reproducible Experiment Infrastructure

Ad hoc benchmarking produces inconsistent results. There is no turnkey system that automates the full cycle of: deploy model, run controlled inference, collect synchronized metrics, tear down environment, and produce analysis -- for both local and distributed modes.

### Summary Table

| Gap | Current State | What KAI Addresses |
|-----|--------------|-------------------|
| Energy-aware AI benchmarking | Not available in standard tools | Measures GPU power, energy (Wh), and correlates with latency/throughput |
| Local vs. K8s comparison | No unified platform exists | Runs identical models in both modes under same conditions |
| Distribution overhead quantification | Overhead exists but is unmeasured | Isolates network, serialization, and orchestration costs |
| Integrated GPU power monitoring | Manual, ad hoc nvidia-smi usage | Automated NVML-based sampling synchronized with inference runs |
| Energy-efficient deployment guidance | No data-driven framework | Produces comparative metrics and efficiency ratios |
| Reproducible experiment infrastructure | Ad hoc scripts, inconsistent setups | Automated experiment runner with structured logging and analysis |

---

## Project Goals

### Primary Goals

1. **Quantify the energy cost of AI inference** under local (single-GPU) execution by measuring GPU power draw, total energy consumption, and inference latency.

2. **Quantify the energy cost of AI inference** under Kubernetes-based distributed execution, including all overhead introduced by containerization, orchestration, and network communication.

3. **Produce a direct, controlled comparison** between the two deployment modes using identical models, inputs, and iteration counts.

4. **Build an automated, reproducible experiment pipeline** that handles deployment, execution, monitoring, data collection, teardown, and analysis without manual intervention.

### Secondary Goals

5. **Develop a model chunking mechanism** that splits a PyTorch model into independent layers groups, each deployable as a standalone microservice.

6. **Implement real-time GPU and CPU monitoring** with 1-second sampling resolution, producing time-series data correlated with inference events.

7. **Create a visualization dashboard** for interactive exploration of experiment results.

8. **Generate publication-quality analysis** including derived metrics (energy per inference, performance per watt) and comparative plots.

---

## Expected Outputs

### Data Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Local experiment results | JSON | Per-run latency, GPU power samples, CPU usage, throughput, total energy |
| Kubernetes experiment results | JSON | Per-run latency (end-to-end and per-chunk), aggregated GPU power across nodes, network overhead, total energy |
| GPU monitoring logs | CSV/JSON | Time-series of power draw (W), utilization (%), memory usage (MB), temperature (C) at 1-second intervals |
| Combined experiment summary | JSON/CSV | Side-by-side metrics for both modes from a single experiment session |

### Analysis Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Total energy consumption (Wh) | Numeric | Energy used per experiment in each mode |
| Average GPU power (W) | Numeric | Mean power draw across the experiment duration |
| Average inference latency (ms) | Numeric | Mean time per inference in each mode |
| Throughput (inferences/sec) | Numeric | Inference rate in each mode |
| Energy per inference (Wh) | Numeric | Energy cost of a single inference |
| Performance-per-watt ratio | Numeric | Throughput divided by average power |
| Summary comparison table | CSV | All metrics side-by-side for local vs. Kubernetes |

### Visualization Outputs

| Output | Format | Description |
|--------|--------|-------------|
| GPU power over time | PNG plot | Line chart showing power draw during inference for both modes |
| GPU utilization over time | PNG plot | Line chart comparing GPU utilization |
| Latency comparison | PNG plot | Bar chart of average latency (local vs. Kubernetes) |
| Energy comparison | PNG plot | Bar chart of total energy consumed |
| Energy efficiency comparison | PNG plot | Bar chart of energy per inference |
| Throughput comparison | PNG plot | Bar chart of inference rate |
| Interactive dashboard | Streamlit app | Web-based exploration of all metrics and plots |

### Infrastructure Outputs

| Output | Description |
|--------|-------------|
| Dockerfiles | Container definitions for model chunks, gateway, and monitor services |
| Kubernetes manifests | Deployment, service, and resource quota YAML files |
| gRPC service definition | Protocol buffer definition for inter-chunk tensor communication |
| Experiment runner CLI | Single command to execute full experiment in local, Kubernetes, or both modes |

---

## Key Metrics Computed

The following metrics are computed for each experiment run and used for comparison:

```
Total Energy (Wh) = Average Power (W) x Total Runtime (s) / 3600

Energy per Inference (Wh) = Total Energy (Wh) / Number of Inferences

Throughput = Number of Inferences / Total Runtime (s)

Performance per Watt = Throughput / Average Power (W)

Network Overhead (K8s only) = End-to-End Latency - Sum of Chunk Compute Times
```

---

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA drivers and CUDA toolkit installed
- Docker with NVIDIA Container Toolkit
- Kubernetes cluster with GPU-equipped nodes and NVIDIA device plugin (for K8s mode)
- Python 3.10 or later

---

## Project Structure

```
project/
  model/              AI model definitions and chunking logic
  monitoring/         GPU and CPU monitoring modules
  kubernetes/         Kubernetes manifests and controller
  docker/             Dockerfiles for all services
  experiments/        Experiment runner scripts
  dashboard/          Streamlit visualization application
  logs/               Experiment output data
  analysis/           Post-experiment analysis and plotting
  proto/              gRPC protocol buffer definitions
```

---

## How to Use (after implementation)

```bash
# Install dependencies
pip install -r requirements.txt

# Run local experiment
python -m experiments.experiment_runner --mode local --model transformer --iterations 100

# Run Kubernetes experiment
python -m experiments.experiment_runner --mode kubernetes --model transformer --num-chunks 4 --iterations 100

# Run both and compare
python -m experiments.experiment_runner --mode both --model transformer --num-chunks 4 --iterations 100

# Analyze results
python -m analysis.analyzer --input logs/

# Generate plots
python -m analysis.plots --input logs/

# Launch dashboard
streamlit run dashboard/app.py
```
