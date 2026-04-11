# KAI: Initial Methodology Proposal

## Abstract

This document presents the **initial research methodology proposal** for KAI (Kubernetes AI Inference), a planned distributed inference platform for large language models (LLMs). The proposal is designed for academic review and project approval, and it describes the intended problem framing, system design, experimental plan, and validation strategy before full implementation.

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

### 1.1 Problem Statement

Large language models are difficult to run on low-cost hardware because they demand high GPU memory, strong compute capacity, and stable inference throughput. Most practical solutions either:

1. Depend on expensive cloud GPUs.
2. Sacrifice quality through aggressive compression.
3. Fail to adapt efficiently in heterogeneous edge or lab clusters.

This project proposes an idea to address these gaps by distributing model inference across commodity devices while optimizing energy, latency, and fairness.

### 1.2 Research Objective (Mathematical Form)

The proposed system will attempt to optimize a multi-objective function:

$$\min \; J = \alpha E + \beta L + \gamma C - \delta F$$

Where:
- $E$ = total energy consumption
- $L$ = end-to-end latency
- $C$ = operational cost
- $F$ = fairness of resource usage
- $\alpha, \beta, \gamma, \delta$ = weighting coefficients

Subject to:
- Memory feasibility per node
- Deadline/latency constraints
- Minimum response quality threshold

### 1.3 Study Goals

The proposal aims to:

1. Design a scalable distributed inference architecture for commodity clusters.
2. Reduce energy and cost compared to monolithic high-end GPU inference.
3. Maintain acceptable latency and output quality.
4. Introduce adaptive scheduling and fair resource allocation.

---

## 2. System Model / Architecture

### 2.1 Proposed System Design

The proposed architecture has two layers:

1. **Control Plane**: gateway, scheduler, monitor, and tuning controller.
2. **Execution Plane**: distributed chunk workers that host model partitions.

### 2.2 Planned Block Pipeline

```
Client Request
  -> Gateway (tokenization + routing)
  -> Scheduler (placement decision)
  -> Chunk Workers (layer-wise forward pass)
  -> Aggregation (token sampling + response)
  -> Monitoring Feedback (power/latency/utilization)
```

### 2.3 Planned Modules

- Gateway for inference entry and output assembly
- Model partitioner for layer assignment
- Scheduler for task-to-worker mapping
- Monitor for telemetry (power, latency, queue depth)
- Controller for adaptive policy updates

---

## 3. Data Description

### 3.1 Data Sources (Planned)

The study will use:

1. Public LLM checkpoints (for inference experiments)
2. Public benchmark prompts/datasets (evaluation)
3. Runtime telemetry collected during experiments

### 3.2 Expected Data Characteristics

- Prompt lengths: short to long (e.g., 1-512 tokens)
- Generation lengths: bounded inference windows
- Time-series telemetry: power, latency, throughput, memory

### 3.3 Features to be Collected

- Request features: prompt size, generation length, temperature
- Resource features: GPU memory, compute load, utilization
- Network features: transfer delay, communication overhead
- Outcome features: response latency, throughput, energy

---

## 4. Data Preprocessing

### 4.1 Cleaning Plan

- Remove incomplete telemetry records
- Filter invalid or failed runs
- Align timestamps across modules

### 4.2 Normalization Plan

- Normalize latency and throughput scales
- Normalize power based on per-device capacity
- Convert metrics to comparable ranges for optimization

### 4.3 Feature Engineering Plan

- Compute energy-efficiency indicators
- Compute fairness index across workers
- Derive queue-pressure and bottleneck indicators

### 4.4 Data Split Plan

- Train/validation/test style split for policy evaluation
- Separate unseen workloads for final testing
- Time-aware split to avoid leakage in sequential telemetry

---

## 5. Proposed Method / Algorithm

### 5.1 Core Method Idea

The proposal combines three ideas:

1. **Layer-wise model partitioning** across available workers.
2. **Adaptive scheduling** based on queue and device conditions.
3. **Energy-aware control** using real-time monitoring feedback.

### 5.2 Planned Algorithm Components

- Fair cost-efficient worker selection strategy
- Dynamic task ordering policy
- Optional optimization solver for constrained assignment
- Heuristic fallback when exact optimization is too expensive

### 5.3 High-Level Pseudocode (Proposed)

```text
For each request:
  Extract workload features
  Rank candidate workers by feasibility and score
  Assign layer chunks based on capacity
  Run distributed forward pass
  Record telemetry
  Update scheduling policy if bottleneck detected
```

---

## 6. Baseline Methods

The proposed system will be compared with:

1. Single-GPU inference baseline
2. Single-machine optimized runtime baseline
3. Distributed baseline without adaptive/energy-aware scheduling

Reason for selection:
- They represent common practical deployment choices.
- They make cost/latency/energy comparisons meaningful.

---

## 7. Implementation Details

### 7.1 Proposed Stack

- Language: Python
- Core frameworks: PyTorch, Transformers
- Communication: gRPC + Protocol Buffers
- Orchestration: Kubernetes (or local multi-process mode)
- Visualization: Streamlit/Plotly dashboard

### 7.2 Prototype Plan

Phase 1: Single-node functional prototype
Phase 2: Multi-node distributed prototype
Phase 3: Adaptive scheduling and optimization integration

### 7.3 Hardware Plan

- Commodity GPU nodes (heterogeneous allowed)
- Moderate RAM and SSD storage
- Standard networked cluster setup

---

## 8. Experimental Setup

### 8.1 Planned Variables

- Batch size
- Precision mode
- Scheduling policy
- Number of workers/chunks

### 8.2 Planned Procedure

1. Run baseline systems.
2. Run proposed system under same workloads.
3. Vary one variable at a time.
4. Repeat multiple runs for statistical stability.

### 8.3 Test Environment Plan

- Controlled hardware profile documentation
- Reproducible run scripts
- Fixed benchmark prompt sets

---

## 9. Evaluation Metrics

Planned metrics:

1. Latency (TTFT and per-token)
2. Throughput (tokens/sec)
3. Energy consumption (Wh)
4. Energy efficiency ratio
5. Fairness index
6. Cost per inference run
7. Output quality consistency

Metric selection rationale:
- Covers both systems performance and sustainability.
- Aligns with practical deployment concerns.

---

## 10. Validation Strategy

### 10.1 Validation Method

- Hold-out testing with unseen prompt sets
- Cross-scenario validation across workload intensity levels

### 10.2 Tuning Strategy

- Grid search for initial parameter ranges
- Bayesian or heuristic refinement for efficient tuning
- Final robustness check under stress scenarios

---

## 11. Workflow / Pipeline

### 11.1 Proposed End-to-End Flow

1. Receive request
2. Tokenize and profile request
3. Select policy and assign resources
4. Execute distributed inference
5. Aggregate output
6. Log telemetry and update control policy

### 11.2 Planned Workflow Diagram

```
Request -> Profile -> Schedule -> Execute -> Aggregate -> Monitor -> Adapt
```

---

## 12. Assumptions

1. Heterogeneous nodes can be profiled reliably.
2. Inter-node communication overhead is measurable and manageable.
3. Benchmark workloads are representative of target usage.
4. Quality loss from optimization techniques remains acceptable.
5. Monitoring data is sufficient for adaptive decisions.

---

## 13. Limitations

1. Initial prototype may not scale to very large clusters.
2. Network variability can affect reproducibility.
3. Exact optimization methods may become costly at larger scale.
4. Commodity hardware diversity may complicate fair comparisons.
5. Full production-level fault tolerance may be outside first milestone.

---

## Expected Academic Contribution

If successful, this work is expected to contribute:

1. A practical methodology for energy-aware distributed LLM inference.
2. A reproducible evaluation framework combining latency, fairness, and energy metrics.
3. A comparative analysis of adaptive scheduling for heterogeneous inference clusters.

---

*Document Type: Initial Proposal*
*Status: Conceptual Methodology for Professor Review*
*Version: 1.0*
*Date: April 2026*
