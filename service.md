# KAI — Service Overview

## What KAI Can Do

KAI is a **distributed AI inference platform** that lets you run large language models across a cluster of low-end PCs using Kubernetes. Here's what it can do:

---

## Distributed LLM Inference

- **Run models too large for one machine** — splits HuggingFace models (GPT-2, LLaMA, Mistral, Phi-2, Falcon, Qwen, etc.) layer-by-layer across multiple nodes. Each node loads only its assigned layers.
- **Autoregressive text generation** with temperature, top-k, top-p sampling, repetition penalty, and streaming output.
- **Single command**: `python kai_cli.py run --model microsoft/phi-2 --prompt "Hello" --max-tokens 100 --stream`

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

## Monitoring & Visualization

- **Real-time GPU/CPU monitoring** at 1-second resolution via NVML and psutil.
- **8 publication-quality plots** — power over time, latency distribution, energy comparison, per-chunk latency, etc.
- **Interactive Streamlit dashboard** for exploring results, comparing experiments, and auto-refreshing live data.

---

## Infrastructure

- **Docker images** for chunk servers, gateway, and monitoring — ready for Kubernetes deployment.
- **Kubernetes controller** — automated deploy, health check, metrics collection, and teardown.
- **gRPC pipeline** — binary tensor serialization between chunks with 256 MB max message size.

---

## CLI Commands

| Command | What It Does |
|---------|-------------|
| `kai_cli.py run` | Generate text with a distributed HuggingFace model |
| `kai_cli.py scan` | Show available GPU/CPU/RAM on your machine or cluster |
| `kai_cli.py partition` | Preview how a model would be split across N nodes |
| `kai_cli.py benchmark` | Run energy benchmarking (local, K8s, or both) |
| `kai_cli.py dashboard` | Launch the Streamlit visualization dashboard |

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

- **55 integration tests** — all passing
  - 25 tests for energy benchmarking (Phases 1-13)
  - 30 tests for distributed inference (Phases 14-18)

---
