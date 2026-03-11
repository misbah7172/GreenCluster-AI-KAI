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

## Where KAI is NOT the Best Tool

KAI solves a specific problem. Here are scenarios where other tools are a better fit:

### 1. You Have a Single Powerful GPU (24GB+ VRAM)

If your machine already has enough VRAM to load the entire model, distributing it across nodes adds unnecessary network latency. Use **Ollama** or **vLLM** instead — they'll be significantly faster because there's zero inter-node communication overhead.

> **Example:** You have an RTX 4090 (24GB) and want to run LLaMA 7B (14GB in fp16). Just load it locally. KAI would be slower here.

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

KAI's core value is **distributing a model across multiple nodes**. If you only have one PC, KAI's distributed features don't help. Instead:

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

KAI distributes models in fp16/fp32 (full precision). If your priority is fitting large models on limited hardware with minimal quality loss through quantization:

- **Ollama / llama.cpp** — Supports GGUF 2/3/4/5/6/8-bit quantization.
- **GPTQ / AWQ** — GPU-optimized quantized inference with near-full-precision quality.
- **bitsandbytes** — 4-bit and 8-bit quantization integrated with HuggingFace.

KAI currently doesn't quantize — it splits the full-precision model across nodes instead. These are complementary approaches (you could quantize AND distribute), but KAI doesn't do quantization natively.

---

## Honest Summary

| KAI's Strength | KAI's Weakness |
|----------------|----------------|
| Pool multiple cheap GPUs to run large models | Slower than single powerful GPU (network overhead) |
| Kubernetes-native with health checks and scaling | Complex setup vs. single-binary tools like Ollama |
| Energy benchmarking (unique feature) | Not designed for production throughput |
| Smart auto-partitioning based on real hardware | Inference only — no training or fine-tuning |
| Private and secure (your own cluster) | Requires multiple machines to be useful |
| Works with any HuggingFace model | No quantization support (full precision only) |
| Full monitoring + dashboard | Overkill for casual local use |

**KAI is best when:** You have 2+ low-end PCs with small GPUs, you want to run a model that doesn't fit on any single one, and you care about energy efficiency and monitoring.

**KAI is NOT best when:** You have one powerful GPU, need production throughput, want zero-setup local chat, or only have one machine.

---
