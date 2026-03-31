"""
KAI CLI — Main entry point for distributed LLM inference on low-end hardware.

Run large AI models across a Kubernetes cluster of budget PCs. Each node
loads only the layers it is responsible for, so no single machine needs
enough VRAM/RAM for the full model.

Subcommands::

    kai run           — Download a model, partition across cluster, generate text.
    kai scan          — Detect cluster resources and show capabilities.
    kai partition     — Preview how a model would be split (dry-run).
    kai benchmark     — Run the original energy benchmarking workflow.
    kai dashboard     — Launch the Streamlit dashboard.
    kai build         — Build Docker images for chunk/gateway/monitor.
    kai prepare       — Download model, chunk weights, save for K8s deployment.

Next-Generation Features::

    kai autotune      — Auto-tune configuration for optimal performance.
    kai speculative   — Run with speculative decoding for faster inference.
    kai hybrid        — Run with hybrid parallelism (tensor + pipeline).
    kai placement     — Generate intelligent placement plan.
    kai energy-loop   — Start the energy feedback control loop.
    kai fault-tolerant — Run with fault-tolerant pipeline.
    kai plugins       — List and manage plugins.

Usage::

    python kai_cli.py run --model sshleifer/tiny-gpt2 --prompt "Hello" --max-tokens 50
    python kai_cli.py scan
    python kai_cli.py partition --model microsoft/phi-2 --num-nodes 3
    python kai_cli.py benchmark --model transformer --mode local
    python kai_cli.py dashboard
    python kai_cli.py build --tag kai:latest
    python kai_cli.py prepare --model sshleifer/tiny-gpt2 --num-chunks 3
    python kai_cli.py autotune --model sshleifer/tiny-gpt2 --objective energy
    python kai_cli.py speculative --model sshleifer/tiny-gpt2 --prompt "Hello"
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import torch

logger = logging.getLogger("kai")


def cmd_run(args):
    """Run distributed inference: download model, partition, generate text."""
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker, LayerChunk
    from model.generation import DistributedGenerator
    from model.resource_detector import ResourceDetector

    quantize = getattr(args, "quantize", None)
    if quantize:
        print(f"[KAI] Quantization requested: {quantize}")

    print(f"[KAI] Loading model: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    # Validate architecture
    try:
        loader.validate_architecture()
    except ValueError as e:
        print(f"[KAI] Error: {e}")
        sys.exit(1)

    # Get model info
    size_est = loader.get_model_size_estimate()
    dtype_key = "float16_mb" if args.dtype in ("float16", "fp16") else "float32_mb"
    est_mb = size_est.get(dtype_key, size_est["float32_mb"])
    print(f"[KAI] Model: ~{size_est['params_millions']:.0f}M params, ~{est_mb:.0f} MB ({args.dtype})")

    # Scan resources
    print("[KAI] Scanning resources...")
    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()
    total_usable = sum(n.usable_memory_mb for n in nodes)
    print(f"[KAI] Cluster: {len(nodes)} node(s), {total_usable:.0f} MB usable")

    # Create chunks
    num_chunks = args.num_chunks or len(nodes)
    print(f"[KAI] Partitioning model into {num_chunks} chunks...")
    chunker = LayerChunker(loader)

    prefetch_engine = None
    weight_manager = None

    if getattr(args, "offload", False):
        # FlexGen-style offloaded chunking
        gpu_budget_mb = args.gpu_budget_mb
        if gpu_budget_mb <= 0:
            # Auto-detect from first GPU node
            for n in nodes:
                if n.has_gpu and n.gpu_vram_mb > 0:
                    gpu_budget_mb = n.gpu_vram_mb * 0.8  # 80% safety margin
                    break
            if gpu_budget_mb <= 0:
                gpu_budget_mb = 512.0  # fallback for CPU-only
        ram_budget_mb = sum(n.usable_memory_mb for n in nodes) * 0.5

        print(f"[KAI] Offloading enabled: GPU={gpu_budget_mb:.0f} MB, RAM={ram_budget_mb:.0f} MB, disk={args.disk_swap_dir}")
        chunks, weight_manager, prefetch_engine = chunker.create_offloaded_chunks(
            gpu_budget_mb=gpu_budget_mb,
            ram_budget_mb=ram_budget_mb,
            disk_dir=args.disk_swap_dir,
            device=args.device,
        )
    elif len(nodes) > 1:
        memory_budgets = [n.usable_memory_mb for n in nodes[:num_chunks]]
        chunks = chunker.create_chunks_by_memory(memory_budgets)
    else:
        chunks = chunker.create_chunks(num_chunks)

    for c in chunks:
        print(f"  Chunk {c.chunk_id}: {c.layer_names} (~{c.estimate_memory_mb():.0f} MB)")

    # Load real weights into chunks
    print("[KAI] Loading model weights...")
    if not getattr(args, "offload", False):
        _load_real_weights(loader, chunks, args.device, quantize=quantize)

    # Generate
    tokenizer = loader.get_tokenizer()
    gen = DistributedGenerator(
        chunks, tokenizer, device=args.device,
        prefetch_engine=prefetch_engine,
        weight_manager=weight_manager,
    )

    print(f"[KAI] Generating (max_tokens={args.max_tokens}, temp={args.temperature})...")
    print("---")

    if args.stream:
        for token_text in gen.generate_stream(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            print(token_text, end="", flush=True)
        print("\n---")
    else:
        result = gen.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(result)
        print("---")

    print("[KAI] Done.")


def cmd_scan(args):
    """Scan cluster resources."""
    from model.resource_detector import ResourceDetector

    print(f"[KAI] Scanning resources (mode={args.mode})...")
    detector = ResourceDetector(mode=args.mode)
    summary = detector.scan_summary()

    print(f"\nCluster Summary:")
    print(f"  Nodes: {summary['num_nodes']}")
    print(f"  GPU nodes: {summary['gpu_nodes']}")
    print(f"  CPU-only nodes: {summary['cpu_only_nodes']}")
    print(f"  Total GPU VRAM: {summary['total_gpu_vram_mb']:.0f} MB")
    print(f"  Total RAM: {summary['total_ram_mb']:.0f} MB")
    print(f"  Total usable: {summary['total_usable_mb']:.0f} MB")
    print()

    for node in summary["nodes"]:
        gpu_str = f"{node['gpu_type']} ({node['gpu_vram_mb']:.0f} MB)" if node["gpu_type"] != "none" else "none"
        print(f"  {node['name']}:")
        print(f"    GPU: {gpu_str}")
        print(f"    RAM: {node['ram_mb']:.0f} MB")
        print(f"    Usable for model: {node['usable_mb']:.0f} MB")

    # Estimate which models can fit
    print("\nModel Compatibility (approximate):")
    model_sizes = {
        "GPT-2 (124M)": 250,
        "Phi-2 (2.7B)": 5400,
        "Llama-2-7B": 14000,
        "Mistral-7B": 14000,
        "Llama-2-13B": 26000,
        "Llama-2-70B": 140000,
    }
    total_usable = summary["total_usable_mb"]
    for model_name, size_mb in model_sizes.items():
        fits = "YES" if total_usable >= size_mb else "NO"
        print(f"  {model_name} (~{size_mb} MB fp16): {fits}")


def cmd_partition(args):
    """Preview model partitioning without deploying."""
    from model.hf_loader import HFModelLoader
    from model.resource_detector import ResourceDetector, NodeInfo
    from model.auto_partitioner import AutoPartitioner

    print(f"[KAI] Loading model config: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    if args.mode == "local":
        detector = ResourceDetector(mode="local")
        nodes = detector.scan()
        # Simulate multiple nodes by splitting local resources
        if args.num_nodes > 1:
            local = nodes[0]
            per_node_mb = local.usable_memory_mb / args.num_nodes
            nodes = [
                NodeInfo(
                    name=f"virtual-node-{i}",
                    gpu_vram_mb=local.gpu_vram_mb / args.num_nodes if local.has_gpu else 0,
                    gpu_type=local.gpu_type,
                    ram_mb=local.ram_mb / args.num_nodes,
                    cpu_cores=max(1, local.cpu_cores // args.num_nodes),
                    has_gpu=local.has_gpu,
                )
                for i in range(args.num_nodes)
            ]
    else:
        detector = ResourceDetector(mode="kubernetes")
        nodes = detector.scan()

    partitioner = AutoPartitioner()
    plan = partitioner.create_plan(loader, nodes)
    print()
    print(plan.summary())
    print()

    issues = partitioner.validate_plan(plan)
    if issues:
        print("Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Plan validation: OK")


def cmd_benchmark(args):
    """Run the original energy benchmarking workflow."""
    from experiments.experiment_runner import run_experiment

    hf_model = getattr(args, "hf_model", None)
    sampling_rate = getattr(args, "sampling_rate", 1.0)
    enable_deas = getattr(args, "enable_deas", False)
    deas_cooldown = getattr(args, "deas_cooldown", 30.0)
    if hf_model:
        print(f"[KAI] Running HF model benchmark: mode={args.mode}, model={hf_model}")
        results = run_experiment(
            mode=args.mode,
            model_type="transformer",
            num_chunks=args.num_chunks,
            iterations=args.iterations,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            hf_model=hf_model,
            monitor_interval=sampling_rate,
            enable_deas=enable_deas,
            deas_cooldown=deas_cooldown,
        )
    else:
        print(f"[KAI] Running benchmark: mode={args.mode}, model={args.model}")
        results = run_experiment(
            mode=args.mode,
            model_type=args.model,
            num_chunks=args.num_chunks,
            iterations=args.iterations,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            monitor_interval=sampling_rate,
            enable_deas=enable_deas,
            deas_cooldown=deas_cooldown,
        )
    print("[KAI] Benchmark complete. Results saved to:", args.output_dir)


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    app_file = "dashboard/app.py" if getattr(args, "legacy", False) else "dashboard/unified_app.py"
    print(f"[KAI] Launching dashboard ({os.path.basename(app_file)})...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.headless", "true",
        "--server.port", str(args.port),
    ]
    subprocess.run(cmd)


def cmd_build(args):
    """Build Docker images for KAI components."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    tag = args.tag
    images = {
        "chunk": os.path.join("docker", "Dockerfile.chunk"),
        "gateway": os.path.join("docker", "Dockerfile.gateway"),
        "monitor": os.path.join("docker", "Dockerfile.monitor"),
    }

    for name, dockerfile in images.items():
        image_tag = f"{tag}-{name}" if tag != "kai:latest" else f"kai-{name}:latest"
        print(f"[KAI] Building {name} image: {image_tag}")
        cmd = [
            "docker", "build",
            "-f", dockerfile,
            "-t", image_tag,
            project_root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[KAI] ERROR building {name}:")
            print(result.stderr)
            sys.exit(1)
        print(f"[KAI] Built: {image_tag}")

        if args.push:
            print(f"[KAI] Pushing {image_tag}...")
            push_result = subprocess.run(
                ["docker", "push", image_tag],
                capture_output=True, text=True,
            )
            if push_result.returncode != 0:
                print(f"[KAI] ERROR pushing {name}:")
                print(push_result.stderr)
                sys.exit(1)
            print(f"[KAI] Pushed: {image_tag}")

    print("[KAI] All images built successfully.")


def cmd_autotune(args):
    """Auto-tune configuration for optimal performance."""
    from model.auto_tuner import AutoTuner, TuningObjective, ConfigurationSpace
    from model.hf_loader import HFModelLoader
    from model.resource_detector import ResourceDetector

    print(f"[KAI] Starting auto-tuning for: {args.model}")
    print(f"[KAI] Objective: {args.objective}")
    print(f"[KAI] Max trials: {args.max_trials}")

    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()

    config_space = ConfigurationSpace(
        num_chunks_range=(1, max(8, len(nodes))),
        precision_options=["fp32", "fp16", "int8", "int4"] if args.all_precisions else ["fp16", "int8"],
        batch_size_values=[1, 2, 4, 8, 16, 32] if args.batch_range else [1, 4, 16],
    )

    tuner = AutoTuner(
        loader, nodes,
        config_space=config_space,
        output_dir=args.output_dir,
    )

    objective_map = {
        "latency": TuningObjective.LATENCY,
        "throughput": TuningObjective.THROUGHPUT,
        "energy": TuningObjective.ENERGY_EFFICIENCY,
        "memory": TuningObjective.MEMORY,
        "balanced": TuningObjective.BALANCED,
    }

    result = tuner.tune(
        objective=objective_map.get(args.objective, TuningObjective.BALANCED),
        max_trials=args.max_trials,
        search_strategy=args.strategy,
        timeout_s=args.timeout if args.timeout > 0 else None,
    )

    print()
    print(result.summary())
    print()
    print(f"[KAI] Results saved to: {args.output_dir}")


def cmd_speculative(args):
    """Run inference with speculative decoding."""
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker
    from model.resource_detector import ResourceDetector
    from model.speculative_decoder import AdaptiveSpeculativeDecoder, VerificationMode

    print(f"[KAI] Loading main model: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    # Scan resources
    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()
    num_chunks = args.num_chunks or len(nodes)

    print(f"[KAI] Partitioning into {num_chunks} chunks...")
    chunker = LayerChunker(loader)
    chunks = chunker.create_chunks(num_chunks)

    print("[KAI] Loading weights...")
    _load_real_weights(loader, chunks, args.device, quantize=args.quantize)

    # Load draft model if specified
    draft_model = None
    if args.draft_model:
        print(f"[KAI] Loading draft model: {args.draft_model}")
        from transformers import AutoModelForCausalLM
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=loader.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.token,
        )
        draft_model.to(args.device)
        draft_model.eval()

    tokenizer = loader.get_tokenizer()

    verification_modes = {
        "strict": VerificationMode.STRICT,
        "threshold": VerificationMode.THRESHOLD,
        "sampling": VerificationMode.SAMPLING,
    }

    speculative = AdaptiveSpeculativeDecoder(
        chunks=chunks,
        tokenizer=tokenizer,
        draft_model=draft_model,
        speculation_length=args.speculation_length,
        verification_mode=verification_modes.get(args.verification, VerificationMode.STRICT),
        device=args.device,
    )

    print(f"[KAI] Speculative generation (speculation_length={args.speculation_length})...")
    print("---")

    tokens = speculative.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    result = tokenizer.decode(tokens, skip_special_tokens=True)
    print(result)
    print("---")

    stats = speculative.get_stats()
    print(f"[KAI] Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"[KAI] Avg speculation length: {stats['avg_speculation_length']:.1f}")
    print("[KAI] Done.")


def cmd_hybrid(args):
    """Run inference with hybrid parallelism."""
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker
    from model.resource_detector import ResourceDetector
    from model.hybrid_parallelism import HybridParallelismEngine, ParallelismMode

    print(f"[KAI] Loading model: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()

    print(f"[KAI] Setting up hybrid parallelism (mode={args.mode})...")

    mode_map = {
        "auto": None,  # Auto-detect
        "pipeline": ParallelismMode.PIPELINE_ONLY,
        "tensor": ParallelismMode.TENSOR_ONLY,
        "hybrid": ParallelismMode.HYBRID,
    }

    engine = HybridParallelismEngine(
        loader,
        nodes,
        mode=mode_map.get(args.mode),
        tensor_parallel_size=args.tensor_parallel,
    )

    tokenizer = loader.get_tokenizer()
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")

    print("[KAI] Running inference...")
    print("---")

    output = engine.forward(input_ids.to(args.device))
    output_ids = output.argmax(dim=-1)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(result)
    print("---")

    print(f"[KAI] Parallelism mode used: {engine.mode.value}")
    print("[KAI] Done.")


def cmd_placement(args):
    """Generate intelligent placement plan."""
    from model.hf_loader import HFModelLoader
    from model.resource_detector import ResourceDetector
    from model.intelligent_placement import IntelligentPlacementEngine, OptimizationObjective

    print(f"[KAI] Loading model config: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    print(f"[KAI] Scanning resources (mode={args.resource_mode})...")
    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()

    objective_map = {
        "latency": OptimizationObjective.LATENCY,
        "energy": OptimizationObjective.ENERGY,
        "memory": OptimizationObjective.MEMORY,
        "balanced": OptimizationObjective.BALANCED,
    }

    engine = IntelligentPlacementEngine(
        nodes,
        objective=objective_map.get(args.objective, OptimizationObjective.BALANCED),
    )

    print(f"[KAI] Generating placement plan (objective={args.objective})...")
    plan = engine.generate_plan(loader)

    print()
    print(plan.summary())
    print()

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        print(f"[KAI] Plan saved to: {args.output}")


def cmd_energy_loop(args):
    """Start the energy feedback control loop."""
    from model.energy_feedback_loop import EnergyFeedbackController

    print("[KAI] Starting energy feedback control loop...")
    print(f"[KAI] Power target: {args.power_target}W")
    print(f"[KAI] Latency target: {args.latency_target}ms")

    controller = EnergyFeedbackController(
        power_target_w=args.power_target,
        latency_target_ms=args.latency_target,
        update_interval=args.interval,
    )

    if args.daemon:
        print("[KAI] Running in daemon mode (Ctrl+C to stop)...")
        try:
            controller.run_forever()
        except KeyboardInterrupt:
            print("\n[KAI] Stopping control loop...")
            controller.stop()
    else:
        print("[KAI] Running single control step...")
        config = controller.step()
        print(f"[KAI] Recommended config: {config}")


def cmd_fault_tolerant(args):
    """Run with fault-tolerant pipeline."""
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker
    from model.resource_detector import ResourceDetector
    from model.fault_tolerant_pipeline import FaultTolerantPipeline

    print(f"[KAI] Loading model: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    detector = ResourceDetector(mode=args.resource_mode)
    nodes = detector.scan()
    num_chunks = args.num_chunks or len(nodes)

    print(f"[KAI] Partitioning into {num_chunks} chunks...")
    chunker = LayerChunker(loader)
    chunks = chunker.create_chunks(num_chunks)

    print("[KAI] Loading weights...")
    _load_real_weights(loader, chunks, args.device, quantize=args.quantize)

    print("[KAI] Setting up fault-tolerant pipeline...")
    pipeline = FaultTolerantPipeline(
        chunks=chunks,
        nodes=nodes,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        health_check_interval=args.health_interval,
    )

    tokenizer = loader.get_tokenizer()
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")

    print("[KAI] Running inference with fault tolerance...")
    print("---")

    output = pipeline.forward(input_ids.to(args.device))
    output_ids = output.argmax(dim=-1)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(result)
    print("---")

    stats = pipeline.get_stats()
    print(f"[KAI] Checkpoints created: {stats['checkpoints_created']}")
    print(f"[KAI] Failures detected: {stats['failures_detected']}")
    print(f"[KAI] Recoveries: {stats['recoveries']}")
    print("[KAI] Done.")


def cmd_plugins(args):
    """List and manage plugins."""
    from model.plugin_architecture import PluginRegistry

    if args.action == "list":
        print("[KAI] Registered Plugins:")
        for category in ["scheduler", "optimizer", "executor", "cache", "placement", "parallelism"]:
            plugins = PluginRegistry.list_plugins(category)
            if plugins:
                print(f"\n  {category}:")
                for name in plugins:
                    plugin = PluginRegistry.get(category, name)
                    desc = getattr(plugin, "__doc__", "No description") or "No description"
                    desc = desc.strip().split("\n")[0][:60]
                    print(f"    - {name}: {desc}")
    elif args.action == "info":
        if not args.name:
            print("[KAI] Error: --name required for info action")
            return
        plugin = PluginRegistry.get(args.category, args.name)
        if plugin:
            print(f"[KAI] Plugin: {args.name}")
            print(f"  Category: {args.category}")
            print(f"  Class: {plugin.__name__}")
            doc = getattr(plugin, "__doc__", "No documentation")
            print(f"  Documentation:\n{doc}")
        else:
            print(f"[KAI] Plugin not found: {args.category}/{args.name}")


def cmd_prepare(args):
    """Download model, chunk weights, and save for K8s deployment."""
    from model.hf_loader import HFModelLoader
    from model.layer_chunker import LayerChunker

    print(f"[KAI] Preparing model: {args.model}")
    loader = HFModelLoader(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )

    try:
        loader.validate_architecture()
    except ValueError as e:
        print(f"[KAI] Error: {e}")
        sys.exit(1)

    size_est = loader.get_model_size_estimate()
    dtype_key = "float16_mb" if args.dtype in ("float16", "fp16") else "float32_mb"
    est_mb = size_est.get(dtype_key, size_est["float32_mb"])
    print(f"[KAI] Model: ~{size_est['params_millions']:.0f}M params, ~{est_mb:.0f} MB ({args.dtype})")

    # Create chunks
    num_chunks = args.num_chunks
    print(f"[KAI] Partitioning into {num_chunks} chunks...")
    chunker = LayerChunker(loader)
    chunks = chunker.create_chunks(num_chunks)

    for c in chunks:
        print(f"  Chunk {c.chunk_id}: {c.layer_names} (~{c.estimate_memory_mb():.0f} MB)")

    # Load weights and save per-chunk
    print("[KAI] Loading and saving chunk weights...")
    _load_real_weights(loader, chunks, "cpu", quantize=getattr(args, "quantize", None))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for chunk in chunks:
        path = chunker.save_chunk_weights(chunk, output_dir)
        print(f"  Saved chunk {chunk.chunk_id} -> {path}")

    # Save chunk manifest
    manifest = {
        "model": args.model,
        "dtype": args.dtype,
        "num_chunks": num_chunks,
        "quantize": getattr(args, "quantize", None),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "layer_names": c.layer_names,
                "memory_mb": round(c.estimate_memory_mb(), 2),
            }
            for c in chunks
        ],
    }
    manifest_path = os.path.join(output_dir, "chunk_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[KAI] Manifest saved: {manifest_path}")
    print("[KAI] Preparation complete. Chunk weights ready for K8s deployment.")


def _load_real_weights(loader, chunks, device, quantize=None):
    """Load actual model weights into chunk modules.

    For small models, loads the full model then distributes layers to chunks.
    For large models (or when system RAM is limited), uses shard-based
    loading via WeightMapper so only needed shards are read from disk.
    """
    import psutil
    from transformers import AutoModelForCausalLM

    model_name = loader.model_name
    dtype = loader.torch_dtype

    # Estimate model size to decide loading strategy
    size_est = loader.get_model_size_estimate()
    dtype_key = "float16_mb" if dtype == torch.float16 else "float32_mb"
    est_mb = size_est.get(dtype_key, size_est["float32_mb"])
    avail_ram_mb = psutil.virtual_memory().available / (1024 ** 2)

    # Use shard-based loading if model is larger than 80% of available RAM
    use_shard_loading = est_mb > avail_ram_mb * 0.8

    if use_shard_loading:
        logger.info(
            "Model (~%.0f MB) exceeds 80%% of available RAM (~%.0f MB). "
            "Using shard-based loading.",
            est_mb, avail_ram_mb,
        )
        _load_weights_shard_based(loader, chunks, device, quantize)
        return

    # Full-model loading for small models
    try:
        real_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=loader.trust_remote_code,
            token=loader.token,
        )
        real_model.eval()
    except Exception as e:
        logger.warning("Could not load full model: %s. Using meta weights.", e)
        return

    # Detect structure and get real layers
    from model.hf_loader import HFModelLoader
    embed, blocks, norm, lm_head = HFModelLoader._detect_structure(real_model)

    # Map layer names to real modules
    real_layer_map = {"embed": embed}
    for i, block in enumerate(blocks):
        real_layer_map[f"layer_{i}"] = block
    if norm is not None:
        real_layer_map["norm"] = norm
    if lm_head is not None:
        real_layer_map["lm_head"] = lm_head

    # Apply quantization if requested
    if quantize:
        from model.quantizer import quantize_module
        for name, module in real_layer_map.items():
            real_layer_map[name] = quantize_module(module, quantize)

    # Replace chunk modules with real-weight versions
    for chunk in chunks:
        for name in chunk.layer_names:
            if name in real_layer_map:
                chunk.layers[name] = real_layer_map[name]
        chunk.to(device)
        chunk.eval()


def _load_weights_shard_based(loader, chunks, device, quantize=None):
    """Load weights using shard-based loading for large models.

    Only loads the checkpoint shards needed for each chunk, avoiding
    loading the full model into RAM.
    """
    from model.weight_utils import WeightMapper

    mapper = WeightMapper(
        loader.model_name,
        token=loader.token,
    )

    for chunk in chunks:
        logger.info("Shard-loading weights for chunk %d: %s", chunk.chunk_id, chunk.layer_names)
        try:
            state_dict = mapper.load_state_dict_for_layers(chunk.layer_names, device=device)
            # Load matching keys into the chunk
            missing, unexpected = chunk.load_state_dict(state_dict, strict=False)
            if missing:
                logger.debug("Chunk %d missing keys (expected for partial load): %d", chunk.chunk_id, len(missing))
            # Apply quantization if requested
            if quantize:
                from model.quantizer import quantize_module
                for name in chunk.layer_names:
                    if name in chunk.layers:
                        chunk.layers[name] = quantize_module(chunk.layers[name], quantize)
            chunk.to(device)
            chunk.eval()
        except Exception as e:
            logger.warning("Shard-loading failed for chunk %d: %s", chunk.chunk_id, e)


def main():
    parser = argparse.ArgumentParser(
        prog="kai",
        description="KAI — Run large AI models on clusters of low-end PCs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Generate text with a distributed model")
    run_parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    run_parser.add_argument("--prompt", required=True, help="Input text prompt")
    run_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    run_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    run_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    run_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    run_parser.add_argument("--num-chunks", type=int, default=None, help="Number of chunks (default: auto)")
    run_parser.add_argument("--dtype", default="float16", help="Weight dtype (float16/bfloat16/float32)")
    run_parser.add_argument("--device", default="cpu", help="Compute device (cpu/cuda:0)")
    run_parser.add_argument("--stream", action="store_true", help="Stream tokens as generated")
    run_parser.add_argument("--resource-mode", default="local", help="Resource scan mode (local/kubernetes)")
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--token", default=None, help="HuggingFace token for gated models")
    run_parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"],
                            help="Quantize model weights (4bit NF4 or 8bit INT8)")
    run_parser.add_argument("--offload", action="store_true",
                            help="Enable FlexGen-style CPU/disk offloading for models exceeding GPU VRAM")
    run_parser.add_argument("--gpu-budget-mb", type=float, default=0,
                            help="GPU VRAM budget in MB for offloading (0 = auto-detect)")
    run_parser.add_argument("--disk-swap-dir", default="/tmp/kai_swap",
                            help="Directory for disk-swapped weights (default: /tmp/kai_swap)")
    run_parser.set_defaults(func=cmd_run)

    # --- scan ---
    scan_parser = subparsers.add_parser("scan", help="Scan cluster resources")
    scan_parser.add_argument("--mode", default="local", choices=["local", "kubernetes"])
    scan_parser.set_defaults(func=cmd_scan)

    # --- partition ---
    part_parser = subparsers.add_parser("partition", help="Preview model partitioning")
    part_parser.add_argument("--model", required=True, help="HuggingFace model name")
    part_parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to simulate")
    part_parser.add_argument("--dtype", default="float16", help="Weight dtype")
    part_parser.add_argument("--mode", default="local", choices=["local", "kubernetes"])
    part_parser.add_argument("--trust-remote-code", action="store_true")
    part_parser.add_argument("--token", default=None)
    part_parser.set_defaults(func=cmd_partition)

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run energy benchmark (original KAI)")
    bench_parser.add_argument("--mode", default="local", choices=["local", "kubernetes", "both"])
    bench_parser.add_argument("--model", default="transformer", choices=["transformer", "cnn"])
    bench_parser.add_argument("--hf-model", default=None,
                              help="HuggingFace model name for benchmarking (overrides --model)")
    bench_parser.add_argument("--num-chunks", type=int, default=2)
    bench_parser.add_argument("--iterations", type=int, default=10)
    bench_parser.add_argument("--batch-size", type=int, default=8)
    bench_parser.add_argument("--output-dir", default="logs")
    bench_parser.add_argument("--sampling-rate", type=float, default=1.0,
                              help="GPU monitoring sampling interval in seconds (e.g. 0.1 for 100ms)")
    bench_parser.add_argument("--enable-deas", action="store_true",
                              help="Enable Dynamic Energy-Aware Scheduling during Kubernetes benchmark")
    bench_parser.add_argument("--deas-cooldown", type=float, default=30.0,
                              help="DEAS minimum seconds between migration attempts (default: 30)")
    bench_parser.set_defaults(func=cmd_benchmark)

    # --- dashboard ---
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501)
    dash_parser.add_argument("--legacy", action="store_true",
                             help="Launch the legacy analysis-only dashboard")
    dash_parser.set_defaults(func=cmd_dashboard)

    # --- build ---
    build_parser = subparsers.add_parser("build", help="Build Docker images for KAI")
    build_parser.add_argument("--tag", default="kai:latest", help="Base image tag")
    build_parser.add_argument("--push", action="store_true", help="Push images after build")
    build_parser.set_defaults(func=cmd_build)

    # --- prepare ---
    prep_parser = subparsers.add_parser("prepare", help="Download model and save chunk weights")
    prep_parser.add_argument("--model", required=True, help="HuggingFace model name")
    prep_parser.add_argument("--num-chunks", type=int, default=3, help="Number of chunks")
    prep_parser.add_argument("--output-dir", default="data/chunks", help="Output directory")
    prep_parser.add_argument("--dtype", default="float16", help="Weight dtype")
    prep_parser.add_argument("--trust-remote-code", action="store_true")
    prep_parser.add_argument("--token", default=None)
    prep_parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"],
                             help="Quantize chunk weights (4bit NF4 or 8bit INT8)")
    prep_parser.set_defaults(func=cmd_prepare)

    # --- autotune ---
    autotune_parser = subparsers.add_parser("autotune", help="Auto-tune configuration for optimal performance")
    autotune_parser.add_argument("--model", required=True, help="HuggingFace model name")
    autotune_parser.add_argument("--objective", default="balanced",
                                  choices=["latency", "throughput", "energy", "memory", "balanced"],
                                  help="Optimization objective")
    autotune_parser.add_argument("--max-trials", type=int, default=20, help="Maximum number of trials")
    autotune_parser.add_argument("--strategy", default="bayesian",
                                  choices=["random", "grid", "bayesian"],
                                  help="Search strategy")
    autotune_parser.add_argument("--timeout", type=float, default=0, help="Timeout in seconds (0=no timeout)")
    autotune_parser.add_argument("--output-dir", default="./tuning_results", help="Output directory")
    autotune_parser.add_argument("--dtype", default="float16", help="Weight dtype")
    autotune_parser.add_argument("--resource-mode", default="local", choices=["local", "kubernetes"])
    autotune_parser.add_argument("--trust-remote-code", action="store_true")
    autotune_parser.add_argument("--token", default=None)
    autotune_parser.add_argument("--all-precisions", action="store_true",
                                  help="Test all precision options (fp32, fp16, int8, int4)")
    autotune_parser.add_argument("--batch-range", action="store_true",
                                  help="Test extended batch size range")
    autotune_parser.set_defaults(func=cmd_autotune)

    # --- speculative ---
    spec_parser = subparsers.add_parser("speculative", help="Run with speculative decoding")
    spec_parser.add_argument("--model", required=True, help="Main HuggingFace model name")
    spec_parser.add_argument("--draft-model", default=None,
                              help="Draft model for speculation (default: auto-created)")
    spec_parser.add_argument("--prompt", required=True, help="Input prompt")
    spec_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    spec_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    spec_parser.add_argument("--speculation-length", type=int, default=5,
                              help="Number of tokens to speculate ahead")
    spec_parser.add_argument("--verification", default="strict",
                              choices=["strict", "threshold", "sampling"],
                              help="Verification mode")
    spec_parser.add_argument("--num-chunks", type=int, default=None)
    spec_parser.add_argument("--dtype", default="float16")
    spec_parser.add_argument("--device", default="cpu")
    spec_parser.add_argument("--resource-mode", default="local", choices=["local", "kubernetes"])
    spec_parser.add_argument("--trust-remote-code", action="store_true")
    spec_parser.add_argument("--token", default=None)
    spec_parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"])
    spec_parser.set_defaults(func=cmd_speculative)

    # --- hybrid ---
    hybrid_parser = subparsers.add_parser("hybrid", help="Run with hybrid parallelism")
    hybrid_parser.add_argument("--model", required=True, help="HuggingFace model name")
    hybrid_parser.add_argument("--prompt", required=True, help="Input prompt")
    hybrid_parser.add_argument("--mode", default="auto",
                                choices=["auto", "pipeline", "tensor", "hybrid"],
                                help="Parallelism mode")
    hybrid_parser.add_argument("--tensor-parallel", type=int, default=2,
                                help="Tensor parallel size (GPUs per tensor group)")
    hybrid_parser.add_argument("--dtype", default="float16")
    hybrid_parser.add_argument("--device", default="cpu")
    hybrid_parser.add_argument("--resource-mode", default="local", choices=["local", "kubernetes"])
    hybrid_parser.add_argument("--trust-remote-code", action="store_true")
    hybrid_parser.add_argument("--token", default=None)
    hybrid_parser.set_defaults(func=cmd_hybrid)

    # --- placement ---
    place_parser = subparsers.add_parser("placement", help="Generate intelligent placement plan")
    place_parser.add_argument("--model", required=True, help="HuggingFace model name")
    place_parser.add_argument("--objective", default="balanced",
                               choices=["latency", "energy", "memory", "balanced"],
                               help="Optimization objective")
    place_parser.add_argument("--output", default=None, help="Output file for placement plan (JSON)")
    place_parser.add_argument("--dtype", default="float16")
    place_parser.add_argument("--resource-mode", default="local", choices=["local", "kubernetes"])
    place_parser.add_argument("--trust-remote-code", action="store_true")
    place_parser.add_argument("--token", default=None)
    place_parser.set_defaults(func=cmd_placement)

    # --- energy-loop ---
    energy_parser = subparsers.add_parser("energy-loop", help="Start energy feedback control loop")
    energy_parser.add_argument("--power-target", type=float, default=100.0,
                                help="Target power consumption in Watts")
    energy_parser.add_argument("--latency-target", type=float, default=100.0,
                                help="Target latency in milliseconds")
    energy_parser.add_argument("--interval", type=float, default=1.0,
                                help="Control loop update interval in seconds")
    energy_parser.add_argument("--daemon", action="store_true",
                                help="Run as daemon (continuous)")
    energy_parser.set_defaults(func=cmd_energy_loop)

    # --- fault-tolerant ---
    ft_parser = subparsers.add_parser("fault-tolerant", help="Run with fault-tolerant pipeline")
    ft_parser.add_argument("--model", required=True, help="HuggingFace model name")
    ft_parser.add_argument("--prompt", required=True, help="Input prompt")
    ft_parser.add_argument("--num-chunks", type=int, default=None)
    ft_parser.add_argument("--checkpoint-interval", type=int, default=5,
                            help="Checkpoint every N layers")
    ft_parser.add_argument("--checkpoint-dir", default="/tmp/kai_checkpoints",
                            help="Directory for checkpoints")
    ft_parser.add_argument("--health-interval", type=float, default=5.0,
                            help="Health check interval in seconds")
    ft_parser.add_argument("--dtype", default="float16")
    ft_parser.add_argument("--device", default="cpu")
    ft_parser.add_argument("--resource-mode", default="local", choices=["local", "kubernetes"])
    ft_parser.add_argument("--trust-remote-code", action="store_true")
    ft_parser.add_argument("--token", default=None)
    ft_parser.add_argument("--quantize", default=None, choices=["4bit", "8bit"])
    ft_parser.set_defaults(func=cmd_fault_tolerant)

    # --- plugins ---
    plugins_parser = subparsers.add_parser("plugins", help="List and manage plugins")
    plugins_parser.add_argument("--action", default="list", choices=["list", "info"],
                                 help="Action to perform")
    plugins_parser.add_argument("--category", default="scheduler",
                                 choices=["scheduler", "optimizer", "executor", "cache", "placement", "parallelism"],
                                 help="Plugin category")
    plugins_parser.add_argument("--name", default=None, help="Plugin name (for info action)")
    plugins_parser.set_defaults(func=cmd_plugins)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
