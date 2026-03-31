"""
Auto-Tuning Benchmark System for KAI.

Automatically tests multiple configurations and finds optimal settings:
- Partition strategies
- Precision modes
- Offloading settings

Outputs best configuration based on:
- Energy efficiency
- Latency
- Throughput

Usage::

    from model.auto_tuner import AutoTuner, TuningObjective
    
    tuner = AutoTuner(model_loader, nodes)
    
    # Run auto-tuning
    result = tuner.tune(
        objective=TuningObjective.ENERGY_EFFICIENCY,
        max_trials=20,
    )
    
    print(result.best_config)
"""

import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TuningObjective(Enum):
    """Optimization objectives for auto-tuning."""
    LATENCY = "latency"               # Minimize inference latency
    THROUGHPUT = "throughput"         # Maximize tokens/second
    ENERGY_EFFICIENCY = "energy"      # Maximize tokens/Joule
    MEMORY = "memory"                 # Minimize memory usage
    BALANCED = "balanced"             # Multi-objective


@dataclass
class ConfigurationSpace:
    """Defines the space of configurations to search."""
    
    # Partition strategies
    num_chunks_range: Tuple[int, int] = (1, 8)
    
    # Precision options
    precision_options: List[str] = field(
        default_factory=lambda: ["fp32", "fp16", "int8", "int4"]
    )
    
    # Batch sizes
    batch_size_range: Tuple[int, int] = (1, 64)
    batch_size_values: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64]
    )
    
    # Offloading
    offload_options: List[bool] = field(
        default_factory=lambda: [False, True]
    )
    gpu_budget_options: List[float] = field(
        default_factory=lambda: [0.5, 0.7, 0.9, 1.0]  # Fraction of VRAM
    )
    
    # Parallelism modes
    parallelism_options: List[str] = field(
        default_factory=lambda: ["pipeline", "tensor", "hybrid"]
    )
    
    def get_num_configurations(self) -> int:
        """Estimate total number of configurations."""
        num_chunks = self.num_chunks_range[1] - self.num_chunks_range[0] + 1
        return (
            num_chunks *
            len(self.precision_options) *
            len(self.batch_size_values) *
            len(self.offload_options) *
            len(self.parallelism_options)
        )
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        return {
            "num_chunks": random.randint(*self.num_chunks_range),
            "precision": random.choice(self.precision_options),
            "batch_size": random.choice(self.batch_size_values),
            "offload_enabled": random.choice(self.offload_options),
            "gpu_budget_fraction": random.choice(self.gpu_budget_options),
            "parallelism_mode": random.choice(self.parallelism_options),
        }
    
    def enumerate_all(self) -> List[Dict[str, Any]]:
        """Enumerate all configurations (for grid search)."""
        configs = []
        
        for num_chunks in range(self.num_chunks_range[0], self.num_chunks_range[1] + 1):
            for precision in self.precision_options:
                for batch_size in self.batch_size_values:
                    for offload in self.offload_options:
                        for parallelism in self.parallelism_options:
                            configs.append({
                                "num_chunks": num_chunks,
                                "precision": precision,
                                "batch_size": batch_size,
                                "offload_enabled": offload,
                                "gpu_budget_fraction": 0.8,
                                "parallelism_mode": parallelism,
                            })
        
        return configs


@dataclass
class TrialResult:
    """Result of a single tuning trial."""
    trial_id: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str = ""
    duration_s: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    @property
    def latency_ms(self) -> float:
        return self.metrics.get("latency_ms", float("inf"))
    
    @property
    def throughput_tokens_per_sec(self) -> float:
        return self.metrics.get("throughput_tokens_per_sec", 0.0)
    
    @property
    def energy_efficiency(self) -> float:
        """Tokens per Joule."""
        return self.metrics.get("tokens_per_joule", 0.0)
    
    @property
    def memory_mb(self) -> float:
        return self.metrics.get("memory_mb", float("inf"))
    
    def get_score(self, objective: TuningObjective) -> float:
        """Get score for the given objective (higher is better)."""
        if objective == TuningObjective.LATENCY:
            return -self.latency_ms  # Negative because lower is better
        elif objective == TuningObjective.THROUGHPUT:
            return self.throughput_tokens_per_sec
        elif objective == TuningObjective.ENERGY_EFFICIENCY:
            return self.energy_efficiency
        elif objective == TuningObjective.MEMORY:
            return -self.memory_mb  # Negative because lower is better
        else:  # BALANCED
            # Weighted combination
            return (
                -0.3 * self.latency_ms / 100 +
                0.3 * self.throughput_tokens_per_sec / 100 +
                0.3 * self.energy_efficiency +
                -0.1 * self.memory_mb / 1000
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 2),
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class TuningResult:
    """Result of the complete auto-tuning process."""
    objective: TuningObjective
    best_config: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_trials: List[TrialResult]
    total_duration_s: float
    
    def summary(self) -> str:
        """Human-readable summary."""
        successful = [t for t in self.all_trials if t.success]
        
        lines = [
            f"Auto-Tuning Results",
            f"  Objective: {self.objective.value}",
            f"  Total trials: {len(self.all_trials)}",
            f"  Successful trials: {len(successful)}",
            f"  Total duration: {self.total_duration_s:.1f}s",
            "",
            "Best Configuration:",
        ]
        
        for key, value in self.best_config.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append("Best Metrics:")
        
        for key, value in self.best_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.3f}")
            else:
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective.value,
            "best_config": self.best_config,
            "best_metrics": self.best_metrics,
            "num_trials": len(self.all_trials),
            "total_duration_s": round(self.total_duration_s, 2),
            "trials": [t.to_dict() for t in self.all_trials],
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TuningResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        trials = [
            TrialResult(
                trial_id=t["trial_id"],
                config=t["config"],
                metrics=t["metrics"],
                timestamp=t["timestamp"],
                duration_s=t["duration_s"],
                success=t["success"],
                error_message=t.get("error_message", ""),
            )
            for t in data["trials"]
        ]
        
        return cls(
            objective=TuningObjective(data["objective"]),
            best_config=data["best_config"],
            best_metrics=data["best_metrics"],
            all_trials=trials,
            total_duration_s=data["total_duration_s"],
        )


class BenchmarkRunner:
    """Runs benchmarks for a specific configuration.
    
    Parameters
    ----------
    model_loader
        HFModelLoader or similar
    nodes : list
        Cluster nodes
    warmup_iterations : int
        Warmup iterations before timing
    benchmark_iterations : int
        Iterations for timing
    """
    
    def __init__(
        self,
        model_loader,
        nodes: List[Any],
        warmup_iterations: int = 2,
        benchmark_iterations: int = 5,
    ):
        self._loader = model_loader
        self._nodes = nodes
        self._warmup_iterations = warmup_iterations
        self._benchmark_iterations = benchmark_iterations
    
    def run_benchmark(
        self,
        config: Dict[str, Any],
        input_tokens: int = 32,
        output_tokens: int = 64,
    ) -> Dict[str, float]:
        """Run benchmark with given configuration.
        
        Parameters
        ----------
        config : dict
            Configuration to benchmark
        input_tokens : int
            Number of input tokens
        output_tokens : int
            Number of tokens to generate
            
        Returns
        -------
        dict
            Benchmark metrics
        """
        metrics = {}
        
        try:
            # Setup based on config
            chunks = self._setup_model(config)
            
            # Create dummy input
            input_tensor = torch.randn(1, input_tokens, 768)  # Simplified
            
            # Warmup
            for _ in range(self._warmup_iterations):
                self._run_inference(chunks, input_tensor)
            
            # Benchmark
            latencies = []
            memory_samples = []
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            for _ in range(self._benchmark_iterations):
                start = time.perf_counter()
                
                self._run_inference(chunks, input_tensor)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
                
                # Sample memory
                if torch.cuda.is_available():
                    memory_samples.append(
                        torch.cuda.memory_allocated() / (1024**2)
                    )
            
            # Calculate metrics
            avg_latency = sum(latencies) / len(latencies)
            throughput = output_tokens / (avg_latency / 1000)
            
            # Estimate energy (simplified)
            power_w = 150.0  # Assumed power draw
            energy_j = power_w * (avg_latency / 1000)
            tokens_per_joule = output_tokens / energy_j if energy_j > 0 else 0
            
            metrics = {
                "latency_ms": avg_latency,
                "latency_std_ms": (
                    (sum((x - avg_latency)**2 for x in latencies) / len(latencies)) ** 0.5
                ),
                "throughput_tokens_per_sec": throughput,
                "tokens_per_joule": tokens_per_joule,
                "memory_mb": sum(memory_samples) / len(memory_samples) if memory_samples else 0,
                "energy_j_per_inference": energy_j,
            }
            
        except Exception as e:
            logger.error("Benchmark failed: %s", e)
            metrics = {
                "latency_ms": float("inf"),
                "throughput_tokens_per_sec": 0.0,
                "tokens_per_joule": 0.0,
                "memory_mb": float("inf"),
                "error": str(e),
            }
        
        return metrics
    
    def _setup_model(self, config: Dict[str, Any]) -> List[Any]:
        """Setup model with configuration."""
        # Simplified - would actually configure chunking, precision, etc.
        return []
    
    def _run_inference(self, chunks: List[Any], input_tensor: torch.Tensor) -> None:
        """Run inference through chunks."""
        # Simplified - would run actual inference
        time.sleep(0.01)  # Simulate work


class SearchStrategy:
    """Base class for search strategies."""
    
    def __init__(self, config_space: ConfigurationSpace):
        self.config_space = config_space
    
    def get_next_config(
        self,
        previous_results: List[TrialResult],
    ) -> Optional[Dict[str, Any]]:
        """Get next configuration to try."""
        raise NotImplementedError
    
    def is_complete(
        self,
        num_trials: int,
        max_trials: int,
    ) -> bool:
        """Check if search is complete."""
        return num_trials >= max_trials


class RandomSearch(SearchStrategy):
    """Random search strategy."""
    
    def __init__(
        self,
        config_space: ConfigurationSpace,
        seed: Optional[int] = None,
    ):
        super().__init__(config_space)
        if seed is not None:
            random.seed(seed)
        self._tried: Set[str] = set()
    
    def get_next_config(
        self,
        previous_results: List[TrialResult],
    ) -> Optional[Dict[str, Any]]:
        for _ in range(100):  # Max attempts
            config = self.config_space.sample_random()
            config_hash = json.dumps(config, sort_keys=True)
            
            if config_hash not in self._tried:
                self._tried.add(config_hash)
                return config
        
        return None


class GridSearch(SearchStrategy):
    """Grid search strategy."""
    
    def __init__(self, config_space: ConfigurationSpace):
        super().__init__(config_space)
        self._all_configs = config_space.enumerate_all()
        self._index = 0
    
    def get_next_config(
        self,
        previous_results: List[TrialResult],
    ) -> Optional[Dict[str, Any]]:
        if self._index >= len(self._all_configs):
            return None
        
        config = self._all_configs[self._index]
        self._index += 1
        return config
    
    def is_complete(
        self,
        num_trials: int,
        max_trials: int,
    ) -> bool:
        return self._index >= len(self._all_configs) or num_trials >= max_trials


class BayesianSearch(SearchStrategy):
    """Bayesian optimization search strategy (simplified)."""
    
    def __init__(
        self,
        config_space: ConfigurationSpace,
        objective: TuningObjective,
        exploration_weight: float = 0.1,
    ):
        super().__init__(config_space)
        self._objective = objective
        self._exploration_weight = exploration_weight
    
    def get_next_config(
        self,
        previous_results: List[TrialResult],
    ) -> Optional[Dict[str, Any]]:
        if len(previous_results) < 5:
            # Initial exploration: random
            return self.config_space.sample_random()
        
        # Simple heuristic: slightly mutate best config
        best_result = max(
            previous_results,
            key=lambda r: r.get_score(self._objective) if r.success else float("-inf")
        )
        
        config = dict(best_result.config)
        
        # Random mutation
        mutation_type = random.choice(["precision", "batch_size", "chunks"])
        
        if mutation_type == "precision" and random.random() < self._exploration_weight:
            config["precision"] = random.choice(self.config_space.precision_options)
        elif mutation_type == "batch_size" and random.random() < self._exploration_weight:
            config["batch_size"] = random.choice(self.config_space.batch_size_values)
        elif mutation_type == "chunks":
            config["num_chunks"] = random.randint(*self.config_space.num_chunks_range)
        
        return config


class AutoTuner:
    """Auto-tuning system for finding optimal KAI configurations.
    
    Parameters
    ----------
    model_loader
        HFModelLoader or similar
    nodes : list
        Cluster nodes
    config_space : ConfigurationSpace, optional
        Search space
    output_dir : str
        Directory for results
    """
    
    def __init__(
        self,
        model_loader,
        nodes: List[Any],
        config_space: Optional[ConfigurationSpace] = None,
        output_dir: str = "./tuning_results",
    ):
        self._loader = model_loader
        self._nodes = nodes
        self._config_space = config_space or ConfigurationSpace()
        self._output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._benchmark_runner = BenchmarkRunner(model_loader, nodes)
    
    def tune(
        self,
        objective: TuningObjective = TuningObjective.BALANCED,
        max_trials: int = 20,
        search_strategy: str = "random",
        timeout_s: Optional[float] = None,
        callback: Optional[Callable[[TrialResult], None]] = None,
    ) -> TuningResult:
        """Run auto-tuning.
        
        Parameters
        ----------
        objective : TuningObjective
            Optimization objective
        max_trials : int
            Maximum number of trials
        search_strategy : str
            Search strategy: "random", "grid", "bayesian"
        timeout_s : float, optional
            Total timeout in seconds
        callback : callable, optional
            Called after each trial
            
        Returns
        -------
        TuningResult
            Tuning results
        """
        logger.info(
            "Starting auto-tuning: objective=%s, max_trials=%d, strategy=%s",
            objective.value, max_trials, search_strategy
        )
        
        # Create search strategy
        if search_strategy == "random":
            strategy = RandomSearch(self._config_space)
        elif search_strategy == "grid":
            strategy = GridSearch(self._config_space)
        elif search_strategy == "bayesian":
            strategy = BayesianSearch(self._config_space, objective)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
        
        # Run trials
        all_trials: List[TrialResult] = []
        start_time = time.perf_counter()
        
        trial_id = 0
        while not strategy.is_complete(len(all_trials), max_trials):
            # Check timeout
            if timeout_s and (time.perf_counter() - start_time) > timeout_s:
                logger.info("Timeout reached")
                break
            
            # Get next configuration
            config = strategy.get_next_config(all_trials)
            if config is None:
                logger.info("No more configurations to try")
                break
            
            # Run trial
            trial_result = self._run_trial(trial_id, config)
            all_trials.append(trial_result)
            
            # Callback
            if callback:
                callback(trial_result)
            
            # Log progress
            logger.info(
                "Trial %d: %s, latency=%.1fms, throughput=%.1f tok/s",
                trial_id,
                "success" if trial_result.success else "failed",
                trial_result.latency_ms,
                trial_result.throughput_tokens_per_sec,
            )
            
            trial_id += 1
        
        total_duration = time.perf_counter() - start_time
        
        # Find best configuration
        successful_trials = [t for t in all_trials if t.success]
        
        if successful_trials:
            best_trial = max(
                successful_trials,
                key=lambda t: t.get_score(objective)
            )
            best_config = best_trial.config
            best_metrics = best_trial.metrics
        else:
            best_config = {}
            best_metrics = {}
        
        result = TuningResult(
            objective=objective,
            best_config=best_config,
            best_metrics=best_metrics,
            all_trials=all_trials,
            total_duration_s=total_duration,
        )
        
        # Save results
        result_path = os.path.join(
            self._output_dir,
            f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        result.save(result_path)
        logger.info("Results saved to %s", result_path)
        
        return result
    
    def _run_trial(
        self,
        trial_id: int,
        config: Dict[str, Any],
    ) -> TrialResult:
        """Run a single trial."""
        start_time = time.perf_counter()
        
        try:
            metrics = self._benchmark_runner.run_benchmark(config)
            success = "error" not in metrics
            error_message = metrics.pop("error", "")
            
        except Exception as e:
            metrics = {}
            success = False
            error_message = str(e)
        
        duration = time.perf_counter() - start_time
        
        return TrialResult(
            trial_id=trial_id,
            config=config,
            metrics=metrics,
            duration_s=duration,
            success=success,
            error_message=error_message,
        )
    
    def compare_configs(
        self,
        configs: List[Dict[str, Any]],
        iterations: int = 5,
    ) -> List[TrialResult]:
        """Compare specific configurations.
        
        Parameters
        ----------
        configs : list[dict]
            Configurations to compare
        iterations : int
            Benchmark iterations per config
            
        Returns
        -------
        list[TrialResult]
            Results for each config
        """
        results = []
        
        old_iterations = self._benchmark_runner._benchmark_iterations
        self._benchmark_runner._benchmark_iterations = iterations
        
        for i, config in enumerate(configs):
            result = self._run_trial(i, config)
            results.append(result)
        
        self._benchmark_runner._benchmark_iterations = old_iterations
        
        return results


def quick_tune(
    model_name: str,
    nodes: Optional[List[Any]] = None,
    objective: TuningObjective = TuningObjective.BALANCED,
    max_trials: int = 10,
) -> TuningResult:
    """Quick auto-tuning with sensible defaults.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model name
    nodes : list, optional
        Cluster nodes (auto-detected if not provided)
    objective : TuningObjective
        Optimization objective
    max_trials : int
        Maximum trials
        
    Returns
    -------
    TuningResult
        Tuning results
    """
    from model.hf_loader import HFModelLoader
    from model.resource_detector import ResourceDetector
    
    # Load model config
    loader = HFModelLoader(model_name)
    
    # Detect nodes
    if nodes is None:
        detector = ResourceDetector(mode="local")
        nodes = detector.scan()
    
    # Run tuning
    tuner = AutoTuner(loader, nodes)
    return tuner.tune(
        objective=objective,
        max_trials=max_trials,
        search_strategy="bayesian",
    )
