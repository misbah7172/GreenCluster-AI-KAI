"""
Simulation Optimizer Module
Optimization tricks for faster and more efficient simulation.

Key Features:
- Simplified repeated layer execution
- Approximate decode phase
- Layer fusion and caching
- Memory-efficient simulation
- Profile-guided optimization
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import threading


class OptimizationLevel(Enum):
    """Simulation optimization levels."""
    NONE = 0  # Full fidelity
    BASIC = 1  # Layer caching
    AGGRESSIVE = 2  # Approximations
    EXTREME = 3  # Heavy approximations (may affect accuracy)


class LayerType(Enum):
    """Types of transformer layers."""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FFN = "ffn"
    NORM = "norm"
    LM_HEAD = "lm_head"
    OTHER = "other"


@dataclass
class LayerProfile:
    """Profile information for a layer."""
    layer_id: str
    layer_type: LayerType
    
    # Timing
    avg_forward_time_ms: float = 0.0
    avg_backward_time_ms: float = 0.0
    
    # Memory
    param_size_mb: float = 0.0
    activation_size_mb: float = 0.0
    
    # Computation
    flops: int = 0
    
    # Optimization potential
    is_repeated: bool = False
    repeat_count: int = 1
    can_fuse: bool = False
    can_approximate: bool = False


@dataclass
class SimulationConfig:
    """Configuration for optimized simulation."""
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    
    # Layer simplification
    merge_repeated_layers: bool = True
    max_unique_layers: int = 4  # Simulate only N unique layers
    
    # Approximation settings
    approximate_attention: bool = False
    attention_approximation_ratio: float = 0.5  # Sample ratio
    approximate_ffn: bool = False
    ffn_approximation_ratio: float = 0.7
    
    # Decode phase approximation
    approximate_decode: bool = True
    decode_sample_interval: int = 10  # Simulate every Nth decode step
    
    # Caching
    cache_layer_outputs: bool = True
    cache_size_mb: int = 1024
    
    # Quantization for simulation
    simulate_in_fp16: bool = True
    simulate_in_int8: bool = False
    
    # Profiling
    enable_profiling: bool = True
    profile_warmup_steps: int = 3


@dataclass
class SimulationResult:
    """Result of optimized simulation."""
    # Timing
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    
    # Estimated metrics (if approximated)
    estimated_full_time_ms: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Accuracy
    approximation_error: float = 0.0
    
    # Statistics
    layers_simulated: int = 0
    layers_skipped: int = 0
    cache_hits: int = 0
    decode_steps_simulated: int = 0
    decode_steps_interpolated: int = 0


class LayerOutputCache:
    """
    Cache for layer outputs to speed up repeated simulations.
    """
    
    def __init__(self, max_size_mb: int = 1024):
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, Any] = {}
        self.cache_sizes: Dict[str, float] = {}
        self.access_count: Dict[str, int] = {}
        self.total_size_mb: float = 0.0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached output."""
        with self._lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
        return None
    
    def put(self, key: str, value: Any, size_mb: float) -> bool:
        """Cache an output. Returns True if cached."""
        with self._lock:
            # Check if fits
            if size_mb > self.max_size_mb:
                return False
            
            # Evict if needed
            while self.total_size_mb + size_mb > self.max_size_mb:
                self._evict_one()
            
            self.cache[key] = value
            self.cache_sizes[key] = size_mb
            self.access_count[key] = 1
            self.total_size_mb += size_mb
            
            return True
    
    def _evict_one(self) -> None:
        """Evict least frequently used entry."""
        if not self.cache:
            return
        
        # Find LFU entry
        lfu_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        
        self.total_size_mb -= self.cache_sizes.get(lfu_key, 0)
        del self.cache[lfu_key]
        del self.cache_sizes[lfu_key]
        del self.access_count[lfu_key]
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self.cache.clear()
            self.cache_sizes.clear()
            self.access_count.clear()
            self.total_size_mb = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(self.access_count.values())
        if total_accesses == 0:
            return 0.0
        # Rough estimate: access_count > 1 means hits
        hits = sum(1 for c in self.access_count.values() if c > 1)
        return hits / len(self.access_count) if self.access_count else 0.0


class RepeatedLayerSimplifier:
    """
    Simplifies repeated identical layers in simulation.
    
    In transformers, many layers are structurally identical.
    We can simulate one and scale the result.
    """
    
    def __init__(self, max_unique_layers: int = 4):
        self.max_unique_layers = max_unique_layers
        self.layer_groups: Dict[str, List[int]] = {}  # group_key -> layer_indices
        self.representative_layers: Dict[str, int] = {}  # group_key -> representative_idx
    
    def analyze_layers(
        self,
        layers: List[Any],
        get_layer_signature: Callable[[Any], str],
    ) -> Dict[str, List[int]]:
        """
        Analyze layers and group identical ones.
        
        Args:
            layers: List of layer modules
            get_layer_signature: Function to get unique signature for a layer
            
        Returns:
            Dict mapping signature -> list of layer indices
        """
        self.layer_groups.clear()
        self.representative_layers.clear()
        
        for idx, layer in enumerate(layers):
            sig = get_layer_signature(layer)
            
            if sig not in self.layer_groups:
                self.layer_groups[sig] = []
                self.representative_layers[sig] = idx
            
            self.layer_groups[sig].append(idx)
        
        return dict(self.layer_groups)
    
    def get_simulation_schedule(self) -> List[Tuple[int, int]]:
        """
        Get optimized simulation schedule.
        
        Returns list of (layer_idx, repeat_count) tuples.
        Only representative layers are simulated, with their repeat count.
        """
        schedule = []
        
        for sig, indices in self.layer_groups.items():
            rep_idx = self.representative_layers[sig]
            count = len(indices)
            schedule.append((rep_idx, count))
        
        # Sort by layer index for sequential execution
        schedule.sort(key=lambda x: x[0])
        
        return schedule
    
    def estimate_time_scaling(
        self,
        representative_time_ms: float,
        repeat_count: int,
    ) -> float:
        """
        Estimate total time for repeated layers.
        
        Uses sub-linear scaling to account for:
        - Memory cache effects
        - Kernel launch overhead amortization
        """
        # Sub-linear scaling factor
        scaling = repeat_count ** 0.95  # Slightly sub-linear
        return representative_time_ms * scaling


class DecodePhaseApproximator:
    """
    Approximates decode phase for faster simulation.
    
    The decode phase generates tokens one at a time, which is slow.
    We can approximate by:
    1. Simulating every Nth step
    2. Interpolating intermediate results
    3. Using statistical models for timing
    """
    
    def __init__(
        self,
        sample_interval: int = 10,
        use_exponential_sampling: bool = True,
    ):
        """
        Initialize decode approximator.
        
        Args:
            sample_interval: Simulate every N decode steps
            use_exponential_sampling: Use exponential decay sampling (more samples early)
        """
        self.sample_interval = sample_interval
        self.use_exponential_sampling = use_exponential_sampling
        
        # Collected timing data
        self.timing_samples: List[Tuple[int, float]] = []  # (step, time_ms)
    
    def get_simulation_steps(self, total_steps: int) -> List[int]:
        """
        Get which decode steps to actually simulate.
        
        Args:
            total_steps: Total number of decode steps
            
        Returns:
            List of step indices to simulate
        """
        if self.use_exponential_sampling:
            return self._exponential_sampling(total_steps)
        else:
            return self._uniform_sampling(total_steps)
    
    def _uniform_sampling(self, total_steps: int) -> List[int]:
        """Uniform sampling of decode steps."""
        steps = [0]  # Always include first step
        
        for i in range(self.sample_interval, total_steps, self.sample_interval):
            steps.append(i)
        
        if total_steps - 1 not in steps:
            steps.append(total_steps - 1)  # Always include last step
        
        return steps
    
    def _exponential_sampling(self, total_steps: int) -> List[int]:
        """
        Exponential sampling - more samples at start (where timing varies more).
        """
        steps = [0]
        
        # Denser sampling in first 20%
        early_steps = int(total_steps * 0.2)
        for i in range(1, early_steps, max(1, self.sample_interval // 3)):
            steps.append(i)
        
        # Sparser sampling for rest
        for i in range(early_steps, total_steps, self.sample_interval):
            if i not in steps:
                steps.append(i)
        
        if total_steps - 1 not in steps:
            steps.append(total_steps - 1)
        
        return sorted(set(steps))
    
    def record_timing(self, step: int, time_ms: float) -> None:
        """Record timing for a simulated step."""
        self.timing_samples.append((step, time_ms))
    
    def estimate_total_time(self, total_steps: int) -> float:
        """
        Estimate total decode time from samples.
        
        Uses linear interpolation and statistical estimation.
        """
        if not self.timing_samples:
            return 0.0
        
        if len(self.timing_samples) == 1:
            return self.timing_samples[0][1] * total_steps
        
        # Sort samples
        samples = sorted(self.timing_samples, key=lambda x: x[0])
        
        total_time = 0.0
        prev_step, prev_time = samples[0]
        
        for step, time_ms in samples[1:]:
            # Interpolate between samples
            step_diff = step - prev_step
            time_per_step = (time_ms + prev_time) / 2  # Average
            total_time += time_per_step * step_diff
            
            prev_step, prev_time = step, time_ms
        
        # Extrapolate remaining steps
        if samples[-1][0] < total_steps - 1:
            remaining = total_steps - 1 - samples[-1][0]
            avg_late_time = samples[-1][1]  # Use last sample
            total_time += avg_late_time * remaining
        
        return total_time
    
    def reset(self) -> None:
        """Reset collected timing data."""
        self.timing_samples.clear()


class AttentionApproximator:
    """
    Approximates attention computation for faster simulation.
    
    Techniques:
    - Sparse attention patterns
    - Low-rank approximation
    - Sample-based estimation
    """
    
    def __init__(
        self,
        approximation_ratio: float = 0.5,
        use_sparse_pattern: bool = True,
    ):
        self.approximation_ratio = approximation_ratio
        self.use_sparse_pattern = use_sparse_pattern
    
    def approximate_attention_time(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        full_attention_time_ms: float,
    ) -> float:
        """
        Estimate attention time with approximation.
        
        Standard attention is O(n²). We estimate time savings from:
        - Sparse patterns (O(n * k) where k << n)
        - Sampling (process subset of heads)
        """
        # Sparse pattern approximation
        if self.use_sparse_pattern:
            # Assume local attention window
            effective_ratio = math.sqrt(self.approximation_ratio)
        else:
            effective_ratio = self.approximation_ratio
        
        # Estimate time (quadratic scaling with sequence length)
        approx_time = full_attention_time_ms * effective_ratio
        
        return approx_time
    
    def get_sparse_pattern(
        self,
        seq_len: int,
        pattern_type: str = "local",
        window_size: int = 128,
    ) -> List[Tuple[int, int]]:
        """
        Get sparse attention pattern indices.
        
        Args:
            seq_len: Sequence length
            pattern_type: "local", "strided", or "random"
            window_size: Window size for local attention
            
        Returns:
            List of (row, col) indices that should be computed
        """
        indices = []
        
        if pattern_type == "local":
            # Local window attention
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2)
                for j in range(start, end):
                    indices.append((i, j))
        
        elif pattern_type == "strided":
            # Strided pattern
            stride = max(1, int(1.0 / self.approximation_ratio))
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    indices.append((i, j))
        
        elif pattern_type == "random":
            # Random sampling
            import random
            num_samples = int(seq_len * seq_len * self.approximation_ratio)
            for _ in range(num_samples):
                i = random.randint(0, seq_len - 1)
                j = random.randint(0, seq_len - 1)
                indices.append((i, j))
        
        return indices


class SimulationOptimizer:
    """
    Main simulation optimizer combining all optimization techniques.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        # Components
        self.cache = LayerOutputCache(max_size_mb=self.config.cache_size_mb)
        self.layer_simplifier = RepeatedLayerSimplifier(
            max_unique_layers=self.config.max_unique_layers
        )
        self.decode_approximator = DecodePhaseApproximator(
            sample_interval=self.config.decode_sample_interval
        )
        self.attention_approximator = AttentionApproximator(
            approximation_ratio=self.config.attention_approximation_ratio
        )
        
        # Profiling data
        self.layer_profiles: Dict[str, LayerProfile] = {}
        self.profiling_enabled = self.config.enable_profiling
    
    def optimize_simulation(
        self,
        model: Any,
        input_data: Any,
        num_decode_steps: int = 100,
    ) -> SimulationResult:
        """
        Run optimized simulation.
        
        Args:
            model: Model to simulate (or mock)
            input_data: Input data
            num_decode_steps: Number of decode steps
            
        Returns:
            SimulationResult with timing and metrics
        """
        result = SimulationResult()
        start_time = time.time()
        
        # Profile if first run
        if self.profiling_enabled and not self.layer_profiles:
            self._profile_model(model)
        
        # Prefill phase
        prefill_start = time.time()
        self._simulate_prefill(model, input_data, result)
        result.prefill_time_ms = (time.time() - prefill_start) * 1000
        
        # Decode phase
        decode_start = time.time()
        self._simulate_decode(model, num_decode_steps, result)
        result.decode_time_ms = (time.time() - decode_start) * 1000
        
        result.total_time_ms = (time.time() - start_time) * 1000
        
        # Estimate full time
        result.estimated_full_time_ms = self._estimate_full_time(
            result, num_decode_steps
        )
        
        return result
    
    def _profile_model(self, model: Any) -> None:
        """Profile model layers."""
        # Get layers from model
        try:
            if hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'transformer'):
                layers = model.transformer.h if hasattr(model.transformer, 'h') else []
            else:
                layers = []
            
            # Analyze layer structure
            def get_signature(layer):
                return str(type(layer).__name__)
            
            if layers:
                self.layer_simplifier.analyze_layers(
                    list(layers), get_signature
                )
        except Exception:
            pass
    
    def _simulate_prefill(
        self,
        model: Any,
        input_data: Any,
        result: SimulationResult
    ) -> None:
        """Simulate prefill phase with optimizations."""
        if self.config.merge_repeated_layers:
            # Use simplified schedule
            schedule = self.layer_simplifier.get_simulation_schedule()
            
            for layer_idx, repeat_count in schedule:
                # Simulate representative layer
                layer_time = self._simulate_layer(layer_idx)
                
                # Scale for repeated layers
                total_time = self.layer_simplifier.estimate_time_scaling(
                    layer_time, repeat_count
                )
                
                result.layers_simulated += 1
                result.layers_skipped += repeat_count - 1
        else:
            # Simulate all layers
            pass
    
    def _simulate_decode(
        self,
        model: Any,
        num_steps: int,
        result: SimulationResult
    ) -> None:
        """Simulate decode phase with approximations."""
        if self.config.approximate_decode:
            # Get steps to simulate
            sim_steps = self.decode_approximator.get_simulation_steps(num_steps)
            result.decode_steps_simulated = len(sim_steps)
            result.decode_steps_interpolated = num_steps - len(sim_steps)
            
            # Simulate selected steps
            for step in sim_steps:
                step_time = self._simulate_decode_step(step)
                self.decode_approximator.record_timing(step, step_time)
        else:
            # Full simulation
            for step in range(num_steps):
                self._simulate_decode_step(step)
                result.decode_steps_simulated += 1
    
    def _simulate_layer(self, layer_idx: int) -> float:
        """Simulate a single layer, returning time in ms."""
        # Check cache
        cache_key = f"layer_{layer_idx}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Simulate (mock timing for now)
        time_ms = 1.0 + layer_idx * 0.1  # Mock timing
        
        # Cache result
        self.cache.put(cache_key, time_ms, 0.001)
        
        return time_ms
    
    def _simulate_decode_step(self, step: int) -> float:
        """Simulate a single decode step, returning time in ms."""
        # Check cache
        cache_key = f"decode_{step}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Simulate (mock timing)
        base_time = 5.0  # Base decode time
        # Early steps might be slightly different
        step_factor = 1.0 if step > 10 else 1.2
        time_ms = base_time * step_factor
        
        return time_ms
    
    def _estimate_full_time(
        self,
        result: SimulationResult,
        num_decode_steps: int
    ) -> float:
        """Estimate what full simulation would have taken."""
        full_decode_time = self.decode_approximator.estimate_total_time(
            num_decode_steps
        )
        
        # Account for skipped layers
        if result.layers_skipped > 0:
            layer_ratio = (result.layers_simulated + result.layers_skipped) / result.layers_simulated
            full_prefill_time = result.prefill_time_ms * layer_ratio
        else:
            full_prefill_time = result.prefill_time_ms
        
        return full_prefill_time + full_decode_time
    
    def reset(self) -> None:
        """Reset optimizer state."""
        self.cache.clear()
        self.decode_approximator.reset()
        self.layer_profiles.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "cache_size_mb": self.cache.total_size_mb,
            "cache_hit_rate": self.cache.hit_rate,
            "layer_groups": len(self.layer_simplifier.layer_groups),
            "profiled_layers": len(self.layer_profiles),
            "optimization_level": self.config.optimization_level.name,
        }


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("optimizer", "simulation")
    class SimulationOptimizerPlugin:
        """Simulation Optimizer Plugin."""
        
        def __init__(self, **kwargs):
            config = SimulationConfig(**kwargs) if kwargs else None
            self.optimizer = SimulationOptimizer(config)
        
        def optimize(self, model, input_data, num_steps=100) -> SimulationResult:
            return self.optimizer.optimize_simulation(model, input_data, num_steps)
        
        def get_stats(self) -> Dict[str, Any]:
            return self.optimizer.get_statistics()

except ImportError:
    pass
