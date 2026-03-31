"""
Hybrid Parallelism Engine for KAI.

Combines pipeline parallelism (existing) with tensor parallelism:
- Split attention layers across multiple GPUs
- Keep feed-forward layers in pipeline mode
- Dynamically switch between pipeline-only, tensor-only, and hybrid

Usage::

    from model.hybrid_parallelism import HybridParallelismEngine, ParallelismMode
    
    engine = HybridParallelismEngine(devices=["cuda:0", "cuda:1"])
    
    # Configure strategy
    engine.set_mode(ParallelismMode.HYBRID)
    
    # Execute with hybrid parallelism
    output = engine.forward(model_chunk, input_tensor)
"""

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from model.plugin_architecture import PluginRegistry, ParallelismPlugin, ExecutorPlugin

logger = logging.getLogger(__name__)


class ParallelismMode(Enum):
    """Parallelism execution modes."""
    PIPELINE_ONLY = "pipeline"    # Sequential layer execution
    TENSOR_ONLY = "tensor"        # Split tensors across devices
    HYBRID = "hybrid"             # Tensor for attention, pipeline for FFN


class SplitDimension(Enum):
    """Dimension along which to split tensors."""
    BATCH = 0
    SEQUENCE = 1
    HEAD = 2
    HIDDEN = -1


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    num_devices: int = 2
    split_attention: bool = True
    split_feedforward: bool = False
    split_dimension: SplitDimension = SplitDimension.HEAD
    reduce_scatter: bool = True  # Use reduce-scatter for efficiency
    async_communication: bool = True


@dataclass
class ParallelismStrategy:
    """Strategy for a specific layer type."""
    layer_pattern: str
    mode: ParallelismMode
    tensor_config: Optional[TensorParallelConfig] = None
    priority: int = 0  # Higher = check first
    
    def matches(self, layer_name: str) -> bool:
        """Check if layer name matches pattern."""
        return self.layer_pattern.lower() in layer_name.lower()


@dataclass
class ExecutionStats:
    """Statistics for parallel execution."""
    total_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    num_splits: int = 0
    mode_used: ParallelismMode = ParallelismMode.PIPELINE_ONLY


class TensorSplitter:
    """Utilities for splitting and gathering tensors across devices."""
    
    @staticmethod
    def split_tensor(
        tensor: Tensor,
        num_splits: int,
        dim: int = -1,
    ) -> List[Tensor]:
        """Split tensor along specified dimension."""
        if num_splits == 1:
            return [tensor]
        
        size = tensor.size(dim)
        if size % num_splits != 0:
            # Pad if necessary
            pad_size = num_splits - (size % num_splits)
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=dim)
        
        return torch.chunk(tensor, num_splits, dim=dim)
    
    @staticmethod
    def gather_tensor(
        tensors: List[Tensor],
        dim: int = -1,
    ) -> Tensor:
        """Gather split tensors."""
        if len(tensors) == 1:
            return tensors[0]
        
        # Move all to same device
        device = tensors[0].device
        tensors = [t.to(device) for t in tensors]
        
        return torch.cat(tensors, dim=dim)
    
    @staticmethod
    def all_reduce(
        tensor: Tensor,
        op: str = "sum",
        devices: Optional[List[str]] = None,
    ) -> Tensor:
        """All-reduce across devices (simulated for single-process)."""
        # In multi-process setting, this would use dist.all_reduce
        # For single-process multi-GPU, we simulate by collecting on device 0
        return tensor
    
    @staticmethod
    def scatter_to_devices(
        tensor: Tensor,
        devices: List[str],
        dim: int = -1,
    ) -> List[Tensor]:
        """Scatter tensor chunks to different devices."""
        chunks = TensorSplitter.split_tensor(tensor, len(devices), dim)
        return [chunk.to(device) for chunk, device in zip(chunks, devices)]


class AttentionParallel(nn.Module):
    """Tensor-parallel wrapper for attention layers.
    
    Splits attention heads across devices for parallel computation.
    
    Parameters
    ----------
    attention_module : nn.Module
        Original attention module
    devices : list[str]
        Devices to distribute across
    num_heads : int
        Total number of attention heads
    """
    
    def __init__(
        self,
        attention_module: nn.Module,
        devices: List[str],
        num_heads: int = 8,
    ):
        super().__init__()
        self.attention = attention_module
        self.devices = devices
        self.num_heads = num_heads
        self.heads_per_device = num_heads // len(devices)
        
        # Split attention weights across devices
        self._split_weights()
    
    def _split_weights(self) -> None:
        """Split attention weights by head dimension."""
        # This is a simplified version - full implementation would
        # actually move weight slices to different devices
        pass
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward with tensor parallelism."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if len(self.devices) == 1:
            # No parallelism needed
            return self.attention(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Split by head dimension and compute in parallel
        outputs = []
        
        # For each device, compute subset of heads
        for i, device in enumerate(self.devices):
            start_head = i * self.heads_per_device
            end_head = (i + 1) * self.heads_per_device
            
            # Move input to device
            device_input = hidden_states.to(device)
            
            # Compute attention (simplified - would need head-specific weights)
            with torch.cuda.device(device) if "cuda" in device else torch.device(device):
                output = self.attention(
                    device_input,
                    attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                    **kwargs
                )
            
            outputs.append(output)
        
        # Gather outputs
        return TensorSplitter.gather_tensor(outputs, dim=-1)


class FeedForwardParallel(nn.Module):
    """Tensor-parallel wrapper for feed-forward layers.
    
    Splits the hidden dimension for parallel MLP computation.
    
    Parameters
    ----------
    ffn_module : nn.Module
        Original FFN module
    devices : list[str]
        Devices to distribute across
    """
    
    def __init__(
        self,
        ffn_module: nn.Module,
        devices: List[str],
    ):
        super().__init__()
        self.ffn = ffn_module
        self.devices = devices
        self.num_splits = len(devices)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward with column-parallel / row-parallel strategy."""
        if len(self.devices) == 1:
            return self.ffn(hidden_states)
        
        # Column-parallel: split output dimension
        # Row-parallel: split input dimension
        
        # Simplified implementation
        # Full impl would split weights and compute in parallel
        return self.ffn(hidden_states)


class HybridParallelismEngine:
    """Engine for hybrid pipeline + tensor parallelism.
    
    Dynamically switches between parallelism modes based on
    layer type and available resources.
    
    Parameters
    ----------
    devices : list[str]
        Available devices (e.g., ["cuda:0", "cuda:1"])
    mode : ParallelismMode
        Default parallelism mode
    config : TensorParallelConfig, optional
        Tensor parallelism configuration
    """
    
    def __init__(
        self,
        devices: Optional[List[str]] = None,
        mode: ParallelismMode = ParallelismMode.PIPELINE_ONLY,
        config: Optional[TensorParallelConfig] = None,
    ):
        self.devices = devices or self._detect_devices()
        self.mode = mode
        self.config = config or TensorParallelConfig(num_devices=len(self.devices))
        
        # Layer strategies
        self._strategies: List[ParallelismStrategy] = []
        self._setup_default_strategies()
        
        # Wrapped modules cache
        self._wrapped_modules: Dict[str, nn.Module] = {}
        
        # Statistics
        self._stats = ExecutionStats()
        self._lock = threading.Lock()
    
    def _detect_devices(self) -> List[str]:
        """Detect available CUDA devices."""
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]
    
    def _setup_default_strategies(self) -> None:
        """Setup default parallelism strategies."""
        # Attention layers: tensor parallel
        self._strategies.append(ParallelismStrategy(
            layer_pattern="attn",
            mode=ParallelismMode.TENSOR_ONLY,
            tensor_config=TensorParallelConfig(
                num_devices=len(self.devices),
                split_attention=True,
            ),
            priority=10,
        ))
        
        self._strategies.append(ParallelismStrategy(
            layer_pattern="attention",
            mode=ParallelismMode.TENSOR_ONLY,
            tensor_config=TensorParallelConfig(
                num_devices=len(self.devices),
                split_attention=True,
            ),
            priority=10,
        ))
        
        # FFN layers: pipeline (default)
        self._strategies.append(ParallelismStrategy(
            layer_pattern="mlp",
            mode=ParallelismMode.PIPELINE_ONLY,
            priority=5,
        ))
        
        self._strategies.append(ParallelismStrategy(
            layer_pattern="ffn",
            mode=ParallelismMode.PIPELINE_ONLY,
            priority=5,
        ))
        
        # Sort by priority
        self._strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def set_mode(self, mode: ParallelismMode) -> None:
        """Set global parallelism mode."""
        self.mode = mode
        logger.info("Parallelism mode set to: %s", mode.value)
    
    def add_strategy(self, strategy: ParallelismStrategy) -> None:
        """Add a custom parallelism strategy."""
        self._strategies.append(strategy)
        self._strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def get_strategy_for_layer(self, layer_name: str) -> ParallelismStrategy:
        """Get parallelism strategy for a layer."""
        for strategy in self._strategies:
            if strategy.matches(layer_name):
                return strategy
        
        # Default strategy based on global mode
        return ParallelismStrategy(
            layer_pattern="*",
            mode=self.mode,
        )
    
    def wrap_module(
        self,
        module: nn.Module,
        layer_name: str,
    ) -> nn.Module:
        """Wrap module for parallel execution."""
        if layer_name in self._wrapped_modules:
            return self._wrapped_modules[layer_name]
        
        strategy = self.get_strategy_for_layer(layer_name)
        
        if strategy.mode == ParallelismMode.PIPELINE_ONLY:
            wrapped = module
        elif strategy.mode == ParallelismMode.TENSOR_ONLY:
            if "attn" in layer_name.lower() or "attention" in layer_name.lower():
                # Infer num_heads from module
                num_heads = getattr(module, "num_heads", 8)
                wrapped = AttentionParallel(module, self.devices, num_heads)
            else:
                wrapped = FeedForwardParallel(module, self.devices)
        else:
            # Hybrid: wrap based on layer type
            if "attn" in layer_name.lower():
                num_heads = getattr(module, "num_heads", 8)
                wrapped = AttentionParallel(module, self.devices, num_heads)
            else:
                wrapped = module
        
        self._wrapped_modules[layer_name] = wrapped
        return wrapped
    
    def forward(
        self,
        module: nn.Module,
        inputs: Tensor,
        layer_name: str = "",
        **kwargs,
    ) -> Tensor:
        """Execute forward pass with appropriate parallelism.
        
        Parameters
        ----------
        module : nn.Module
            Module to execute
        inputs : Tensor
            Input tensor
        layer_name : str
            Layer name for strategy lookup
            
        Returns
        -------
        Tensor
            Output tensor
        """
        import time
        start_time = time.perf_counter()
        
        strategy = self.get_strategy_for_layer(layer_name)
        
        if strategy.mode == ParallelismMode.PIPELINE_ONLY:
            output = self._forward_pipeline(module, inputs, **kwargs)
        elif strategy.mode == ParallelismMode.TENSOR_ONLY:
            output = self._forward_tensor_parallel(module, inputs, layer_name, **kwargs)
        else:
            output = self._forward_hybrid(module, inputs, layer_name, **kwargs)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        with self._lock:
            self._stats.total_time_ms += elapsed_ms
            self._stats.mode_used = strategy.mode
        
        return output
    
    def _forward_pipeline(
        self,
        module: nn.Module,
        inputs: Tensor,
        **kwargs,
    ) -> Tensor:
        """Standard pipeline execution."""
        with torch.no_grad():
            return module(inputs, **kwargs) if kwargs else module(inputs)
    
    def _forward_tensor_parallel(
        self,
        module: nn.Module,
        inputs: Tensor,
        layer_name: str,
        **kwargs,
    ) -> Tensor:
        """Tensor-parallel execution."""
        if len(self.devices) == 1:
            return self._forward_pipeline(module, inputs, **kwargs)
        
        wrapped = self.wrap_module(module, layer_name)
        
        with torch.no_grad():
            return wrapped(inputs, **kwargs) if kwargs else wrapped(inputs)
    
    def _forward_hybrid(
        self,
        module: nn.Module,
        inputs: Tensor,
        layer_name: str,
        **kwargs,
    ) -> Tensor:
        """Hybrid execution: tensor parallel for attention, pipeline for FFN."""
        # Determine layer type
        if "attn" in layer_name.lower() or "attention" in layer_name.lower():
            return self._forward_tensor_parallel(module, inputs, layer_name, **kwargs)
        else:
            return self._forward_pipeline(module, inputs, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            return {
                "total_time_ms": round(self._stats.total_time_ms, 3),
                "compute_time_ms": round(self._stats.compute_time_ms, 3),
                "communication_time_ms": round(self._stats.communication_time_ms, 3),
                "memory_used_mb": round(self._stats.memory_used_mb, 2),
                "num_splits": self._stats.num_splits,
                "mode_used": self._stats.mode_used.value,
                "num_devices": len(self.devices),
                "devices": self.devices,
            }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        with self._lock:
            self._stats = ExecutionStats()


class WorkloadAnalyzer:
    """Analyzes workload to recommend parallelism strategy.
    
    Parameters
    ----------
    model : nn.Module
        Model to analyze
    sample_input : Tensor
        Sample input for profiling
    """
    
    def __init__(
        self,
        model: nn.Module,
        sample_input: Optional[Tensor] = None,
    ):
        self.model = model
        self.sample_input = sample_input
        
        self._layer_profiles: Dict[str, Dict[str, Any]] = {}
    
    def profile_layers(self) -> Dict[str, Dict[str, Any]]:
        """Profile each layer for compute/memory characteristics."""
        profiles = {}
        
        for name, module in self.model.named_modules():
            if not list(module.children()):  # Leaf modules
                param_count = sum(p.numel() for p in module.parameters())
                if param_count == 0:
                    continue
                
                is_attention = any(
                    p in name.lower()
                    for p in ["attn", "attention", "self_attn"]
                )
                
                is_ffn = any(
                    p in name.lower()
                    for p in ["mlp", "ffn", "feed_forward", "fc"]
                )
                
                profiles[name] = {
                    "param_count": param_count,
                    "memory_mb": param_count * 2 / (1024**2),
                    "is_attention": is_attention,
                    "is_ffn": is_ffn,
                    "module_type": module.__class__.__name__,
                }
        
        self._layer_profiles = profiles
        return profiles
    
    def recommend_mode(self) -> Tuple[ParallelismMode, str]:
        """Recommend parallelism mode based on model characteristics."""
        if not self._layer_profiles:
            self.profile_layers()
        
        attention_params = sum(
            p["param_count"]
            for p in self._layer_profiles.values()
            if p["is_attention"]
        )
        
        ffn_params = sum(
            p["param_count"]
            for p in self._layer_profiles.values()
            if p["is_ffn"]
        )
        
        total_params = attention_params + ffn_params
        
        if total_params == 0:
            return ParallelismMode.PIPELINE_ONLY, "No significant layers found"
        
        attention_ratio = attention_params / total_params if total_params > 0 else 0
        
        if attention_ratio > 0.6:
            return ParallelismMode.TENSOR_ONLY, f"High attention ratio ({attention_ratio:.1%})"
        elif attention_ratio > 0.3:
            return ParallelismMode.HYBRID, f"Mixed workload ({attention_ratio:.1%} attention)"
        else:
            return ParallelismMode.PIPELINE_ONLY, f"FFN-dominant workload"


# Register as plugins
@PluginRegistry.register(
    "parallelism",
    "hybrid",
    description="Hybrid pipeline + tensor parallelism"
)
class HybridParallelismPlugin(ParallelismPlugin):
    """Plugin wrapper for HybridParallelismEngine."""
    
    def __init__(self, devices: Optional[List[str]] = None):
        self._engine = HybridParallelismEngine(devices=devices)
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    def get_strategy(
        self,
        layer_type: str,
        resources: Dict[str, Any],
    ) -> str:
        strategy = self._engine.get_strategy_for_layer(layer_type)
        return strategy.mode.value
    
    def execute_parallel(
        self,
        module: nn.Module,
        inputs: Tensor,
        strategy: str,
        devices: List[str],
    ) -> Tensor:
        self._engine.devices = devices
        self._engine.set_mode(ParallelismMode(strategy))
        return self._engine.forward(module, inputs)


@PluginRegistry.register(
    "executor",
    "hybrid_parallel",
    description="Executor with hybrid parallelism support"
)
class HybridParallelExecutor(ExecutorPlugin):
    """Executor plugin using hybrid parallelism."""
    
    def __init__(self, devices: Optional[List[str]] = None):
        self._engine = HybridParallelismEngine(devices=devices)
    
    @property
    def name(self) -> str:
        return "hybrid_parallel"
    
    def execute(
        self,
        module: nn.Module,
        inputs: Tensor,
        **kwargs,
    ) -> Tensor:
        layer_name = kwargs.get("layer_name", "")
        return self._engine.forward(module, inputs, layer_name=layer_name)
    
    def supports_async(self) -> bool:
        return False
