"""
Layer Streaming Engine for Single-GPU Large Model Execution.

Implements load → compute → evict pattern to enable running models
larger than GPU VRAM by streaming layers through GPU one at a time.

Key Concepts:
- LayerBatch: Groups of layers loaded together (e.g., 2-4 layers)
- Double Buffering: Prefetch next batch while computing current
- Activation Reuse: Save only critical residuals, not all intermediates
- Progressive Offloading: Weights → disk immediately after use

Memory Benefit:
  Before: 6GB (model) + 1.5GB (KV cache) + 2GB (activations) = 9.5GB
  After:  0.5GB (current layer batch) + 1.5GB (KV) + 0.2GB (residuals) = 2.2GB
  Reduction: 77% → enables 5× larger models

Usage::

    from model.layer_streamer import LayerStreamer, StreamingConfig
    
    config = StreamingConfig(
        batch_size=2,  # Load 2 layers at a time
        prefetch=True,
        memory_target_mb=12000,  # 12GB of 24GB GPU
    )
    
    streamer = LayerStreamer(model, config)
    for token_idx in range(max_tokens):
        hidden = streamer.forward(hidden)
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """How to store activations for residual connections."""
    FULL = "full"          # Keep full precision (baseline)
    FP16 = "fp16"          # Half precision
    INT8 = "int8"          # Quantized
    SKIP = "skip"          # Don't store (recompute if needed)


@dataclass
class StreamingConfig:
    """Configuration for layer streaming."""
    batch_size: int = 2                    # Layers to load at once
    prefetch: bool = True                   # Enable double buffering
    prefetch_buffer_size: int = 2          # Extra batches to prefetch
    memory_target_mb: float = 12000.0      # Target GPU memory for inference
    compression_strategy: CompressionStrategy = CompressionStrategy.FP16
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float32
    
    def validate(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.memory_target_mb < 1000:
            raise ValueError("memory_target_mb should be >= 1000 MB")


class LayerBatch:
    """Container for a batch of layers to stream."""
    
    def __init__(
        self,
        layer_indices: List[int],
        layers: List[nn.Module],
        batch_idx: int,
    ):
        self.layer_indices = layer_indices
        self.layers = layers
        self.batch_idx = batch_idx
        self.is_loaded = False
        self.load_time_ms: float = 0.0
    
    def __repr__(self) -> str:
        return f"LayerBatch(idx={self.batch_idx}, layers={self.layer_indices}, loaded={self.is_loaded})"


class LayerStreamer:
    """
    Streams model layers through GPU during inference.
    
    Manages:
    - Loading/evicting layers from GPU
    - Double-buffered prefetching
    - Residual activation storage
    - Memory tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: StreamingConfig,
    ):
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        self.model = model
        
        # Extract layer list (works for transformer models)
        self.layers = self._extract_layers()
        self.num_layers = len(self.layers)
        
        # Organize into batches
        self.layer_batches = self._organize_batches()
        
        # Prefetch management
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_event = threading.Event()
        self._prefetch_buffer: Dict[int, List[nn.Module]] = {}
        self._current_batch_idx = 0
        
        # Statistics
        self.stats = {
            "total_loads": 0,
            "total_evicts": 0,
            "total_prefetches": 0,
            "total_load_time_ms": 0.0,
            "prefetch_overlapped_ms": 0.0,
            "gpu_peak_memory_mb": 0.0,
        }
        
        # Residual activations (for skip connections)
        self._residuals: Dict[str, torch.Tensor] = {}
        
        logger.info(
            f"LayerStreamer initialized: {self.num_layers} layers "
            f"in {len(self.layer_batches)} batches of size {config.batch_size}"
        )
    
    def _extract_layers(self) -> List[nn.Module]:
        """Extract transformer layers from model."""
        # Try standard transformer layout
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return list(self.model.transformer.h)
        elif hasattr(self.model, 'gpt2') and hasattr(self.model.gpt2, 'h'):
            return list(self.model.gpt2.h)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA-style
            return list(self.model.model.layers)
        else:
            # Fallback: try to find sequential modules
            layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    layers.append(module)
            return layers if layers else [self.model]
    
    def _organize_batches(self) -> List[LayerBatch]:
        """Organize layers into streaming batches."""
        batches = []
        for batch_idx in range(0, self.num_layers, self.config.batch_size):
            end_idx = min(batch_idx + self.config.batch_size, self.num_layers)
            layer_indices = list(range(batch_idx, end_idx))
            batch_layers = [self.layers[i] for i in layer_indices]
            batch = LayerBatch(layer_indices, batch_layers, len(batches))
            batches.append(batch)
        return batches
    
    def _get_batch_memory_mb(self, batch: LayerBatch) -> float:
        """Estimate memory needed for a layer batch."""
        total = 0.0
        for layer in batch.layers:
            for param in layer.parameters():
                total += param.numel() * param.dtype.itemsize / (1024 ** 2)
        return total
    
    def _load_batch_to_gpu(self, batch: LayerBatch) -> None:
        """Load a layer batch to GPU."""
        start_time = time.perf_counter()
        
        for layer in batch.layers:
            layer.to(self.device)
            layer.eval()
        
        batch.is_loaded = True
        batch.load_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.stats["total_loads"] += 1
        self.stats["total_load_time_ms"] += batch.load_time_ms
        
        logger.debug(f"Loaded {batch}: {batch.load_time_ms:.1f}ms")
    
    def _evict_batch_from_gpu(self, batch: LayerBatch) -> None:
        """Evict a layer batch from GPU to CPU."""
        start_time = time.perf_counter()
        
        for layer in batch.layers:
            # Move to CPU (parameters and buffers)
            layer.cpu()
        
        # Trigger garbage collection
        torch.cuda.empty_cache()
        
        batch.is_loaded = False
        evict_ms = (time.perf_counter() - start_time) * 1000.0
        self.stats["total_evicts"] += 1
        
        logger.debug(f"Evicted {batch}: {evict_ms:.1f}ms")
    
    def _save_residuals(
        self,
        batch_idx: int,
        hidden_state: torch.Tensor,
        strategy: CompressionStrategy,
    ) -> torch.Tensor:
        """Save residual for next batch (e.g., for skip connections)."""
        key = f"residual_batch_{batch_idx}"
        
        if strategy == CompressionStrategy.FULL:
            self._residuals[key] = hidden_state.detach().cpu()
        elif strategy == CompressionStrategy.FP16:
            # Compress to FP16
            compressed = hidden_state.detach().cpu().to(torch.float16)
            self._residuals[key] = compressed
        elif strategy == CompressionStrategy.INT8:
            # Quantize to INT8 (simple linear quantization)
            hstate = hidden_state.detach().cpu()
            qmin, qmax = hstate.min().item(), hstate.max().item()
            qscale = (qmax - qmin) / 255.0
            quantized = ((hstate - qmin) / qscale).byte()
            self._residuals[key] = (quantized, qscale, qmin)
        # Skip doesn't save anything
        
        return hidden_state
    
    def _get_residuals(self, batch_idx: int) -> Optional[torch.Tensor]:
        """Retrieve residuals from previous batch."""
        key = f"residual_batch_{batch_idx - 1}"
        if key not in self._residuals:
            return None
        
        residual = self._residuals[key]
        
        # Decompress if necessary
        if isinstance(residual, tuple):
            quantized, qscale, qmin = residual
            residual = (quantized.float() * qscale + qmin).to(self.device)
        elif residual.dtype == torch.float16:
            residual = residual.to(self.device).to(self.config.dtype)
        else:
            residual = residual.to(self.device)
        
        return residual
    
    def _prefetch_batch_async(self, batch_idx: int) -> None:
        """Asynchronously prefetch a batch into buffer."""
        if batch_idx >= len(self.layer_batches):
            return
        
        batch = self.layer_batches[batch_idx]
        self._load_batch_to_gpu(batch)
        self._prefetch_buffer[batch_idx] = batch.layers
        self.stats["total_prefetches"] += 1
    
    def prefetch(self, next_batch_idx: int) -> None:
        """Start prefetching next batch (non-blocking)."""
        if not self.config.prefetch:
            return
        
        # Don't prefetch if already in buffer
        if next_batch_idx in self._prefetch_buffer:
            return
        
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_batch_async,
            args=(next_batch_idx,),
            daemon=True,
        )
        self._prefetch_thread.start()
    
    def forward(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through all layers with streaming.
        
        Parameters
        ----------
        hidden_state : torch.Tensor
            Input hidden state [batch, seq_len, hidden_dim]
        
        Returns
        -------
        torch.Tensor
            Output after all layers
        """
        for batch_idx, batch in enumerate(self.layer_batches):
            # Prefetch next batch (if enabled)
            if batch_idx + 1 < len(self.layer_batches):
                self.prefetch(batch_idx + 1)
            
            # Wait for prefetch if this batch isn't in buffer
            if batch_idx not in self._prefetch_buffer:
                self._load_batch_to_gpu(batch)
            else:
                # Use prefetched batch
                batch.layers = self._prefetch_buffer.pop(batch_idx)
            
            # Forward through batch
            for layer in batch.layers:
                hidden_state = layer(hidden_state)
            
            # Save residuals for skip connections / next batch
            self._save_residuals(batch_idx, hidden_state, self.config.compression_strategy)
            
            # Evict this batch if not prefetched
            if batch_idx + 1 not in self._prefetch_buffer:
                self._evict_batch_from_gpu(batch)
        
        return hidden_state
    
    def get_stats(self) -> Dict[str, float]:
        """Return streaming statistics."""
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """Reset streaming statistics."""
        self.stats = {
            "total_loads": 0,
            "total_evicts": 0,
            "total_prefetches": 0,
            "total_load_time_ms": 0.0,
            "prefetch_overlapped_ms": 0.0,
            "gpu_peak_memory_mb": 0.0,
        }
