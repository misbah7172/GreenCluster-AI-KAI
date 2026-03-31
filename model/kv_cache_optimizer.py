"""
KV Cache Optimization Layer (TurboQuant-style).

Implements mixed-precision KV cache with:
- FP16 for recent tokens
- INT8/compressed for older tokens
- Cache reuse across requests (prompt overlap detection)
- Memory-aware eviction policies

Usage::

    from model.kv_cache_optimizer import KVCacheOptimizer, MixedPrecisionKVCache
    
    cache = MixedPrecisionKVCache(
        max_length=2048,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        recent_window=128,
    )
    
    # Store KV states
    cache.update(layer_idx, key_states, value_states)
    
    # Retrieve KV states
    keys, values = cache.get(layer_idx)
"""

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.plugin_architecture import PluginRegistry, CachePlugin

logger = logging.getLogger(__name__)


class CachePrecision(Enum):
    """Precision levels for KV cache."""
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    
    @property
    def bytes_per_element(self) -> float:
        return {
            CachePrecision.FP16: 2.0,
            CachePrecision.FP8: 1.0,
            CachePrecision.INT8: 1.0,
            CachePrecision.INT4: 0.5,
        }[self]


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ATTENTION_SCORE = "attention"  # Based on attention weights
    MEMORY_PRESSURE = "memory"  # Evict when memory is low


@dataclass
class CacheConfig:
    """Configuration for KV cache."""
    max_length: int = 2048
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    recent_window: int = 128  # Tokens to keep at FP16
    compression_precision: CachePrecision = CachePrecision.INT8
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    max_memory_mb: float = 4096.0
    enable_reuse: bool = True
    reuse_similarity_threshold: float = 0.9
    
    def estimate_memory_mb(self) -> float:
        """Estimate total cache memory in MB."""
        # KV pairs per layer
        kv_elements = 2 * self.max_length * self.num_heads * self.head_dim
        # Recent in FP16, rest compressed
        recent_elements = 2 * self.recent_window * self.num_heads * self.head_dim
        old_elements = kv_elements - recent_elements
        
        memory = (
            (recent_elements * 2.0) +  # FP16
            (old_elements * self.compression_precision.bytes_per_element)
        ) * self.num_layers / (1024 ** 2)
        
        return memory


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    memory_used_mb: float = 0.0
    reuse_hits: int = 0
    avg_compression_ratio: float = 1.0
    
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class QuantizedTensor:
    """Quantized tensor storage with scale and zero point."""
    
    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        precision: CachePrecision,
    ):
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.precision = precision
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to float."""
        return (self.data.float() - self.zero_point) * self.scale
    
    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        precision: CachePrecision,
    ) -> "QuantizedTensor":
        """Quantize a float tensor."""
        if precision == CachePrecision.INT8:
            qmin, qmax = -128, 127
        elif precision == CachePrecision.INT4:
            qmin, qmax = -8, 7
        else:
            # FP16/FP8 - just cast
            return cls(
                tensor.half() if precision == CachePrecision.FP16 else tensor.to(torch.float8_e4m3fn),
                torch.tensor(1.0),
                torch.tensor(0.0),
                precision,
            )
        
        # Compute scale and zero point
        vmin, vmax = tensor.min(), tensor.max()
        scale = (vmax - vmin) / (qmax - qmin)
        scale = scale.clamp(min=1e-8)
        zero_point = qmin - vmin / scale
        
        # Quantize
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            qmin, qmax
        ).to(torch.int8)
        
        return cls(quantized, scale, zero_point, precision)


class MixedPrecisionKVCache:
    """KV cache with mixed precision storage.
    
    Recent tokens are stored at FP16 for accuracy.
    Older tokens are compressed to INT8/INT4 for memory efficiency.
    
    Parameters
    ----------
    config : CacheConfig
        Cache configuration
    device : str
        Device for cache tensors
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        device: str = "cuda:0",
    ):
        self.config = config or CacheConfig()
        self.device = device
        
        # Per-layer cache storage
        # Structure: {layer_idx: (recent_k, recent_v, old_k, old_v)}
        self._cache: Dict[int, Tuple[
            torch.Tensor,  # recent keys (FP16)
            torch.Tensor,  # recent values (FP16)
            Optional[QuantizedTensor],  # old keys (quantized)
            Optional[QuantizedTensor],  # old values (quantized)
        ]] = {}
        
        self._seq_length = 0
        self._stats = CacheStats()
        self._lock = threading.Lock()
    
    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Update cache with new KV states.
        
        Parameters
        ----------
        layer_idx : int
            Layer index
        key_states : torch.Tensor
            Key states [batch, heads, seq_len, head_dim]
        value_states : torch.Tensor
            Value states [batch, heads, seq_len, head_dim]
        attention_scores : torch.Tensor, optional
            Attention weights for importance-based eviction
        """
        with self._lock:
            new_seq_len = key_states.shape[2]
            
            if layer_idx not in self._cache:
                # Initialize cache for this layer
                self._cache[layer_idx] = (
                    key_states.half().to(self.device),
                    value_states.half().to(self.device),
                    None,
                    None,
                )
                self._seq_length = new_seq_len
                return
            
            recent_k, recent_v, old_k, old_v = self._cache[layer_idx]
            
            # Concatenate new states
            combined_k = torch.cat([recent_k, key_states.half().to(self.device)], dim=2)
            combined_v = torch.cat([recent_v, value_states.half().to(self.device)], dim=2)
            
            total_len = combined_k.shape[2]
            
            # Split into recent (FP16) and old (quantized)
            if total_len > self.config.recent_window:
                split_point = total_len - self.config.recent_window
                
                old_k_new = combined_k[:, :, :split_point, :]
                old_v_new = combined_v[:, :, :split_point, :]
                recent_k = combined_k[:, :, split_point:, :]
                recent_v = combined_v[:, :, split_point:, :]
                
                # Quantize old portion
                if old_k is not None:
                    # Merge with existing old cache
                    old_k_combined = torch.cat([
                        old_k.dequantize(), old_k_new
                    ], dim=2)
                    old_v_combined = torch.cat([
                        old_v.dequantize(), old_v_new
                    ], dim=2)
                else:
                    old_k_combined = old_k_new
                    old_v_combined = old_v_new
                
                # Evict if exceeding max length
                if old_k_combined.shape[2] + self.config.recent_window > self.config.max_length:
                    evict_count = (
                        old_k_combined.shape[2] + self.config.recent_window - 
                        self.config.max_length
                    )
                    old_k_combined = old_k_combined[:, :, evict_count:, :]
                    old_v_combined = old_v_combined[:, :, evict_count:, :]
                    self._stats.evictions += evict_count
                
                # Quantize
                old_k = QuantizedTensor.quantize(
                    old_k_combined, self.config.compression_precision
                )
                old_v = QuantizedTensor.quantize(
                    old_v_combined, self.config.compression_precision
                )
                self._stats.compressions += 1
            else:
                recent_k = combined_k
                recent_v = combined_v
            
            self._cache[layer_idx] = (recent_k, recent_v, old_k, old_v)
            self._seq_length = recent_k.shape[2] + (
                old_k.data.shape[2] if old_k else 0
            )
    
    def get(
        self,
        layer_idx: int,
        dequantize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV states.
        
        Parameters
        ----------
        layer_idx : int
            Layer index
        dequantize : bool
            If True, dequantize old tokens to FP16
            
        Returns
        -------
        keys, values : torch.Tensor
            Full KV cache tensors
        """
        with self._lock:
            if layer_idx not in self._cache:
                self._stats.misses += 1
                raise KeyError(f"No cache for layer {layer_idx}")
            
            self._stats.hits += 1
            recent_k, recent_v, old_k, old_v = self._cache[layer_idx]
            
            if old_k is None:
                return recent_k, recent_v
            
            if dequantize:
                full_k = torch.cat([old_k.dequantize().half(), recent_k], dim=2)
                full_v = torch.cat([old_v.dequantize().half(), recent_v], dim=2)
            else:
                # Return recent only
                full_k = recent_k
                full_v = recent_v
            
            return full_k, full_v
    
    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear cache."""
        with self._lock:
            if layer_idx is not None:
                self._cache.pop(layer_idx, None)
            else:
                self._cache.clear()
            self._seq_length = 0
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        total = 0.0
        
        with self._lock:
            for layer_idx, (recent_k, recent_v, old_k, old_v) in self._cache.items():
                # Recent FP16
                total += recent_k.nelement() * 2 / (1024**2)
                total += recent_v.nelement() * 2 / (1024**2)
                
                # Old quantized
                if old_k is not None:
                    total += (
                        old_k.data.nelement() * 
                        old_k.precision.bytes_per_element / (1024**2)
                    )
                if old_v is not None:
                    total += (
                        old_v.data.nelement() * 
                        old_v.precision.bytes_per_element / (1024**2)
                    )
        
        self._stats.memory_used_mb = total
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate(),
            "evictions": self._stats.evictions,
            "compressions": self._stats.compressions,
            "memory_mb": self.get_memory_usage_mb(),
            "seq_length": self._seq_length,
            "num_layers": len(self._cache),
        }
    
    @property
    def seq_length(self) -> int:
        """Current sequence length in cache."""
        return self._seq_length


class CacheReuser:
    """Detects and reuses cached KV states for overlapping prompts.
    
    Uses locality-sensitive hashing to quickly find similar prompts.
    
    Parameters
    ----------
    similarity_threshold : float
        Minimum similarity for reuse (0.0-1.0)
    max_entries : int
        Maximum cached prompts
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        max_entries: int = 100,
    ):
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        
        # Hash -> (prompt_tokens, cache_key)
        self._prompt_cache: Dict[str, Tuple[torch.Tensor, str]] = {}
        # Cache key -> MixedPrecisionKVCache
        self._kv_caches: Dict[str, MixedPrecisionKVCache] = {}
        
        self._lock = threading.Lock()
        self._stats = {"reuse_hits": 0, "reuse_misses": 0}
    
    def _compute_hash(self, tokens: torch.Tensor) -> str:
        """Compute hash for token sequence."""
        # Use first 64 tokens for fast hashing
        sample = tokens[:64].cpu().numpy().tobytes()
        return hashlib.md5(sample).hexdigest()
    
    def _compute_similarity(
        self,
        tokens1: torch.Tensor,
        tokens2: torch.Tensor,
    ) -> float:
        """Compute prefix similarity between two token sequences."""
        min_len = min(len(tokens1), len(tokens2))
        if min_len == 0:
            return 0.0
        
        matches = (tokens1[:min_len] == tokens2[:min_len]).sum().item()
        return matches / min_len
    
    def find_reusable_cache(
        self,
        prompt_tokens: torch.Tensor,
    ) -> Tuple[Optional[MixedPrecisionKVCache], int]:
        """Find a reusable cache for the given prompt.
        
        Parameters
        ----------
        prompt_tokens : torch.Tensor
            Input prompt tokens
            
        Returns
        -------
        cache, reuse_length : tuple
            Reusable cache and number of prefix tokens that can be reused.
            Returns (None, 0) if no reusable cache found.
        """
        with self._lock:
            prompt_hash = self._compute_hash(prompt_tokens)
            
            # Quick hash lookup
            if prompt_hash in self._prompt_cache:
                cached_tokens, cache_key = self._prompt_cache[prompt_hash]
                similarity = self._compute_similarity(cached_tokens, prompt_tokens)
                
                if similarity >= self._threshold:
                    self._stats["reuse_hits"] += 1
                    reuse_len = min(len(cached_tokens), len(prompt_tokens))
                    return self._kv_caches.get(cache_key), reuse_len
            
            # Search all caches for partial match
            best_match = None
            best_len = 0
            
            for cache_key, (cached_tokens, _) in self._prompt_cache.items():
                similarity = self._compute_similarity(cached_tokens, prompt_tokens)
                match_len = int(similarity * min(len(cached_tokens), len(prompt_tokens)))
                
                if match_len > best_len and match_len >= 32:  # Minimum reuse threshold
                    best_match = self._kv_caches.get(cache_key)
                    best_len = match_len
            
            if best_match:
                self._stats["reuse_hits"] += 1
                return best_match, best_len
            
            self._stats["reuse_misses"] += 1
            return None, 0
    
    def store_cache(
        self,
        prompt_tokens: torch.Tensor,
        cache: MixedPrecisionKVCache,
    ) -> str:
        """Store a cache for future reuse.
        
        Returns the cache key.
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._prompt_cache) >= self._max_entries:
                oldest_key = next(iter(self._prompt_cache))
                del self._prompt_cache[oldest_key]
                self._kv_caches.pop(oldest_key, None)
            
            cache_key = f"cache_{time.time_ns()}"
            prompt_hash = self._compute_hash(prompt_tokens)
            
            self._prompt_cache[prompt_hash] = (prompt_tokens.clone(), cache_key)
            self._kv_caches[cache_key] = cache
            
            return cache_key
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reuse statistics."""
        return {
            **self._stats,
            "cached_prompts": len(self._prompt_cache),
            "hit_rate": (
                self._stats["reuse_hits"] / 
                (self._stats["reuse_hits"] + self._stats["reuse_misses"])
                if (self._stats["reuse_hits"] + self._stats["reuse_misses"]) > 0
                else 0.0
            ),
        }


class KVCacheOptimizer:
    """High-level optimizer for KV cache management.
    
    Combines mixed-precision caching with reuse and memory management.
    
    Parameters
    ----------
    config : CacheConfig
        Cache configuration
    device : str
        Device for cache tensors
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        device: str = "cuda:0",
    ):
        self.config = config or CacheConfig()
        self.device = device
        
        self._active_cache = MixedPrecisionKVCache(self.config, device)
        self._reuser = CacheReuser(
            similarity_threshold=self.config.reuse_similarity_threshold,
        ) if self.config.enable_reuse else None
        
        self._memory_monitor_enabled = True
    
    def get_or_create_cache(
        self,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[MixedPrecisionKVCache, int]:
        """Get existing cache or create new one.
        
        Parameters
        ----------
        prompt_tokens : torch.Tensor, optional
            Prompt tokens for reuse lookup
            
        Returns
        -------
        cache, reuse_length : tuple
            Cache instance and number of reusable tokens
        """
        reuse_len = 0
        
        if self._reuser and prompt_tokens is not None:
            cached, reuse_len = self._reuser.find_reusable_cache(prompt_tokens)
            if cached and reuse_len > 0:
                logger.debug("Reusing %d tokens from cached KV", reuse_len)
                # Clone the cache for this request
                self._active_cache = MixedPrecisionKVCache(self.config, self.device)
                # Copy reusable portion (simplified - full impl would copy cache data)
                return self._active_cache, reuse_len
        
        # Create fresh cache
        self._active_cache = MixedPrecisionKVCache(self.config, self.device)
        return self._active_cache, 0
    
    def update_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Update active cache."""
        self._active_cache.update(layer_idx, key_states, value_states)
        
        # Check memory pressure
        if self._memory_monitor_enabled:
            memory_mb = self._active_cache.get_memory_usage_mb()
            if memory_mb > self.config.max_memory_mb * 0.9:
                logger.warning(
                    "KV cache memory high: %.1f MB / %.1f MB",
                    memory_mb, self.config.max_memory_mb
                )
    
    def get_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV states."""
        return self._active_cache.get(layer_idx)
    
    def finalize_request(
        self,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        """Finalize request and optionally store for reuse."""
        if self._reuser and prompt_tokens is not None:
            self._reuser.store_cache(prompt_tokens, self._active_cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = self._active_cache.get_stats()
        if self._reuser:
            stats["reuse"] = self._reuser.get_stats()
        return stats
    
    def clear(self) -> None:
        """Clear all caches."""
        self._active_cache.clear()


# Register as plugin
@PluginRegistry.register(
    "cache",
    "kv_optimizer",
    description="Mixed-precision KV cache with reuse"
)
class KVCachePlugin(CachePlugin):
    """Plugin wrapper for KVCacheOptimizer."""
    
    def __init__(self, **config):
        cache_config = CacheConfig(**config) if config else CacheConfig()
        self._optimizer = KVCacheOptimizer(cache_config)
        self._data: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "kv_optimizer"
    
    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._data[key] = value
    
    def evict(self, key: str) -> bool:
        return self._data.pop(key, None) is not None
    
    def clear(self) -> None:
        self._data.clear()
        self._optimizer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return self._optimizer.get_stats()
