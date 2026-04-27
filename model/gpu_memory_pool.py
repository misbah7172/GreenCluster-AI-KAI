"""
GPU Memory Pool for preventing fragmentation.

Pre-allocates and reuses GPU memory buffers instead of frequent
allocation/deallocation, which causes fragmentation and allocation
stalls.

Usage::

    from model.gpu_memory_pool import GPUMemoryPool
    
    pool = GPUMemoryPool(gpu_budget_mb=12000)
    
    # Reserve typical buffer sizes
    pool.reserve(name="hidden_state", shape=(1, 1, 4096))
    pool.reserve(name="attention_matrix", shape=(32, 2048, 2048))
    
    # Allocate from pool
    hidden = pool.alloc("hidden_state")
    # ... use ...
    pool.free("hidden_state", hidden)
    
    # Get stats
    stats = pool.get_stats()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class BufferSpec:
    """Specification for a preallocated buffer."""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32
    device: str = "cuda:0"
    pool_size: int = 3  # How many to pre-allocate
    
    def element_count(self) -> int:
        """Total elements in buffer."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def memory_bytes(self) -> int:
        """Memory per buffer in bytes."""
        dtype_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int4: 0.5,  # Half-byte per element
        }.get(self.dtype, 4)
        return int(self.element_count() * dtype_size)
    
    def memory_mb(self) -> float:
        """Memory per buffer in MB."""
        return self.memory_bytes() / (1024 ** 2)


class GPUMemoryPool:
    """
    Pre-allocates GPU memory buffers to prevent fragmentation.
    
    Maintains pools of buffers for common shapes and types.
    Allocate/free operations pop/push from pools.
    """
    
    def __init__(self, gpu_budget_mb: float, device: str = "cuda:0"):
        self.gpu_budget_mb = gpu_budget_mb
        self.device = device
        
        # {name -> List[tensor]}
        self._free_pools: Dict[str, List[torch.Tensor]] = {}
        # {tensor.data_ptr() -> name}
        self._allocated: Dict[int, str] = {}
        # {name -> BufferSpec}
        self._specs: Dict[str, BufferSpec] = {}
        # {name -> bytes allocated}
        self._pool_usage: Dict[str, int] = {}
        
        self.stats = {
            "total_allocations": 0,
            "total_frees": 0,
            "total_reserves": 0,
            "fragmentation_index": 0.0,
        }
    
    def reserve(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        pool_size: int = 3,
    ) -> None:
        """Reserve buffers for a given shape."""
        spec = BufferSpec(name, shape, dtype, self.device, pool_size)
        self._specs[name] = spec
        self._free_pools[name] = []
        
        total_mb = spec.memory_mb() * pool_size
        logger.info(
            f"Reserving pool '{name}': {pool_size} buffers of "
            f"shape {shape} = {total_mb:.1f} MB"
        )
        
        # Pre-allocate
        for _ in range(pool_size):
            try:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self._free_pools[name].append(buf)
            except RuntimeError as e:
                logger.warning(f"Failed to pre-allocate buffer for {name}: {e}")
        
        self._pool_usage[name] = 0
        self.stats["total_reserves"] += 1
    
    def alloc(
        self,
        name: str,
        zero: bool = False,
    ) -> torch.Tensor:
        """
        Allocate buffer from pool.
        
        Parameters
        ----------
        name : str
            Buffer pool name (must have been reserved)
        zero : bool
            If True, zero the buffer before returning
        
        Returns
        -------
        torch.Tensor
            Tensor from pool (or newly allocated if pool exhausted)
        """
        if name not in self._specs:
            raise ValueError(f"Unknown buffer pool: {name}")
        
        # Try to get from free pool
        if self._free_pools[name]:
            buf = self._free_pools[name].pop()
        else:
            # Allocate new (pool exhausted)
            spec = self._specs[name]
            buf = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
            logger.warning(f"Pool '{name}' exhausted, allocated new buffer")
        
        # Track allocation
        self._allocated[id(buf)] = name
        self._pool_usage[name] += buf.numel() * buf.dtype.itemsize
        self.stats["total_allocations"] += 1
        
        # Zero if requested
        if zero:
            buf.zero_()
        
        return buf
    
    def free(self, name: str, tensor: torch.Tensor) -> None:
        """
        Return buffer to pool.
        
        Parameters
        ----------
        name : str
            Buffer pool name
        tensor : torch.Tensor
            Tensor to return to pool
        """
        if name not in self._specs:
            logger.warning(f"Unknown pool {name}, discarding tensor")
            return
        
        # Verify it matches spec
        spec = self._specs[name]
        if tensor.shape != spec.shape or tensor.dtype != spec.dtype:
            logger.warning(
                f"Tensor shape/dtype mismatch for pool {name}: "
                f"expected {spec.shape}/{spec.dtype}, got {tensor.shape}/{tensor.dtype}"
            )
            return
        
        # Return to pool
        self._free_pools[name].append(tensor)
        self._pool_usage[name] -= tensor.numel() * tensor.dtype.itemsize
        self.stats["total_frees"] += 1
    
    def get_pool_status(self, name: str) -> Dict[str, int]:
        """Get status of specific pool."""
        if name not in self._specs:
            return {}
        
        spec = self._specs[name]
        free_count = len(self._free_pools[name])
        allocated_count = spec.pool_size - free_count  # Approximate
        
        return {
            "name": name,
            "free_buffers": free_count,
            "allocated_buffers": allocated_count,
            "total_buffers": spec.pool_size,
            "per_buffer_mb": spec.memory_mb(),
            "total_reserved_mb": spec.memory_mb() * spec.pool_size,
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive pool statistics."""
        total_mb = sum(
            self._specs[name].memory_mb() * len(self._free_pools[name])
            for name in self._specs
        )
        
        utilization = {}
        for name in self._specs:
            status = self.get_pool_status(name)
            utilization[name] = status
        
        return {
            "total_allocations": self.stats["total_allocations"],
            "total_frees": self.stats["total_frees"],
            "total_reserves": self.stats["total_reserves"],
            "total_free_memory_mb": total_mb,
            "pool_utilization": utilization,
        }
    
    def clear(self) -> None:
        """Clear all pools."""
        self._free_pools.clear()
        self._allocated.clear()
        self._specs.clear()
        self._pool_usage.clear()
        torch.cuda.empty_cache()
