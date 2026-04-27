"""
Adaptive Batch Size Controller.

Dynamically adjusts batch size based on memory pressure, latency targets,
and throughput goals. Implements closed-loop control with safety guardrails.

Usage::

    from model.adaptive_batch_controller import AdaptiveBatchController, BatchConfig
    
    config = BatchConfig(
        initial_batch=8,
        min_batch=1,
        max_batch=32,
        target_latency_ms=100.0,
        memory_threshold=0.85,
    )
    
    controller = AdaptiveBatchController(config)
    
    for epoch in range(10):
        try:
            batch_size = controller.get_batch_size()
            loss = train_batch(batch_size)
            controller.record_success(latency_ms=45.0, memory_mb=15000.0)
        except torch.cuda.OutOfMemoryError:
            controller.record_failure()
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BatchAdjustmentMode(Enum):
    """How to adjust batch size."""
    CONSERVATIVE = "conservative"  # +/- 1 at a time
    MODERATE = "moderate"          # +/- 2 at a time
    AGGRESSIVE = "aggressive"      # +/- 4 at a time


@dataclass
class BatchConfig:
    """Configuration for adaptive batch size control."""
    initial_batch: int = 8
    min_batch: int = 1
    max_batch: int = 32
    
    # Control parameters
    mode: BatchAdjustmentMode = BatchAdjustmentMode.MODERATE
    growth_factor: float = 1.1  # Grow slower than shrink
    shrink_factor: float = 0.7  # Shrink faster than grow
    
    # Target thresholds
    target_latency_ms: float = 100.0
    latency_window: int = 5  # # of recent samples
    memory_threshold: float = 0.85  # Fraction of budget
    
    # Stability
    success_streak_to_grow: int = 3  # Growth only after N successes
    failure_streak_to_shrink: int = 2  # Shrink on N failures


class AdaptiveBatchController:
    """
    Adapts batch size based on observed performance and constraints.
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        
        # Current state
        self.current_batch = self.config.initial_batch
        self.previous_batch = self.current_batch
        
        # Performance history
        self.latency_history: list = []
        self.memory_history: list = []
        self.success_streak = 0
        self.failure_streak = 0
        
        # Statistics
        self.stats = {
            "total_adjustments": 0,
            "total_grows": 0,
            "total_shrinks": 0,
            "total_successes": 0,
            "total_failures": 0,
            "avg_latency_ms": 0.0,
            "avg_memory_mb": 0.0,
        }
        
        logger.info(
            f"AdaptiveBatchController initialized: "
            f"initial_batch={self.config.initial_batch}, "
            f"range=[{self.config.min_batch}, {self.config.max_batch}]"
        )
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch
    
    def record_success(
        self,
        latency_ms: float,
        memory_mb: float,
    ) -> None:
        """
        Record successful training step.
        
        Parameters
        ----------
        latency_ms : float
            Step latency in milliseconds
        memory_mb : float
            Peak memory usage in MB
        """
        self.latency_history.append(latency_ms)
        self.memory_history.append(memory_mb)
        
        self.success_streak += 1
        self.failure_streak = 0
        self.stats["total_successes"] += 1
        
        # Update averages
        window = self.config.latency_window
        self.stats["avg_latency_ms"] = (
            sum(self.latency_history[-window:]) / len(self.latency_history[-window:])
        )
        self.stats["avg_memory_mb"] = (
            sum(self.memory_history[-window:]) / len(self.memory_history[-window:])
        )
        
        # Decide whether to grow
        self._check_growth()
    
    def record_failure(self) -> None:
        """
        Record failed training step (OOM or other error).
        """
        self.failure_streak += 1
        self.success_streak = 0
        self.stats["total_failures"] += 1
        
        # Immediately shrink
        if self.failure_streak >= self.config.failure_streak_to_shrink:
            self._shrink_batch()
    
    def _check_growth(self) -> None:
        """Check if we can grow batch size."""
        # Need streak of successes
        if self.success_streak < self.config.success_streak_to_grow:
            return
        
        avg_latency = self.stats["avg_latency_ms"]
        avg_memory = self.stats["avg_memory_mb"]
        
        # Don't grow if latency already at target
        if avg_latency >= self.config.target_latency_ms:
            return
        
        # Don't grow if memory pressure is high
        if avg_memory > self.config.memory_threshold * 24000:  # Assuming 24GB
            return
        
        self._grow_batch()
    
    def _grow_batch(self) -> None:
        """Increase batch size."""
        self.previous_batch = self.current_batch
        
        # Apply growth factor
        new_batch = int(self.current_batch * self.config.growth_factor)
        self.current_batch = min(new_batch, self.config.max_batch)
        
        if self.current_batch > self.previous_batch:
            logger.info(
                f"Growing batch: {self.previous_batch} → {self.current_batch} "
                f"(streak={self.success_streak})"
            )
            self.stats["total_grows"] += 1
            self.stats["total_adjustments"] += 1
            self.success_streak = 0  # Reset streak
    
    def _shrink_batch(self) -> None:
        """Decrease batch size."""
        self.previous_batch = self.current_batch
        
        # Apply shrink factor
        new_batch = max(int(self.current_batch * self.config.shrink_factor), 1)
        self.current_batch = max(new_batch, self.config.min_batch)
        
        if self.current_batch < self.previous_batch:
            logger.info(
                f"Shrinking batch: {self.previous_batch} → {self.current_batch} "
                f"(failure_streak={self.failure_streak})"
            )
            self.stats["total_shrinks"] += 1
            self.stats["total_adjustments"] += 1
            self.failure_streak = 0  # Reset streak
    
    def get_stats(self) -> Dict[str, any]:
        """Get controller statistics."""
        stats = dict(self.stats)
        stats["current_batch"] = self.current_batch
        stats["latency_history_length"] = len(self.latency_history)
        stats["memory_history_length"] = len(self.memory_history)
        stats["success_streak"] = self.success_streak
        stats["failure_streak"] = self.failure_streak
        return stats
    
    def reset(self) -> None:
        """Reset controller to initial state."""
        self.current_batch = self.config.initial_batch
        self.latency_history.clear()
        self.memory_history.clear()
        self.success_streak = 0
        self.failure_streak = 0
