"""
OOM Guardian - Prevents out-of-memory crashes through preemptive action.

Monitors memory pressure and takes corrective actions before crashes occur:
- Reduces batch size
- Enables aggressive offloading
- Downgrades precision
- Triggers garbage collection

Usage::

    from model.oom_guardian import OOMGuardian, OOMGuardianConfig
    
    guardian = OOMGuardian(
        gpu_budget_mb=24000,
        warning_threshold=0.85,
        critical_threshold=0.95,
    )
    
    # Check before operation
    estimated_memory = hidden.numel() * 4  # bytes
    if not guardian.can_allocate(estimated_memory):
        # Take evasive action
        reduce_batch_size()
    
    # Record metrics for analysis
    guardian.record_memory_state(current_mb, peak_mb)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class OOMAction(Enum):
    """Actions that can be taken to prevent OOM."""
    REDUCE_BATCH = "reduce_batch"
    ENABLE_OFFLOAD = "enable_offload"
    DOWNGRADE_PRECISION = "downgrade_precision"
    AGGRESSIVE_GC = "aggressive_gc"
    EMERGENCY_EVICT = "emergency_evict"
    CHECKPOINT_SAVE = "checkpoint_save"


class MemoryPressureLevel(Enum):
    """Memory pressure classification."""
    LOW = "low"        # < 70% utilization
    MODERATE = "moderate"  # 70-85%
    WARNING = "warning"      # 85-92%
    CRITICAL = "critical"    # > 92%


@dataclass
class OOMGuardianConfig:
    """Configuration for OOM guardian."""
    gpu_budget_mb: float = 24000.0
    warning_threshold: float = 0.85  # % of budget
    critical_threshold: float = 0.95  # % of budget
    headroom_mb: float = 500.0  # Safety margin
    enable_preemptive_gc: bool = True
    gc_interval_steps: int = 10
    max_batch_size: int = 32
    min_batch_size: int = 1


class OOMGuardian:
    """
    Prevents out-of-memory crashes through monitoring and preemptive action.
    """
    
    def __init__(self, config: Optional[OOMGuardianConfig] = None):
        self.config = config or OOMGuardianConfig()
        
        # Memory tracking
        self.current_usage_mb = 0.0
        self.peak_usage_mb = 0.0
        self.last_recorded_step = 0
        
        # Action history
        self.action_history: List[tuple] = []  # (step, action, reason)
        self.oom_near_misses = 0
        
        # Callbacks for actions
        self._action_callbacks: Dict[OOMAction, List[Callable]] = {
            action: [] for action in OOMAction
        }
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "total_warnings": 0,
            "total_critical": 0,
            "total_interventions": 0,
            "total_oom_prevented": 0,
        }
        
        logger.info(
            f"OOMGuardian initialized: budget={config.gpu_budget_mb:.0f}MB, "
            f"warning_threshold={config.warning_threshold:.1%}, "
            f"critical_threshold={config.critical_threshold:.1%}"
        )
    
    def register_callback(
        self,
        action: OOMAction,
        callback: Callable,
    ) -> None:
        """
        Register callback for an action.
        
        Parameters
        ----------
        action : OOMAction
            Action type
        callback : Callable
            Function to call when action triggered.
            Signature: callback() -> None
        """
        self._action_callbacks[action].append(callback)
    
    def get_pressure_level(self) -> MemoryPressureLevel:
        """Classify current memory pressure."""
        utilization = self.current_usage_mb / self.config.gpu_budget_mb
        
        if utilization < 0.70:
            return MemoryPressureLevel.LOW
        elif utilization < 0.85:
            return MemoryPressureLevel.MODERATE
        elif utilization < 0.95:
            return MemoryPressureLevel.WARNING
        else:
            return MemoryPressureLevel.CRITICAL
    
    def can_allocate(
        self,
        size_bytes: int,
        margin_mb: float = 100.0,
    ) -> bool:
        """
        Check if allocation would be safe.
        
        Parameters
        ----------
        size_bytes : int
            Size of allocation in bytes
        margin_mb : float
            Extra safety margin in MB
        
        Returns
        -------
        bool
            True if safe to allocate
        """
        self.stats["total_checks"] += 1
        
        size_mb = size_bytes / (1024 ** 2)
        required_total = self.current_usage_mb + size_mb + margin_mb
        available = self.config.gpu_budget_mb - self.config.headroom_mb
        
        if required_total > available:
            pressure = self.get_pressure_level()
            self.stats["total_warnings"] += 1
            
            if pressure == MemoryPressureLevel.CRITICAL:
                self.stats["total_critical"] += 1
                logger.warning(
                    f"CRITICAL memory pressure: {self.current_usage_mb:.0f}MB + "
                    f"{size_mb:.0f}MB would exceed {available:.0f}MB available"
                )
                self._trigger_action(OOMAction.AGGRESSIVE_GC, "memory critical")
                return False
            elif pressure == MemoryPressureLevel.WARNING:
                logger.warning(
                    f"WARNING: Memory pressure at {required_total/available:.1%} "
                    f"({self.current_usage_mb:.0f}MB / {available:.0f}MB)"
                )
                self._trigger_action(OOMAction.REDUCE_BATCH, "memory warning")
                return True
        
        return True
    
    def record_memory_state(
        self,
        current_mb: float,
        peak_mb: Optional[float] = None,
    ) -> None:
        """Record current memory state."""
        self.current_usage_mb = current_mb
        if peak_mb is not None:
            self.peak_usage_mb = max(self.peak_usage_mb, peak_mb)
        else:
            self.peak_usage_mb = max(self.peak_usage_mb, current_mb)
    
    def _trigger_action(
        self,
        action: OOMAction,
        reason: str,
        step: int = 0,
    ) -> None:
        """
        Trigger an action and call registered callbacks.
        
        Parameters
        ----------
        action : OOMAction
            Action to take
        reason : str
            Reason for action
        step : int
            Current step/iteration
        """
        logger.info(f"OOMGuardian triggering {action.value}: {reason}")
        
        self.action_history.append((step, action, reason))
        self.stats["total_interventions"] += 1
        
        # Call all registered callbacks for this action
        for callback in self._action_callbacks[action]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in OOMGuardian callback: {e}")
    
    def emergency_gc(self) -> None:
        """Perform emergency garbage collection."""
        logger.info("Emergency GC triggered")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def get_stats(self) -> Dict[str, any]:
        """Get guardian statistics."""
        stats = dict(self.stats)
        stats["current_memory_mb"] = self.current_usage_mb
        stats["peak_memory_mb"] = self.peak_usage_mb
        stats["pressure_level"] = self.get_pressure_level().value
        stats["action_history_length"] = len(self.action_history)
        return stats
    
    def get_recent_actions(self, n: int = 10) -> List[tuple]:
        """Get recent actions taken."""
        return self.action_history[-n:]
    
    def reset(self) -> None:
        """Reset guardian state."""
        self.current_usage_mb = 0.0
        self.peak_usage_mb = 0.0
        self.action_history.clear()
        self.oom_near_misses = 0
