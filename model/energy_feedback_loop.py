"""
Energy Feedback Control Loop for KAI.

Implements a closed-loop optimizer that continuously monitors:
- Power usage
- Throughput
- Latency

And dynamically adjusts:
- Batch size
- GPU power limits
- Precision strategy
- Offloading thresholds

Goal: Minimize energy per token while maintaining output quality.

Usage::

    from model.energy_feedback_loop import EnergyFeedbackController
    
    controller = EnergyFeedbackController(
        target_power_w=200,
        target_latency_ms=100,
    )
    
    controller.start()
    
    # Get current recommendations
    config = controller.get_optimal_config()
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from model.plugin_architecture import PluginRegistry, OptimizerPlugin

logger = logging.getLogger(__name__)


class ControlAction(Enum):
    """Actions the controller can take."""
    INCREASE_BATCH = "increase_batch"
    DECREASE_BATCH = "decrease_batch"
    INCREASE_POWER_LIMIT = "increase_power_limit"
    DECREASE_POWER_LIMIT = "decrease_power_limit"
    INCREASE_PRECISION = "increase_precision"
    DECREASE_PRECISION = "decrease_precision"
    ENABLE_OFFLOAD = "enable_offload"
    DISABLE_OFFLOAD = "disable_offload"
    NO_ACTION = "no_action"


@dataclass
class EnergyMetrics:
    """Energy and performance metrics snapshot."""
    timestamp: str = ""
    power_w: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    batch_size: int = 1
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    @property
    def energy_per_token_j(self) -> float:
        """Energy consumption per token in Joules."""
        if self.throughput_tokens_per_sec <= 0:
            return float("inf")
        return self.power_w / self.throughput_tokens_per_sec
    
    @property
    def efficiency_score(self) -> float:
        """Overall efficiency score (higher is better)."""
        if self.power_w <= 0 or self.latency_ms <= 0:
            return 0.0
        return self.throughput_tokens_per_sec / (self.power_w * self.latency_ms / 1000)
    
    @property
    def memory_pressure(self) -> float:
        """Memory pressure (0.0-1.0)."""
        if self.memory_total_mb <= 0:
            return 0.0
        return self.memory_used_mb / self.memory_total_mb
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "power_w": round(self.power_w, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "latency_ms": round(self.latency_ms, 3),
            "batch_size": self.batch_size,
            "memory_used_mb": round(self.memory_used_mb, 2),
            "memory_total_mb": round(self.memory_total_mb, 2),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 2),
            "energy_per_token_j": round(self.energy_per_token_j, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "memory_pressure": round(self.memory_pressure, 3),
        }


@dataclass
class ControlConfig:
    """Configuration that can be adjusted by the controller."""
    batch_size: int = 1
    gpu_power_limit_w: int = 250
    precision: str = "fp16"  # fp32, fp16, int8, int4
    offload_enabled: bool = False
    offload_threshold_mb: float = 1000.0  # Offload when free VRAM < this
    
    # Constraints
    min_batch_size: int = 1
    max_batch_size: int = 64
    min_power_limit_w: int = 100
    max_power_limit_w: int = 350
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "gpu_power_limit_w": self.gpu_power_limit_w,
            "precision": self.precision,
            "offload_enabled": self.offload_enabled,
            "offload_threshold_mb": self.offload_threshold_mb,
        }


@dataclass
class ControllerState:
    """Internal state of the feedback controller."""
    current_config: ControlConfig = field(default_factory=ControlConfig)
    last_action: ControlAction = ControlAction.NO_ACTION
    action_history: List[Tuple[str, ControlAction]] = field(default_factory=list)
    stable_since: float = 0.0  # Timestamp when config became stable
    adjustments_count: int = 0
    
    def record_action(self, action: ControlAction) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.action_history.append((timestamp, action))
        self.last_action = action
        self.adjustments_count += 1
        
        # Keep last 100 actions
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]


class PIDController:
    """PID controller for continuous optimization.
    
    Parameters
    ----------
    kp : float
        Proportional gain
    ki : float
        Integral gain
    kd : float
        Derivative gain
    setpoint : float
        Target value
    """
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.1,
        setpoint: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()
    
    def update(self, measured: float) -> float:
        """Compute control output.
        
        Parameters
        ----------
        measured : float
            Current measured value
            
        Returns
        -------
        float
            Control output (adjustment amount)
        """
        now = time.monotonic()
        dt = now - self._last_time
        if dt <= 0:
            dt = 0.001
        
        error = self.setpoint - measured
        
        # Proportional
        p_term = self.kp * error
        
        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = max(-10.0, min(10.0, self._integral))
        i_term = self.ki * self._integral
        
        # Derivative
        d_term = self.kd * (error - self._last_error) / dt
        
        self._last_error = error
        self._last_time = now
        
        return p_term + i_term + d_term
    
    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()


class EnergyFeedbackController:
    """Closed-loop energy optimization controller.
    
    Continuously monitors system metrics and adjusts configuration
    to minimize energy per token while maintaining quality and latency.
    
    Parameters
    ----------
    target_power_w : float
        Target power consumption in Watts
    target_latency_ms : float
        Target latency per inference in milliseconds
    target_efficiency : float
        Target tokens per Joule (higher is better)
    update_interval_s : float
        How often to check and adjust
    """
    
    PRECISION_ORDER = ["fp32", "fp16", "bf16", "int8", "int4"]
    
    def __init__(
        self,
        target_power_w: float = 200.0,
        target_latency_ms: float = 100.0,
        target_efficiency: Optional[float] = None,
        update_interval_s: float = 5.0,
    ):
        self._target_power = target_power_w
        self._target_latency = target_latency_ms
        self._target_efficiency = target_efficiency
        self._update_interval = update_interval_s
        
        # State
        self._state = ControllerState()
        self._metrics_history: deque = deque(maxlen=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # PID controllers
        self._power_pid = PIDController(kp=0.02, ki=0.005, kd=0.01, setpoint=target_power_w)
        self._latency_pid = PIDController(kp=0.05, ki=0.01, kd=0.02, setpoint=target_latency_ms)
        
        # Callbacks for metric collection
        self._metric_callback: Optional[Callable[[], EnergyMetrics]] = None
        self._config_apply_callback: Optional[Callable[[ControlConfig], None]] = None
    
    def set_metric_callback(self, callback: Callable[[], EnergyMetrics]) -> None:
        """Set callback to collect current metrics."""
        self._metric_callback = callback
    
    def set_config_callback(self, callback: Callable[[ControlConfig], None]) -> None:
        """Set callback to apply configuration changes."""
        self._config_apply_callback = callback
    
    def start(self) -> None:
        """Start the feedback control loop."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="energy-feedback",
        )
        self._thread.start()
        logger.info("EnergyFeedbackController started")
    
    def stop(self) -> None:
        """Stop the feedback control loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("EnergyFeedbackController stopped")
    
    def get_optimal_config(self) -> ControlConfig:
        """Get current optimal configuration."""
        with self._lock:
            return self._state.current_config
    
    def get_metrics_history(self, n: int = 100) -> List[EnergyMetrics]:
        """Get recent metrics history."""
        with self._lock:
            return list(self._metrics_history)[-n:]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        with self._lock:
            return {
                "config": self._state.current_config.to_dict(),
                "last_action": self._state.last_action.value,
                "adjustments_count": self._state.adjustments_count,
                "is_stable": self._is_stable(),
                "targets": {
                    "power_w": self._target_power,
                    "latency_ms": self._target_latency,
                    "efficiency": self._target_efficiency,
                },
            }
    
    def record_metrics(self, metrics: EnergyMetrics) -> None:
        """Manually record metrics (for external monitoring)."""
        with self._lock:
            self._metrics_history.append(metrics)
    
    def _control_loop(self) -> None:
        """Main feedback control loop."""
        while self._running:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                
                if metrics:
                    with self._lock:
                        self._metrics_history.append(metrics)
                    
                    # Decide action
                    action = self._decide_action(metrics)
                    
                    # Apply action
                    if action != ControlAction.NO_ACTION:
                        self._apply_action(action)
            
            except Exception as e:
                logger.error("Control loop error: %s", e)
            
            time.sleep(self._update_interval)
    
    def _collect_metrics(self) -> Optional[EnergyMetrics]:
        """Collect current metrics."""
        if self._metric_callback:
            return self._metric_callback()
        
        # Default: try to get GPU metrics via NVML
        try:
            return self._collect_gpu_metrics()
        except Exception:
            return None
    
    def _collect_gpu_metrics(self) -> EnergyMetrics:
        """Collect metrics from GPU via NVML."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            pynvml.nvmlShutdown()
            
            return EnergyMetrics(
                power_w=power_w,
                memory_used_mb=mem_info.used / (1024**2),
                memory_total_mb=mem_info.total / (1024**2),
                gpu_utilization_pct=util.gpu,
                batch_size=self._state.current_config.batch_size,
            )
        except Exception:
            # Return dummy metrics if NVML not available
            return EnergyMetrics(
                power_w=150.0,
                throughput_tokens_per_sec=50.0,
                latency_ms=50.0,
                batch_size=self._state.current_config.batch_size,
            )
    
    def _decide_action(self, metrics: EnergyMetrics) -> ControlAction:
        """Decide what action to take based on metrics."""
        config = self._state.current_config
        
        # Calculate control signals
        power_signal = self._power_pid.update(metrics.power_w)
        latency_signal = self._latency_pid.update(metrics.latency_ms)
        
        # Priority: latency > power > efficiency
        
        # Check latency constraint
        if metrics.latency_ms > self._target_latency * 1.2:
            # Latency too high
            if config.batch_size > config.min_batch_size:
                return ControlAction.DECREASE_BATCH
            elif config.precision != "fp16":
                return ControlAction.DECREASE_PRECISION
            elif config.offload_enabled:
                return ControlAction.DISABLE_OFFLOAD
            elif config.gpu_power_limit_w < config.max_power_limit_w:
                return ControlAction.INCREASE_POWER_LIMIT
        
        # Check power constraint
        if metrics.power_w > self._target_power * 1.1:
            # Power too high
            if config.gpu_power_limit_w > config.min_power_limit_w:
                return ControlAction.DECREASE_POWER_LIMIT
            elif config.batch_size > config.min_batch_size:
                return ControlAction.DECREASE_BATCH
        
        # Check memory pressure
        if metrics.memory_pressure > 0.9:
            # Memory critical
            if not config.offload_enabled:
                return ControlAction.ENABLE_OFFLOAD
            elif self.PRECISION_ORDER.index(config.precision) < len(self.PRECISION_ORDER) - 1:
                return ControlAction.DECREASE_PRECISION
        
        # Optimization mode: can we be more efficient?
        if self._is_stable():
            # Try to improve efficiency
            if metrics.power_w < self._target_power * 0.8:
                # Underutilized: increase batch
                if config.batch_size < config.max_batch_size:
                    return ControlAction.INCREASE_BATCH
            
            if metrics.latency_ms < self._target_latency * 0.5:
                # Very fast: can trade latency for efficiency
                if config.batch_size < config.max_batch_size:
                    return ControlAction.INCREASE_BATCH
        
        return ControlAction.NO_ACTION
    
    def _apply_action(self, action: ControlAction) -> None:
        """Apply a control action."""
        with self._lock:
            config = self._state.current_config
            
            if action == ControlAction.INCREASE_BATCH:
                config.batch_size = min(
                    config.batch_size * 2,
                    config.max_batch_size
                )
            
            elif action == ControlAction.DECREASE_BATCH:
                config.batch_size = max(
                    config.batch_size // 2,
                    config.min_batch_size
                )
            
            elif action == ControlAction.INCREASE_POWER_LIMIT:
                config.gpu_power_limit_w = min(
                    config.gpu_power_limit_w + 25,
                    config.max_power_limit_w
                )
            
            elif action == ControlAction.DECREASE_POWER_LIMIT:
                config.gpu_power_limit_w = max(
                    config.gpu_power_limit_w - 25,
                    config.min_power_limit_w
                )
            
            elif action == ControlAction.INCREASE_PRECISION:
                idx = self.PRECISION_ORDER.index(config.precision)
                if idx > 0:
                    config.precision = self.PRECISION_ORDER[idx - 1]
            
            elif action == ControlAction.DECREASE_PRECISION:
                idx = self.PRECISION_ORDER.index(config.precision)
                if idx < len(self.PRECISION_ORDER) - 1:
                    config.precision = self.PRECISION_ORDER[idx + 1]
            
            elif action == ControlAction.ENABLE_OFFLOAD:
                config.offload_enabled = True
            
            elif action == ControlAction.DISABLE_OFFLOAD:
                config.offload_enabled = False
            
            self._state.record_action(action)
            self._state.stable_since = 0.0  # Reset stability
            
            logger.info("Applied action: %s -> config: %s", action.value, config.to_dict())
        
        # Apply configuration externally
        if self._config_apply_callback:
            self._config_apply_callback(config)
    
    def _is_stable(self) -> bool:
        """Check if system has been stable."""
        if self._state.stable_since == 0.0:
            self._state.stable_since = time.monotonic()
            return False
        
        # Stable for at least 30 seconds
        return (time.monotonic() - self._state.stable_since) > 30.0


class PowerLimitManager:
    """Manages GPU power limits via NVML.
    
    Parameters
    ----------
    device_index : int
        GPU device index
    """
    
    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._handle = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize NVML."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning("NVML initialization failed: %s", e)
            return False
    
    def get_power_limit(self) -> Tuple[int, int, int]:
        """Get current, min, max power limits in Watts."""
        if not self._initialized:
            return (250, 100, 350)  # Defaults
        
        try:
            import pynvml
            current = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) // 1000
            limits = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self._handle)
            return (current, limits[0] // 1000, limits[1] // 1000)
        except Exception:
            return (250, 100, 350)
    
    def set_power_limit(self, limit_w: int) -> bool:
        """Set GPU power limit."""
        if not self._initialized:
            return False
        
        try:
            import pynvml
            _, min_limit, max_limit = self.get_power_limit()
            limit_w = max(min_limit, min(limit_w, max_limit))
            pynvml.nvmlDeviceSetPowerManagementLimit(self._handle, limit_w * 1000)
            logger.info("Set GPU power limit to %d W", limit_w)
            return True
        except Exception as e:
            logger.error("Failed to set power limit: %s", e)
            return False
    
    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


# Register as plugin
@PluginRegistry.register(
    "optimizer",
    "energy_feedback",
    description="Closed-loop energy optimization"
)
class EnergyFeedbackPlugin(OptimizerPlugin):
    """Plugin wrapper for EnergyFeedbackController."""
    
    def __init__(
        self,
        target_power_w: float = 200.0,
        target_latency_ms: float = 100.0,
    ):
        self._controller = EnergyFeedbackController(
            target_power_w=target_power_w,
            target_latency_ms=target_latency_ms,
        )
    
    @property
    def name(self) -> str:
        return "energy_feedback"
    
    def optimize(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
    ) -> torch.nn.Module:
        # Start the controller
        self._controller.start()
        
        # Apply initial config if provided
        optimal = self._controller.get_optimal_config()
        
        # The controller will continuously optimize
        # This method returns the model unchanged
        return model
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return self._controller.get_optimal_config().to_dict()
    
    def shutdown(self) -> None:
        self._controller.stop()
