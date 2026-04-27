"""
Energy Feedback Control Loop for KAI.

This module implements a production-oriented adaptive controller that monitors
power, latency, throughput, memory pressure, and GPU utilization, then applies
safe control actions to improve energy efficiency without violating service
objectives.

Backward-compatible API surface:
- EnergyFeedbackController
- EnergyFeedbackConfig
- ControlConfig
- PIDController
- EnergyMetrics
- ControlAction
- EnergyFeedbackPlugin
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, pvariance
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import torch

from model.plugin_architecture import OptimizerPlugin, PluginRegistry

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


class ActionRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EnergyFeedbackConfig:
    """Controller configuration and tuning parameters.

    The field names preserve the old controller semantics while adding
    predictive-control and safety knobs.
    """

    power_target_w: float = 200.0
    latency_target_ms: float = 100.0
    target_efficiency: Optional[float] = None
    update_interval_s: float = 5.0

    # PID and smoothing
    deadband_power_w: float = 3.0
    deadband_latency_ms: float = 5.0
    pid_output_min: float = -50.0
    pid_output_max: float = 50.0
    pid_rate_limit: float = 12.0
    smoothing_window: int = 5

    # Stability / prediction
    stability_window: int = 6
    stability_variance_power: float = 25.0
    stability_variance_latency: float = 100.0
    prediction_horizon_steps: int = 3
    prediction_slope_weight: float = 1.0

    # Objective weights
    power_weight: float = 1.0
    latency_weight: float = 1.5
    throughput_weight: float = 0.5

    # Guardrails
    max_latency_ms: float = 250.0
    min_throughput_tokens_per_sec: float = 1.0
    max_memory_pressure: float = 0.92
    max_action_repeat: int = 3
    action_cooldown_s: float = 2.0

    # Adaptive sampling
    min_update_interval_s: float = 0.5
    max_update_interval_s: float = 5.0
    unstable_interval_s: float = 0.5

    # Power estimation fallback
    fallback_idle_power_w: float = 45.0
    fallback_max_power_w: float = 180.0

    # Risk weights
    low_risk_weight: float = 0.05
    medium_risk_weight: float = 0.2
    high_risk_weight: float = 0.45

    # Compatibility aliases for old callers/tests
    power_limit_w: Optional[float] = None
    offload_threshold: float = 1000.0

    def __post_init__(self) -> None:
        if self.power_limit_w is None:
            self.power_limit_w = self.power_target_w

    @property
    def target_power_w(self) -> float:
        return self.power_target_w

    @target_power_w.setter
    def target_power_w(self, value: float) -> None:
        self.power_target_w = value
        self.power_limit_w = value

    @property
    def target_latency_ms(self) -> float:
        return self.latency_target_ms

    @target_latency_ms.setter
    def target_latency_ms(self, value: float) -> None:
        self.latency_target_ms = value


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
    worker_id: Optional[str] = None
    gpu_id: Optional[int] = None
    is_estimated: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def energy_per_token_j(self) -> float:
        if self.throughput_tokens_per_sec <= 0:
            return float("inf")
        return self.power_w / self.throughput_tokens_per_sec

    @property
    def efficiency_score(self) -> float:
        if self.power_w <= 0 or self.latency_ms <= 0:
            return 0.0
        return self.throughput_tokens_per_sec / (self.power_w * self.latency_ms / 1000.0)

    @property
    def memory_pressure(self) -> float:
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
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "is_estimated": self.is_estimated,
            "notes": dict(self.notes),
        }


@dataclass
class ControlConfig:
    """Runtime configuration controlled by the feedback loop."""

    batch_size: int = 1
    power_limit_w: int = 250
    precision: str = "fp16"  # fp32, fp16, bf16, int8, int4
    offload_enabled: bool = False
    offload_threshold: float = 1000.0

    min_batch_size: int = 1
    max_batch_size: int = 64
    min_power_limit_w: int = 100
    max_power_limit_w: int = 350

    def __post_init__(self) -> None:
        self.batch_size = int(self.batch_size)
        self.power_limit_w = int(self.power_limit_w)
        self.offload_threshold = float(self.offload_threshold)

    @property
    def gpu_power_limit_w(self) -> int:
        return self.power_limit_w

    @gpu_power_limit_w.setter
    def gpu_power_limit_w(self, value: int) -> None:
        self.power_limit_w = int(value)

    @property
    def offload_threshold_mb(self) -> float:
        return self.offload_threshold

    @offload_threshold_mb.setter
    def offload_threshold_mb(self, value: float) -> None:
        self.offload_threshold = float(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "power_limit_w": self.power_limit_w,
            "gpu_power_limit_w": self.power_limit_w,
            "precision": self.precision,
            "offload_enabled": self.offload_enabled,
            "offload_threshold": self.offload_threshold,
            "offload_threshold_mb": self.offload_threshold,
        }


@dataclass
class ControllerState:
    """Internal state of the feedback controller."""

    current_config: ControlConfig = field(default_factory=ControlConfig)
    last_action: ControlAction = ControlAction.NO_ACTION
    action_history: List[Tuple[str, ControlAction]] = field(default_factory=list)
    outcome_history: Dict[str, List[bool]] = field(default_factory=dict)
    stability_since: float = 0.0
    last_action_at: float = 0.0
    adjustments_count: int = 0

    def record_action(self, action: ControlAction) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.action_history.append((timestamp, action))
        self.last_action = action
        self.last_action_at = time.monotonic()
        self.adjustments_count += 1
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

    def record_outcome(self, state_key: str, success: bool) -> None:
        outcomes = self.outcome_history.setdefault(state_key, [])
        outcomes.append(success)
        if len(outcomes) > 50:
            self.outcome_history[state_key] = outcomes[-50:]


class PIDController:
    """PID controller with deadband, rate limiting, and output clamping."""

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.1,
        setpoint: float = 0.0,
        deadband: float = 0.0,
        output_limits: Tuple[float, float] = (-100.0, 100.0),
        rate_limit: float = 10.0,
        smoothing_window: int = 5,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.deadband = abs(deadband)
        self.output_limits = output_limits
        self.rate_limit = abs(rate_limit)
        self.smoothing_window = max(1, smoothing_window)

        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()
        self._last_output = 0.0
        self._measurements: Deque[float] = deque(maxlen=self.smoothing_window)

    def update(self, measured: float) -> float:
        self._measurements.append(measured)
        smoothed = mean(self._measurements)

        now = time.monotonic()
        dt = now - self._last_time
        if dt <= 0:
            dt = 0.001

        error = self.setpoint - smoothed
        if abs(error) <= self.deadband:
            error = 0.0

        p_term = self.kp * error
        self._integral += error * dt
        self._integral = max(-10.0, min(10.0, self._integral))
        i_term = self.ki * self._integral
        d_term = self.kd * (error - self._last_error) / dt

        output = p_term + i_term + d_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        delta = output - self._last_output
        if delta > self.rate_limit:
            output = self._last_output + self.rate_limit
        elif delta < -self.rate_limit:
            output = self._last_output - self.rate_limit

        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self._last_error = error
        self._last_time = now
        self._last_output = output
        return output

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.monotonic()
        self._last_output = 0.0
        self._measurements.clear()


@dataclass
class ControlDecision:
    action: ControlAction
    score: float
    benefit: float
    cost: float
    risk: float
    reason: str
    risk_level: ActionRiskLevel
    predicted_metrics: EnergyMetrics


class EnergyFeedbackController:
    """Closed-loop adaptive energy optimization controller."""

    PRECISION_ORDER = ["fp32", "fp16", "bf16", "int8", "int4"]

    def __init__(
        self,
        config: Optional[EnergyFeedbackConfig] = None,
        target_power_w: float = 200.0,
        target_latency_ms: float = 100.0,
        target_efficiency: Optional[float] = None,
        update_interval_s: float = 5.0,
        power_target_w: Optional[float] = None,
        latency_target_ms: Optional[float] = None,
        update_interval: Optional[float] = None,
    ):
        if config is None:
            config = EnergyFeedbackConfig(
                power_target_w=power_target_w if power_target_w is not None else target_power_w,
                latency_target_ms=latency_target_ms if latency_target_ms is not None else target_latency_ms,
                target_efficiency=target_efficiency,
                update_interval_s=update_interval if update_interval is not None else update_interval_s,
            )
        self.config = config
        self._target_power = config.power_target_w
        self._target_latency = config.latency_target_ms
        self._target_efficiency = config.target_efficiency
        self._update_interval = config.update_interval_s

        self._state = ControllerState()
        self._metrics_history: Deque[EnergyMetrics] = deque(maxlen=200)
        self._decision_history: Deque[ControlDecision] = deque(maxlen=200)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._current_interval = config.update_interval_s

        self._power_pid = PIDController(
            kp=0.02,
            ki=0.005,
            kd=0.01,
            setpoint=self._target_power,
            deadband=config.deadband_power_w,
            output_limits=(config.pid_output_min, config.pid_output_max),
            rate_limit=config.pid_rate_limit,
            smoothing_window=config.smoothing_window,
        )
        self._latency_pid = PIDController(
            kp=0.05,
            ki=0.01,
            kd=0.02,
            setpoint=self._target_latency,
            deadband=config.deadband_latency_ms,
            output_limits=(config.pid_output_min, config.pid_output_max),
            rate_limit=config.pid_rate_limit,
            smoothing_window=config.smoothing_window,
        )

        self._metric_callback: Optional[Callable[[], EnergyMetrics]] = None
        self._config_apply_callback: Optional[Callable[[ControlConfig], None]] = None
        self._scheduler_signal_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_worker_signals: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_metric_callback(self, callback: Callable[[], EnergyMetrics]) -> None:
        self._metric_callback = callback

    def set_config_callback(self, callback: Callable[[ControlConfig], None]) -> None:
        self._config_apply_callback = callback

    def set_scheduler_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._scheduler_signal_callback = callback

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True, name="energy-feedback")
        self._thread.start()
        logger.info("EnergyFeedbackController started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("EnergyFeedbackController stopped")

    def run_forever(self) -> None:
        self.start()
        while self._running:
            time.sleep(0.2)

    def step(self) -> ControlConfig:
        metrics = self._collect_metrics()
        if metrics is None:
            metrics = self._estimate_metrics_without_nvml()
        self._process_metrics(metrics)
        return self.get_optimal_config()

    def get_optimal_config(self) -> ControlConfig:
        with self._lock:
            return self._clone_config(self._state.current_config)

    def get_metrics_history(self, n: int = 100) -> List[EnergyMetrics]:
        with self._lock:
            return list(self._metrics_history)[-n:]

    def get_state(self) -> Dict[str, Any]:
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
                "current_interval_s": self._current_interval,
            }

    def record_metrics(self, metrics: EnergyMetrics) -> None:
        with self._lock:
            self._metrics_history.append(metrics)

    def get_decision_history(self, n: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._decision_to_dict(d) for d in list(self._decision_history)[-n:]]

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        while self._running:
            try:
                metrics = self._collect_metrics()
                if metrics is None:
                    metrics = self._estimate_metrics_without_nvml()
                self._process_metrics(metrics)
            except Exception:
                logger.exception("Energy feedback control loop error")
            time.sleep(self._current_interval)

    def _process_metrics(self, metrics: EnergyMetrics) -> None:
        with self._lock:
            self._metrics_history.append(metrics)

        decision = self._select_best_action(metrics)
        if decision.action != ControlAction.NO_ACTION:
            self._apply_decision(decision, metrics)
        else:
            self._adapt_sampling_rate(metrics, stable=self._is_stable())

        self._emit_scheduler_signals(metrics)

    # ------------------------------------------------------------------
    # Metrics collection and estimation
    # ------------------------------------------------------------------

    def _collect_metrics(self) -> Optional[EnergyMetrics]:
        if self._metric_callback:
            return self._metric_callback()

        try:
            return self._collect_gpu_metrics()
        except Exception:
            return None

    def _collect_gpu_metrics(self) -> EnergyMetrics:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            pynvml.nvmlShutdown()
            return EnergyMetrics(
                power_w=power_w,
                throughput_tokens_per_sec=self._estimate_throughput_from_util(util.gpu),
                latency_ms=self._estimate_latency_from_util(util.gpu),
                batch_size=self._state.current_config.batch_size,
                memory_used_mb=mem_info.used / (1024**2),
                memory_total_mb=mem_info.total / (1024**2),
                gpu_utilization_pct=util.gpu,
                notes={"power_limit_w": power_limit_w},
            )
        except Exception:
            return self._estimate_metrics_without_nvml()

    def _estimate_metrics_without_nvml(self) -> EnergyMetrics:
        util_pct = self._estimate_utilization_without_nvml()
        throughput = self._estimate_throughput_from_util(util_pct)
        latency = self._estimate_latency_from_util(util_pct)
        power = self._estimate_power_without_nvml(util_pct, throughput, latency)
        memory_used, memory_total = self._estimate_memory_without_nvml(util_pct)
        return EnergyMetrics(
            power_w=power,
            throughput_tokens_per_sec=throughput,
            latency_ms=latency,
            batch_size=self._state.current_config.batch_size,
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
            gpu_utilization_pct=util_pct,
            is_estimated=True,
            notes={"source": "heuristic_fallback"},
        )

    def _estimate_utilization_without_nvml(self) -> float:
        if torch.cuda.is_available():
            mem_allocated = 0.0
            mem_reserved = 0.0
            try:
                mem_allocated = torch.cuda.memory_allocated() / max(1.0, torch.cuda.get_device_properties(0).total_memory)
                mem_reserved = torch.cuda.memory_reserved() / max(1.0, torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                pass
            util = 40.0 + 60.0 * max(mem_allocated, mem_reserved)
            return max(5.0, min(98.0, util))
        return 15.0 + min(40.0, self._state.current_config.batch_size * 3.0)

    def _estimate_throughput_from_util(self, util_pct: float) -> float:
        base = 35.0 if torch.cuda.is_available() else 12.0
        batch_factor = 1.0 + (self._state.current_config.batch_size - 1) * 0.12
        return max(0.1, base * (util_pct / 100.0) * batch_factor)

    def _estimate_latency_from_util(self, util_pct: float) -> float:
        base = 120.0 if torch.cuda.is_available() else 220.0
        batch_penalty = 1.0 + (self._state.current_config.batch_size - 1) * 0.08
        if self._state.current_config.offload_enabled:
            batch_penalty *= 1.12
        precision_factor = {"fp32": 1.25, "fp16": 1.0, "bf16": 1.0, "int8": 0.9, "int4": 0.85}.get(
            self._state.current_config.precision,
            1.0,
        )
        return max(1.0, base * batch_penalty * precision_factor * (1.2 - util_pct / 150.0))

    def _estimate_power_without_nvml(self, util_pct: float, throughput: float, latency: float) -> float:
        cfg = self._state.current_config
        workload_factor = 1.0 + (cfg.batch_size - 1) * 0.05
        precision_factor = {"fp32": 1.1, "fp16": 1.0, "bf16": 0.98, "int8": 0.88, "int4": 0.82}.get(
            cfg.precision,
            1.0,
        )
        offload_penalty = 1.06 if cfg.offload_enabled else 1.0
        estimated = self.config.fallback_idle_power_w + (self.config.fallback_max_power_w - self.config.fallback_idle_power_w) * (util_pct / 100.0)
        estimated *= workload_factor * precision_factor * offload_penalty
        estimated *= 1.0 + min(0.15, latency / 1000.0)
        return max(self.config.fallback_idle_power_w, min(self.config.fallback_max_power_w, estimated))

    def _estimate_memory_without_nvml(self, util_pct: float) -> Tuple[float, float]:
        total = 8192.0 if torch.cuda.is_available() else 4096.0
        used = total * min(0.98, 0.25 + util_pct / 130.0 + self._state.current_config.batch_size * 0.03)
        return used, total

    # ------------------------------------------------------------------
    # Prediction / stability / optimization
    # ------------------------------------------------------------------

    def _select_best_action(self, metrics: EnergyMetrics) -> ControlDecision:
        predicted = self._predict_next_metrics(metrics)
        stable = self._is_stable()
        objective = self._compute_objective(metrics)
        predicted_objective = self._compute_objective(predicted)
        load_mode = self._get_load_mode(metrics)

        candidates: List[ControlDecision] = []
        for action in self._candidate_actions(metrics, stable=stable):
            candidate = self._evaluate_action(action, metrics, predicted, objective, predicted_objective, load_mode, stable)
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return ControlDecision(
                action=ControlAction.NO_ACTION,
                score=0.0,
                benefit=0.0,
                cost=0.0,
                risk=0.0,
                reason="no_safe_action_available",
                risk_level=ActionRiskLevel.LOW,
                predicted_metrics=predicted,
            )

        candidates.sort(key=lambda d: (d.score, -self._action_severity_rank(d.action)), reverse=True)
        best = candidates[0]
        logger.info(
            "Energy controller selected action=%s score=%.3f benefit=%.3f cost=%.3f risk=%.3f reason=%s metrics=%s",
            best.action.value,
            best.score,
            best.benefit,
            best.cost,
            best.risk,
            best.reason,
            metrics.to_dict(),
        )
        return best

    def _predict_next_metrics(self, metrics: EnergyMetrics) -> EnergyMetrics:
        history = list(self._metrics_history)[-self.config.smoothing_window :]
        if len(history) < 2:
            return metrics

        def _slope(values: List[float]) -> float:
            n = len(values)
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = sum(values) / n
            denom = sum((x - x_mean) ** 2 for x in xs)
            if denom == 0:
                return 0.0
            return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom

        power_values = [m.power_w for m in history]
        latency_values = [m.latency_ms for m in history]
        throughput_values = [m.throughput_tokens_per_sec for m in history]
        memory_values = [m.memory_used_mb for m in history]
        util_values = [m.gpu_utilization_pct for m in history]

        power_slope = _slope(power_values)
        latency_slope = _slope(latency_values)
        throughput_slope = _slope(throughput_values)
        memory_slope = _slope(memory_values)
        util_slope = _slope(util_values)

        horizon = max(1, self.config.prediction_horizon_steps)
        predicted = EnergyMetrics(
            power_w=max(0.0, metrics.power_w + power_slope * horizon),
            throughput_tokens_per_sec=max(0.0, metrics.throughput_tokens_per_sec + throughput_slope * horizon),
            latency_ms=max(0.0, metrics.latency_ms + latency_slope * horizon),
            batch_size=metrics.batch_size,
            memory_used_mb=max(0.0, metrics.memory_used_mb + memory_slope * horizon),
            memory_total_mb=metrics.memory_total_mb,
            gpu_utilization_pct=max(0.0, min(100.0, metrics.gpu_utilization_pct + util_slope * horizon)),
            worker_id=metrics.worker_id,
            gpu_id=metrics.gpu_id,
            is_estimated=True,
            notes={
                "prediction": True,
                "power_slope": power_slope,
                "latency_slope": latency_slope,
                "throughput_slope": throughput_slope,
                "memory_slope": memory_slope,
                "util_slope": util_slope,
            },
        )
        return predicted

    def _compute_objective(self, metrics: EnergyMetrics) -> float:
        throughput_penalty = max(0.0, self.config.min_throughput_tokens_per_sec - metrics.throughput_tokens_per_sec)
        power_term = self.config.power_weight * metrics.power_w
        latency_term = self.config.latency_weight * metrics.latency_ms
        throughput_term = self.config.throughput_weight * throughput_penalty * 100.0
        if self._get_load_mode(metrics) == "high_load":
            latency_term *= 1.35
            throughput_term *= 1.2
        elif self._get_load_mode(metrics) == "low_load":
            power_term *= 1.2
        return power_term + latency_term + throughput_term

    def _get_load_mode(self, metrics: EnergyMetrics) -> str:
        if metrics.latency_ms > self._target_latency * 1.15 or metrics.gpu_utilization_pct > 85.0:
            return "high_load"
        if metrics.gpu_utilization_pct < 45.0 and metrics.latency_ms < self._target_latency * 0.85:
            return "low_load"
        return "balanced"

    def _candidate_actions(self, metrics: EnergyMetrics, stable: bool) -> List[ControlAction]:
        candidates = [ControlAction.INCREASE_BATCH, ControlAction.DECREASE_BATCH, ControlAction.INCREASE_POWER_LIMIT, ControlAction.DECREASE_POWER_LIMIT]
        if stable:
            candidates.extend([ControlAction.DECREASE_PRECISION, ControlAction.ENABLE_OFFLOAD, ControlAction.DISABLE_OFFLOAD, ControlAction.INCREASE_PRECISION])
        else:
            candidates.extend([ControlAction.ENABLE_OFFLOAD, ControlAction.DECREASE_PRECISION])
        if metrics.memory_pressure > 0.85:
            candidates.append(ControlAction.ENABLE_OFFLOAD)
        if metrics.throughput_tokens_per_sec < self.config.min_throughput_tokens_per_sec:
            candidates = [ControlAction.INCREASE_BATCH, ControlAction.INCREASE_POWER_LIMIT] + candidates
        # Preserve order while removing duplicates.
        seen = set()
        deduped = []
        for action in candidates:
            if action not in seen:
                deduped.append(action)
                seen.add(action)
        return deduped

    def _evaluate_action(
        self,
        action: ControlAction,
        metrics: EnergyMetrics,
        predicted: EnergyMetrics,
        current_objective: float,
        predicted_objective: float,
        load_mode: str,
        stable: bool,
    ) -> Optional[ControlDecision]:
        config = self._state.current_config
        new_metrics = self._simulate_action_metrics(action, metrics, predicted)
        if not self._safety_check(new_metrics):
            return None

        objective_after = self._compute_objective(new_metrics)
        benefit = max(0.0, current_objective - objective_after)
        cost = self._estimate_action_cost(action, metrics, new_metrics, load_mode)
        risk_level = self._risk_level(action)
        risk = self._risk_weight(risk_level) + self._historical_penalty(action, metrics)

        if action in (ControlAction.DECREASE_PRECISION, ControlAction.ENABLE_OFFLOAD) and not stable:
            risk += 0.25
        if action in (ControlAction.INCREASE_BATCH, ControlAction.DECREASE_BATCH) and metrics.latency_ms > self._target_latency * 1.4:
            benefit += 5.0
        if action in (ControlAction.DECREASE_POWER_LIMIT, ControlAction.INCREASE_POWER_LIMIT):
            benefit += max(0.0, metrics.power_w - self._target_power) * 0.1

        score = benefit - cost - risk
        if action == ControlAction.NO_ACTION:
            score = -1.0

        reason = self._build_reason(action, metrics, predicted, load_mode, stable, benefit, cost, risk)
        return ControlDecision(
            action=action,
            score=score,
            benefit=benefit,
            cost=cost,
            risk=risk,
            reason=reason,
            risk_level=risk_level,
            predicted_metrics=new_metrics,
        )

    def _simulate_action_metrics(self, action: ControlAction, metrics: EnergyMetrics, predicted: EnergyMetrics) -> EnergyMetrics:
        cfg = self._clone_config(self._state.current_config)
        if action == ControlAction.INCREASE_BATCH:
            cfg.batch_size = min(cfg.max_batch_size, max(cfg.min_batch_size, cfg.batch_size + 1))
        elif action == ControlAction.DECREASE_BATCH:
            cfg.batch_size = max(cfg.min_batch_size, cfg.batch_size - 1)
        elif action == ControlAction.INCREASE_POWER_LIMIT:
            cfg.power_limit_w = min(cfg.max_power_limit_w, cfg.power_limit_w + 25)
        elif action == ControlAction.DECREASE_POWER_LIMIT:
            cfg.power_limit_w = max(cfg.min_power_limit_w, cfg.power_limit_w - 25)
        elif action == ControlAction.INCREASE_PRECISION:
            idx = max(0, self.PRECISION_ORDER.index(cfg.precision) - 1) if cfg.precision in self.PRECISION_ORDER else 1
            cfg.precision = self.PRECISION_ORDER[idx]
        elif action == ControlAction.DECREASE_PRECISION:
            idx = min(len(self.PRECISION_ORDER) - 1, self.PRECISION_ORDER.index(cfg.precision) + 1) if cfg.precision in self.PRECISION_ORDER else 2
            cfg.precision = self.PRECISION_ORDER[idx]
        elif action == ControlAction.ENABLE_OFFLOAD:
            cfg.offload_enabled = True
        elif action == ControlAction.DISABLE_OFFLOAD:
            cfg.offload_enabled = False

        power = predicted.power_w
        latency = predicted.latency_ms
        throughput = predicted.throughput_tokens_per_sec
        memory = predicted.memory_used_mb
        util = predicted.gpu_utilization_pct

        if action == ControlAction.INCREASE_BATCH:
            latency *= 1.04
            throughput *= 1.08
            power *= 1.03
        elif action == ControlAction.DECREASE_BATCH:
            latency *= 0.96
            throughput *= 0.95
            power *= 0.92
        elif action == ControlAction.INCREASE_POWER_LIMIT:
            latency *= 0.97
            power *= 1.05
        elif action == ControlAction.DECREASE_POWER_LIMIT:
            latency *= 1.06
            power *= 0.90
        elif action == ControlAction.INCREASE_PRECISION:
            latency *= 0.98
            power *= 1.01
            memory *= 1.01
        elif action == ControlAction.DECREASE_PRECISION:
            latency *= 1.03
            power *= 0.92
            memory *= 0.88
        elif action == ControlAction.ENABLE_OFFLOAD:
            latency *= 1.08
            power *= 0.91
            memory *= 0.82
        elif action == ControlAction.DISABLE_OFFLOAD:
            latency *= 0.97
            power *= 1.04
            memory *= 1.08

        if cfg.offload_enabled:
            memory *= 0.92
            latency *= 1.02

        util = max(0.0, min(100.0, util * (memory / max(1.0, metrics.memory_used_mb))))
        return EnergyMetrics(
            timestamp=metrics.timestamp,
            power_w=max(0.0, power),
            throughput_tokens_per_sec=max(0.0, throughput),
            latency_ms=max(0.0, latency),
            batch_size=cfg.batch_size,
            memory_used_mb=max(0.0, memory),
            memory_total_mb=metrics.memory_total_mb,
            gpu_utilization_pct=util,
            worker_id=metrics.worker_id,
            gpu_id=metrics.gpu_id,
            is_estimated=True,
            notes={"action": action.value},
        )

    def _estimate_action_cost(self, action: ControlAction, metrics: EnergyMetrics, new_metrics: EnergyMetrics, load_mode: str) -> float:
        latency_delta = max(0.0, new_metrics.latency_ms - metrics.latency_ms)
        power_delta = max(0.0, new_metrics.power_w - metrics.power_w)
        disruption = 0.0
        if action in (ControlAction.DECREASE_PRECISION, ControlAction.ENABLE_OFFLOAD, ControlAction.DISABLE_OFFLOAD):
            disruption += 8.0
        if action in (ControlAction.INCREASE_BATCH, ControlAction.DECREASE_BATCH):
            disruption += 2.0
        if action in (ControlAction.DECREASE_POWER_LIMIT, ControlAction.INCREASE_POWER_LIMIT):
            disruption += 3.0
        if load_mode == "high_load" and action in (ControlAction.ENABLE_OFFLOAD, ControlAction.DECREASE_PRECISION):
            disruption += 6.0
        return latency_delta * 0.2 + power_delta * 0.05 + disruption

    def _risk_level(self, action: ControlAction) -> ActionRiskLevel:
        if action in (ControlAction.INCREASE_BATCH, ControlAction.DECREASE_BATCH):
            return ActionRiskLevel.LOW
        if action in (ControlAction.INCREASE_POWER_LIMIT, ControlAction.DECREASE_POWER_LIMIT):
            return ActionRiskLevel.MEDIUM
        if action in (ControlAction.INCREASE_PRECISION, ControlAction.DECREASE_PRECISION, ControlAction.ENABLE_OFFLOAD, ControlAction.DISABLE_OFFLOAD):
            return ActionRiskLevel.HIGH
        return ActionRiskLevel.LOW

    def _risk_weight(self, risk_level: ActionRiskLevel) -> float:
        if risk_level == ActionRiskLevel.LOW:
            return self.config.low_risk_weight
        if risk_level == ActionRiskLevel.MEDIUM:
            return self.config.medium_risk_weight
        return self.config.high_risk_weight

    def _historical_penalty(self, action: ControlAction, metrics: EnergyMetrics) -> float:
        state_key = self._state_signature(metrics)
        recent = self._state.outcome_history.get(f"{state_key}:{action.value}", [])
        if not recent:
            return 0.0
        failures = sum(1 for item in recent if not item)
        return min(0.4, failures / max(1, len(recent)) * 0.35)

    def _build_reason(
        self,
        action: ControlAction,
        metrics: EnergyMetrics,
        predicted: EnergyMetrics,
        load_mode: str,
        stable: bool,
        benefit: float,
        cost: float,
        risk: float,
    ) -> str:
        trigger_bits = []
        if metrics.latency_ms > self._target_latency * 1.1 or predicted.latency_ms > self._target_latency:
            trigger_bits.append("latency_pressure")
        if metrics.power_w > self._target_power * 1.1 or predicted.power_w > self._target_power:
            trigger_bits.append("power_pressure")
        if metrics.memory_pressure > self.config.max_memory_pressure:
            trigger_bits.append("memory_pressure")
        if load_mode == "high_load":
            trigger_bits.append("high_load_mode")
        if not stable:
            trigger_bits.append("not_stable")
        if not trigger_bits:
            trigger_bits.append("efficiency_optimization")
        return f"{action.value}: triggers={','.join(trigger_bits)} benefit={benefit:.3f} cost={cost:.3f} risk={risk:.3f}"

    def _apply_decision(self, decision: ControlDecision, metrics: EnergyMetrics) -> None:
        with self._lock:
            prev_config = self._clone_config(self._state.current_config)
            new_config = self._clone_config(self._state.current_config)
            self._apply_action_to_config(new_config, decision.action)

            if not self._safety_check(metrics, config=new_config):
                logger.info("Skipping action %s due to safety guardrails", decision.action.value)
                return

            self._state.current_config = new_config
            self._state.record_action(decision.action)
            self._decision_history.append(decision)
            self._state.record_outcome(
                f"{self._state_signature(metrics)}:{decision.action.value}",
                success=True,
            )
            self._update_stability(metrics)
            self._adapt_sampling_rate(metrics, stable=self._is_stable())

        try:
            if self._config_apply_callback:
                self._config_apply_callback(self.get_optimal_config())
        except Exception:
            logger.exception("Config apply callback failed; rolling back")
            with self._lock:
                self._state.current_config = prev_config
                self._state.record_outcome(
                    f"{self._state_signature(metrics)}:{decision.action.value}",
                    success=False,
                )
            if self._config_apply_callback:
                try:
                    self._config_apply_callback(self.get_optimal_config())
                except Exception:
                    pass
            return

        logger.info("Applied action=%s new_config=%s", decision.action.value, self._state.current_config.to_dict())

    # ------------------------------------------------------------------
    # Stability / adaptation / safety
    # ------------------------------------------------------------------

    def _update_stability(self, metrics: EnergyMetrics) -> None:
        history = list(self._metrics_history)[-self.config.stability_window :]
        if len(history) < self.config.stability_window:
            self._state.stability_since = 0.0
            return

        power_values = [m.power_w for m in history]
        latency_values = [m.latency_ms for m in history]
        power_var = pvariance(power_values) if len(power_values) > 1 else 0.0
        latency_var = pvariance(latency_values) if len(latency_values) > 1 else 0.0
        if power_var < self.config.stability_variance_power and latency_var < self.config.stability_variance_latency:
            if self._state.stability_since == 0.0:
                self._state.stability_since = time.monotonic()
        else:
            self._state.stability_since = 0.0

    def _is_stable(self) -> bool:
        if self._state.stability_since == 0.0:
            return False
        return (time.monotonic() - self._state.stability_since) >= 20.0

    def _adapt_sampling_rate(self, metrics: EnergyMetrics, stable: bool) -> None:
        if not stable or metrics.latency_ms > self._target_latency * 1.15 or metrics.power_w > self._target_power * 1.1:
            self._current_interval = max(self.config.min_update_interval_s, self._current_interval * 0.75)
        else:
            self._current_interval = min(self.config.max_update_interval_s, self._current_interval * 1.15)

    def _safety_check(self, metrics: EnergyMetrics, config: Optional[ControlConfig] = None) -> bool:
        cfg = config or self._state.current_config
        if metrics.latency_ms > self.config.max_latency_ms:
            return False
        if metrics.throughput_tokens_per_sec < self.config.min_throughput_tokens_per_sec:
            return False
        if metrics.memory_pressure > self.config.max_memory_pressure:
            return False
        if cfg.batch_size < cfg.min_batch_size or cfg.batch_size > cfg.max_batch_size:
            return False
        if cfg.power_limit_w < cfg.min_power_limit_w or cfg.power_limit_w > cfg.max_power_limit_w:
            return False
        return True

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _emit_scheduler_signals(self, metrics: EnergyMetrics) -> None:
        if not self._scheduler_signal_callback:
            return

        signals: Dict[str, Any] = {
            "timestamp": metrics.timestamp,
            "worker_id": metrics.worker_id,
            "gpu_id": metrics.gpu_id,
            "overloaded_worker": metrics.worker_id if metrics.latency_ms > self._target_latency * 1.2 or metrics.memory_pressure > self.config.max_memory_pressure else None,
            "inefficient_node": metrics.worker_id if metrics.power_w > self._target_power * 1.1 and metrics.throughput_tokens_per_sec < self.config.min_throughput_tokens_per_sec * 1.2 else None,
            "metrics": metrics.to_dict(),
            "recommended_action": self._state.last_action.value,
        }

        # Avoid chatty duplicate signals for the same worker.
        worker_key = metrics.worker_id or f"gpu-{metrics.gpu_id if metrics.gpu_id is not None else 'local'}"
        last = self._last_worker_signals.get(worker_key, 0.0)
        if time.monotonic() - last < 2.0:
            return
        self._last_worker_signals[worker_key] = time.monotonic()
        try:
            self._scheduler_signal_callback(signals)
        except Exception:
            logger.exception("Scheduler signal callback failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_action_to_config(self, config: ControlConfig, action: ControlAction) -> None:
        if action == ControlAction.INCREASE_BATCH:
            config.batch_size = min(config.max_batch_size, config.batch_size + 1)
        elif action == ControlAction.DECREASE_BATCH:
            config.batch_size = max(config.min_batch_size, config.batch_size - 1)
        elif action == ControlAction.INCREASE_POWER_LIMIT:
            config.power_limit_w = min(config.max_power_limit_w, config.power_limit_w + 25)
        elif action == ControlAction.DECREASE_POWER_LIMIT:
            config.power_limit_w = max(config.min_power_limit_w, config.power_limit_w - 25)
        elif action == ControlAction.INCREASE_PRECISION:
            idx = self.PRECISION_ORDER.index(config.precision) if config.precision in self.PRECISION_ORDER else 1
            config.precision = self.PRECISION_ORDER[max(0, idx - 1)]
        elif action == ControlAction.DECREASE_PRECISION:
            idx = self.PRECISION_ORDER.index(config.precision) if config.precision in self.PRECISION_ORDER else 1
            config.precision = self.PRECISION_ORDER[min(len(self.PRECISION_ORDER) - 1, idx + 1)]
        elif action == ControlAction.ENABLE_OFFLOAD:
            config.offload_enabled = True
        elif action == ControlAction.DISABLE_OFFLOAD:
            config.offload_enabled = False

    def _clone_config(self, config: ControlConfig) -> ControlConfig:
        return ControlConfig(
            batch_size=config.batch_size,
            power_limit_w=config.power_limit_w,
            precision=config.precision,
            offload_enabled=config.offload_enabled,
            offload_threshold=config.offload_threshold,
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size,
            min_power_limit_w=config.min_power_limit_w,
            max_power_limit_w=config.max_power_limit_w,
        )

    def _state_signature(self, metrics: EnergyMetrics) -> str:
        worker = metrics.worker_id or "local"
        gpu = f"gpu{metrics.gpu_id}" if metrics.gpu_id is not None else "gpu-na"
        power_bucket = int(metrics.power_w // 10)
        latency_bucket = int(metrics.latency_ms // 10)
        util_bucket = int(metrics.gpu_utilization_pct // 10)
        return f"{worker}:{gpu}:p{power_bucket}:l{latency_bucket}:u{util_bucket}:b{self._state.current_config.batch_size}:pwr{self._state.current_config.power_limit_w}:prec{self._state.current_config.precision}"

    def _action_severity_rank(self, action: ControlAction) -> int:
        if action in (ControlAction.INCREASE_BATCH, ControlAction.DECREASE_BATCH):
            return 1
        if action in (ControlAction.INCREASE_POWER_LIMIT, ControlAction.DECREASE_POWER_LIMIT):
            return 2
        if action in (ControlAction.INCREASE_PRECISION, ControlAction.DECREASE_PRECISION, ControlAction.ENABLE_OFFLOAD, ControlAction.DISABLE_OFFLOAD):
            return 3
        return 0

    def _decision_to_dict(self, decision: ControlDecision) -> Dict[str, Any]:
        return {
            "action": decision.action.value,
            "score": decision.score,
            "benefit": decision.benefit,
            "cost": decision.cost,
            "risk": decision.risk,
            "reason": decision.reason,
            "risk_level": decision.risk_level.value,
            "predicted_metrics": decision.predicted_metrics.to_dict(),
        }

    # Backward-compatible alias for existing code.
    def get_optimization_config(self) -> Dict[str, Any]:
        return self.get_optimal_config().to_dict()


class PowerLimitManager:
    """Manages GPU power limits via NVML."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._handle = None
        self._initialized = False

    def initialize(self) -> bool:
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
        if not self._initialized:
            return (250, 100, 350)
        try:
            import pynvml

            current = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) // 1000
            limits = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self._handle)
            return (current, limits[0] // 1000, limits[1] // 1000)
        except Exception:
            return (250, 100, 350)

    def set_power_limit(self, limit_w: int) -> bool:
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
        if self._initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


@PluginRegistry.register(
    "optimizer",
    "energy_feedback",
    description="Closed-loop energy optimization",
)
class EnergyFeedbackPlugin(OptimizerPlugin):
    """Plugin wrapper for EnergyFeedbackController."""

    def __init__(
        self,
        target_power_w: float = 200.0,
        target_latency_ms: float = 100.0,
        config: Optional[EnergyFeedbackConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = EnergyFeedbackConfig(
                power_target_w=target_power_w,
                latency_target_ms=target_latency_ms,
            )
        self._controller = EnergyFeedbackController(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "energy_feedback"

    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        self._controller.start()
        self._controller.get_optimal_config()
        return model

    def get_optimization_config(self) -> Dict[str, Any]:
        return self._controller.get_optimal_config().to_dict()

    def shutdown(self) -> None:
        self._controller.stop()
