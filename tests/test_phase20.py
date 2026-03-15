"""
Integration tests for Phase 20: Real-Time Energy Instrumentation & Trigger Framework.

Tests cover:
  - High-frequency GPU sampling (ring buffer, sub-second intervals)
  - TDP auto-detection and tdp_pct metric
  - Trapezoidal energy calculation
  - Event bus (pub/sub, level filtering, thread safety)
  - Power threshold service (OPTIMAL / WARNING / CRITICAL classification)
  - Monitor HTTP service (--sampling-rate, threshold endpoints)
  - CLI --sampling-rate flag

Run with: python -m pytest tests/test_phase20.py -v
"""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest


# ------------------------------------------------------------------
# High-Frequency Sampling & Ring Buffer
# ------------------------------------------------------------------

class TestHighFrequencySampling:
    """GPUMonitor accepts sub-second intervals and uses a ring buffer."""

    def test_gpu_monitor_accepts_100ms_interval(self):
        """GPUMonitor constructor accepts interval=0.1 (100ms)."""
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(interval=0.1)
        assert mon.interval == 0.1

    def test_gpu_monitor_has_ring_buffer(self):
        """GPUMonitor has _ring_buffer attribute."""
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(buffer_size=100)
        assert hasattr(mon, '_ring_buffer')
        assert mon._ring_buffer.maxlen == 100

    def test_gpu_monitor_has_tdp_watts(self):
        """GPUMonitor stores tdp_watts parameter."""
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(tdp_watts=150.0)
        assert mon.get_tdp_watts() == 150.0

    def test_gpu_monitor_spike_callback_param(self):
        """GPUMonitor accepts spike_callback parameter."""
        from monitoring.gpu_monitor import GPUMonitor
        cb = MagicMock()
        mon = GPUMonitor(spike_callback=cb)
        assert mon._spike_callback is cb

    def test_metrics_collector_passes_interval(self):
        """MetricsCollector passes interval to GPUMonitor."""
        from monitoring.metrics import MetricsCollector
        with patch("monitoring.metrics.GPUMonitor") as MockGPU:
            with patch("monitoring.metrics.CPUMonitor"):
                mc = MetricsCollector(interval=0.1, enable_threshold=False)
                MockGPU.assert_called_once()
                call_kwargs = MockGPU.call_args
                assert call_kwargs.kwargs.get("interval", call_kwargs[1].get("interval")) == 0.1

    def test_monitor_service_sampling_rate_arg(self):
        """monitor_service --help shows --sampling-rate."""
        result = subprocess.run(
            [sys.executable, "-m", "monitoring.monitor_service", "--help"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--sampling-rate" in result.stdout


class TestRingBuffer:
    """Ring buffer captures recent samples within bounded memory."""

    def test_ring_buffer_bounded_size(self):
        """Ring buffer does not exceed buffer_size samples."""
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(buffer_size=5)
        # Manually simulate appending samples
        for i in range(10):
            sample = {"power_w": float(i)}
            mon._ring_buffer.append(sample)
        assert len(mon._ring_buffer) == 5

    def test_get_recent_samples_all(self):
        """get_recent_samples(0) returns all buffered samples."""
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(buffer_size=10)
        for i in range(7):
            with mon._lock:
                mon._ring_buffer.append({"power_w": float(i)})
        recent = mon.get_recent_samples(0)
        assert len(recent) == 7

    def test_get_recent_samples_n(self):
        """get_recent_samples(3) returns last 3 samples."""
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(buffer_size=10)
        for i in range(7):
            with mon._lock:
                mon._ring_buffer.append({"power_w": float(i)})
        recent = mon.get_recent_samples(3)
        assert len(recent) == 3
        assert recent[-1]["power_w"] == 6.0


# ------------------------------------------------------------------
# Trapezoidal Energy Calculation
# ------------------------------------------------------------------

class TestTrapezoidalEnergy:
    """MetricsCollector computes energy via trapezoidal integration."""

    def test_trapezoidal_known_values(self):
        """Trapezoidal integration produces correct Wh for known data."""
        from monitoring.metrics import MetricsCollector
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        # 100W for 1 hour = 100 Wh exactly
        samples = [
            {"timestamp": t0.isoformat(), "power_w": 100.0},
            {"timestamp": (t0 + timedelta(hours=1)).isoformat(), "power_w": 100.0},
        ]
        energy = MetricsCollector._compute_energy_trapezoidal(samples)
        assert abs(energy - 100.0) < 0.01

    def test_trapezoidal_linear_ramp(self):
        """Linear ramp from 0W to 100W over 1h = 50 Wh."""
        from monitoring.metrics import MetricsCollector
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        samples = [
            {"timestamp": t0.isoformat(), "power_w": 0.0},
            {"timestamp": (t0 + timedelta(hours=1)).isoformat(), "power_w": 100.0},
        ]
        energy = MetricsCollector._compute_energy_trapezoidal(samples)
        assert abs(energy - 50.0) < 0.01

    def test_trapezoidal_single_sample_returns_zero(self):
        """Falls back to 0.0 with < 2 samples."""
        from monitoring.metrics import MetricsCollector
        energy = MetricsCollector._compute_energy_trapezoidal(
            [{"timestamp": "2025-01-01T00:00:00+00:00", "power_w": 100.0}]
        )
        assert energy == 0.0

    def test_trapezoidal_empty_returns_zero(self):
        """Returns 0.0 for empty sample list."""
        from monitoring.metrics import MetricsCollector
        assert MetricsCollector._compute_energy_trapezoidal([]) == 0.0


# ------------------------------------------------------------------
# Event Bus
# ------------------------------------------------------------------

class TestEventBus:
    """Thread-safe pub/sub for threshold events."""

    def test_event_bus_importable(self):
        """EventBus, ThresholdLevel, ThresholdEvent are importable."""
        from monitoring.event_bus import EventBus, ThresholdLevel, ThresholdEvent
        assert EventBus is not None
        assert ThresholdLevel.OPTIMAL.value == "optimal"

    def test_event_bus_publish_subscribe(self):
        """Publish delivers events to subscribers."""
        from monitoring.event_bus import EventBus, ThresholdEvent, ThresholdLevel
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))
        bus.start()

        event = ThresholdEvent(
            level=ThresholdLevel.CRITICAL,
            node_name="test",
            power_w=150.0,
            tdp_w=200.0,
            tdp_pct=75.0,
        )
        bus.publish(event)
        time.sleep(0.3)
        bus.stop()

        assert len(received) == 1
        assert received[0].level == ThresholdLevel.CRITICAL

    def test_event_bus_level_filter(self):
        """Subscribers with level_filter only receive matching events."""
        from monitoring.event_bus import EventBus, ThresholdEvent, ThresholdLevel
        bus = EventBus()
        critical_events = []
        all_events = []
        bus.subscribe(lambda e: critical_events.append(e), level_filter=ThresholdLevel.CRITICAL)
        bus.subscribe(lambda e: all_events.append(e))
        bus.start()

        bus.publish(ThresholdEvent(
            level=ThresholdLevel.OPTIMAL, node_name="a",
            power_w=50.0, tdp_w=200.0, tdp_pct=25.0,
        ))
        bus.publish(ThresholdEvent(
            level=ThresholdLevel.CRITICAL, node_name="b",
            power_w=180.0, tdp_w=200.0, tdp_pct=90.0,
        ))
        time.sleep(0.3)
        bus.stop()

        assert len(critical_events) == 1
        assert len(all_events) == 2

    def test_event_bus_history(self):
        """get_event_history returns stored events."""
        from monitoring.event_bus import EventBus, ThresholdEvent, ThresholdLevel
        bus = EventBus()
        bus.start()
        for i in range(5):
            bus.publish(ThresholdEvent(
                level=ThresholdLevel.WARNING, node_name=f"n{i}",
                power_w=100.0, tdp_w=200.0, tdp_pct=50.0,
            ))
        time.sleep(0.3)
        bus.stop()

        history = bus.get_event_history(3)
        assert len(history) == 3

    def test_event_bus_threadsafe(self):
        """Concurrent publish/subscribe does not crash."""
        from monitoring.event_bus import EventBus, ThresholdEvent, ThresholdLevel
        bus = EventBus()
        bus.start()
        counter = {"n": 0}
        lock = threading.Lock()

        def on_event(e):
            with lock:
                counter["n"] += 1

        bus.subscribe(on_event)

        def publisher():
            for _ in range(50):
                bus.publish(ThresholdEvent(
                    level=ThresholdLevel.WARNING, node_name="test",
                    power_w=100.0, tdp_w=200.0, tdp_pct=50.0,
                ))

        threads = [threading.Thread(target=publisher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(0.5)
        bus.stop()

        assert counter["n"] == 250

    def test_threshold_event_to_dict(self):
        """ThresholdEvent.to_dict() returns serializable dict."""
        from monitoring.event_bus import ThresholdEvent, ThresholdLevel
        event = ThresholdEvent(
            level=ThresholdLevel.CRITICAL, node_name="test",
            power_w=150.0, tdp_w=200.0, tdp_pct=75.0,
        )
        d = event.to_dict()
        assert d["level"] == "critical"
        assert d["node_name"] == "test"
        assert "timestamp" in d


# ------------------------------------------------------------------
# Power Threshold Service
# ------------------------------------------------------------------

class TestPowerThresholdService:
    """Classification of power levels relative to TDP."""

    def test_threshold_service_importable(self):
        """PowerThresholdService is importable."""
        from monitoring.threshold_service import PowerThresholdService
        assert PowerThresholdService is not None

    def test_classify_optimal(self):
        """Power < 70% TDP classified as OPTIMAL."""
        from monitoring.threshold_service import PowerThresholdService
        from monitoring.event_bus import EventBus, ThresholdLevel
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(tdp_watts=100.0)
        bus = EventBus()
        svc = PowerThresholdService(mon, bus, tdp_watts=100.0)
        assert svc.classify(60.0) == ThresholdLevel.OPTIMAL

    def test_classify_warning(self):
        """Power >= 70%, < 80% TDP classified as WARNING."""
        from monitoring.threshold_service import PowerThresholdService
        from monitoring.event_bus import EventBus, ThresholdLevel
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(tdp_watts=100.0)
        bus = EventBus()
        svc = PowerThresholdService(mon, bus, tdp_watts=100.0)
        assert svc.classify(75.0) == ThresholdLevel.WARNING

    def test_classify_critical(self):
        """Power >= 80% TDP classified as CRITICAL."""
        from monitoring.threshold_service import PowerThresholdService
        from monitoring.event_bus import EventBus, ThresholdLevel
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(tdp_watts=100.0)
        bus = EventBus()
        svc = PowerThresholdService(mon, bus, tdp_watts=100.0)
        assert svc.classify(85.0) == ThresholdLevel.CRITICAL

    def test_classify_boundary_70(self):
        """Exactly 70% TDP is WARNING."""
        from monitoring.threshold_service import PowerThresholdService
        from monitoring.event_bus import EventBus, ThresholdLevel
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(tdp_watts=100.0)
        bus = EventBus()
        svc = PowerThresholdService(mon, bus, tdp_watts=100.0)
        assert svc.classify(70.0) == ThresholdLevel.WARNING

    def test_classify_boundary_80(self):
        """Exactly 80% TDP is CRITICAL."""
        from monitoring.threshold_service import PowerThresholdService
        from monitoring.event_bus import EventBus, ThresholdLevel
        from monitoring.gpu_monitor import GPUMonitor
        mon = GPUMonitor(tdp_watts=100.0)
        bus = EventBus()
        svc = PowerThresholdService(mon, bus, tdp_watts=100.0)
        assert svc.classify(80.0) == ThresholdLevel.CRITICAL


# ------------------------------------------------------------------
# CLI --sampling-rate
# ------------------------------------------------------------------

class TestCLISamplingRate:
    """CLI benchmark subcommand accepts --sampling-rate."""

    def test_benchmark_sampling_rate_flag(self):
        """benchmark --help shows --sampling-rate."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "benchmark", "--help"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "--sampling-rate" in result.stdout
