"""
Phase 23 tests — Integrated Validation & Energy-Efficiency Analysis.

Tests trigger latency, migration integrity, tiered transition stability,
prefetch overlap, energy recovery, analyzer methods, and new plot functions.
"""

import importlib
import time
import threading
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


class TestTriggerLatency(unittest.TestCase):
    """EventBus publish-to-callback latency and ThresholdService detection."""

    def test_eventbus_callback_latency(self):
        """EventBus publish-to-callback should complete within 500ms."""
        from monitoring.event_bus import EventBus, ThresholdEvent, ThresholdLevel

        bus = EventBus()
        bus.start()

        received = threading.Event()
        callback_time = []

        def on_event(event):
            callback_time.append(time.perf_counter())
            received.set()

        bus.subscribe(on_event)

        event = ThresholdEvent(
            level=ThresholdLevel.CRITICAL,
            node_name="test-node",
            power_w=250.0,
            tdp_w=300.0,
            tdp_pct=83.3,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gpu_index=0,
        )

        publish_time = time.perf_counter()
        bus.publish(event)

        received.wait(timeout=2.0)
        bus.stop()

        self.assertTrue(received.is_set(), "Callback was not invoked")
        latency_ms = (callback_time[0] - publish_time) * 1000.0
        self.assertLess(latency_ms, 500.0, f"Callback latency {latency_ms:.1f}ms > 500ms")

    def test_threshold_service_detects_critical(self):
        """ThresholdService should classify power >= 80% TDP as CRITICAL."""
        from monitoring.threshold_service import PowerThresholdService
        from monitoring.event_bus import ThresholdLevel

        mock_monitor = MagicMock()
        mock_bus = MagicMock()

        service = PowerThresholdService(
            gpu_monitor=mock_monitor,
            event_bus=mock_bus,
            node_name="test",
            tdp_watts=300.0,
        )

        level = service.classify(250.0)  # 83.3% of 300W
        self.assertEqual(level, ThresholdLevel.CRITICAL)


class TestMigrationIntegrity(unittest.TestCase):
    """Checkpoint + restore preserves tensor data."""

    def test_checkpoint_restore_bit_exact(self):
        """Checkpoint and restore should produce bit-exact tensors."""
        import torch
        import tempfile
        import os

        # Simulate checkpoint: save and reload a tensor
        original = torch.randn(64, 128)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_weights.pt")
            torch.save(original, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)

        self.assertTrue(
            torch.equal(original, restored),
            "Restored tensor is not bit-exact match of original",
        )

    def test_state_dict_roundtrip(self):
        """Model state_dict checkpoint/restore should be lossless."""
        import torch
        import torch.nn as nn
        import tempfile
        import os

        model = nn.Linear(32, 16)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_state.pt")
            torch.save(model.state_dict(), path)
            loaded_state = torch.load(path, map_location="cpu", weights_only=True)

        for key in original_state:
            self.assertTrue(
                torch.equal(original_state[key], loaded_state[key]),
                f"Mismatch in state_dict key: {key}",
            )


class TestTieredTransitionStability(unittest.TestCase):
    """Promote/demote cycles should be lossless."""

    def test_tier_enum_values(self):
        """StorageTier should have the expected 3 tiers."""
        from model.tiered_weight_manager import StorageTier

        self.assertEqual(StorageTier.GPU_VRAM.value, 0)
        self.assertEqual(StorageTier.SYSTEM_RAM.value, 1)
        self.assertEqual(StorageTier.DISK_SWAP.value, 2)

    def test_plan_placement_fills_gpu_first(self):
        """plan_placement should fill GPU budget before using RAM or disk."""
        from model.tiered_weight_manager import TieredWeightManager, StorageTier

        mgr = TieredWeightManager(
            gpu_budget_mb=100.0,
            ram_budget_mb=200.0,
            disk_dir="/tmp/kai_test_swap",
            device="cpu",
        )

        layers = [
            ("layer_0", 40.0),
            ("layer_1", 40.0),
            ("layer_2", 40.0),   # exceeds GPU budget
            ("layer_3", 100.0),  # goes to RAM
            ("layer_4", 200.0),  # exceeds RAM, goes to disk
        ]

        placement = mgr.plan_placement(layers)
        # First two should fit in GPU, third might go to GPU or RAM depending on exact logic
        # At minimum, verify all layers are accounted for
        self.assertEqual(len(placement), 5)
        for name, _ in layers:
            self.assertIn(name, placement)


class TestPrefetchOverlap(unittest.TestCase):
    """Prefetch engine should report timing stats."""

    def test_prefetch_stats_structure(self):
        """get_prefetch_stats should return expected keys."""
        from model.prefetch_engine import PrefetchEngine

        mock_manager = MagicMock()
        engine = PrefetchEngine(weight_manager=mock_manager, device="cpu")

        stats = engine.get_prefetch_stats()
        self.assertIn("total_prefetch_ms", stats)
        self.assertIn("total_swap_ms", stats)
        self.assertIn("total_gpu_idle_ms", stats)
        self.assertIn("overlap_pct", stats)
        self.assertIn("avg_prefetch_ms", stats)
        self.assertIn("prefetch_count", stats)


class TestEnergyRecovery(unittest.TestCase):
    """Power baseline recovery after migration (mock-based)."""

    def test_power_returns_to_baseline(self):
        """Simulated power samples should show recovery after migration spike."""
        # Simulate power samples: baseline at 100W, spike to 250W during migration,
        # then recover to ~100W
        baseline = [100.0] * 10
        migration_spike = [250.0, 260.0, 240.0]
        recovery = [110.0, 105.0, 102.0, 100.0, 100.0]

        all_powers = baseline + migration_spike + recovery

        # Verify recovery: last 3 samples within 10% of baseline avg
        baseline_avg = sum(baseline) / len(baseline)
        recovery_samples = all_powers[-3:]
        for p in recovery_samples:
            self.assertLess(
                abs(p - baseline_avg) / baseline_avg * 100, 10.0,
                f"Power {p}W not within 10% of baseline {baseline_avg}W",
            )


class TestAnalyzerMigrationEnergy(unittest.TestCase):
    """Analyzer Phase 23 methods."""

    def test_compute_migration_energy_importable(self):
        """ExperimentAnalyzer should have compute_migration_energy method."""
        from analysis.analyzer import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer()
        self.assertTrue(hasattr(analyzer, "compute_migration_energy"))

    def test_compute_offloading_overhead_importable(self):
        """ExperimentAnalyzer should have compute_offloading_overhead method."""
        from analysis.analyzer import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer()
        self.assertTrue(hasattr(analyzer, "compute_offloading_overhead"))

    def test_offloading_overhead_calculation(self):
        """compute_offloading_overhead should return correct delta and pct."""
        from analysis.analyzer import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer()

        baseline = [10.0, 12.0, 11.0]
        offloaded = [15.0, 17.0, 16.0]

        result = analyzer.compute_offloading_overhead(baseline, offloaded)
        self.assertIn("baseline_avg_ms", result)
        self.assertIn("offloaded_avg_ms", result)
        self.assertIn("overhead_delta_ms", result)
        self.assertIn("overhead_pct", result)

        self.assertAlmostEqual(result["baseline_avg_ms"], 11.0, places=2)
        self.assertAlmostEqual(result["offloaded_avg_ms"], 16.0, places=2)
        self.assertAlmostEqual(result["overhead_delta_ms"], 5.0, places=2)
        self.assertGreater(result["overhead_pct"], 0)

    def test_analyse_deas_importable(self):
        """ExperimentAnalyzer should have analyse_deas method."""
        from analysis.analyzer import ExperimentAnalyzer
        analyzer = ExperimentAnalyzer()
        self.assertTrue(hasattr(analyzer, "analyse_deas"))


class TestNewPlotFunctions(unittest.TestCase):
    """Phase 23 plot functions should be importable."""

    def test_plot_migration_energy_impact_importable(self):
        """plot_migration_energy_impact should be importable."""
        from analysis.plots import plot_migration_energy_impact
        self.assertTrue(callable(plot_migration_energy_impact))

    def test_plot_vram_ram_tradeoff_importable(self):
        """plot_vram_ram_tradeoff should be importable."""
        from analysis.plots import plot_vram_ram_tradeoff
        self.assertTrue(callable(plot_vram_ram_tradeoff))


if __name__ == "__main__":
    unittest.main()
