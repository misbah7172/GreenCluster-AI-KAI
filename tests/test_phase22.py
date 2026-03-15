"""
Tests for Phase 22: FlexGen-Style CPU/Disk Offloading.

Tests cover:
  - TieredWeightManager: StorageTier enum, plan_placement GPU-first fill,
    capacity tracking, tier introspection
  - PrefetchEngine: double-buffered prefetching API surface
  - Offloaded generation: CLI --offload flag, DistributedGenerator kwargs,
    _forward_all_chunks_offloaded method

Run with: python -m pytest tests/test_phase22.py -v
"""

import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch


# ===================================================================
# TieredWeightManager
# ===================================================================

class TestTieredWeightManager(unittest.TestCase):
    """TieredWeightManager places layers across GPU, RAM, and disk."""

    def test_importable(self):
        """TieredWeightManager is importable from model.tiered_weight_manager."""
        from model.tiered_weight_manager import TieredWeightManager
        self.assertIsNotNone(TieredWeightManager)

    def test_storage_tier_enum_values(self):
        """StorageTier enum has GPU_VRAM=0, SYSTEM_RAM=1, DISK_SWAP=2."""
        from model.tiered_weight_manager import StorageTier
        self.assertEqual(StorageTier.GPU_VRAM.value, 0)
        self.assertEqual(StorageTier.SYSTEM_RAM.value, 1)
        self.assertEqual(StorageTier.DISK_SWAP.value, 2)

    def test_plan_placement_fills_gpu_first(self):
        """plan_placement assigns layers to GPU VRAM before RAM or disk."""
        from model.tiered_weight_manager import TieredWeightManager, StorageTier

        mgr = TieredWeightManager(
            gpu_budget_mb=500.0,
            ram_budget_mb=1000.0,
            disk_dir="/tmp/kai_test_swap",
        )

        layers = [
            ("layer_0", 200.0),
            ("layer_1", 200.0),
            ("layer_2", 200.0),  # exceeds 500 MB GPU budget
            ("layer_3", 600.0),  # exceeds remaining RAM
        ]
        placements = mgr.plan_placement(layers)

        self.assertEqual(placements["layer_0"].current_tier, StorageTier.GPU_VRAM)
        self.assertEqual(placements["layer_1"].current_tier, StorageTier.GPU_VRAM)
        # layer_2 should spill to RAM (100 MB GPU free < 200 MB needed)
        self.assertEqual(placements["layer_2"].current_tier, StorageTier.SYSTEM_RAM)
        # layer_3 needs 600 MB but only 800 MB RAM remains — fits in RAM
        self.assertEqual(placements["layer_3"].current_tier, StorageTier.SYSTEM_RAM)

    def test_plan_placement_spills_to_disk(self):
        """Layers exceeding both GPU and RAM budgets go to DISK_SWAP."""
        from model.tiered_weight_manager import TieredWeightManager, StorageTier

        mgr = TieredWeightManager(
            gpu_budget_mb=100.0,
            ram_budget_mb=100.0,
            disk_dir="/tmp/kai_test_swap",
        )

        layers = [
            ("layer_0", 80.0),   # GPU
            ("layer_1", 80.0),   # RAM (only 20 MB GPU left)
            ("layer_2", 80.0),   # Disk (only 20 MB RAM left)
        ]
        placements = mgr.plan_placement(layers)

        self.assertEqual(placements["layer_0"].current_tier, StorageTier.GPU_VRAM)
        self.assertEqual(placements["layer_1"].current_tier, StorageTier.SYSTEM_RAM)
        self.assertEqual(placements["layer_2"].current_tier, StorageTier.DISK_SWAP)
        self.assertIsNotNone(placements["layer_2"].disk_path)

    def test_capacity_tracking_after_plan(self):
        """Tier used_mb and free_mb update after plan_placement."""
        from model.tiered_weight_manager import TieredWeightManager, StorageTier

        mgr = TieredWeightManager(
            gpu_budget_mb=300.0,
            ram_budget_mb=500.0,
            disk_dir="/tmp/kai_test_swap",
        )
        mgr.plan_placement([("a", 100.0), ("b", 150.0)])

        gpu_cap = mgr._tiers[StorageTier.GPU_VRAM]
        self.assertAlmostEqual(gpu_cap.used_mb, 250.0)
        self.assertAlmostEqual(gpu_cap.free_mb, 50.0)

    def test_get_tier_stats_returns_list(self):
        """get_tier_stats returns a list with one entry per tier."""
        from model.tiered_weight_manager import TieredWeightManager

        mgr = TieredWeightManager(
            gpu_budget_mb=100.0,
            ram_budget_mb=100.0,
            disk_dir="/tmp/kai_test_swap",
        )
        stats = mgr.get_tier_stats()
        self.assertIsInstance(stats, list)
        self.assertEqual(len(stats), 3)
        tier_names = {s["tier"] for s in stats}
        self.assertIn("GPU_VRAM", tier_names)
        self.assertIn("SYSTEM_RAM", tier_names)
        self.assertIn("DISK_SWAP", tier_names)


# ===================================================================
# PrefetchEngine
# ===================================================================

class TestPrefetchEngine(unittest.TestCase):
    """PrefetchEngine provides double-buffered async weight loading."""

    def test_importable(self):
        """PrefetchEngine is importable from model.prefetch_engine."""
        from model.prefetch_engine import PrefetchEngine
        self.assertIsNotNone(PrefetchEngine)

    def test_has_prefetch_layer_method(self):
        """PrefetchEngine instances expose a prefetch_layer method."""
        from model.prefetch_engine import PrefetchEngine
        mock_wm = MagicMock()
        engine = PrefetchEngine(weight_manager=mock_wm, device="cpu")
        self.assertTrue(callable(getattr(engine, "prefetch_layer", None)))

    def test_has_wait_and_swap_method(self):
        """PrefetchEngine instances expose a wait_and_swap method."""
        from model.prefetch_engine import PrefetchEngine
        mock_wm = MagicMock()
        engine = PrefetchEngine(weight_manager=mock_wm, device="cpu")
        self.assertTrue(callable(getattr(engine, "wait_and_swap", None)))

    def test_has_get_active_buffer_method(self):
        """PrefetchEngine instances expose a get_active_buffer method."""
        from model.prefetch_engine import PrefetchEngine
        mock_wm = MagicMock()
        engine = PrefetchEngine(weight_manager=mock_wm, device="cpu")
        self.assertTrue(callable(getattr(engine, "get_active_buffer", None)))

    def test_get_prefetch_stats_returns_dict(self):
        """get_prefetch_stats returns a dict with expected keys."""
        from model.prefetch_engine import PrefetchEngine
        mock_wm = MagicMock()
        engine = PrefetchEngine(weight_manager=mock_wm, device="cpu")
        stats = engine.get_prefetch_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_prefetch_ms", stats)
        self.assertIn("prefetch_count", stats)
        self.assertIn("overlap_pct", stats)


# ===================================================================
# Offloaded Generation (CLI + DistributedGenerator)
# ===================================================================

class TestOffloadedGeneration(unittest.TestCase):
    """CLI --offload flag and DistributedGenerator offload integration."""

    def test_cli_shows_offload_flag(self):
        """CLI run --help lists the --offload option."""
        result = subprocess.run(
            [sys.executable, "kai_cli.py", "run", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--offload", result.stdout)

    def test_distributed_generator_accepts_offload_kwargs(self):
        """DistributedGenerator.__init__ accepts prefetch_engine and weight_manager."""
        from model.generation import DistributedGenerator

        mock_chunk = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_pe = MagicMock()
        mock_wm = MagicMock()

        gen = DistributedGenerator(
            chunks=[mock_chunk],
            tokenizer=mock_tokenizer,
            device="cpu",
            prefetch_engine=mock_pe,
            weight_manager=mock_wm,
        )
        self.assertIs(gen._prefetch_engine, mock_pe)
        self.assertIs(gen._weight_manager, mock_wm)

    def test_forward_all_chunks_offloaded_method_exists(self):
        """DistributedGenerator has _forward_all_chunks_offloaded method."""
        from model.generation import DistributedGenerator
        self.assertTrue(
            callable(getattr(DistributedGenerator, "_forward_all_chunks_offloaded", None))
        )


if __name__ == "__main__":
    unittest.main()
