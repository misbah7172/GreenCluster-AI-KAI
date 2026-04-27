"""
Unit tests for Phase 21: Dynamic Energy-Aware Scheduling & Live Migration.

Tests cover:
  - Energy-Efficiency Ratio (EER) in MetricsCollector.compute_summary
  - DEASScheduler cluster EER, rebalance logic, cooldown, and migration planning
  - InferenceServicer Pause / Checkpoint / Resume RPCs and paused flag
  - InferenceGateway relinking, topology, and GatewayHTTPHandler POST /relink

Run with: python -m pytest tests/test_phase21.py -v
"""

import io
import json
import time
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# TestEER -- energy_efficiency_ratio in MetricsCollector.compute_summary
# ---------------------------------------------------------------------------

class TestEER(unittest.TestCase):
    """MetricsCollector.compute_summary includes energy_efficiency_ratio."""

    def _make_collector(self):
        """Create a MetricsCollector with GPU/CPU monitors disabled."""
        from monitoring.metrics import MetricsCollector
        with patch("monitoring.metrics.GPUMonitor"), \
             patch("monitoring.metrics.CPUMonitor"):
            mc = MetricsCollector(
                enable_gpu=False,
                enable_cpu=False,
                enable_threshold=False,
            )
        return mc

    def test_summary_contains_eer_key(self):
        """compute_summary dict has 'energy_efficiency_ratio' key."""
        mc = self._make_collector()
        summary = mc.compute_summary()
        self.assertIn("energy_efficiency_ratio", summary)

    def test_eer_zero_when_no_data(self):
        """EER is 0.0 when there are no GPU samples or inferences."""
        mc = self._make_collector()
        summary = mc.compute_summary()
        self.assertEqual(summary["energy_efficiency_ratio"], 0.0)

    def test_eer_equals_throughput_over_power(self):
        """EER = throughput / avg_power when both are positive."""
        from monitoring.metrics import MetricsCollector
        from datetime import datetime, timezone, timedelta

        with patch("monitoring.metrics.GPUMonitor"), \
             patch("monitoring.metrics.CPUMonitor"):
            mc = MetricsCollector(
                enable_gpu=False,
                enable_cpu=False,
                enable_threshold=False,
            )

        # Simulate 10 inferences in 2 seconds -> throughput = 5 inf/s
        mc._start_time = 0.0
        mc._end_time = 2.0
        for _ in range(10):
            mc.record_latency(100.0)

        # Fake GPU samples so avg_power_w is 200 W
        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        fake_samples = [
            {
                "timestamp": t0.isoformat(),
                "power_w": 200.0,
                "gpu_utilization_pct": 80.0,
                "memory_used_mb": 1000.0,
                "temperature_c": 60.0,
            },
            {
                "timestamp": (t0 + timedelta(seconds=2)).isoformat(),
                "power_w": 200.0,
                "gpu_utilization_pct": 80.0,
                "memory_used_mb": 1000.0,
                "temperature_c": 60.0,
            },
        ]
        mc.get_gpu_samples = lambda: fake_samples

        summary = mc.compute_summary()
        expected_eer = 5.0 / 200.0  # 0.025
        self.assertAlmostEqual(summary["energy_efficiency_ratio"], expected_eer, places=4)


# ---------------------------------------------------------------------------
# TestDEASScheduler -- import, EER calc, rebalance, cooldown, migration plan
# ---------------------------------------------------------------------------

class TestDEASScheduler(unittest.TestCase):
    """DEASScheduler from model.deas_scheduler."""

    def test_importable(self):
        """DEASScheduler, MigrationPlan, NodeEnergyProfile are importable."""
        from model.deas_scheduler import (
            DEASScheduler, MigrationPlan, NodeEnergyProfile, MigrationState,
        )
        self.assertIsNotNone(DEASScheduler)
        self.assertIsNotNone(MigrationPlan)
        self.assertIsNotNone(NodeEnergyProfile)
        self.assertIsNotNone(MigrationState)

    def _make_scheduler(self, cooldown_s=30.0):
        bus = MagicMock()
        partitioner = MagicMock()
        from model.deas_scheduler import DEASScheduler
        return DEASScheduler(
            event_bus=bus,
            auto_partitioner=partitioner,
            controller=None,
            cooldown_s=cooldown_s,
        )

    def _profile(self, name, power, throughput, level, chunks=None, memory=4096.0):
        from model.deas_scheduler import NodeEnergyProfile
        eer = throughput / power if power > 0 else 0.0
        return NodeEnergyProfile(
            node_name=name,
            avg_power_w=power,
            throughput_inf_per_sec=throughput,
            eer=eer,
            current_chunks=chunks if chunks is not None else [],
            threshold_level=level,
            usable_memory_mb=memory,
        )

    def test_compute_cluster_eer_correct(self):
        """compute_cluster_eer = sum(throughput) / sum(power)."""
        from model.deas_scheduler import DEASScheduler
        profiles = [
            self._profile("a", 100.0, 10.0, "optimal"),
            self._profile("b", 200.0, 30.0, "warning"),
        ]
        eer = DEASScheduler.compute_cluster_eer(profiles)
        # (10 + 30) / (100 + 200) = 40 / 300
        self.assertAlmostEqual(eer, 40.0 / 300.0, places=6)

    def test_compute_cluster_eer_zero_power(self):
        """compute_cluster_eer returns 0.0 when total power is zero."""
        from model.deas_scheduler import DEASScheduler
        profiles = [
            self._profile("a", 0.0, 0.0, "optimal"),
        ]
        self.assertEqual(DEASScheduler.compute_cluster_eer(profiles), 0.0)

    def test_should_rebalance_detects_critical(self):
        """should_rebalance returns True when a node is CRITICAL."""
        sched = self._make_scheduler()
        profiles = [
            self._profile("a", 100.0, 10.0, "optimal"),
            self._profile("b", 250.0, 5.0, "critical"),
        ]
        self.assertTrue(sched.should_rebalance(profiles))

    def test_should_rebalance_false_without_critical(self):
        """should_rebalance returns False when no node is CRITICAL."""
        sched = self._make_scheduler()
        profiles = [
            self._profile("a", 100.0, 10.0, "optimal"),
            self._profile("b", 150.0, 12.0, "warning"),
        ]
        self.assertFalse(sched.should_rebalance(profiles))

    def test_cooldown_prevents_rapid_triggers(self):
        """After a rebalance, _on_critical_event is suppressed during cooldown."""
        sched = self._make_scheduler(cooldown_s=60.0)
        # Simulate a recent rebalance by setting _last_rebalance_time
        sched._last_rebalance_time = time.monotonic()
        event = MagicMock()
        event.node_name = "hot-node"
        event.power_w = 300.0
        event.tdp_pct = 95.0

        # The handler should skip because cooldown has not elapsed
        sched._on_critical_event(event)
        from model.deas_scheduler import MigrationState
        self.assertEqual(sched.state, MigrationState.IDLE)

    def test_plan_migration_returns_migration_plans(self):
        """plan_migration returns a list of MigrationPlan objects."""
        from model.deas_scheduler import MigrationPlan
        sched = self._make_scheduler()
        profiles = [
            self._profile("crit-node", 300.0, 5.0, "critical", chunks=[0, 1]),
            self._profile("cool-node", 80.0, 15.0, "optimal", memory=8192.0),
        ]
        plans = sched.plan_migration(profiles)
        self.assertGreater(len(plans), 0)
        for plan in plans:
            self.assertIsInstance(plan, MigrationPlan)
            self.assertEqual(plan.source_node, "crit-node")
            self.assertEqual(plan.target_node, "cool-node")
            self.assertEqual(plan.reason, "critical_threshold")

    def test_scheduler_signal_triggers_rebalance_via_controller(self):
        """Energy controller signal should trigger controller.trigger_rebalance."""
        bus = MagicMock()
        partitioner = MagicMock()
        controller = MagicMock()
        controller.trigger_rebalance.return_value = {"rebalanced": True}

        from model.deas_scheduler import DEASScheduler, MigrationState
        sched = DEASScheduler(
            event_bus=bus,
            auto_partitioner=partitioner,
            controller=controller,
            cooldown_s=0.0,
        )

        signal = {
            "overloaded_worker": "node-a",
            "metrics": {
                "power_w": 260.0,
                "latency_ms": 210.0,
                "throughput_tokens_per_sec": 1.5,
            },
        }
        sched.handle_scheduler_signal(signal)

        controller.trigger_rebalance.assert_called_once()
        self.assertEqual(sched.state, MigrationState.IDLE)

    def test_scheduler_signal_respects_cooldown(self):
        """Scheduler signals should be ignored while DEAS cooldown is active."""
        bus = MagicMock()
        partitioner = MagicMock()
        controller = MagicMock()

        from model.deas_scheduler import DEASScheduler
        sched = DEASScheduler(
            event_bus=bus,
            auto_partitioner=partitioner,
            controller=controller,
            cooldown_s=60.0,
        )
        sched._last_rebalance_time = time.monotonic()

        signal = {
            "inefficient_node": "node-b",
            "metrics": {
                "power_w": 280.0,
                "latency_ms": 220.0,
                "throughput_tokens_per_sec": 1.0,
            },
        }
        sched.handle_scheduler_signal(signal)

        controller.trigger_rebalance.assert_not_called()

    def test_bind_energy_controller_registers_callback(self):
        """bind_energy_controller should register DEAS callback on controller."""
        bus = MagicMock()
        partitioner = MagicMock()
        from model.deas_scheduler import DEASScheduler

        sched = DEASScheduler(
            event_bus=bus,
            auto_partitioner=partitioner,
            controller=None,
            cooldown_s=30.0,
        )

        energy_controller = MagicMock()
        sched.bind_energy_controller(energy_controller)

        energy_controller.set_scheduler_callback.assert_called_once()
        cb = energy_controller.set_scheduler_callback.call_args[0][0]
        self.assertTrue(callable(cb))

    def test_k8s_controller_start_deas_binds_energy_controller(self):
        """KAIController.start_deas wires energy controller callback into DEAS."""
        module_path = Path(__file__).resolve().parents[1] / "kubernetes" / "controller.py"
        spec = importlib.util.spec_from_file_location("kai_k8s_controller", str(module_path))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        KAIController = mod.KAIController

        # Bypass KAIController.__init__ (kube config) for unit testing.
        ctrl = object.__new__(KAIController)
        event_bus = MagicMock()
        partitioner = MagicMock()
        energy_controller = MagicMock()

        with patch("model.deas_scheduler.DEASScheduler") as mock_deas_cls:
            deas_instance = MagicMock()
            mock_deas_cls.return_value = deas_instance

            KAIController.start_deas(
                ctrl,
                event_bus=event_bus,
                auto_partitioner=partitioner,
                cooldown_s=5.0,
                energy_controller=energy_controller,
            )

            deas_instance.bind_energy_controller.assert_called_once_with(energy_controller)
            deas_instance.start.assert_called_once()


# ---------------------------------------------------------------------------
# TestMigrationRPCs -- Pause / Checkpoint / Resume on InferenceServicer
# ---------------------------------------------------------------------------

class TestMigrationRPCs(unittest.TestCase):
    """InferenceServicer from model.chunk_server exposes migration RPCs."""

    def _make_servicer(self):
        """Instantiate InferenceServicer with all heavy deps mocked out."""
        import sys

        # Pre-mock proto modules that use bare imports (generated proto code
        # uses ``import inference_pb2`` rather than ``from proto import ...``)
        mock_pb2 = MagicMock()
        mock_pb2_grpc = MagicMock()
        # The base servicer class must behave like a normal class
        mock_pb2_grpc.InferenceServiceServicer = type("InferenceServiceServicer", (), {})

        saved = {}
        mock_targets = {
            "inference_pb2": mock_pb2,
            "proto.inference_pb2": mock_pb2,
            "proto.inference_pb2_grpc": mock_pb2_grpc,
        }
        for mod_name, fake in mock_targets.items():
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = fake

        # Remove cached model.chunk_server so it re-imports with our mocks
        sys.modules.pop("model.chunk_server", None)

        try:
            import model.chunk_server as cs

            # Patch _build_model to avoid importing real model code
            original_build = cs._build_model
            fake_model = MagicMock()
            cs._build_model = lambda *a, **kw: fake_model

            # Patch both the original module AND the imported name in chunk_server
            import model.chunker
            original_chunk = model.chunker.chunk_model
            fake_chunk = MagicMock()
            fake_chunk.layer_names = ["layer0"]
            fake_chunk.to.return_value = fake_chunk
            fake_chunk.eval.return_value = None
            mock_chunk_fn = lambda *a, **kw: [fake_chunk, fake_chunk]
            model.chunker.chunk_model = mock_chunk_fn
            cs.chunk_model = mock_chunk_fn

            servicer = cs.InferenceServicer(
                chunk_id=0,
                num_chunks=2,
                model_type="transformer",
                weights_dir="",
                device="cpu",
            )

            cs._build_model = original_build
            model.chunker.chunk_model = original_chunk
            return servicer
        finally:
            for mod_name, original in saved.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original

    def test_has_pause_method(self):
        """InferenceServicer has a Pause method."""
        servicer = self._make_servicer()
        self.assertTrue(callable(getattr(servicer, "Pause", None)))

    def test_has_checkpoint_method(self):
        """InferenceServicer has a Checkpoint method."""
        servicer = self._make_servicer()
        self.assertTrue(callable(getattr(servicer, "Checkpoint", None)))

    def test_has_resume_method(self):
        """InferenceServicer has a Resume method."""
        servicer = self._make_servicer()
        self.assertTrue(callable(getattr(servicer, "Resume", None)))

    def test_pause_sets_paused_flag(self):
        """Calling Pause sets _paused to True."""
        servicer = self._make_servicer()
        self.assertFalse(servicer._paused)

        request = MagicMock()
        context = MagicMock()
        servicer.Pause(request, context)

        self.assertTrue(servicer._paused)

    def test_paused_flag_blocks_inference(self):
        """When _paused is True, Infer aborts with UNAVAILABLE."""
        servicer = self._make_servicer()
        servicer._paused = True

        request = MagicMock()
        context = MagicMock()
        # gRPC context.abort() raises, so simulate that
        context.abort.side_effect = Exception("aborted")

        with self.assertRaises(Exception):
            servicer.Infer(request, context)
        context.abort.assert_called_once()
        # First argument should reference paused state
        args = context.abort.call_args
        self.assertIn("paused", str(args).lower())


# ---------------------------------------------------------------------------
# TestGatewayRelinking -- update_chunk_host, topology, POST /relink
# ---------------------------------------------------------------------------

class TestGatewayRelinking(unittest.TestCase):
    """InferenceGateway live-relinking and GatewayHTTPHandler /relink."""

    @classmethod
    def _ensure_gateway_importable(cls):
        """Pre-mock proto modules so model.gateway can be imported."""
        import sys
        for mod_name in ("inference_pb2", "proto.inference_pb2",
                         "proto.inference_pb2_grpc"):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = MagicMock()
        # Ensure model.gateway is freshly imported with mocks
        if "model.gateway" not in sys.modules:
            pass  # will be imported naturally

    def _make_gateway(self, hosts=None):
        """Create an InferenceGateway with gRPC mocked."""
        import sys
        if hosts is None:
            hosts = ["node-0:50051", "node-1:50052"]

        self._ensure_gateway_importable()

        with patch("model.gateway.grpc") as mock_grpc, \
             patch("model.gateway.inference_pb2_grpc") as mock_pb2_grpc:
            mock_grpc.insecure_channel.return_value = MagicMock()
            mock_pb2_grpc.InferenceServiceStub.return_value = MagicMock()
            mock_grpc._GRPC_OPTIONS = []

            from model.gateway import InferenceGateway
            gw = InferenceGateway(hosts)
        return gw

    def test_has_update_chunk_host(self):
        """InferenceGateway has update_chunk_host method."""
        gw = self._make_gateway()
        self.assertTrue(callable(getattr(gw, "update_chunk_host", None)))

    def test_get_chain_topology_returns_mapping(self):
        """get_chain_topology returns list of chunk_index -> host dicts."""
        gw = self._make_gateway(["a:50051", "b:50052", "c:50053"])
        topo = gw.get_chain_topology()
        self.assertEqual(len(topo), 3)
        for i, entry in enumerate(topo):
            self.assertEqual(entry["chunk_index"], i)
            self.assertIn("host", entry)
        self.assertEqual(topo[0]["host"], "a:50051")
        self.assertEqual(topo[2]["host"], "c:50053")

    def test_update_chunk_host_changes_topology(self):
        """update_chunk_host replaces the host for a given chunk index."""
        self._ensure_gateway_importable()

        with patch("model.gateway.grpc") as mock_grpc, \
             patch("model.gateway.inference_pb2_grpc") as mock_pb2_grpc:
            mock_grpc.insecure_channel.return_value = MagicMock()
            mock_pb2_grpc.InferenceServiceStub.return_value = MagicMock()

            from model.gateway import InferenceGateway
            gw = InferenceGateway(["node-0:50051", "node-1:50052"])
            gw.update_chunk_host(1, "node-2:50053")

        topo = gw.get_chain_topology()
        self.assertEqual(topo[1]["host"], "node-2:50053")

    def test_gateway_http_handler_has_relink(self):
        """GatewayHTTPHandler.do_POST handles /relink path."""
        self._ensure_gateway_importable()
        from model.gateway import GatewayHTTPHandler
        handler = GatewayHTTPHandler
        self.assertTrue(callable(getattr(handler, "do_POST", None)))

        # Verify by inspecting source that /relink is handled (string check)
        import inspect
        source = inspect.getsource(handler.do_POST)
        self.assertIn("/relink", source)


if __name__ == "__main__":
    unittest.main()
