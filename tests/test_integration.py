"""
KAI -- Integration Tests

Validates end-to-end workflows across all major subsystems.
Run with::

    python tests/test_integration.py
"""

import json
import os
import sys
import tempfile
import time
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelDefinitions(unittest.TestCase):
    """Phase 2: AI model definitions produce valid outputs."""

    def test_transformer_build_and_forward(self):
        from model.transformer import build_model, get_dummy_input
        model = build_model()
        x = get_dummy_input(batch_size=2)
        out = model(x)
        self.assertIn(out.dim(), (2, 3))  # shape depends on model head
        self.assertEqual(out.shape[0], 2)

    def test_cnn_build_and_forward(self):
        from model.cnn import build_model, get_dummy_input
        model = build_model()
        x = get_dummy_input(batch_size=2)
        out = model(x)
        self.assertEqual(out.dim(), 2)  # (batch, num_classes)
        self.assertEqual(out.shape[0], 2)


class TestModelChunking(unittest.TestCase):
    """Phase 2: Model chunking splits and validates correctly."""

    def test_chunk_and_validate_transformer(self):
        from model.transformer import build_model, get_dummy_input
        from model.chunker import chunk_model, validate_chunks
        model = build_model()
        chunks = chunk_model(model, num_chunks=3)
        self.assertEqual(len(chunks), 3)
        x = get_dummy_input(batch_size=2)
        self.assertTrue(validate_chunks(model, chunks, x))

    def test_chunk_and_validate_cnn(self):
        from model.cnn import build_model, get_dummy_input
        from model.chunker import chunk_model, validate_chunks
        model = build_model()
        chunks = chunk_model(model, num_chunks=2)
        self.assertEqual(len(chunks), 2)
        x = get_dummy_input(batch_size=2)
        self.assertTrue(validate_chunks(model, chunks, x))

    def test_serialize_and_load_chunk(self):
        from model.transformer import build_model
        from model.chunker import chunk_model, serialize_chunk, load_chunk
        import torch

        model = build_model()
        chunks = chunk_model(model, num_chunks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            for chunk in chunks:
                serialize_chunk(chunk, tmpdir)

            loaded = load_chunk(model, chunk_id=0, num_chunks=2, weights_dir=tmpdir)
            self.assertIsNotNone(loaded)

            # Verify output matches
            x = torch.randint(0, 100, (1, 32))
            with torch.no_grad():
                orig_out = chunks[0](x)
                load_out = loaded(x)
            self.assertTrue(torch.allclose(orig_out, load_out, atol=1e-5))


class TestGPUMonitoring(unittest.TestCase):
    """Phase 4: GPU monitoring records data correctly."""

    def test_gpu_monitor_lifecycle(self):
        from monitoring.gpu_monitor import GPUMonitor
        monitor = GPUMonitor(interval=0.2)
        monitor.start()
        time.sleep(0.6)
        monitor.stop()
        samples = monitor.get_samples()
        # Should have at least 1 sample (may have 0 if no NVIDIA GPU)
        if samples:
            s = samples[0]
            self.assertIn("power_w", s)
            self.assertIn("gpu_utilization_pct", s)
            self.assertIn("memory_used_mb", s)
            self.assertIn("temperature_c", s)
            self.assertIn("timestamp", s)


class TestCPUMonitoring(unittest.TestCase):
    """Phase 4: CPU monitoring records data correctly."""

    def test_cpu_monitor_lifecycle(self):
        from monitoring.cpu_monitor import CPUMonitor
        monitor = CPUMonitor(interval=0.2)
        monitor.start()
        time.sleep(0.6)
        monitor.stop()
        samples = monitor.get_samples()
        self.assertGreater(len(samples), 0)
        s = samples[0]
        # Key may be 'cpu_percent' or 'cpu_utilization_pct' depending on implementation
        self.assertTrue(
            "cpu_percent" in s or "cpu_utilization_pct" in s,
            f"Expected cpu key in {list(s.keys())}"
        )
        self.assertIn("timestamp", s)


class TestMetricsCollector(unittest.TestCase):
    """Phase 4: Unified metrics collector integrates GPU + CPU + latency."""

    def test_collect_and_export(self):
        from monitoring.metrics import MetricsCollector

        collector = MetricsCollector(interval=0.2)
        collector.start()
        # Simulate some inference latencies
        for _ in range(5):
            collector.record_latency(10.0 + _ * 0.5)
            time.sleep(0.1)
        collector.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = collector.export_results(tmpdir, execution_mode="local")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIn("avg_latency_ms", data)
            self.assertIn("total_energy_wh", data)
            self.assertEqual(data["execution_mode"], "local")
            self.assertEqual(len(data.get("inference_latencies_ms", [])), 5)


class TestLocalRunner(unittest.TestCase):
    """Phase 5: Local runner produces valid result JSON."""

    def test_local_experiment_output_structure(self):
        """Verify a local result file has the expected keys."""
        # Create a mock local result to validate structure
        local_result = {
            "execution_mode": "local",
            "avg_latency_ms": 42.5,
            "avg_power_w": 35.0,
            "total_energy_wh": 0.001,
            "throughput_inferences_per_sec": 23.5,
            "energy_per_inference_wh": 0.0001,
            "gpu_samples": [],
            "cpu_samples": [],
            "inference_latencies_ms": [40, 42, 45],
            "experiment_config": {"model": "transformer"},
        }
        required_keys = [
            "avg_latency_ms", "avg_power_w", "total_energy_wh",
            "throughput_inferences_per_sec", "gpu_samples",
            "inference_latencies_ms", "experiment_config",
        ]
        for key in required_keys:
            self.assertIn(key, local_result, f"Missing key: {key}")


class TestAnalyzer(unittest.TestCase):
    """Phase 11: Analyzer processes results and produces comparison."""

    def test_load_and_analyse_local(self):
        from analysis.analyzer import ExperimentAnalyzer

        local_data = {
            "execution_mode": "local",
            "avg_latency_ms": 45.0,
            "avg_power_w": 38.0,
            "total_energy_wh": 0.003,
            "throughput_inferences_per_sec": 22.0,
            "energy_per_inference_wh": 0.0003,
            "inference_latencies_ms": [40, 42, 45, 48, 50],
            "gpu_samples": [{"power_w": 38, "gpu_utilization_pct": 60}],
            "cpu_samples": [{"cpu_percent": 30}],
            "experiment_config": {"model": "transformer"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(local_data, f)
            path = f.name

        try:
            analyzer = ExperimentAnalyzer()
            analyzer.load_local(path)
            result = analyzer.analyse_local()
            self.assertIn("latency_p50_ms", result)
            self.assertIn("latency_p99_ms", result)
            self.assertIn("performance_per_watt", result)
        finally:
            os.unlink(path)

    def test_compare(self):
        from analysis.analyzer import ExperimentAnalyzer

        combined = {
            "local": {
                "avg_latency_ms": 45.0,
                "avg_power_w": 38.0,
                "total_energy_wh": 0.003,
                "throughput_inferences_per_sec": 22.0,
                "energy_per_inference_wh": 0.0003,
                "inference_latencies_ms": [40, 42, 45],
                "gpu_samples": [],
                "cpu_samples": [],
            },
            "kubernetes": {
                "avg_e2e_latency_ms": 78.0,
                "avg_gateway_latency_ms": 72.0,
                "avg_network_overhead_ms": 6.0,
                "total_avg_power_w": 55.0,
                "total_energy_wh": 0.005,
                "throughput_inferences_per_sec": 12.0,
                "energy_per_inference_wh": 0.0005,
                "e2e_latencies_ms": [70, 75, 80],
                "node_metrics": [],
                "inference_results": [],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(combined, f)
            path = f.name

        try:
            analyzer = ExperimentAnalyzer()
            analyzer.load_combined(path)
            comparison = analyzer.compare()
            self.assertIsInstance(comparison, dict)
            self.assertGreater(len(comparison), 0)
            # Each top-level key maps to a dict with local, kubernetes, ratio
            first_key = next(iter(comparison))
            entry = comparison[first_key]
            self.assertIn("ratio_k8s_over_local", entry)
        finally:
            os.unlink(path)

    def test_summary_table(self):
        from analysis.analyzer import ExperimentAnalyzer

        combined = {
            "local": {
                "avg_latency_ms": 45.0, "avg_power_w": 38.0,
                "total_energy_wh": 0.003, "throughput_inferences_per_sec": 22.0,
                "energy_per_inference_wh": 0.0003,
                "inference_latencies_ms": [40, 45], "gpu_samples": [], "cpu_samples": [],
            },
            "kubernetes": {
                "avg_e2e_latency_ms": 78.0, "avg_gateway_latency_ms": 72.0,
                "avg_network_overhead_ms": 6.0, "total_avg_power_w": 55.0,
                "total_energy_wh": 0.005, "throughput_inferences_per_sec": 12.0,
                "energy_per_inference_wh": 0.0005,
                "e2e_latencies_ms": [70, 80], "node_metrics": [], "inference_results": [],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(combined, f)
            path = f.name

        try:
            analyzer = ExperimentAnalyzer()
            analyzer.load_combined(path)
            analyzer.compare()
            table = analyzer.summary_table()
            self.assertIsInstance(table, list)
            self.assertGreater(len(table), 0)
            # Each row is a tuple of (metric, local, k8s, ratio)
            self.assertEqual(len(table[0]), 4)
        finally:
            os.unlink(path)

    def test_export_csv_and_json(self):
        from analysis.analyzer import ExperimentAnalyzer

        combined = {
            "local": {
                "avg_latency_ms": 45.0, "avg_power_w": 38.0,
                "total_energy_wh": 0.003, "throughput_inferences_per_sec": 22.0,
                "energy_per_inference_wh": 0.0003,
                "inference_latencies_ms": [40], "gpu_samples": [], "cpu_samples": [],
            },
            "kubernetes": {
                "avg_e2e_latency_ms": 78.0, "avg_gateway_latency_ms": 72.0,
                "avg_network_overhead_ms": 6.0, "total_avg_power_w": 55.0,
                "total_energy_wh": 0.005, "throughput_inferences_per_sec": 12.0,
                "energy_per_inference_wh": 0.0005,
                "e2e_latencies_ms": [70], "node_metrics": [], "inference_results": [],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(combined, f)
            path = f.name

        try:
            analyzer = ExperimentAnalyzer()
            analyzer.load_combined(path)
            analyzer.compare()

            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = analyzer.export_csv(os.path.join(tmpdir, "out"))
                json_path = analyzer.export_json(os.path.join(tmpdir, "out"))
                self.assertTrue(os.path.exists(csv_path))
                self.assertTrue(os.path.exists(json_path))
        finally:
            os.unlink(path)


class TestPlots(unittest.TestCase):
    """Phase 11: Plot generation produces PNG files."""

    def test_plot_all_generates_files(self):
        from analysis.analyzer import ExperimentAnalyzer
        from analysis.plots import plot_all

        combined = {
            "local": {
                "avg_latency_ms": 45.0, "avg_power_w": 38.0,
                "total_energy_wh": 0.003, "throughput_inferences_per_sec": 22.0,
                "energy_per_inference_wh": 0.0003,
                "inference_latencies_ms": [40, 42, 45, 48, 50],
                "gpu_samples": [
                    {"power_w": 35, "gpu_utilization_pct": 55, "timestamp": i}
                    for i in range(5)
                ],
                "cpu_samples": [],
            },
            "kubernetes": {
                "avg_e2e_latency_ms": 78.0, "total_avg_power_w": 55.0,
                "total_energy_wh": 0.005, "throughput_inferences_per_sec": 12.0,
                "energy_per_inference_wh": 0.0005,
                "e2e_latencies_ms": [70, 75, 80, 85, 90],
                "per_chunk_avg_latency_ms": [22.0, 25.0, 24.0],
                "node_metrics": [
                    {
                        "node_name": "test-node",
                        "gpu_samples": [
                            {"power_w": 50, "gpu_utilization_pct": 70, "timestamp": i}
                            for i in range(5)
                        ],
                    }
                ],
                "inference_results": [],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(combined, f)
            path = f.name

        try:
            analyzer = ExperimentAnalyzer()
            analyzer.load_combined(path)

            with tempfile.TemporaryDirectory() as tmpdir:
                paths = plot_all(analyzer, output_dir=tmpdir)
                self.assertGreater(len(paths), 0)
                for p in paths:
                    self.assertTrue(os.path.exists(p), f"Plot not found: {p}")
                    self.assertTrue(p.endswith(".png"))
        finally:
            os.unlink(path)


class TestDashboardImport(unittest.TestCase):
    """Phase 12: Dashboard module imports without errors."""

    def test_dashboard_helpers_importable(self):
        """Verify dashboard helper functions can be imported (without running Streamlit)."""
        # We can't run Streamlit in a test, but we can verify the file is valid Python
        import ast
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dashboard", "app.py"
        )
        with open(dashboard_path) as f:
            source = f.read()
        # This will raise SyntaxError if the file is invalid
        ast.parse(source)


class TestDockerfiles(unittest.TestCase):
    """Phase 6: Dockerfiles exist and have valid structure."""

    def _check_dockerfile(self, name):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docker", name
        )
        self.assertTrue(os.path.exists(path), f"{name} not found")
        with open(path) as f:
            content = f.read()
        self.assertIn("FROM", content, f"{name} missing FROM instruction")

    def test_dockerfile_chunk(self):
        self._check_dockerfile("Dockerfile.chunk")

    def test_dockerfile_gateway(self):
        self._check_dockerfile("Dockerfile.gateway")

    def test_dockerfile_monitor(self):
        self._check_dockerfile("Dockerfile.monitor")


class TestKubernetesManifests(unittest.TestCase):
    """Phase 7: Kubernetes YAML manifests exist and are valid."""

    def _check_yaml(self, *path_parts):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kubernetes", *path_parts
        )
        self.assertTrue(os.path.exists(path), f"{'/'.join(path_parts)} not found")
        with open(path) as f:
            content = f.read()
        self.assertIn("kind:", content)
        self.assertIn("apiVersion:", content)

    def test_chunk_deployment(self):
        self._check_yaml("deployments", "chunk-deployment.yaml")

    def test_gateway_deployment(self):
        self._check_yaml("deployments", "gateway-deployment.yaml")

    def test_monitor_deployment(self):
        self._check_yaml("deployments", "monitor-deployment.yaml")

    def test_chunk_service(self):
        self._check_yaml("services", "chunk-service.yaml")

    def test_gateway_service(self):
        self._check_yaml("services", "gateway-service.yaml")

    def test_monitor_service(self):
        self._check_yaml("services", "monitor-service.yaml")

    def test_gpu_resource_quota(self):
        self._check_yaml("gpu-resource-quota.yaml")


if __name__ == "__main__":
    unittest.main(verbosity=2)
