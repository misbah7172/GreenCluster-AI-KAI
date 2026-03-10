"""
Monitoring HTTP service.

Runs GPU and CPU monitors and exposes metrics via a simple HTTP API.
Designed to be deployed as a sidecar container or DaemonSet in Kubernetes,
or a standalone service in Docker Compose.

Endpoints:
    GET /metrics          -- JSON snapshot of all collected samples
    GET /metrics/gpu      -- GPU samples only
    GET /metrics/cpu      -- CPU samples only
    GET /metrics/summary  -- Aggregated summary
    GET /health           -- Readiness check
    POST /start           -- Start monitoring
    POST /stop            -- Stop monitoring and return summary
    POST /reset           -- Clear all samples and restart

Environment variables:
    MONITOR_PORT=9090
    GPU_INDEX=0
    SAMPLE_INTERVAL=1.0
    ENABLE_GPU=true
"""

import argparse
import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class MonitorHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing monitoring metrics."""

    collector: MetricsCollector = None
    _running: bool = False

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "healthy",
                "monitoring_active": self._running,
            })

        elif self.path == "/metrics":
            self._json_response(200, {
                "gpu_samples": self.collector.get_gpu_samples(),
                "cpu_samples": self.collector.get_cpu_samples(),
                "inference_latencies_ms": self.collector.get_inference_latencies(),
            })

        elif self.path == "/metrics/gpu":
            self._json_response(200, {
                "gpu_samples": self.collector.get_gpu_samples(),
            })

        elif self.path == "/metrics/cpu":
            self._json_response(200, {
                "cpu_samples": self.collector.get_cpu_samples(),
            })

        elif self.path == "/metrics/summary":
            summary = self.collector.compute_summary(execution_mode="kubernetes")
            # Remove raw samples from summary for a lighter response
            for key in ("gpu_samples", "cpu_samples", "inference_latencies_ms"):
                summary.pop(key, None)
            self._json_response(200, summary)

        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/start":
            if not self.__class__._running:
                self.collector.start()
                self.__class__._running = True
                self._json_response(200, {"status": "monitoring started"})
            else:
                self._json_response(200, {"status": "already running"})

        elif self.path == "/stop":
            if self.__class__._running:
                self.collector.stop()
                self.__class__._running = False
                summary = self.collector.compute_summary(execution_mode="kubernetes")
                self._json_response(200, summary)
            else:
                self._json_response(200, {"status": "not running"})

        elif self.path == "/reset":
            if self.__class__._running:
                self.collector.stop()
            # Re-create the collector to clear state
            gpu_index = self.collector._gpu_monitor.gpu_index if self.collector._gpu_monitor else 0
            interval = self.collector._gpu_monitor.interval if self.collector._gpu_monitor else 1.0
            enable_gpu = self.collector.enable_gpu
            self.__class__.collector = MetricsCollector(
                gpu_index=gpu_index,
                interval=interval,
                enable_gpu=enable_gpu,
                enable_cpu=True,
            )
            self.collector.start()
            self.__class__._running = True
            self._json_response(200, {"status": "reset and restarted"})

        else:
            self._json_response(404, {"error": "not found"})

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.info(format, *args)


def serve(gpu_index: int = 0, interval: float = 1.0, port: int = 9090,
          enable_gpu: bool = True, auto_start: bool = True):
    """Start the monitor HTTP server.

    Parameters
    ----------
    gpu_index : int
        GPU device index.
    interval : float
        Sampling interval in seconds.
    port : int
        HTTP port.
    enable_gpu : bool
        Whether to enable GPU monitoring.
    auto_start : bool
        Start monitoring immediately on server launch.
    """
    collector = MetricsCollector(
        gpu_index=gpu_index,
        interval=interval,
        enable_gpu=enable_gpu,
        enable_cpu=True,
    )
    MonitorHTTPHandler.collector = collector

    if auto_start:
        collector.start()
        MonitorHTTPHandler._running = True

    server = HTTPServer(("0.0.0.0", port), MonitorHTTPHandler)
    logger.info(
        "Monitor serving on port %d (gpu=%d, interval=%.1fs, auto_start=%s)",
        port, gpu_index, interval, auto_start,
    )
    return server, collector


def main():
    parser = argparse.ArgumentParser(description="Monitoring HTTP service")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--no-gpu", action="store_true", default=False)
    args = parser.parse_args()

    port = args.port or int(os.environ.get("MONITOR_PORT", "9090"))
    gpu_index = args.gpu_index if args.gpu_index is not None else int(os.environ.get("GPU_INDEX", "0"))
    interval = args.interval if args.interval is not None else float(os.environ.get("SAMPLE_INTERVAL", "1.0"))
    enable_gpu = not args.no_gpu and os.environ.get("ENABLE_GPU", "true").lower() == "true"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [monitor] %(levelname)s %(message)s",
    )

    server, _ = serve(
        gpu_index=gpu_index,
        interval=interval,
        port=port,
        enable_gpu=enable_gpu,
    )

    logger.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down monitor service.")
        server.shutdown()


if __name__ == "__main__":
    main()
