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
    GET /metrics/threshold -- Current threshold level + recent events
    GET /metrics/events   -- Recent threshold events (optional ?n=50)
    GET /health           -- Readiness check
    POST /start           -- Start monitoring
    POST /stop            -- Stop monitoring and return summary
    POST /reset           -- Clear all samples and restart

Environment variables:
    MONITOR_PORT=9090
    GPU_INDEX=0
    SAMPLE_INTERVAL=1.0
    SAMPLING_RATE=1.0        (alias for SAMPLE_INTERVAL)
    ENABLE_GPU=true
    TDP_WATTS=0.0
    ENABLE_THRESHOLD=false
"""

import argparse
import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Module-level state for re-creation during /reset
_collector_kwargs: dict = {}


class MonitorHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing monitoring metrics."""

    collector: MetricsCollector = None
    _running: bool = False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._json_response(200, {
                "status": "healthy",
                "monitoring_active": self._running,
            })

        elif path == "/metrics":
            self._json_response(200, {
                "gpu_samples": self.collector.get_gpu_samples(),
                "cpu_samples": self.collector.get_cpu_samples(),
                "inference_latencies_ms": self.collector.get_inference_latencies(),
            })

        elif path == "/metrics/gpu":
            self._json_response(200, {
                "gpu_samples": self.collector.get_gpu_samples(),
            })

        elif path == "/metrics/cpu":
            self._json_response(200, {
                "cpu_samples": self.collector.get_cpu_samples(),
            })

        elif path == "/metrics/summary":
            summary = self.collector.compute_summary(execution_mode="kubernetes")
            # Remove raw samples from summary for a lighter response
            for key in ("gpu_samples", "cpu_samples", "inference_latencies_ms"):
                summary.pop(key, None)
            self._json_response(200, summary)

        elif path == "/metrics/threshold":
            self._handle_threshold()

        elif path == "/metrics/events":
            n = int(qs.get("n", ["50"])[0])
            self._handle_events(n)

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
            self.__class__.collector = MetricsCollector(**_collector_kwargs)
            self.collector.start()
            self.__class__._running = True
            self._json_response(200, {"status": "reset and restarted"})

        else:
            self._json_response(404, {"error": "not found"})

    # ------------------------------------------------------------------
    # Threshold helpers
    # ------------------------------------------------------------------

    def _handle_threshold(self):
        """GET /metrics/threshold — current level + recent events."""
        bus = self.collector.event_bus
        ts = (
            self.collector._threshold_service
            if hasattr(self.collector, "_threshold_service")
            else None
        )
        if bus is None or ts is None:
            self._json_response(200, {
                "enabled": False,
                "current_level": None,
                "recent_events": [],
            })
            return

        current = ts.get_current_level()
        recent = bus.get_event_history(10)
        self._json_response(200, {
            "enabled": True,
            "current_level": current.value if current else None,
            "recent_events": [e.to_dict() for e in recent],
        })

    def _handle_events(self, n: int):
        """GET /metrics/events?n=N — last N threshold events."""
        bus = self.collector.event_bus
        if bus is None:
            self._json_response(200, {"events": []})
            return

        events = bus.get_event_history(n)
        self._json_response(200, {
            "events": [e.to_dict() for e in events],
        })

    # ------------------------------------------------------------------

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.info(format, *args)


def serve(
    gpu_index: int = 0,
    interval: float = 1.0,
    port: int = 9090,
    enable_gpu: bool = True,
    auto_start: bool = True,
    tdp_watts: float = 0.0,
    enable_threshold: bool = False,
    node_name: str = "local",
):
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
    tdp_watts : float
        GPU TDP in watts (0 = auto-detect).
    enable_threshold : bool
        Enable the power threshold service.
    node_name : str
        Node name used in threshold events.
    """
    global _collector_kwargs
    _collector_kwargs = dict(
        gpu_index=gpu_index,
        interval=interval,
        enable_gpu=enable_gpu,
        enable_cpu=True,
        enable_threshold=enable_threshold,
        tdp_watts=tdp_watts,
        node_name=node_name,
    )

    collector = MetricsCollector(**_collector_kwargs)
    MonitorHTTPHandler.collector = collector

    if auto_start:
        collector.start()
        MonitorHTTPHandler._running = True

    server = HTTPServer(("0.0.0.0", port), MonitorHTTPHandler)
    logger.info(
        "Monitor serving on port %d (gpu=%d, interval=%.2fs, threshold=%s)",
        port, gpu_index, interval, enable_threshold,
    )
    return server, collector


def main():
    parser = argparse.ArgumentParser(description="Monitoring HTTP service")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--sampling-rate", type=float, default=None,
                        help="Sampling interval in seconds (alias for --interval)")
    parser.add_argument("--no-gpu", action="store_true", default=False)
    parser.add_argument("--tdp-watts", type=float, default=None,
                        help="GPU TDP in watts (0 = auto-detect)")
    parser.add_argument("--enable-threshold", action="store_true", default=False,
                        help="Enable power threshold service")
    args = parser.parse_args()

    port = args.port or int(os.environ.get("MONITOR_PORT", "9090"))
    gpu_index = (
        args.gpu_index
        if args.gpu_index is not None
        else int(os.environ.get("GPU_INDEX", "0"))
    )
    # Sampling rate / interval: CLI > env SAMPLING_RATE > env SAMPLE_INTERVAL > default
    interval = (
        args.sampling_rate
        or args.interval
        or float(os.environ.get("SAMPLING_RATE", os.environ.get("SAMPLE_INTERVAL", "1.0")))
    )
    enable_gpu = not args.no_gpu and os.environ.get("ENABLE_GPU", "true").lower() == "true"
    tdp_watts = (
        args.tdp_watts
        if args.tdp_watts is not None
        else float(os.environ.get("TDP_WATTS", "0.0"))
    )
    enable_threshold = (
        args.enable_threshold
        or os.environ.get("ENABLE_THRESHOLD", "false").lower() == "true"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [monitor] %(levelname)s %(message)s",
    )

    server, _ = serve(
        gpu_index=gpu_index,
        interval=interval,
        port=port,
        enable_gpu=enable_gpu,
        tdp_watts=tdp_watts,
        enable_threshold=enable_threshold,
    )

    logger.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down monitor service.")
        server.shutdown()


if __name__ == "__main__":
    main()
