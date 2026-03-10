"""
Inference gateway service.

Receives inference requests, forwards the input tensor through a chain
of model chunk gRPC services in sequence, and returns the final output.
Acts as the single entry point for Kubernetes-mode inference.

The gateway discovers chunk endpoints from environment variables:
    CHUNK_HOSTS=chunk-0:50051,chunk-1:50052,chunk-2:50053
    NUM_CHUNKS=3
    MODEL_TYPE=transformer
    GATEWAY_PORT=8080

Usage::

    python -m model.gateway \
        --chunk-hosts chunk-0:50051,chunk-1:50052 \
        --port 8080
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List

import torch
import grpc

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from proto import inference_pb2
from proto import inference_pb2_grpc

logger = logging.getLogger(__name__)

# 256 MB max message size for large intermediate tensors
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _deserialize_tensor(data: bytes) -> torch.Tensor:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


class InferenceGateway:
    """Orchestrates inference across a chain of chunk gRPC services."""

    def __init__(self, chunk_hosts: List[str]):
        """
        Parameters
        ----------
        chunk_hosts : list[str]
            Ordered list of ``host:port`` strings for each chunk service.
        """
        self.chunk_hosts = chunk_hosts
        self.stubs = []

        for host in chunk_hosts:
            channel = grpc.insecure_channel(host, options=_GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            self.stubs.append(stub)
            logger.info("Connected to chunk at %s", host)

    def health_check_all(self) -> List[dict]:
        """Check health of all chunk services."""
        results = []
        for i, stub in enumerate(self.stubs):
            try:
                resp = stub.HealthCheck(inference_pb2.HealthRequest())
                results.append({
                    "chunk_id": resp.chunk_id,
                    "ready": resp.ready,
                    "status": resp.status,
                    "host": self.chunk_hosts[i],
                })
            except grpc.RpcError as e:
                results.append({
                    "chunk_id": i,
                    "ready": False,
                    "status": str(e),
                    "host": self.chunk_hosts[i],
                })
        return results

    def run_inference(self, input_tensor: torch.Tensor, request_id: str = "") -> dict:
        """Pass a tensor through the full chunk pipeline.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The original model input.
        request_id : str
            Unique request identifier for tracing.

        Returns
        -------
        dict
            Contains ``output_tensor``, ``total_time_ms``,
            ``chunk_times_ms`` (list), and ``request_id``.
        """
        if not request_id:
            request_id = uuid.uuid4().hex[:12]

        current_tensor = input_tensor
        chunk_times = []
        total_start = time.perf_counter()

        for i, stub in enumerate(self.stubs):
            tensor_bytes = _serialize_tensor(current_tensor)

            req = inference_pb2.InferRequest(
                tensor_data=tensor_bytes,
                chunk_id=i,
                request_id=request_id,
                tensor_shape=list(current_tensor.shape),
                tensor_dtype=str(current_tensor.dtype).replace("torch.", ""),
            )

            resp = stub.Infer(req)
            current_tensor = _deserialize_tensor(resp.tensor_data)
            chunk_times.append(resp.inference_time_ms)

            logger.debug(
                "Chunk %d: %.2f ms, output shape %s",
                i, resp.inference_time_ms, list(current_tensor.shape),
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "output_tensor": current_tensor,
            "total_time_ms": total_ms,
            "chunk_times_ms": chunk_times,
            "network_overhead_ms": total_ms - sum(chunk_times),
            "request_id": request_id,
        }


class GatewayHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that accepts inference requests via POST.

    POST /infer
        Body: raw tensor bytes (torch.save format)
        Response: JSON with timing info and output tensor shape

    GET /health
        Response: JSON health status of all chunks
    """

    gateway: InferenceGateway = None  # set by serve()

    def do_GET(self):
        if self.path == "/health":
            health = self.gateway.health_check_all()
            all_ready = all(h["ready"] for h in health)
            self._json_response(
                200 if all_ready else 503,
                {"status": "healthy" if all_ready else "degraded", "chunks": health},
            )
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/infer":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                input_tensor = _deserialize_tensor(body)
            except Exception as e:
                self._json_response(400, {"error": f"Invalid tensor data: {e}"})
                return

            request_id = self.headers.get("X-Request-ID", "")
            result = self.gateway.run_inference(input_tensor, request_id)

            self._json_response(200, {
                "request_id": result["request_id"],
                "output_shape": list(result["output_tensor"].shape),
                "total_time_ms": round(result["total_time_ms"], 3),
                "chunk_times_ms": [round(t, 3) for t in result["chunk_times_ms"]],
                "network_overhead_ms": round(result["network_overhead_ms"], 3),
            })
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


def serve(chunk_hosts: List[str], port: int = 8080):
    """Start the gateway HTTP server.

    Parameters
    ----------
    chunk_hosts : list[str]
        Ordered list of chunk service ``host:port`` addresses.
    port : int
        Port for the HTTP gateway.
    """
    gateway = InferenceGateway(chunk_hosts)
    GatewayHTTPHandler.gateway = gateway

    server = HTTPServer(("0.0.0.0", port), GatewayHTTPHandler)
    logger.info("Gateway serving on port %d, %d chunks", port, len(chunk_hosts))
    return server, gateway


def main():
    parser = argparse.ArgumentParser(description="Inference gateway service")
    parser.add_argument(
        "--chunk-hosts", type=str, default=None,
        help="Comma-separated chunk host:port list (e.g. chunk-0:50051,chunk-1:50052)",
    )
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    chunk_hosts_str = args.chunk_hosts or os.environ.get("CHUNK_HOSTS", "localhost:50051")
    port = args.port or int(os.environ.get("GATEWAY_PORT", "8080"))
    chunk_hosts = [h.strip() for h in chunk_hosts_str.split(",")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [gateway] %(levelname)s %(message)s",
    )

    server, _ = serve(chunk_hosts, port)

    logger.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down gateway.")
        server.shutdown()


if __name__ == "__main__":
    main()
