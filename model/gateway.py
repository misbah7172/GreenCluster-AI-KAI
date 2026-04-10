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
import hashlib
import sys
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs

import torch
import grpc

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from proto import inference_pb2
from proto import inference_pb2_grpc
from monitoring.telemetry import (
    get_default_collector, RoutingDecision, InferenceMetrics
)
from model.latency_probe import get_default_prober

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

    def __init__(
        self,
        chunk_hosts: List[str],
        route_policy: str = "deterministic-latency",
        link_latency_ms: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        chunk_hosts : list[str]
            Ordered list of ``host:port`` strings for each chunk service.
        """
        self.chunk_hosts = list(chunk_hosts)
        # Support multiple candidates per chunk via "hostA|hostB" syntax.
        self._candidate_hosts: List[List[str]] = [
            [h.strip() for h in host.split("|") if h.strip()]
            for host in chunk_hosts
        ]
        self._route_policy = route_policy
        self._link_latency_ms = link_latency_ms or {}
        self.stubs = []
        self._channels = []
        self._chain_lock = threading.Lock()
        self._last_route_hash = ""

        # Build an initial deterministic route across chunks.
        self._recompute_route_locked()
        for host in self.chunk_hosts:
            channel = grpc.insecure_channel(host, options=_GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            self._channels.append(channel)
            self.stubs.append(stub)
            logger.info("Connected to chunk at %s", host)

    def _probe_host_latency_ms(self, host: str, samples: int = 2, timeout_s: float = 1.5) -> float:
        """Probe host RTT using HealthCheck RPC and return median-like average."""
        vals: List[float] = []
        channel = grpc.insecure_channel(host, options=_GRPC_OPTIONS)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        try:
            for _ in range(max(1, samples)):
                t0 = time.perf_counter()
                try:
                    stub.HealthCheck(inference_pb2.HealthRequest(), timeout=timeout_s)
                    vals.append((time.perf_counter() - t0) * 1000.0)
                except grpc.RpcError:
                    vals.append(timeout_s * 1000.0)
        finally:
            try:
                channel.close()
            except Exception:
                pass

        vals.sort()
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def calibrate_latency_map(self, samples: int = 2) -> Dict[str, float]:
        """Probe candidate hosts and refresh link latency map.

        Uses endpoint RTT to estimate pair costs deterministically and updates
        routing map in-place.
        """
        with self._chain_lock:
            unique_hosts = sorted({h for cands in self._candidate_hosts for h in cands})
            host_rtt: Dict[str, float] = {}
            for host in unique_hosts:
                host_rtt[host] = self._probe_host_latency_ms(host, samples=samples)

            updated: Dict[str, float] = {}
            for i in range(1, len(self._candidate_hosts)):
                left = sorted(set(self._candidate_hosts[i - 1]))
                right = sorted(set(self._candidate_hosts[i]))
                for src in left:
                    for dst in right:
                        # Estimate directional pair latency from observed endpoint RTT.
                        updated[f"{src}->{dst}"] = (host_rtt[src] + host_rtt[dst]) / 2.0

            self._link_latency_ms.update(updated)
            self._recompute_route_locked()
            return dict(updated)

    @staticmethod
    def _host_key(host: str) -> str:
        """Return host identity without port for topology comparisons."""
        return host.split(":", 1)[0]

    def _pair_latency_ms(self, src_host: str, dst_host: str) -> float:
        """Get expected latency between two hosts.

        Lookup order:
        1. Explicit `src->dst` value from `link_latency_ms`.
        2. Explicit `dst->src` value.
        3. Same-node tiny latency.
        4. Deterministic synthetic LAN latency.
        """
        key = f"{src_host}->{dst_host}"
        rev = f"{dst_host}->{src_host}"
        if key in self._link_latency_ms:
            return float(self._link_latency_ms[key])
        if rev in self._link_latency_ms:
            return float(self._link_latency_ms[rev])

        if self._host_key(src_host) == self._host_key(dst_host):
            return 0.05

        seed = hashlib.sha256(f"{src_host}|{dst_host}".encode("utf-8")).hexdigest()
        val = int(seed[:8], 16)
        # Deterministic 0.4..2.0 ms synthetic fallback.
        return 0.4 + (val % 160) / 100.0

    def _choose_host_for_chunk(self, chunk_index: int, prev_host: Optional[str]) -> str:
        """Select host deterministically for a chunk.

        With `deterministic-latency`, minimize pair latency to previous chunk.
        Tie-breaker is lexical order to eliminate random switching.
        """
        candidates = sorted(set(self._candidate_hosts[chunk_index]))
        if not candidates:
            raise ValueError(f"No candidate hosts for chunk {chunk_index}")

        if prev_host is None or self._route_policy != "deterministic-latency":
            return candidates[0]

        best = min(
            candidates,
            key=lambda h: (self._pair_latency_ms(prev_host, h), h),
        )
        return best

    def _recompute_route_locked(self) -> bool:
        """Compute and apply current route. Returns True if route changed."""
        route: List[str] = []
        prev: Optional[str] = None
        for idx in range(len(self._candidate_hosts)):
            selected = self._choose_host_for_chunk(idx, prev)
            route.append(selected)
            prev = selected

        route_hash = "|".join(route)
        changed = route_hash != self._last_route_hash
        self._last_route_hash = route_hash

        if (
            not self.chunk_hosts
            or len(self.chunk_hosts) != len(route)
            or len(self._channels) != len(route)
        ):
            self.chunk_hosts = list(route)
            return True

        # Apply host changes safely by reusing update path.
        for i, host in enumerate(route):
            if self.chunk_hosts[i] != host:
                self._replace_chunk_stub_locked(i, host)

        return changed

    def _replace_chunk_stub_locked(self, chunk_index: int, new_host: str) -> None:
        """Internal host replacement. Caller must hold `_chain_lock`."""
        old_host = self.chunk_hosts[chunk_index]
        try:
            self._channels[chunk_index].close()
        except Exception:
            pass

        channel = grpc.insecure_channel(new_host, options=_GRPC_OPTIONS)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        self._channels[chunk_index] = channel
        self.stubs[chunk_index] = stub
        self.chunk_hosts[chunk_index] = new_host

        logger.info("Route update chunk %d: %s -> %s", chunk_index, old_host, new_host)

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

    def run_inference(self, input_tensor: torch.Tensor, request_id: str = "", 
                      model_name: str = "unknown") -> dict:
        """Pass a tensor through the full chunk pipeline with telemetry.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The original model input.
        request_id : str
            Unique request identifier for tracing.
        model_name : str
            Name of the model being served (for telemetry).

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
        network_bytes = 0
        inference_start = time.time()

        with self._chain_lock:
            self._recompute_route_locked()
            for i, stub in enumerate(self.stubs):
                # Record routing decision for this chunk
                selected_host = self.chunk_hosts[i]
                candidates = sorted(set(self._candidate_hosts[i]))
                latencies = {
                    h: self._pair_latency_ms(self.chunk_hosts[i-1] if i > 0 else "local", h)
                    for h in candidates
                }
                
                routing_decision = RoutingDecision(
                    timestamp=time.time(),
                    chunk_index=i,
                    selected_host=selected_host,
                    candidate_hosts=candidates,
                    latencies_ms=latencies,
                    decision_latency_ms=(time.perf_counter() - total_start) * 1000,
                    inference_id=request_id,
                    model_name=model_name,
                    method="deterministic-latency" if self._route_policy == "deterministic-latency" else self._route_policy,
                )
                get_default_collector().record_routing_decision(routing_decision)
                
                # Record latency sample for this host
                if selected_host:
                    get_default_collector().record_latency_sample(selected_host, latencies.get(selected_host, 0.0))
                
                tensor_bytes = _serialize_tensor(current_tensor)
                network_bytes += len(tensor_bytes)

                req = inference_pb2.InferRequest(
                    tensor_data=tensor_bytes,
                    chunk_id=i,
                    request_id=request_id,
                    tensor_shape=list(current_tensor.shape),
                    tensor_dtype=str(current_tensor.dtype).replace("torch.", ""),
                )

                resp = stub.Infer(req)
                current_tensor = _deserialize_tensor(resp.tensor_data)
                network_bytes += len(resp.tensor_data)
                chunk_times.append(resp.inference_time_ms)

                logger.debug(
                    "Chunk %d: %.2f ms, output shape %s",
                    i, resp.inference_time_ms, list(current_tensor.shape),
                )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        inference_end = time.time()
        
        # Record overall inference metrics
        metrics = InferenceMetrics(
            inference_id=request_id,
            model_name=model_name,
            start_time=inference_start,
            end_time=inference_end,
            chunk_count=len(self.stubs),
            total_tokens=int(current_tensor.numel()),
            gpu_memory_mb=0,  # TODO: capture GPU memory if available
            cpu_memory_mb=0,
            network_bytes_sent=network_bytes,
            network_bytes_received=network_bytes,
        )
        get_default_collector().record_inference(metrics)

        return {
            "output_tensor": current_tensor,
            "total_time_ms": total_ms,
            "chunk_times_ms": chunk_times,
            "network_overhead_ms": total_ms - sum(chunk_times),
            "request_id": request_id,
        }

    def update_chunk_host(self, chunk_index: int, new_host: str) -> None:
        """Replace the gRPC stub for a specific chunk index.

        Used during live migration to point at the new node.
        Thread-safe: acquires the chain lock to prevent mid-inference rewiring.

        Parameters
        ----------
        chunk_index : int
            Index of the chunk to re-link (0-based).
        new_host : str
            New ``host:port`` address for the chunk service.
        """
        with self._chain_lock:
            if chunk_index < 0 or chunk_index >= len(self.stubs):
                raise IndexError(
                    f"chunk_index {chunk_index} out of range [0, {len(self.stubs)})"
                )
            self._candidate_hosts[chunk_index] = [new_host]
            self._replace_chunk_stub_locked(chunk_index, new_host)

    def get_chain_topology(self) -> List[Dict[str, str]]:
        """Return current chunk → host mapping."""
        with self._chain_lock:
            return [
                {"chunk_index": i, "host": host}
                for i, host in enumerate(self.chunk_hosts)
            ]

    def get_route_status(self) -> Dict[str, Any]:
        """Return selected route and candidates."""
        with self._chain_lock:
            links: List[Dict[str, Any]] = []
            for i in range(1, len(self.chunk_hosts)):
                src = self.chunk_hosts[i - 1]
                dst = self.chunk_hosts[i]
                links.append(
                    {
                        "from": src,
                        "to": dst,
                        "latency_ms": round(self._pair_latency_ms(src, dst), 3),
                    }
                )

            return {
                "policy": self._route_policy,
                "selected": [
                    {"chunk_index": i, "host": h}
                    for i, h in enumerate(self.chunk_hosts)
                ],
                "candidates": [
                    {"chunk_index": i, "hosts": list(cands)}
                    for i, cands in enumerate(self._candidate_hosts)
                ],
                "links": links,
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
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            health = self.gateway.health_check_all()
            all_ready = all(h["ready"] for h in health)
            self._json_response(
                200 if all_ready else 503,
                {"status": "healthy" if all_ready else "degraded", "chunks": health},
            )
        elif path == "/topology":
            topo = self.gateway.get_chain_topology()
            self._json_response(200, {"topology": topo})
        elif path == "/routing":
            self._json_response(200, self.gateway.get_route_status())
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))

        if self.path == "/infer":
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

        elif self.path == "/relink":
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode())
                chunk_index = int(data["chunk_index"])
                new_host = str(data["new_host"])
                self.gateway.update_chunk_host(chunk_index, new_host)
                self._json_response(200, {
                    "status": "relinked",
                    "chunk_index": chunk_index,
                    "new_host": new_host,
                })
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                self._json_response(400, {"error": str(e)})

        elif self.path == "/probe-latency":
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body.decode()) if body else {}
                samples = int(data.get("samples", 2))
                updated = self.gateway.calibrate_latency_map(samples=samples)
                self._json_response(200, {
                    "status": "ok",
                    "updated_links": len(updated),
                    "routing": self.gateway.get_route_status(),
                })
            except Exception as e:
                self._json_response(400, {"error": str(e)})

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


def serve(
    chunk_hosts: List[str],
    port: int = 8080,
    route_policy: str = "deterministic-latency",
    link_latency_ms: Optional[Dict[str, float]] = None,
):
    """Start the gateway HTTP server.

    Parameters
    ----------
    chunk_hosts : list[str]
        Ordered list of chunk service ``host:port`` addresses.
    port : int
        Port for the HTTP gateway.
    """
    gateway = InferenceGateway(
        chunk_hosts,
        route_policy=route_policy,
        link_latency_ms=link_latency_ms,
    )
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
    route_policy = os.environ.get("KAI_GATEWAY_ROUTE_POLICY", "deterministic-latency")
    raw_latency = os.environ.get("KAI_LINK_LATENCY_MS", "")
    link_latency_ms: Dict[str, float] = {}
    if raw_latency:
        try:
            parsed = json.loads(raw_latency)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    try:
                        link_latency_ms[str(k)] = float(v)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Invalid KAI_LINK_LATENCY_MS JSON: %s", e)

    chunk_hosts = [h.strip() for h in chunk_hosts_str.split(",")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [gateway] %(levelname)s %(message)s",
    )

    server, _ = serve(
        chunk_hosts,
        port,
        route_policy=route_policy,
        link_latency_ms=link_latency_ms,
    )

    logger.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down gateway.")
        server.shutdown()


if __name__ == "__main__":
    main()
