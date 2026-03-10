# Phase 6: Docker Setup -- Documentation

## Objective

Containerize the three core services of the KAI platform (model chunk server, inference gateway, monitoring service) so they can run as isolated microservices in Docker and later be deployed to Kubernetes.

## What Was Built

### 1. Inference Gateway (`model/gateway.py`)

An HTTP service that acts as the single entry point for distributed inference.

- **InferenceGateway class** -- Connects to an ordered list of chunk gRPC servers and chains inference calls sequentially (chunk 0 -> chunk 1 -> ... -> chunk N). Each chunk receives the previous chunk's output tensor.
- **GatewayHTTPHandler** -- stdlib `BaseHTTPRequestHandler` exposing two endpoints:
  - `POST /infer` -- Accepts a serialized PyTorch tensor, runs the full chunk pipeline, returns JSON with `output_shape`, `total_time_ms`, per-chunk `chunk_times_ms`, and `network_overhead_ms`.
  - `GET /health` -- Queries each chunk's `HealthCheck` RPC and returns aggregated status; returns HTTP 200 if all ready, 503 if any unavailable.
- **Configuration** -- Reads `CHUNK_HOSTS` (comma-separated `host:port` list) and `GATEWAY_PORT` (default 8080) from environment variables or CLI arguments.
- **Message limits** -- 256 MB max gRPC message size to handle large intermediate tensors.

### 2. Monitor HTTP Service (`monitoring/monitor_service.py`)

A standalone HTTP service wrapping the existing `MetricsCollector` for remote metric access.

- **Endpoints**:
  - `GET /health` -- Returns `{"status": "healthy", "monitoring_active": bool}`.
  - `GET /metrics` -- Full raw samples (GPU + CPU + inference latencies).
  - `GET /metrics/gpu` -- GPU samples only.
  - `GET /metrics/cpu` -- CPU samples only.
  - `GET /metrics/summary` -- Aggregated summary (avg power, utilization, energy, throughput, etc.) without raw sample arrays.
  - `POST /start` -- Start monitoring if not already running.
  - `POST /stop` -- Stop monitoring and return summary.
  - `POST /reset` -- Clear all samples and restart fresh.
- **Configuration** -- `MONITOR_PORT` (default 9090), `GPU_INDEX` (default 0), `SAMPLE_INTERVAL` (default 1.0s), `ENABLE_GPU` (default true).
- **Auto-start** -- Monitoring begins automatically when the service starts.

### 3. Dockerfile.chunk (`docker/Dockerfile.chunk`)

Container for running a single model chunk gRPC server.

- **Base image**: `nvidia/cuda:12.2.2-runtime-ubuntu22.04` (GPU-capable).
- **Python 3.11** installed via apt.
- **Dependencies**: torch, grpcio, protobuf, pynvml (minimal set).
- **Copies**: `model/`, `proto/`, `monitoring/` directories.
- **Environment variables**: `CHUNK_ID`, `NUM_CHUNKS`, `MODEL_TYPE`, `WEIGHTS_DIR`, `PORT`.
- **Entrypoint**: `python3 -m model.chunk_server`.
- **Health check**: gRPC HealthCheck call via Python inline script.

### 4. Dockerfile.gateway (`docker/Dockerfile.gateway`)

Container for the inference gateway HTTP service.

- **Base image**: `python:3.11-slim` (no GPU needed -- gateway only forwards tensors).
- **Dependencies**: torch, grpcio, protobuf.
- **Copies**: `model/`, `proto/` directories.
- **Environment variables**: `CHUNK_HOSTS`, `GATEWAY_PORT`.
- **Entrypoint**: `python3 -m model.gateway`.
- **Health check**: HTTP GET to `/health`.

### 5. Dockerfile.monitor (`docker/Dockerfile.monitor`)

Container for the monitoring HTTP service.

- **Base image**: `nvidia/cuda:12.2.2-runtime-ubuntu22.04` (needs NVML for GPU monitoring).
- **Python 3.11** installed via apt.
- **Dependencies**: pynvml, psutil (minimal set).
- **Copies**: `monitoring/` directory only.
- **Environment variables**: `MONITOR_PORT`, `GPU_INDEX`, `SAMPLE_INTERVAL`, `ENABLE_GPU`.
- **Entrypoint**: `python3 -m monitoring.monitor_service`.
- **Health check**: HTTP GET to `/health`.

### 6. Docker Compose (`docker/docker-compose.yml`)

Multi-container orchestration file for local testing.

- **Services**:
  - `chunk-0`, `chunk-1`, `chunk-2` -- Three chunk servers (ports 50051--50053), all using `Dockerfile.chunk` with NVIDIA GPU reservations and a shared `chunk-data` volume.
  - `gateway` -- Inference gateway (port 8080) using `Dockerfile.gateway`, depends on all three chunk services.
  - `monitor` -- Monitoring service (port 9090) using `Dockerfile.monitor` with GPU access.
- **Volumes**: `chunk-data` shared volume for chunk weight files.
- **GPU access**: All GPU-requiring services use `deploy.resources.reservations.devices` for NVIDIA GPU passthrough.

## How It Was Done

1. **Gateway design** -- Studied the existing `chunk_server.py` gRPC interface (InferRequest/InferResponse messages) and built the gateway to chain stub calls sequentially. Tensor serialization uses `torch.save`/`torch.load` over `io.BytesIO` for byte-level transport, matching the protocol established in Phase 3.

2. **Monitor service design** -- Wrapped the existing `MetricsCollector` (from Phase 4) in a `BaseHTTPRequestHandler` to provide REST access. The service auto-starts monitoring on launch and supports runtime control (start/stop/reset) for flexible experiment management in Kubernetes.

3. **Dockerfile strategy** -- Each Dockerfile installs only the dependencies its service needs:
   - Chunk server needs CUDA runtime + PyTorch + gRPC.
   - Gateway needs only PyTorch + gRPC (CPU-only, no CUDA).
   - Monitor needs only CUDA runtime + pynvml + psutil.
   This keeps images smaller and follows the principle of least privilege.

4. **Docker Compose** -- Configured for local multi-container testing before moving to Kubernetes. Service names (`chunk-0`, `chunk-1`, `chunk-2`) serve as DNS hostnames in the Docker network, matching the `CHUNK_HOSTS` format expected by the gateway.

## Verification

### Gateway End-to-End Test

- Launched 3 chunk servers (ports 50071--50073) and 1 gateway (port 8082) as subprocesses.
- Health check returned all 3 chunks as `ready=true`, status `healthy`.
- Sent a `POST /infer` with a random tensor `[2, 16]` (batch=2, seq_len=16).
- Result: output shape `[2, 10]` (correct for 10-class transformer), total inference 144.3ms across 3 chunks, 13.6ms network overhead.
- Test PASSED.

### Monitor Service Test

- Started monitor service on port 9091 with 0.5s sampling interval.
- `GET /health` returned `{"status": "healthy", "monitoring_active": true}`.
- `GET /metrics/summary` returned all expected keys: `execution_mode`, `total_runtime_s`, `avg_power_w`, `avg_gpu_utilization_pct`, `avg_gpu_memory_used_mb`, `avg_gpu_temperature_c`, `total_energy_wh`, `energy_per_inference_wh`, `avg_cpu_utilization_pct`, `avg_latency_ms`, `throughput_inferences_per_sec`.
- Test PASSED.

## Files Created

| File | Purpose |
|------|---------|
| `model/gateway.py` | HTTP inference gateway with gRPC chunk chain |
| `monitoring/monitor_service.py` | HTTP monitoring service wrapping MetricsCollector |
| `docker/Dockerfile.chunk` | Container image for chunk gRPC server |
| `docker/Dockerfile.gateway` | Container image for inference gateway |
| `docker/Dockerfile.monitor` | Container image for monitoring service |
| `docker/docker-compose.yml` | Multi-container local test orchestration |

## Architecture Diagram

```
Client Request
      |
      v
  [Gateway :8080]          (Dockerfile.gateway)
      |
      +--gRPC--> [Chunk-0 :50051]   (Dockerfile.chunk)
      |               |
      +--gRPC--> [Chunk-1 :50052]   (Dockerfile.chunk)
      |               |
      +--gRPC--> [Chunk-2 :50053]   (Dockerfile.chunk)
      |
      v
  JSON Response

  [Monitor :9090]          (Dockerfile.monitor)
      |
      +-- GPU metrics (pynvml)
      +-- CPU metrics (psutil)
```
