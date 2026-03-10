# Phase 7: Kubernetes Configuration -- Documentation

## Objective

Create all Kubernetes manifests to deploy the chunked inference pipeline as a distributed system on a Kubernetes cluster with GPU-equipped nodes.

## What Was Built

### 1. Chunk Deployment (`kubernetes/deployments/chunk-deployment.yaml`)

A Deployment template for model chunk gRPC servers.

- **Kind**: Deployment (replicas: 1 per chunk)
- **Image**: `kai-chunk:latest` (built from `docker/Dockerfile.chunk`)
- **Container port**: 50051 (gRPC)
- **Environment variables**: `CHUNK_ID`, `NUM_CHUNKS`, `MODEL_TYPE`, `WEIGHTS_DIR`, `PORT` -- all configurable per chunk instance
- **GPU resources**: Requests and limits `nvidia.com/gpu: 1` per pod
- **CPU/Memory**: Requests 500m/1Gi, limits 2/4Gi
- **Volume**: `emptyDir` mounted at `/data/chunks` for chunk weight files
- **Health checks**:
  - Readiness probe: gRPC HealthCheck call (initial delay 15s, period 10s)
  - Liveness probe: gRPC HealthCheck call (initial delay 30s, period 15s)
- **Pod anti-affinity**: Prefers spreading chunk pods across different nodes (topology key: `kubernetes.io/hostname`) to maximize GPU distribution
- **GPU toleration**: Tolerates `nvidia.com/gpu` NoSchedule taint so pods can be scheduled on GPU-tainted nodes
- **Labels**: `app: kai`, `component: chunk`, `chunk-id: "N"` -- the `chunk-id` label differentiates chunks and connects each deployment to its corresponding service

The template shows chunk 0. For additional chunks, the `chunk-id` label, deployment name, and `CHUNK_ID` env var are changed. The controller.py script (Phase 8) will create these programmatically for any number of chunks.

### 2. Gateway Deployment (`kubernetes/deployments/gateway-deployment.yaml`)

A single-replica Deployment for the inference gateway HTTP service.

- **Kind**: Deployment (replicas: 1)
- **Image**: `kai-gateway:latest` (built from `docker/Dockerfile.gateway`)
- **Container port**: 8080 (HTTP)
- **No GPU required** -- the gateway only serializes/deserializes tensors and forwards gRPC calls
- **Environment variables**:
  - `GATEWAY_PORT=8080`
  - `CHUNK_HOSTS=kai-chunk-0:50051,kai-chunk-1:50051,kai-chunk-2:50051` -- references the ClusterIP service DNS names
- **CPU/Memory**: Requests 250m/512Mi, limits 1/2Gi
- **Health checks**:
  - Readiness probe: HTTP GET `/health` (initial delay 10s)
  - Liveness probe: HTTP GET `/health` (initial delay 20s)

### 3. Monitor DaemonSet (`kubernetes/deployments/monitor-deployment.yaml`)

A DaemonSet that runs one monitor pod per GPU node.

- **Kind**: DaemonSet (runs on every matching node)
- **Image**: `kai-monitor:latest` (built from `docker/Dockerfile.monitor`)
- **Container port**: 9090 (HTTP)
- **Environment variables**: `MONITOR_PORT`, `GPU_INDEX`, `SAMPLE_INTERVAL`, `ENABLE_GPU`
- **GPU resources**: Requests and limits `nvidia.com/gpu: 1`
- **CPU/Memory**: Requests 100m/256Mi (lightweight)
- **nodeSelector**: `nvidia.com/gpu.present: "true"` -- only schedules on nodes with GPUs
- **GPU toleration**: Tolerates `nvidia.com/gpu` NoSchedule taint
- **Health checks**: HTTP GET `/health` (readiness 5s, liveness 10s initial delay)

Using a DaemonSet instead of a Deployment ensures that every GPU node is monitored regardless of chunk scheduling.

### 4. Chunk Services (`kubernetes/services/chunk-service.yaml`)

Three ClusterIP Services (one per chunk) for internal gRPC routing.

- **Names**: `kai-chunk-0`, `kai-chunk-1`, `kai-chunk-2`
- **Type**: ClusterIP (internal only)
- **Port**: 50051 -> 50051
- **Selectors**: Match pod labels `app: kai`, `component: chunk`, `chunk-id: "N"`

The gateway references these service names in its `CHUNK_HOSTS` env var. Kubernetes DNS resolves `kai-chunk-0` to the ClusterIP of chunk 0's service, enabling the gateway to reach each chunk by name.

### 5. Gateway Service (`kubernetes/services/gateway-service.yaml`)

A NodePort Service exposing the gateway externally.

- **Name**: `kai-gateway`
- **Type**: NodePort
- **Port mapping**: 8080 -> 8080, nodePort 30080
- **Selector**: `app: kai`, `component: gateway`

External access at `http://<node-ip>:30080`. The k8s_runner.py (Phase 9) will use this endpoint to send inference requests from outside the cluster.

### 6. Monitor Service (`kubernetes/services/monitor-service.yaml`)

A ClusterIP Service for the monitoring DaemonSet.

- **Name**: `kai-monitor`
- **Type**: ClusterIP (internal only -- the controller queries pods directly by IP)
- **Port**: 9090 -> 9090
- **Selector**: `app: kai`, `component: monitor`

### 7. GPU Resource Quota (`kubernetes/gpu-resource-quota.yaml`)

A ResourceQuota to cap GPU consumption within the `kai` namespace.

- **Namespace**: `kai`
- **GPU limits**: 4 GPUs max (requests and limits)
- **CPU limits**: 8 requests / 16 limits
- **Memory limits**: 16Gi requests / 32Gi limits
- **Pod limit**: 10 pods max

Prevents the benchmarking workloads from consuming all cluster resources in a shared environment.

## How It Was Done

1. **Naming convention** -- All resources use the prefix `kai-` with a component suffix (e.g., `kai-chunk-0`, `kai-gateway`, `kai-monitor`). Pod labels follow a consistent scheme: `app: kai`, `component: <role>`, plus `chunk-id` for chunks.

2. **Service-Deployment linking** -- Each service's `selector` exactly matches the corresponding deployment's `spec.template.metadata.labels`. This was programmatically verified after creation.

3. **DNS-based service discovery** -- The gateway's `CHUNK_HOSTS` env var uses Kubernetes service DNS names (`kai-chunk-0:50051`, etc.) rather than IP addresses. This lets the gateway resolve chunk endpoints automatically without hardcoded IPs.

4. **GPU scheduling** -- All GPU-requiring pods (chunks, monitors) specify `nvidia.com/gpu: 1` in both requests and limits, and tolerate the standard NVIDIA GPU taint. The monitor DaemonSet additionally uses a `nodeSelector` to only run on GPU-equipped nodes.

5. **Anti-affinity for chunks** -- The chunk deployment uses `preferredDuringSchedulingIgnoredDuringExecution` pod anti-affinity to spread chunks across nodes when possible, maximizing the benefit of distributed inference.

6. **DaemonSet for monitoring** -- Chose DaemonSet over Deployment for the monitor to ensure per-node coverage. Each GPU node gets exactly one monitor pod, providing node-level granularity for power and utilization metrics.

7. **Docker Compose alignment** -- Port numbers (50051 for gRPC, 8080 for gateway HTTP, 9090 for monitor HTTP), image names, and environment variables are consistent with the Docker Compose setup from Phase 6.

## Verification

- All 7 YAML files parsed successfully using `yaml.safe_load_all()`.
- Label selector verification confirmed that every service selector matches its deployment/DaemonSet pod template labels exactly:
  - `kai-chunk-0` service -> `{app: kai, component: chunk, chunk-id: "0"}` -- Match OK
  - `kai-gateway` service -> `{app: kai, component: gateway}` -- Match OK
  - `kai-monitor` service -> `{app: kai, component: monitor}` -- Match OK
- Resource kinds verified: 1 Deployment (chunk), 1 Deployment (gateway), 1 DaemonSet (monitor), 5 Services (3 chunk + 1 gateway + 1 monitor), 1 ResourceQuota.

## Files Created

| File | Kind | Purpose |
|------|------|---------|
| `kubernetes/deployments/chunk-deployment.yaml` | Deployment | Model chunk gRPC server (1 per chunk) |
| `kubernetes/deployments/gateway-deployment.yaml` | Deployment | Inference gateway HTTP service |
| `kubernetes/deployments/monitor-deployment.yaml` | DaemonSet | GPU/CPU monitor (1 per GPU node) |
| `kubernetes/services/chunk-service.yaml` | Service x3 | ClusterIP services for chunk gRPC |
| `kubernetes/services/gateway-service.yaml` | Service | NodePort service for external access |
| `kubernetes/services/monitor-service.yaml` | Service | ClusterIP service for monitor |
| `kubernetes/gpu-resource-quota.yaml` | ResourceQuota | GPU allocation caps for `kai` namespace |

## Deployment Flow

```
1. Create namespace:
     kubectl create namespace kai

2. Apply resource quota:
     kubectl apply -f kubernetes/gpu-resource-quota.yaml -n kai

3. Deploy chunk services (internal DNS):
     kubectl apply -f kubernetes/services/chunk-service.yaml -n kai

4. Deploy chunk servers:
     kubectl apply -f kubernetes/deployments/chunk-deployment.yaml -n kai
     (repeat for chunk-1, chunk-2 with modified CHUNK_ID)

5. Deploy gateway service and deployment:
     kubectl apply -f kubernetes/services/gateway-service.yaml -n kai
     kubectl apply -f kubernetes/deployments/gateway-deployment.yaml -n kai

6. Deploy monitor DaemonSet and service:
     kubectl apply -f kubernetes/services/monitor-service.yaml -n kai
     kubectl apply -f kubernetes/deployments/monitor-deployment.yaml -n kai

7. Verify:
     kubectl get pods -n kai
     curl http://<node-ip>:30080/health
```

This manual flow will be automated by controller.py in Phase 8.
