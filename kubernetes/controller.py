"""
Kubernetes controller for the KAI inference pipeline.

Automates deployment, readiness polling, inference requests,
metrics collection, and teardown of the distributed model-chunk
pipeline on a Kubernetes cluster.

Functions:
    deploy_chunks(num_chunks, ...)   -- Create chunk Deployments + Services
    deploy_gateway(num_chunks, ...)  -- Create gateway Deployment + Service
    deploy_monitor()                 -- Create monitor DaemonSet + Service
    wait_for_ready(timeout)          -- Poll until all pods are Running
    send_inference_request(data)     -- POST tensor to the gateway
    collect_metrics()                -- GET metrics from all monitor pods
    teardown()                       -- Delete all KAI resources

Usage::

    python -m kubernetes.controller deploy --num-chunks 3 --model transformer
    python -m kubernetes.controller status
    python -m kubernetes.controller teardown
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

try:
    # The local 'kubernetes/' directory shadows the pip package when running
    # from the project root.  We temporarily remove the project root from
    # sys.path while importing the pip package, then restore it.
    import importlib
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _saved_path = sys.path[:]
    sys.path = [p for p in sys.path if os.path.abspath(p) != _project_root]
    # Also remove any cached reference to our local kubernetes dir
    for _key in list(sys.modules.keys()):
        if _key == "kubernetes" or _key.startswith("kubernetes."):
            del sys.modules[_key]
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    sys.path = _saved_path
except ImportError:
    sys.path = _saved_path if '_saved_path' in dir() else sys.path
    print("ERROR: 'kubernetes' package not installed. Run: pip install kubernetes")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAMESPACE = "kai"
APP_LABEL = "kai"
CHUNK_IMAGE = "kai-chunk:latest"
GATEWAY_IMAGE = "kai-gateway:latest"
MONITOR_IMAGE = "kai-monitor:latest"
GRPC_PORT = 50051
GATEWAY_PORT = 8080
GATEWAY_NODE_PORT = 30080
MONITOR_PORT = 9090


def _parse_node_selector(raw: str) -> Dict[str, str]:
    """Parse node selector string like 'k=v,k2=v2' into a dict."""
    result: Dict[str, str] = {}
    if not raw:
        return result
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kube_config():
    """Load kubeconfig from default location or in-cluster config."""
    try:
        config.load_kube_config()
        logger.info("Loaded kubeconfig from default location")
    except config.ConfigException:
        config.load_incluster_config()
        logger.info("Loaded in-cluster config")


def _ensure_namespace(core_v1: client.CoreV1Api):
    """Create the namespace if it does not exist."""
    try:
        core_v1.read_namespace(NAMESPACE)
        logger.info("Namespace '%s' already exists", NAMESPACE)
    except ApiException as e:
        if e.status == 404:
            ns = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=NAMESPACE)
            )
            core_v1.create_namespace(ns)
            logger.info("Created namespace '%s'", NAMESPACE)
        else:
            raise


def _delete_if_exists(func, name, namespace=NAMESPACE, **kwargs):
    """Call a delete function, ignoring 404 (not found)."""
    try:
        func(name=name, namespace=namespace, **kwargs)
        logger.info("Deleted %s/%s", namespace, name)
    except ApiException as e:
        if e.status == 404:
            logger.debug("%s/%s not found, skipping", namespace, name)
        else:
            raise


# ---------------------------------------------------------------------------
# Deployment builders
# ---------------------------------------------------------------------------

def _build_chunk_deployment(
    chunk_id: int,
    num_chunks: int,
    model_type: str = "transformer",
    image: str = CHUNK_IMAGE,
    rdma_enabled: bool = False,
    nccl_enabled: bool = False,
) -> client.V1Deployment:
    """Build a Deployment object for a single model chunk."""

    name = f"kai-chunk-{chunk_id}"
    labels = {
        "app": APP_LABEL,
        "component": "chunk",
        "chunk-id": str(chunk_id),
    }

    chunk_env = [
        client.V1EnvVar(name="CHUNK_ID", value=str(chunk_id)),
        client.V1EnvVar(name="NUM_CHUNKS", value=str(num_chunks)),
        client.V1EnvVar(name="MODEL_TYPE", value=model_type),
        client.V1EnvVar(name="WEIGHTS_DIR", value="/data/chunks"),
        client.V1EnvVar(name="PORT", value=str(GRPC_PORT)),
    ]

    if nccl_enabled:
        chunk_env.extend([
            client.V1EnvVar(name="NCCL_IB_DISABLE", value="0" if rdma_enabled else "1"),
            client.V1EnvVar(name="NCCL_ASYNC_ERROR_HANDLING", value="1"),
            client.V1EnvVar(name="NCCL_DEBUG", value=os.environ.get("KAI_NCCL_DEBUG", "WARN")),
        ])
        iface = os.environ.get("KAI_NCCL_SOCKET_IFNAME", "")
        if iface:
            chunk_env.append(client.V1EnvVar(name="NCCL_SOCKET_IFNAME", value=iface))

    container = client.V1Container(
        name="chunk-server",
        image=image,
        image_pull_policy="IfNotPresent",
        ports=[client.V1ContainerPort(container_port=GRPC_PORT, name="grpc")],
        env=chunk_env,
        resources=client.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi", "nvidia.com/gpu": "1"},
            limits={"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"},
        ),
        volume_mounts=[
            client.V1VolumeMount(name="chunk-data", mount_path="/data/chunks"),
        ],
    )

    anti_affinity = client.V1PodAntiAffinity(
        preferred_during_scheduling_ignored_during_execution=[
            client.V1WeightedPodAffinityTerm(
                weight=100,
                pod_affinity_term=client.V1PodAffinityTerm(
                    label_selector=client.V1LabelSelector(
                        match_labels={"app": APP_LABEL, "component": "chunk"},
                    ),
                    topology_key="kubernetes.io/hostname",
                ),
            )
        ]
    )

    node_selector = {"nvidia.com/gpu.present": "true"}
    node_selector.update(_parse_node_selector(os.environ.get("KAI_CHUNK_NODE_SELECTOR", "")))
    if rdma_enabled:
        node_selector.update(_parse_node_selector(os.environ.get("KAI_RDMA_NODE_SELECTOR", "rdma.capable=true")))

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=client.V1PodSpec(
            containers=[container],
            volumes=[
                client.V1Volume(name="chunk-data", empty_dir=client.V1EmptyDirVolumeSource()),
            ],
            affinity=client.V1Affinity(pod_anti_affinity=anti_affinity),
            node_selector=node_selector,
            tolerations=[
                client.V1Toleration(
                    key="nvidia.com/gpu", operator="Exists", effect="NoSchedule",
                ),
            ],
        ),
    )

    return client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name, namespace=NAMESPACE, labels=labels),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels=labels),
            template=template,
        ),
    )


def _build_chunk_service(chunk_id: int) -> client.V1Service:
    """Build a ClusterIP Service for a single chunk."""
    name = f"kai-chunk-{chunk_id}"
    labels = {
        "app": APP_LABEL,
        "component": "chunk",
        "chunk-id": str(chunk_id),
    }
    return client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=name, namespace=NAMESPACE, labels=labels),
        spec=client.V1ServiceSpec(
            type="ClusterIP",
            selector=labels,
            ports=[
                client.V1ServicePort(name="grpc", port=GRPC_PORT, target_port=GRPC_PORT),
            ],
        ),
    )


def _build_gateway_deployment(
    num_chunks: int,
    image: str = GATEWAY_IMAGE,
    rdma_enabled: bool = False,
    nccl_enabled: bool = False,
) -> client.V1Deployment:
    """Build the gateway Deployment."""

    labels = {"app": APP_LABEL, "component": "gateway"}
    chunk_hosts = ",".join(
        f"kai-chunk-{i}:{GRPC_PORT}" for i in range(num_chunks)
    )

    gateway_env = [
        client.V1EnvVar(name="GATEWAY_PORT", value=str(GATEWAY_PORT)),
        client.V1EnvVar(name="CHUNK_HOSTS", value=chunk_hosts),
        client.V1EnvVar(
            name="KAI_GATEWAY_ROUTE_POLICY",
            value=os.environ.get("KAI_GATEWAY_ROUTE_POLICY", "deterministic-latency"),
        ),
    ]

    latency_json = os.environ.get("KAI_LINK_LATENCY_MS", "")
    if latency_json:
        gateway_env.append(
            client.V1EnvVar(name="KAI_LINK_LATENCY_MS", value=latency_json)
        )

    if nccl_enabled:
        gateway_env.extend([
            client.V1EnvVar(name="KAI_NCCL_ENABLED", value="true"),
            client.V1EnvVar(name="KAI_RDMA_ENABLED", value="true" if rdma_enabled else "false"),
        ])

    gateway_selector = _parse_node_selector(os.environ.get("KAI_GATEWAY_NODE_SELECTOR", ""))

    container = client.V1Container(
        name="gateway",
        image=image,
        image_pull_policy="IfNotPresent",
        ports=[client.V1ContainerPort(container_port=GATEWAY_PORT, name="http")],
        env=gateway_env,
        resources=client.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        readiness_probe=client.V1Probe(
            http_get=client.V1HTTPGetAction(path="/health", port=GATEWAY_PORT),
            initial_delay_seconds=10,
            period_seconds=10,
            timeout_seconds=5,
        ),
        liveness_probe=client.V1Probe(
            http_get=client.V1HTTPGetAction(path="/health", port=GATEWAY_PORT),
            initial_delay_seconds=20,
            period_seconds=15,
            timeout_seconds=5,
        ),
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=client.V1PodSpec(containers=[container], node_selector=gateway_selector or None),
    )

    return client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(
            name="kai-gateway", namespace=NAMESPACE, labels=labels,
        ),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels=labels),
            template=template,
        ),
    )


def _build_gateway_service() -> client.V1Service:
    """Build the NodePort Service for the gateway."""
    labels = {"app": APP_LABEL, "component": "gateway"}
    return client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name="kai-gateway", namespace=NAMESPACE, labels=labels,
        ),
        spec=client.V1ServiceSpec(
            type="NodePort",
            selector=labels,
            ports=[
                client.V1ServicePort(
                    name="http",
                    port=GATEWAY_PORT,
                    target_port=GATEWAY_PORT,
                    node_port=GATEWAY_NODE_PORT,
                ),
            ],
        ),
    )


def _build_monitor_daemonset(
    image: str = MONITOR_IMAGE,
    sampling_rate: float = 1.0,
    tdp_watts: float = 0.0,
    enable_threshold: bool = False,
) -> dict:
    """Build the monitor DaemonSet as a raw dict.

    The kubernetes Python client does not have a first-class DaemonSet model
    in all versions, so we use a dict that can be passed to create_namespaced_daemon_set.
    """
    labels = {"app": APP_LABEL, "component": "monitor"}
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {
            "name": "kai-monitor",
            "namespace": NAMESPACE,
            "labels": labels,
        },
        "spec": {
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [{
                        "name": "monitor",
                        "image": image,
                        "imagePullPolicy": "IfNotPresent",
                        "ports": [{"containerPort": MONITOR_PORT, "name": "http"}],
                        "env": [
                            {"name": "MONITOR_PORT", "value": str(MONITOR_PORT)},
                            {"name": "GPU_INDEX", "value": "0"},
                            {"name": "SAMPLING_RATE", "value": str(sampling_rate)},
                            {"name": "ENABLE_GPU", "value": "true"},
                            {"name": "TDP_WATTS", "value": str(tdp_watts)},
                            {"name": "ENABLE_THRESHOLD", "value": str(enable_threshold).lower()},
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "100m",
                                "memory": "256Mi",
                                "nvidia.com/gpu": "1",
                            },
                            "limits": {
                                "cpu": "500m",
                                "memory": "512Mi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": MONITOR_PORT},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10,
                            "timeoutSeconds": 3,
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": MONITOR_PORT},
                            "initialDelaySeconds": 10,
                            "periodSeconds": 15,
                            "timeoutSeconds": 3,
                        },
                    }],
                    "tolerations": [{
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }],
                    "nodeSelector": {"nvidia.com/gpu.present": "true"},
                },
            },
        },
    }


def _build_monitor_service() -> client.V1Service:
    """Build the ClusterIP Service for the monitor DaemonSet."""
    labels = {"app": APP_LABEL, "component": "monitor"}
    return client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name="kai-monitor", namespace=NAMESPACE, labels=labels,
        ),
        spec=client.V1ServiceSpec(
            type="ClusterIP",
            selector=labels,
            ports=[
                client.V1ServicePort(
                    name="http", port=MONITOR_PORT, target_port=MONITOR_PORT,
                ),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# KAIController
# ---------------------------------------------------------------------------

class KAIController:
    """Orchestrates the full KAI pipeline on Kubernetes.

    Parameters
    ----------
    namespace : str
        Kubernetes namespace (default: ``kai``).
    """

    def __init__(self, namespace: str = NAMESPACE):
        self.namespace = namespace
        _load_kube_config()
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self._num_chunks: int = 0

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def deploy_chunks(
        self,
        num_chunks: int,
        model_type: str = "transformer",
        image: str = CHUNK_IMAGE,
        rdma_enabled: bool = False,
        nccl_enabled: bool = False,
    ) -> None:
        """Create Deployments and Services for N chunks.

        Existing resources with the same names are deleted first
        to ensure idempotency.

        Parameters
        ----------
        num_chunks : int
            Number of model chunks (2-8).
        model_type : str
            ``"transformer"`` or ``"cnn"``.
        image : str
            Docker image for the chunk server.
        """
        _ensure_namespace(self.core_v1)
        self._num_chunks = num_chunks

        for i in range(num_chunks):
            name = f"kai-chunk-{i}"

            # Delete existing resources for idempotency
            _delete_if_exists(self.apps_v1.delete_namespaced_deployment, name)
            _delete_if_exists(self.core_v1.delete_namespaced_service, name)

            # Create service first (DNS must be available for gateway)
            svc = _build_chunk_service(i)
            self.core_v1.create_namespaced_service(self.namespace, svc)
            logger.info("Created Service %s", name)

            # Create deployment
            dep = _build_chunk_deployment(
                i,
                num_chunks,
                model_type,
                image,
                rdma_enabled=rdma_enabled,
                nccl_enabled=nccl_enabled,
            )
            self.apps_v1.create_namespaced_deployment(self.namespace, dep)
            logger.info("Created Deployment %s", name)

        logger.info("Deployed %d chunk(s) (model=%s)", num_chunks, model_type)

    def deploy_gateway(
        self,
        num_chunks: int,
        image: str = GATEWAY_IMAGE,
        rdma_enabled: bool = False,
        nccl_enabled: bool = False,
    ) -> None:
        """Create the gateway Deployment and NodePort Service.

        Parameters
        ----------
        num_chunks : int
            Total chunks (used to build CHUNK_HOSTS env var).
        image : str
            Docker image for the gateway.
        """
        _ensure_namespace(self.core_v1)
        self._num_chunks = num_chunks

        _delete_if_exists(self.apps_v1.delete_namespaced_deployment, "kai-gateway")
        _delete_if_exists(self.core_v1.delete_namespaced_service, "kai-gateway")

        svc = _build_gateway_service()
        self.core_v1.create_namespaced_service(self.namespace, svc)
        logger.info("Created Service kai-gateway (NodePort %d)", GATEWAY_NODE_PORT)

        dep = _build_gateway_deployment(
            num_chunks,
            image,
            rdma_enabled=rdma_enabled,
            nccl_enabled=nccl_enabled,
        )
        self.apps_v1.create_namespaced_deployment(self.namespace, dep)
        logger.info("Created Deployment kai-gateway")

    def deploy_monitor(
        self,
        image: str = MONITOR_IMAGE,
        sampling_rate: float = 1.0,
        tdp_watts: float = 0.0,
        enable_threshold: bool = False,
    ) -> None:
        """Create the monitor DaemonSet and Service.

        Parameters
        ----------
        image : str
            Docker image for the monitor service.
        sampling_rate : float
            GPU sampling interval in seconds (default 1.0).
        tdp_watts : float
            GPU TDP in watts (0 = auto-detect).
        enable_threshold : bool
            Enable the power threshold service on each monitor pod.
        """
        _ensure_namespace(self.core_v1)

        # Delete existing
        _delete_if_exists(
            self.apps_v1.delete_namespaced_daemon_set, "kai-monitor",
        )
        _delete_if_exists(self.core_v1.delete_namespaced_service, "kai-monitor")

        svc = _build_monitor_service()
        self.core_v1.create_namespaced_service(self.namespace, svc)
        logger.info("Created Service kai-monitor")

        ds_body = _build_monitor_daemonset(
            image,
            sampling_rate=sampling_rate,
            tdp_watts=tdp_watts,
            enable_threshold=enable_threshold,
        )
        self.apps_v1.create_namespaced_daemon_set(self.namespace, ds_body)
        logger.info("Created DaemonSet kai-monitor")

    def deploy_all(
        self,
        num_chunks: int = 3,
        model_type: str = "transformer",
        rdma_enabled: bool = False,
        nccl_enabled: bool = False,
    ) -> None:
        """Deploy the full pipeline: chunks + gateway + monitor.

        Parameters
        ----------
        num_chunks : int
            Number of model chunks.
        model_type : str
            ``"transformer"`` or ``"cnn"``.
        """
        self.deploy_chunks(
            num_chunks,
            model_type,
            rdma_enabled=rdma_enabled,
            nccl_enabled=nccl_enabled,
        )
        self.deploy_gateway(
            num_chunks,
            rdma_enabled=rdma_enabled,
            nccl_enabled=nccl_enabled,
        )
        self.deploy_monitor()
        logger.info("Full pipeline deployed (%d chunks, model=%s)", num_chunks, model_type)

    # ------------------------------------------------------------------
    # Wait for readiness
    # ------------------------------------------------------------------

    def wait_for_ready(self, timeout: float = 300.0, poll_interval: float = 5.0) -> bool:
        """Poll pod status until all KAI pods are Running and Ready.

        Parameters
        ----------
        timeout : float
            Maximum wait time in seconds.
        poll_interval : float
            Seconds between polls.

        Returns
        -------
        bool
            True if all pods became ready within the timeout.
        """
        logger.info("Waiting for all KAI pods to be ready (timeout=%ds)...", int(timeout))
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            pods = self.core_v1.list_namespaced_pod(
                self.namespace, label_selector="app=kai",
            )

            total = len(pods.items)
            ready_count = 0
            pending_names = []

            for pod in pods.items:
                phase = pod.status.phase
                conditions = pod.status.conditions or []
                is_ready = any(
                    c.type == "Ready" and c.status == "True" for c in conditions
                )
                if phase == "Running" and is_ready:
                    ready_count += 1
                else:
                    pending_names.append(f"{pod.metadata.name}({phase})")

            if total > 0 and ready_count == total:
                logger.info("All %d pods are ready", total)
                return True

            elapsed = int(time.monotonic() - start)
            logger.info(
                "  [%ds] %d/%d ready, pending: %s",
                elapsed, ready_count, total,
                ", ".join(pending_names) if pending_names else "none",
            )
            time.sleep(poll_interval)

        logger.error("Timed out waiting for pods after %ds", int(timeout))
        return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of all KAI pods."""
        pods = self.core_v1.list_namespaced_pod(
            self.namespace, label_selector="app=kai",
        )
        result = []
        for pod in pods.items:
            conditions = pod.status.conditions or []
            is_ready = any(
                c.type == "Ready" and c.status == "True" for c in conditions
            )
            result.append({
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "ready": is_ready,
                "node": pod.spec.node_name,
                "ip": pod.status.pod_ip,
                "component": pod.metadata.labels.get("component", "unknown"),
            })
        return {"namespace": self.namespace, "pods": result}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_gateway_url(self) -> str:
        """Determine the gateway URL from the NodePort service.

        Returns
        -------
        str
            URL like ``http://<node-ip>:30080``.
        """
        # Get a node IP
        nodes = self.core_v1.list_node()
        node_ip = None
        for node in nodes.items:
            for addr in node.status.addresses:
                if addr.type in ("InternalIP", "ExternalIP"):
                    node_ip = addr.address
                    break
            if node_ip:
                break

        if not node_ip:
            raise RuntimeError("Could not determine any node IP")

        return f"http://{node_ip}:{GATEWAY_NODE_PORT}"

    def send_inference_request(
        self,
        input_data: bytes,
        gateway_url: Optional[str] = None,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """Send a tensor to the gateway and return the inference result.

        Parameters
        ----------
        input_data : bytes
            Serialized tensor (``torch.save`` format).
        gateway_url : str, optional
            Override the auto-detected gateway URL.
        request_id : str
            Optional request ID for tracing.

        Returns
        -------
        dict
            JSON response with ``output_shape``, ``total_time_ms``,
            ``chunk_times_ms``, ``network_overhead_ms``, ``request_id``.
        """
        url = gateway_url or self.get_gateway_url()
        headers = {"Content-Type": "application/octet-stream"}
        if request_id:
            headers["X-Request-ID"] = request_id

        req = urllib.request.Request(
            f"{url}/infer",
            data=input_data,
            headers=headers,
            method="POST",
        )

        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read().decode())

    def check_gateway_health(
        self, gateway_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check the health of the gateway and all chunks.

        Returns
        -------
        dict
            Gateway health response.
        """
        url = gateway_url or self.get_gateway_url()
        req = urllib.request.Request(f"{url}/health")
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())

    def probe_gateway_latency(
        self,
        gateway_url: Optional[str] = None,
        samples: int = 2,
    ) -> Dict[str, Any]:
        """Trigger gateway latency probing and route recalibration."""
        url = gateway_url or self.get_gateway_url()
        body = json.dumps({"samples": int(samples)}).encode()
        req = urllib.request.Request(
            f"{url}/probe-latency",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read().decode())

    # ------------------------------------------------------------------
    # Metrics collection
    # ------------------------------------------------------------------

    def collect_metrics(self) -> List[Dict[str, Any]]:
        """Gather metrics from all monitor pods.

        Queries each monitor pod's ``/metrics/summary`` endpoint
        directly via pod IP.

        Returns
        -------
        list[dict]
            One summary dict per monitor pod.
        """
        pods = self.core_v1.list_namespaced_pod(
            self.namespace,
            label_selector="app=kai,component=monitor",
        )

        results = []
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            pod_name = pod.metadata.name
            node_name = pod.spec.node_name

            if not pod_ip:
                logger.warning("Monitor pod %s has no IP, skipping", pod_name)
                continue

            try:
                req = urllib.request.Request(
                    f"http://{pod_ip}:{MONITOR_PORT}/metrics/summary"
                )
                resp = urllib.request.urlopen(req, timeout=10)
                summary = json.loads(resp.read().decode())
                summary["pod_name"] = pod_name
                summary["node_name"] = node_name
                results.append(summary)
                logger.info(
                    "Collected metrics from %s (node=%s)", pod_name, node_name,
                )
            except (urllib.error.URLError, TimeoutError) as e:
                logger.warning(
                    "Failed to collect metrics from %s: %s", pod_name, e,
                )
                results.append({
                    "pod_name": pod_name,
                    "node_name": node_name,
                    "error": str(e),
                })

        return results

    def collect_threshold_events(self) -> List[Dict[str, Any]]:
        """Gather threshold events from all monitor pods.

        Queries each monitor pod's ``/metrics/events`` endpoint.

        Returns
        -------
        list[dict]
            Combined list of threshold events from all pods.
        """
        pods = self.core_v1.list_namespaced_pod(
            self.namespace,
            label_selector="app=kai,component=monitor",
        )

        all_events: List[Dict[str, Any]] = []
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            pod_name = pod.metadata.name
            if not pod_ip:
                continue
            try:
                req = urllib.request.Request(
                    f"http://{pod_ip}:{MONITOR_PORT}/metrics/events?n=100"
                )
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode())
                events = data.get("events", [])
                for ev in events:
                    ev["pod_name"] = pod_name
                all_events.extend(events)
            except (urllib.error.URLError, TimeoutError) as e:
                logger.warning(
                    "Failed to collect threshold events from %s: %s", pod_name, e,
                )

        return all_events

    # ------------------------------------------------------------------
    # DEAS (Dynamic Energy-Aware Scheduling)
    # ------------------------------------------------------------------

    def start_deas(
        self,
        event_bus,
        auto_partitioner,
        cooldown_s: float = 30.0,
        energy_controller=None,
    ) -> None:
        """Create and start the DEAS scheduler.

        Parameters
        ----------
        event_bus : EventBus
            Event bus from the monitoring layer.
        auto_partitioner : AutoPartitioner
            Partitioner for recalculation on rebalance.
        cooldown_s : float
            Minimum seconds between migration attempts.
        energy_controller : EnergyFeedbackController, optional
            If provided, DEAS binds to controller scheduler signals so
            overloaded/inefficient worker events can trigger rebalance.
        """
        from model.deas_scheduler import DEASScheduler

        self._deas = DEASScheduler(
            event_bus=event_bus,
            auto_partitioner=auto_partitioner,
            controller=self,
            cooldown_s=cooldown_s,
        )
        if energy_controller is not None:
            self._deas.bind_energy_controller(energy_controller)
        self._deas.start()
        logger.info("DEAS scheduler started (cooldown=%.1fs)", cooldown_s)

    def stop_deas(self) -> None:
        """Stop the DEAS scheduler if running."""
        deas = getattr(self, "_deas", None)
        if deas is not None:
            deas.stop()
            self._deas = None
            logger.info("DEAS scheduler stopped")

    def get_node_energy_profiles(self) -> List[Dict[str, Any]]:
        """Query all monitor pods and build per-node energy profiles.

        Returns
        -------
        list[dict]
            One profile dict per node with ``node_name``, ``avg_power_w``,
            ``throughput``, ``eer``, ``current_chunks``,
            ``threshold_level``, and ``usable_memory_mb`` keys.
        """
        metrics = self.collect_metrics()
        profiles: List[Dict[str, Any]] = []

        # Build chunk-to-node mapping
        chunk_pods = self.core_v1.list_namespaced_pod(
            self.namespace,
            label_selector="app=kai,component=chunk",
        )
        node_chunks: Dict[str, List[int]] = {}
        for pod in chunk_pods.items:
            node = pod.spec.node_name or "unknown"
            chunk_id_str = pod.metadata.labels.get("chunk-id", "-1")
            node_chunks.setdefault(node, []).append(int(chunk_id_str))

        for m in metrics:
            if "error" in m:
                continue
            node = m.get("node_name", "unknown")
            avg_power = m.get("avg_power_w", 0.0)
            throughput = m.get("throughput_inf_per_sec", 0.0)
            eer = throughput / avg_power if avg_power > 0 else 0.0

            # Determine threshold level from latest data
            tdp_pct = m.get("tdp_pct", 0.0)
            if tdp_pct >= 80.0:
                level = "critical"
            elif tdp_pct >= 70.0:
                level = "warning"
            else:
                level = "optimal"

            profiles.append({
                "node_name": node,
                "avg_power_w": avg_power,
                "throughput": throughput,
                "eer": round(eer, 6),
                "current_chunks": node_chunks.get(node, []),
                "threshold_level": level,
                "usable_memory_mb": m.get("usable_memory_mb", 0.0),
            })

        return profiles

    def trigger_rebalance(self) -> Dict[str, Any]:
        """Manually trigger a DEAS rebalance evaluation.

        Returns
        -------
        dict
            Result with ``rebalanced`` flag, ``profiles``, and
            ``migration_plans`` if applicable.
        """
        from model.deas_scheduler import NodeEnergyProfile

        raw_profiles = self.get_node_energy_profiles()

        # Convert to NodeEnergyProfile objects for the scheduler
        profiles = [
            NodeEnergyProfile(
                node_name=p["node_name"],
                avg_power_w=p["avg_power_w"],
                throughput_inf_per_sec=p["throughput"],
                eer=p["eer"],
                current_chunks=p["current_chunks"],
                threshold_level=p["threshold_level"],
                usable_memory_mb=p["usable_memory_mb"],
            )
            for p in raw_profiles
        ]

        deas = getattr(self, "_deas", None)
        if deas is None:
            return {
                "rebalanced": False,
                "reason": "DEAS not started",
                "profiles": raw_profiles,
            }

        should = deas.should_rebalance(profiles)
        if not should:
            return {
                "rebalanced": False,
                "reason": "no_critical_nodes",
                "profiles": raw_profiles,
            }

        plans = deas.plan_migration(profiles)
        results = []
        for plan in plans:
            ok = deas.execute_migration(plan)
            results.append({
                "chunk_id": plan.chunk_id,
                "source": plan.source_node,
                "target": plan.target_node,
                "success": ok,
            })

        return {
            "rebalanced": True,
            "profiles": raw_profiles,
            "migration_plans": results,
        }

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    def start_monitoring(self) -> None:
        """Send POST /start to all monitor pods to begin recording."""
        self._monitor_action("/start", "start")

    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Send POST /stop to all monitor pods and return summaries."""
        return self._monitor_action("/stop", "stop")

    def reset_monitoring(self) -> None:
        """Send POST /reset to all monitor pods to clear and restart."""
        self._monitor_action("/reset", "reset")

    def _monitor_action(self, path: str, action: str) -> List[Dict[str, Any]]:
        """Send a POST to all monitor pods."""
        pods = self.core_v1.list_namespaced_pod(
            self.namespace,
            label_selector="app=kai,component=monitor",
        )
        results = []
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            if not pod_ip:
                continue
            try:
                req = urllib.request.Request(
                    f"http://{pod_ip}:{MONITOR_PORT}{path}",
                    data=b"",
                    method="POST",
                )
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode())
                results.append(data)
                logger.info("Monitor %s on %s: OK", action, pod.metadata.name)
            except (urllib.error.URLError, TimeoutError) as e:
                logger.warning(
                    "Monitor %s failed on %s: %s", action, pod.metadata.name, e,
                )
        return results

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Delete all KAI resources from the namespace.

        Removes deployments, services, and the DaemonSet. Does not
        delete the namespace itself to avoid removing any user data.
        """
        logger.info("Tearing down KAI resources in namespace '%s'...", self.namespace)

        # Discover and delete chunk deployments + services
        # (supports any number of chunks)
        deps = self.apps_v1.list_namespaced_deployment(
            self.namespace, label_selector="app=kai,component=chunk",
        )
        for dep in deps.items:
            name = dep.metadata.name
            _delete_if_exists(self.apps_v1.delete_namespaced_deployment, name)
            _delete_if_exists(self.core_v1.delete_namespaced_service, name)

        # Delete gateway
        _delete_if_exists(self.apps_v1.delete_namespaced_deployment, "kai-gateway")
        _delete_if_exists(self.core_v1.delete_namespaced_service, "kai-gateway")

        # Delete monitor DaemonSet
        _delete_if_exists(self.apps_v1.delete_namespaced_daemon_set, "kai-monitor")
        _delete_if_exists(self.core_v1.delete_namespaced_service, "kai-monitor")

        # Delete resource quota
        _delete_if_exists(self.core_v1.delete_namespaced_resource_quota, "kai-gpu-quota")

        logger.info("Teardown complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KAI Kubernetes Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Command to execute")

    # deploy
    deploy_p = sub.add_parser("deploy", help="Deploy the full pipeline")
    deploy_p.add_argument("--num-chunks", type=int, default=3, help="Number of chunks (2-8)")
    deploy_p.add_argument("--model", type=str, default="transformer", choices=["transformer", "cnn"])
    deploy_p.add_argument("--wait", action="store_true", help="Wait for pods to be ready")
    deploy_p.add_argument("--timeout", type=int, default=300, help="Readiness timeout (seconds)")
    deploy_p.add_argument("--rdma", action="store_true", help="Enable RDMA-aware scheduling profile")
    deploy_p.add_argument("--nccl", action="store_true", help="Enable NCCL env tuning profile")

    # status
    sub.add_parser("status", help="Show pod status")

    # health
    health_p = sub.add_parser("health", help="Check gateway health")
    health_p.add_argument("--gateway-url", type=str, default=None)

    # metrics
    sub.add_parser("metrics", help="Collect metrics from monitor pods")

    # probe-latency
    probe_p = sub.add_parser("probe-latency", help="Probe gateway latencies and refresh route")
    probe_p.add_argument("--gateway-url", type=str, default=None)
    probe_p.add_argument("--samples", type=int, default=2)

    # teardown
    sub.add_parser("teardown", help="Remove all KAI resources")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [controller] %(levelname)s %(message)s",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    ctrl = KAIController()

    if args.command == "deploy":
        ctrl.deploy_all(
            num_chunks=args.num_chunks,
            model_type=args.model,
            rdma_enabled=bool(args.rdma),
            nccl_enabled=bool(args.nccl),
        )
        if args.wait:
            ok = ctrl.wait_for_ready(timeout=args.timeout)
            if not ok:
                sys.exit(1)

    elif args.command == "status":
        status = ctrl.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "health":
        health = ctrl.check_gateway_health(gateway_url=args.gateway_url)
        print(json.dumps(health, indent=2))

    elif args.command == "metrics":
        metrics = ctrl.collect_metrics()
        print(json.dumps(metrics, indent=2))

    elif args.command == "probe-latency":
        result = ctrl.probe_gateway_latency(
            gateway_url=args.gateway_url,
            samples=args.samples,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "teardown":
        ctrl.teardown()


if __name__ == "__main__":
    main()
