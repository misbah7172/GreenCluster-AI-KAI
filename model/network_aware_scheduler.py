"""
Network-Aware Scheduling for KAI.

Enhances DEAS scheduler with network awareness:
- Track inter-node latency and bandwidth usage
- Avoid placing consecutive layers on high-latency nodes
- Prefer local grouping of dependent layers

Usage::

    from model.network_aware_scheduler import NetworkAwareScheduler
    
    scheduler = NetworkAwareScheduler(event_bus, auto_partitioner)
    scheduler.start()
    
    # Get network-aware placement
    plan = scheduler.compute_network_aware_plan(layers, nodes)
"""

import logging
import threading
import time
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from model.plugin_architecture import PluginRegistry, SchedulerPlugin
from model.deas_scheduler import DEASScheduler, NodeEnergyProfile, MigrationPlan
from model.intelligent_placement import NetworkTopology, NetworkLink

logger = logging.getLogger(__name__)


class NetworkState(Enum):
    """Network health states."""
    HEALTHY = "healthy"       # < 50% utilization
    CONGESTED = "congested"   # 50-80% utilization
    SATURATED = "saturated"   # > 80% utilization


@dataclass
class NetworkMetrics:
    """Real-time network metrics between two nodes."""
    source_node: str
    target_node: str
    latency_ms: float
    bandwidth_used_gbps: float
    bandwidth_total_gbps: float
    packet_loss_pct: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    @property
    def utilization(self) -> float:
        """Bandwidth utilization (0.0-1.0)."""
        if self.bandwidth_total_gbps <= 0:
            return 0.0
        return self.bandwidth_used_gbps / self.bandwidth_total_gbps
    
    @property
    def state(self) -> NetworkState:
        """Current network state."""
        util = self.utilization
        if util < 0.5:
            return NetworkState.HEALTHY
        elif util < 0.8:
            return NetworkState.CONGESTED
        else:
            return NetworkState.SATURATED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "latency_ms": round(self.latency_ms, 3),
            "bandwidth_used_gbps": round(self.bandwidth_used_gbps, 2),
            "bandwidth_total_gbps": round(self.bandwidth_total_gbps, 2),
            "utilization": round(self.utilization, 2),
            "state": self.state.value,
            "packet_loss_pct": round(self.packet_loss_pct, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class NetworkAwarePlacement:
    """Network-aware layer placement."""
    layer_idx: int
    layer_name: str
    node_name: str
    group_id: int  # Layers in same group should be on same node
    expected_transfer_ms: float = 0.0
    network_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "node_name": self.node_name,
            "group_id": self.group_id,
            "expected_transfer_ms": round(self.expected_transfer_ms, 3),
            "network_penalty": round(self.network_penalty, 3),
        }


class NetworkMonitor:
    """Monitors network conditions between nodes.
    
    Periodically measures latency and estimates bandwidth usage.
    
    Parameters
    ----------
    nodes : list[str]
        Node names to monitor
    interval_s : float
        Measurement interval in seconds
    """
    
    def __init__(
        self,
        nodes: List[str],
        interval_s: float = 5.0,
    ):
        self._nodes = list(nodes)
        self._interval = interval_s
        
        # Current metrics: (src, dst) -> NetworkMetrics
        self._metrics: Dict[Tuple[str, str], NetworkMetrics] = {}
        self._history: Dict[Tuple[str, str], List[NetworkMetrics]] = {}
        self._history_size = 100
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks for network state changes
        self._callbacks: List[Callable[[NetworkMetrics], None]] = []

        # Optional explicit link metrics map from env.
        # Format:
        # {
        #   "nodeA->nodeB": {"latency_ms": 0.3, "bandwidth_total_gbps": 100, "bandwidth_used_gbps": 20}
        # }
        self._link_overrides = self._load_link_overrides_from_env()

    @staticmethod
    def _load_link_overrides_from_env() -> Dict[str, Dict[str, float]]:
        raw = os.environ.get("KAI_NETWORK_LINKS_JSON", "")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {
                    str(k): dict(v)
                    for k, v in parsed.items()
                    if isinstance(v, dict)
                }
        except Exception as e:
            logger.warning("Invalid KAI_NETWORK_LINKS_JSON value: %s", e)
        return {}
    
    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="network-monitor",
        )
        self._thread.start()
        logger.info("NetworkMonitor started for %d nodes", len(self._nodes))
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def subscribe(self, callback: Callable[[NetworkMetrics], None]) -> None:
        """Subscribe to network state changes."""
        self._callbacks.append(callback)
    
    def get_metrics(
        self,
        source: str,
        target: str,
    ) -> Optional[NetworkMetrics]:
        """Get current metrics between two nodes."""
        with self._lock:
            return self._metrics.get((source, target))
    
    def get_all_metrics(self) -> List[NetworkMetrics]:
        """Get all current metrics."""
        with self._lock:
            return list(self._metrics.values())
    
    def get_latency(self, source: str, target: str) -> float:
        """Get current latency between nodes."""
        if source == target:
            return 0.01  # Intra-node
        
        metrics = self.get_metrics(source, target)
        if metrics:
            return metrics.latency_ms
        
        # Default if unknown
        return 1.0
    
    def get_topology(self) -> NetworkTopology:
        """Build NetworkTopology from current measurements."""
        topology = NetworkTopology()
        
        with self._lock:
            for (src, dst), metrics in self._metrics.items():
                topology.add_link(
                    src, dst,
                    metrics.latency_ms,
                    metrics.bandwidth_total_gbps,
                )
        
        return topology
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._measure_all()
            except Exception as e:
                logger.error("Network measurement error: %s", e)
            
            time.sleep(self._interval)
    
    def _measure_all(self) -> None:
        """Measure all node pairs."""
        for i, src in enumerate(self._nodes):
            for j, dst in enumerate(self._nodes):
                if i >= j:
                    continue
                
                metrics = self._measure_link(src, dst)
                
                with self._lock:
                    key = (src, dst)
                    old_metrics = self._metrics.get(key)
                    self._metrics[key] = metrics
                    self._metrics[(dst, src)] = NetworkMetrics(
                        source_node=dst,
                        target_node=src,
                        latency_ms=metrics.latency_ms,
                        bandwidth_used_gbps=metrics.bandwidth_used_gbps,
                        bandwidth_total_gbps=metrics.bandwidth_total_gbps,
                        packet_loss_pct=metrics.packet_loss_pct,
                    )
                    
                    # Track history
                    if key not in self._history:
                        self._history[key] = []
                    self._history[key].append(metrics)
                    if len(self._history[key]) > self._history_size:
                        self._history[key] = self._history[key][-self._history_size:]
                
                # Notify on state change
                if old_metrics and old_metrics.state != metrics.state:
                    for callback in self._callbacks:
                        try:
                            callback(metrics)
                        except Exception:
                            pass
    
    def _measure_link(self, source: str, target: str) -> NetworkMetrics:
        """Measure a single link with deterministic behavior.

        No random jitter is introduced. If explicit link metrics are provided,
        those values are used. Otherwise, deterministic synthetic values are
        derived from the node pair identity.
        """
        key = f"{source}->{target}"
        rev = f"{target}->{source}"
        override = self._link_overrides.get(key) or self._link_overrides.get(rev)

        if override:
            return NetworkMetrics(
                source_node=source,
                target_node=target,
                latency_ms=float(override.get("latency_ms", 1.0)),
                bandwidth_used_gbps=float(override.get("bandwidth_used_gbps", 0.0)),
                bandwidth_total_gbps=float(override.get("bandwidth_total_gbps", 10.0)),
                packet_loss_pct=float(override.get("packet_loss_pct", 0.0)),
            )

        # Deterministic synthetic fallback for environments without measurements.
        digest = hashlib.sha256(f"{source}|{target}".encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        latency = 0.6 + (seed % 140) / 100.0  # 0.6 .. 2.0 ms
        bandwidth_total = 10.0
        bandwidth_used = 2.0 + ((seed >> 8) % 40) / 10.0  # 2.0 .. 6.0 Gbps
        packet_loss = ((seed >> 16) % 5) / 100.0  # 0.00 .. 0.04%

        return NetworkMetrics(
            source_node=source,
            target_node=target,
            latency_ms=latency,
            bandwidth_used_gbps=bandwidth_used,
            bandwidth_total_gbps=bandwidth_total,
            packet_loss_pct=packet_loss,
        )


class NetworkAwareScheduler(DEASScheduler):
    """Enhanced DEAS scheduler with network awareness.
    
    Extends DEASScheduler to consider network conditions when
    making scheduling decisions.
    
    Parameters
    ----------
    event_bus : EventBus
        Event bus for threshold events
    auto_partitioner : AutoPartitioner
        Partitioner for recalculation
    network_monitor : NetworkMonitor, optional
        Network monitor (created if not provided)
    max_latency_ms : float
        Maximum acceptable inter-node latency
    min_bandwidth_gbps : float
        Minimum acceptable bandwidth
    """
    
    def __init__(
        self,
        event_bus,
        auto_partitioner,
        controller=None,
        cooldown_s: float = 30.0,
        network_monitor: Optional[NetworkMonitor] = None,
        max_latency_ms: float = 5.0,
        min_bandwidth_gbps: float = 1.0,
    ):
        super().__init__(event_bus, auto_partitioner, controller, cooldown_s)
        
        self._network_monitor = network_monitor
        self._max_latency = max_latency_ms
        self._min_bandwidth = min_bandwidth_gbps
        
        # Layer grouping for network efficiency
        self._layer_groups: Dict[int, int] = {}  # layer_idx -> group_id
    
    def set_network_monitor(self, monitor: NetworkMonitor) -> None:
        """Set the network monitor."""
        self._network_monitor = monitor
    
    def compute_network_aware_plan(
        self,
        layers: List[Tuple[str, Any]],
        nodes: List[Any],
        activation_sizes_mb: Optional[List[float]] = None,
    ) -> List[NetworkAwarePlacement]:
        """Compute network-aware layer placement.
        
        Groups dependent layers to minimize network transfers.
        
        Parameters
        ----------
        layers : list[(name, module)]
            Model layers
        nodes : list[NodeInfo]
            Available nodes
        activation_sizes_mb : list[float], optional
            Size of activations between layers
            
        Returns
        -------
        list[NetworkAwarePlacement]
            Network-optimized placement
        """
        if not nodes:
            raise ValueError("No nodes available")
        
        n_layers = len(layers)
        activation_sizes = activation_sizes_mb or [1.0] * n_layers
        
        # Get network topology
        if self._network_monitor:
            topology = self._network_monitor.get_topology()
        else:
            topology = NetworkTopology()
        
        # Group layers to minimize transfers
        groups = self._compute_layer_groups(layers, nodes, activation_sizes, topology)
        
        # Assign groups to nodes
        placements: List[NetworkAwarePlacement] = []
        node_memory_used: Dict[str, float] = {n.name: 0.0 for n in nodes}
        group_to_node: Dict[int, str] = {}
        
        for i, (name, module) in enumerate(layers):
            group_id = groups[i]
            
            # Estimate memory
            params = sum(p.numel() for p in module.parameters())
            memory_mb = params * 2 / (1024**2)  # FP16
            
            # Determine node
            if group_id in group_to_node:
                node_name = group_to_node[group_id]
            else:
                # Find best node for this group
                node_name = self._find_best_node_for_group(
                    layers, groups, group_id, nodes, node_memory_used, topology
                )
                group_to_node[group_id] = node_name
            
            # Calculate transfer time from previous layer
            transfer_ms = 0.0
            network_penalty = 0.0
            
            if i > 0 and placements[-1].node_name != node_name:
                prev_node = placements[-1].node_name
                link = topology.get_link(prev_node, node_name)
                transfer_ms = link.transfer_time_ms(activation_sizes[i])
                
                # Penalty for high-latency transfers
                if link.latency_ms > self._max_latency:
                    network_penalty = (link.latency_ms - self._max_latency) * 10
            
            placements.append(NetworkAwarePlacement(
                layer_idx=i,
                layer_name=name,
                node_name=node_name,
                group_id=group_id,
                expected_transfer_ms=transfer_ms,
                network_penalty=network_penalty,
            ))
            
            node_memory_used[node_name] += memory_mb
        
        return placements
    
    def should_rebalance_for_network(
        self,
        current_placements: List[NetworkAwarePlacement],
    ) -> bool:
        """Check if rebalancing is needed due to network conditions."""
        if not self._network_monitor:
            return False
        
        total_penalty = sum(p.network_penalty for p in current_placements)
        
        # Rebalance if network penalty exceeds threshold
        return total_penalty > 100.0
    
    def plan_network_aware_migration(
        self,
        current_placements: List[NetworkAwarePlacement],
        current_profiles: List[NodeEnergyProfile],
    ) -> List[MigrationPlan]:
        """Generate migration plans considering network conditions."""
        # First get base DEAS migrations
        base_plans = self.plan_migration(current_profiles)
        
        if not self._network_monitor:
            return base_plans
        
        # Filter out migrations that would increase network overhead
        filtered_plans: List[MigrationPlan] = []
        
        for plan in base_plans:
            # Check network quality of new path
            metrics = self._network_monitor.get_metrics(
                plan.source_node, plan.target_node
            )
            
            if metrics is None:
                filtered_plans.append(plan)
                continue
            
            # Skip if network is saturated
            if metrics.state == NetworkState.SATURATED:
                logger.debug(
                    "Skipping migration %d → %s due to saturated network",
                    plan.chunk_id, plan.target_node
                )
                continue
            
            # Skip if latency is too high
            if metrics.latency_ms > self._max_latency * 2:
                logger.debug(
                    "Skipping migration %d → %s due to high latency (%.1f ms)",
                    plan.chunk_id, plan.target_node, metrics.latency_ms
                )
                continue
            
            filtered_plans.append(plan)
        
        return filtered_plans
    
    def _compute_layer_groups(
        self,
        layers: List[Tuple[str, Any]],
        nodes: List[Any],
        activation_sizes: List[float],
        topology: NetworkTopology,
    ) -> List[int]:
        """Group layers to minimize network transfers.
        
        Layers with large activations or high dependency should be grouped.
        """
        n_layers = len(layers)
        groups = list(range(n_layers))  # Start with each layer in own group
        
        # Calculate average memory per node
        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for _, m in layers
        )
        avg_memory_per_group = total_params * 2 / (1024**2) / len(nodes)  # FP16
        
        # Merge groups where activation transfer would be expensive
        current_group_memory = 0.0
        current_group_id = 0
        
        for i in range(n_layers):
            name, module = layers[i]
            params = sum(p.numel() for p in module.parameters())
            memory_mb = params * 2 / (1024**2)
            
            # Check if should start new group
            if current_group_memory + memory_mb > avg_memory_per_group * 1.5:
                current_group_id += 1
                current_group_memory = 0.0
            
            groups[i] = current_group_id
            current_group_memory += memory_mb
            
            # Keep attention + FFN in same group
            if "attn" in name.lower() and i + 1 < n_layers:
                next_name, _ = layers[i + 1]
                if "mlp" in next_name.lower() or "ffn" in next_name.lower():
                    groups[i + 1] = current_group_id
        
        return groups
    
    def _find_best_node_for_group(
        self,
        layers: List[Tuple[str, Any]],
        groups: List[int],
        target_group: int,
        nodes: List[Any],
        node_memory_used: Dict[str, float],
        topology: NetworkTopology,
    ) -> str:
        """Find best node for a layer group."""
        # Calculate group memory requirement
        group_memory = 0.0
        for i, (_, module) in enumerate(layers):
            if groups[i] == target_group:
                params = sum(p.numel() for p in module.parameters())
                group_memory += params * 2 / (1024**2)
        
        best_node = None
        best_score = float("inf")
        
        for node in nodes:
            available = node.usable_memory_mb - node_memory_used[node.name]
            
            if available < group_memory:
                continue
            
            # Score based on:
            # 1. Memory headroom (prefer nodes with more space)
            # 2. Network quality (if we know previous group's node)
            
            memory_score = (node.usable_memory_mb - available) / node.usable_memory_mb
            
            # Prefer GPU nodes
            gpu_bonus = 0 if node.has_gpu else 0.5
            
            score = memory_score + gpu_bonus
            
            if score < best_score:
                best_score = score
                best_node = node
        
        if best_node is None:
            # Fallback: use node with most available memory
            best_node = max(
                nodes,
                key=lambda n: n.usable_memory_mb - node_memory_used[n.name]
            )
        
        return best_node.name


# Register as plugin
@PluginRegistry.register(
    "scheduler",
    "network_aware",
    description="Network-aware layer scheduling"
)
class NetworkAwareSchedulerPlugin(SchedulerPlugin):
    """Plugin wrapper for NetworkAwareScheduler."""
    
    def __init__(
        self,
        max_latency_ms: float = 5.0,
        min_bandwidth_gbps: float = 1.0,
    ):
        self._max_latency = max_latency_ms
        self._min_bandwidth = min_bandwidth_gbps
        self._network_monitor: Optional[NetworkMonitor] = None
    
    @property
    def name(self) -> str:
        return "network_aware"
    
    def initialize_monitor(self, nodes: List[str]) -> None:
        """Initialize network monitor for given nodes."""
        self._network_monitor = NetworkMonitor(nodes)
        self._network_monitor.start()
    
    def schedule(
        self,
        layers: List[Any],
        nodes: List[Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[int]]:
        constraints = constraints or {}
        
        # Get network topology
        if self._network_monitor:
            topology = self._network_monitor.get_topology()
        else:
            topology = NetworkTopology()
        
        # Simple grouping-based scheduling
        n_layers = len(layers)
        n_nodes = len(nodes)
        
        if n_nodes == 0:
            raise ValueError("No nodes available")
        
        # Calculate layers per node
        layers_per_node = max(1, n_layers // n_nodes)
        
        mapping: Dict[str, List[int]] = {}
        for i, node in enumerate(nodes):
            node_name = node.name if hasattr(node, 'name') else str(i)
            mapping[node_name] = []
        
        node_names = list(mapping.keys())
        
        for i in range(n_layers):
            node_idx = min(i // layers_per_node, n_nodes - 1)
            mapping[node_names[node_idx]].append(i)
        
        return mapping
    
    def rebalance(
        self,
        current_mapping: Dict[str, List[int]],
        metrics: Dict[str, Any],
    ) -> Optional[Dict[str, List[int]]]:
        # Check if network conditions warrant rebalancing
        if not self._network_monitor:
            return None
        
        all_metrics = self._network_monitor.get_all_metrics()
        
        # Count saturated links
        saturated = sum(1 for m in all_metrics if m.state == NetworkState.SATURATED)
        
        if saturated > len(all_metrics) * 0.3:
            # Too many saturated links, might need rebalancing
            logger.warning(
                "Network congestion detected: %d/%d links saturated",
                saturated, len(all_metrics)
            )
            # Return None to trigger full reschedule
            return None
        
        return None
    
    def shutdown(self) -> None:
        if self._network_monitor:
            self._network_monitor.stop()
