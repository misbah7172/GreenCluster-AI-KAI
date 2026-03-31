"""
Intelligent Model Placement Engine.

Replaces static partitioning with optimization-based placement considering:
- GPU VRAM and CPU RAM
- Network latency between nodes
- Node energy efficiency (EER)
- Bandwidth constraints

Usage::

    from model.intelligent_placement import IntelligentPlacementEngine
    
    engine = IntelligentPlacementEngine()
    plan = engine.compute_placement(
        layers=model_layers,
        nodes=cluster_nodes,
        network_topology=topology,
    )
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from model.plugin_architecture import PluginRegistry, PlacementPlugin
from model.resource_detector import NodeInfo

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Placement optimization objectives."""
    LATENCY = "latency"        # Minimize end-to-end latency
    ENERGY = "energy"          # Minimize energy consumption
    MEMORY = "memory"          # Balance memory usage
    BALANCED = "balanced"      # Multi-objective


@dataclass
class NetworkLink:
    """Network link between two nodes."""
    source_node: str
    target_node: str
    latency_ms: float          # Round-trip latency
    bandwidth_gbps: float      # Available bandwidth
    congestion_factor: float = 1.0  # 1.0 = no congestion
    
    @property
    def effective_bandwidth(self) -> float:
        """Bandwidth adjusted for congestion."""
        return self.bandwidth_gbps / self.congestion_factor
    
    def transfer_time_ms(self, size_mb: float) -> float:
        """Estimate transfer time for given data size."""
        transfer_time = (size_mb * 8) / (self.effective_bandwidth * 1000)  # ms
        return self.latency_ms + transfer_time


@dataclass
class NetworkTopology:
    """Network topology of the cluster."""
    links: Dict[Tuple[str, str], NetworkLink] = field(default_factory=dict)
    local_latency_ms: float = 0.01  # Intra-node latency
    
    def add_link(
        self,
        source: str,
        target: str,
        latency_ms: float,
        bandwidth_gbps: float,
    ) -> None:
        """Add a bidirectional link."""
        self.links[(source, target)] = NetworkLink(
            source, target, latency_ms, bandwidth_gbps
        )
        self.links[(target, source)] = NetworkLink(
            target, source, latency_ms, bandwidth_gbps
        )
    
    def get_link(self, source: str, target: str) -> NetworkLink:
        """Get link between nodes."""
        if source == target:
            return NetworkLink(source, target, self.local_latency_ms, float("inf"))
        
        key = (source, target)
        if key in self.links:
            return self.links[key]
        
        # Default link for unknown topology
        return NetworkLink(source, target, 1.0, 10.0)
    
    def get_latency_matrix(self, nodes: List[str]) -> torch.Tensor:
        """Get latency matrix for nodes."""
        n = len(nodes)
        matrix = torch.zeros(n, n)
        
        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes):
                matrix[i, j] = self.get_link(n1, n2).latency_ms
        
        return matrix
    
    @classmethod
    def from_ping_measurements(
        cls,
        nodes: List[str],
        measurements: Dict[Tuple[str, str], float],
        default_bandwidth: float = 10.0,
    ) -> "NetworkTopology":
        """Create topology from ping measurements."""
        topology = cls()
        
        for (src, dst), latency in measurements.items():
            topology.add_link(src, dst, latency, default_bandwidth)
        
        return topology


@dataclass 
class LayerInfo:
    """Information about a model layer for placement."""
    layer_name: str
    layer_idx: int
    parameter_count: int
    memory_mb: float
    compute_flops: float = 0.0
    is_attention: bool = False
    activation_size_mb: float = 0.0  # Size of output activations
    
    @property
    def total_memory_mb(self) -> float:
        """Total memory including activations."""
        return self.memory_mb + self.activation_size_mb


@dataclass
class PlacementAssignment:
    """Placement assignment for a single layer."""
    layer_name: str
    layer_idx: int
    node_name: str
    memory_mb: float
    expected_latency_ms: float
    expected_energy_j: float
    transfer_time_ms: float = 0.0
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_idx": self.layer_idx,
            "node_name": self.node_name,
            "memory_mb": round(self.memory_mb, 2),
            "expected_latency_ms": round(self.expected_latency_ms, 3),
            "expected_energy_j": round(self.expected_energy_j, 3),
            "transfer_time_ms": round(self.transfer_time_ms, 3),
            "reasons": self.reasons,
        }


@dataclass
class PlacementPlan:
    """Complete placement plan for a model."""
    model_name: str
    total_layers: int
    assignments: List[PlacementAssignment]
    total_memory_mb: float
    expected_latency_ms: float
    expected_energy_j: float
    network_overhead_ms: float
    objective: OptimizationObjective
    
    def get_node_assignments(self) -> Dict[str, List[int]]:
        """Get layer indices assigned to each node."""
        result: Dict[str, List[int]] = {}
        for a in self.assignments:
            if a.node_name not in result:
                result[a.node_name] = []
            result[a.node_name].append(a.layer_idx)
        return result
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Placement Plan for {self.model_name}",
            f"  Objective: {self.objective.value}",
            f"  Total layers: {self.total_layers}",
            f"  Total memory: {self.total_memory_mb:.0f} MB",
            f"  Expected latency: {self.expected_latency_ms:.1f} ms",
            f"  Network overhead: {self.network_overhead_ms:.1f} ms",
            f"  Expected energy: {self.expected_energy_j:.2f} J",
            "",
            "Node Assignments:",
        ]
        
        node_assignments = self.get_node_assignments()
        for node, layers in node_assignments.items():
            memory = sum(
                a.memory_mb for a in self.assignments
                if a.node_name == node
            )
            lines.append(
                f"  {node}: layers {min(layers)}-{max(layers)} "
                f"({len(layers)} layers, {memory:.0f} MB)"
            )
        
        return "\n".join(lines)


class IntelligentPlacementEngine:
    """Computes optimal layer-to-node placement.
    
    Uses multi-objective optimization considering:
    - Memory constraints
    - Network latency  
    - Energy efficiency
    - Load balancing
    
    Parameters
    ----------
    objective : OptimizationObjective
        Primary optimization objective
    latency_weight : float
        Weight for latency in balanced mode
    energy_weight : float
        Weight for energy in balanced mode
    memory_weight : float
        Weight for memory balance in balanced mode
    """
    
    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        latency_weight: float = 0.4,
        energy_weight: float = 0.3,
        memory_weight: float = 0.3,
    ):
        self.objective = objective
        self._weights = {
            "latency": latency_weight,
            "energy": energy_weight,
            "memory": memory_weight,
        }
    
    def compute_placement(
        self,
        layers: List[LayerInfo],
        nodes: List[NodeInfo],
        network: Optional[NetworkTopology] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> PlacementPlan:
        """Compute optimal placement for layers.
        
        Parameters
        ----------
        layers : list[LayerInfo]
            Model layers to place
        nodes : list[NodeInfo]
            Available cluster nodes
        network : NetworkTopology, optional
            Network topology
        constraints : dict, optional
            Additional constraints
            
        Returns
        -------
        PlacementPlan
            Optimized placement plan
        """
        if not nodes:
            raise ValueError("No nodes available for placement")
        
        if not layers:
            raise ValueError("No layers to place")
        
        network = network or NetworkTopology()
        constraints = constraints or {}
        
        # Sort nodes by capability (GPU nodes first, then by memory)
        sorted_nodes = sorted(
            nodes,
            key=lambda n: (n.has_gpu, n.usable_memory_mb),
            reverse=True,
        )
        
        # Choose algorithm based on objective
        if self.objective == OptimizationObjective.LATENCY:
            assignments = self._optimize_latency(layers, sorted_nodes, network)
        elif self.objective == OptimizationObjective.ENERGY:
            assignments = self._optimize_energy(layers, sorted_nodes, network)
        elif self.objective == OptimizationObjective.MEMORY:
            assignments = self._optimize_memory(layers, sorted_nodes)
        else:
            assignments = self._optimize_balanced(layers, sorted_nodes, network)
        
        # Calculate totals
        total_memory = sum(a.memory_mb for a in assignments)
        total_latency = sum(a.expected_latency_ms for a in assignments)
        total_energy = sum(a.expected_energy_j for a in assignments)
        network_overhead = sum(a.transfer_time_ms for a in assignments)
        
        return PlacementPlan(
            model_name=constraints.get("model_name", "unknown"),
            total_layers=len(layers),
            assignments=assignments,
            total_memory_mb=total_memory,
            expected_latency_ms=total_latency + network_overhead,
            expected_energy_j=total_energy,
            network_overhead_ms=network_overhead,
            objective=self.objective,
        )
    
    def _optimize_latency(
        self,
        layers: List[LayerInfo],
        nodes: List[NodeInfo],
        network: NetworkTopology,
    ) -> List[PlacementAssignment]:
        """Optimize for minimum latency."""
        assignments: List[PlacementAssignment] = []
        node_memory_used: Dict[str, float] = {n.name: 0.0 for n in nodes}
        
        # Group consecutive layers on same node to minimize transfers
        current_node_idx = 0
        group_start = 0
        
        for i, layer in enumerate(layers):
            node = nodes[current_node_idx]
            
            # Check if layer fits on current node
            if node_memory_used[node.name] + layer.memory_mb > node.usable_memory_mb:
                # Move to next node
                current_node_idx = min(current_node_idx + 1, len(nodes) - 1)
                node = nodes[current_node_idx]
                group_start = i
            
            # Calculate transfer time from previous layer
            transfer_time = 0.0
            if i > 0 and assignments[i-1].node_name != node.name:
                prev_node = assignments[i-1].node_name
                link = network.get_link(prev_node, node.name)
                transfer_time = link.transfer_time_ms(layer.activation_size_mb)
            
            # Estimate compute latency (simplified)
            compute_latency = self._estimate_compute_latency(layer, node)
            
            assignments.append(PlacementAssignment(
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx,
                node_name=node.name,
                memory_mb=layer.memory_mb,
                expected_latency_ms=compute_latency,
                expected_energy_j=self._estimate_energy(layer, node),
                transfer_time_ms=transfer_time,
                reasons=["latency_optimized", f"grouped_layers_{group_start}-{i}"],
            ))
            
            node_memory_used[node.name] += layer.memory_mb
        
        return assignments
    
    def _optimize_energy(
        self,
        layers: List[LayerInfo],
        nodes: List[NodeInfo],
        network: NetworkTopology,
    ) -> List[PlacementAssignment]:
        """Optimize for minimum energy consumption."""
        assignments: List[PlacementAssignment] = []
        node_memory_used: Dict[str, float] = {n.name: 0.0 for n in nodes}
        
        # Calculate EER for each node
        node_eer: Dict[str, float] = {}
        for node in nodes:
            # Higher EER = more efficient (throughput / power)
            # Estimate based on GPU/CPU type
            if node.has_gpu:
                node_eer[node.name] = 1.0  # GPU baseline
            else:
                node_eer[node.name] = 0.3  # CPU less efficient
        
        # Sort nodes by EER (most efficient first)
        nodes_by_eer = sorted(nodes, key=lambda n: node_eer[n.name], reverse=True)
        
        for layer in layers:
            # Find most efficient node that can fit the layer
            best_node = None
            best_score = float("inf")
            
            for node in nodes_by_eer:
                if node_memory_used[node.name] + layer.memory_mb <= node.usable_memory_mb:
                    energy = self._estimate_energy(layer, node)
                    
                    # Penalize network transfers
                    if assignments and assignments[-1].node_name != node.name:
                        prev_node = assignments[-1].node_name
                        link = network.get_link(prev_node, node.name)
                        # Network energy overhead
                        energy += link.transfer_time_ms * 0.01  # ~10W for network
                    
                    if energy < best_score:
                        best_score = energy
                        best_node = node
            
            if best_node is None:
                # Fallback: use node with most memory
                best_node = max(nodes, key=lambda n: n.usable_memory_mb - node_memory_used[n.name])
            
            # Calculate transfer time
            transfer_time = 0.0
            if assignments and assignments[-1].node_name != best_node.name:
                prev_node = assignments[-1].node_name
                link = network.get_link(prev_node, best_node.name)
                transfer_time = link.transfer_time_ms(layer.activation_size_mb)
            
            assignments.append(PlacementAssignment(
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx,
                node_name=best_node.name,
                memory_mb=layer.memory_mb,
                expected_latency_ms=self._estimate_compute_latency(layer, best_node),
                expected_energy_j=best_score,
                transfer_time_ms=transfer_time,
                reasons=["energy_optimized", f"eer={node_eer[best_node.name]:.2f}"],
            ))
            
            node_memory_used[best_node.name] += layer.memory_mb
        
        return assignments
    
    def _optimize_memory(
        self,
        layers: List[LayerInfo],
        nodes: List[NodeInfo],
    ) -> List[PlacementAssignment]:
        """Optimize for balanced memory usage."""
        assignments: List[PlacementAssignment] = []
        node_memory_used: Dict[str, float] = {n.name: 0.0 for n in nodes}
        
        total_memory = sum(l.memory_mb for l in layers)
        total_node_memory = sum(n.usable_memory_mb for n in nodes)
        
        # Calculate target memory per node (proportional to capacity)
        target_memory: Dict[str, float] = {}
        for node in nodes:
            target_memory[node.name] = (
                node.usable_memory_mb / total_node_memory * total_memory
            )
        
        for layer in layers:
            # Find node with most "budget" remaining
            best_node = None
            best_headroom = float("-inf")
            
            for node in nodes:
                remaining = target_memory[node.name] - node_memory_used[node.name]
                fits = node_memory_used[node.name] + layer.memory_mb <= node.usable_memory_mb
                
                if fits and remaining > best_headroom:
                    best_headroom = remaining
                    best_node = node
            
            if best_node is None:
                # Fallback
                best_node = max(nodes, key=lambda n: n.usable_memory_mb - node_memory_used[n.name])
            
            assignments.append(PlacementAssignment(
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx,
                node_name=best_node.name,
                memory_mb=layer.memory_mb,
                expected_latency_ms=self._estimate_compute_latency(layer, best_node),
                expected_energy_j=self._estimate_energy(layer, best_node),
                reasons=["memory_balanced"],
            ))
            
            node_memory_used[best_node.name] += layer.memory_mb
        
        return assignments
    
    def _optimize_balanced(
        self,
        layers: List[LayerInfo],
        nodes: List[NodeInfo],
        network: NetworkTopology,
    ) -> List[PlacementAssignment]:
        """Multi-objective optimization."""
        assignments: List[PlacementAssignment] = []
        node_memory_used: Dict[str, float] = {n.name: 0.0 for n in nodes}
        
        for i, layer in enumerate(layers):
            best_node = None
            best_score = float("inf")
            best_transfer_time = 0.0
            
            for node in nodes:
                if node_memory_used[node.name] + layer.memory_mb > node.usable_memory_mb:
                    continue
                
                # Compute individual scores
                latency = self._estimate_compute_latency(layer, node)
                energy = self._estimate_energy(layer, node)
                memory_balance = abs(
                    node_memory_used[node.name] / node.usable_memory_mb - 0.5
                )
                
                # Network penalty
                transfer_time = 0.0
                if assignments and assignments[-1].node_name != node.name:
                    prev_node = assignments[-1].node_name
                    link = network.get_link(prev_node, node.name)
                    transfer_time = link.transfer_time_ms(layer.activation_size_mb)
                    latency += transfer_time
                    energy += transfer_time * 0.01
                
                # Weighted score (lower is better)
                score = (
                    self._weights["latency"] * latency +
                    self._weights["energy"] * energy * 100 +  # Scale energy
                    self._weights["memory"] * memory_balance * 10
                )
                
                if score < best_score:
                    best_score = score
                    best_node = node
                    best_transfer_time = transfer_time
            
            if best_node is None:
                best_node = max(nodes, key=lambda n: n.usable_memory_mb - node_memory_used[n.name])
            
            assignments.append(PlacementAssignment(
                layer_name=layer.layer_name,
                layer_idx=layer.layer_idx,
                node_name=best_node.name,
                memory_mb=layer.memory_mb,
                expected_latency_ms=self._estimate_compute_latency(layer, best_node),
                expected_energy_j=self._estimate_energy(layer, best_node),
                transfer_time_ms=best_transfer_time,
                reasons=["balanced_optimization"],
            ))
            
            node_memory_used[best_node.name] += layer.memory_mb
        
        return assignments
    
    def _estimate_compute_latency(
        self,
        layer: LayerInfo,
        node: NodeInfo,
    ) -> float:
        """Estimate compute latency for layer on node."""
        # Simplified model: ~1ms per 100M params on GPU, 10ms on CPU
        base_latency = layer.parameter_count / 1e8
        
        if node.has_gpu:
            return max(0.1, base_latency)
        else:
            return max(1.0, base_latency * 10)
    
    def _estimate_energy(
        self,
        layer: LayerInfo,
        node: NodeInfo,
    ) -> float:
        """Estimate energy consumption in Joules."""
        # Simplified: GPU ~200W, CPU ~100W
        latency_s = self._estimate_compute_latency(layer, node) / 1000
        
        if node.has_gpu:
            power_w = 200.0
        else:
            power_w = 100.0
        
        return power_w * latency_s


# Register as plugin
@PluginRegistry.register(
    "placement",
    "intelligent",
    description="Multi-objective placement optimization"
)
class IntelligentPlacementPlugin(PlacementPlugin):
    """Plugin wrapper for IntelligentPlacementEngine."""
    
    def __init__(
        self,
        objective: str = "balanced",
        **kwargs,
    ):
        self._engine = IntelligentPlacementEngine(
            objective=OptimizationObjective(objective),
            **kwargs,
        )
    
    @property
    def name(self) -> str:
        return "intelligent"
    
    def compute_placement(
        self,
        layers: List[Any],
        nodes: List[Any],
        constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Convert to LayerInfo if needed
        layer_infos = []
        for i, layer in enumerate(layers):
            if isinstance(layer, LayerInfo):
                layer_infos.append(layer)
            elif isinstance(layer, tuple):
                name, module = layer
                params = sum(p.numel() for p in module.parameters())
                layer_infos.append(LayerInfo(
                    layer_name=name,
                    layer_idx=i,
                    parameter_count=params,
                    memory_mb=params * 2 / (1024**2),  # FP16
                ))
            else:
                layer_infos.append(LayerInfo(
                    layer_name=str(layer),
                    layer_idx=i,
                    parameter_count=0,
                    memory_mb=0,
                ))
        
        network = constraints.get("network")
        if network and not isinstance(network, NetworkTopology):
            network = None
        
        plan = self._engine.compute_placement(
            layer_infos, nodes, network, constraints
        )
        
        return [a.to_dict() for a in plan.assignments]


def measure_network_topology(
    nodes: List[NodeInfo],
    ping_count: int = 5,
) -> NetworkTopology:
    """Measure network topology by pinging nodes.
    
    This is a simplified implementation - real implementation
    would use actual network measurements.
    """
    import random
    
    topology = NetworkTopology()
    
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            
            # Simulated latency (would be actual ping in production)
            if n1.name == n2.name:
                latency = 0.01
            else:
                # Random latency 0.5-5ms for LAN
                latency = random.uniform(0.5, 5.0)
            
            # Assume 10Gbps LAN
            bandwidth = 10.0
            
            topology.add_link(n1.name, n2.name, latency, bandwidth)
    
    return topology
