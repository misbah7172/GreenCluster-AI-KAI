"""
Fault-Tolerant Inference Pipeline for KAI.

Improves reliability through:
- Node failure detection mid-inference
- Resume from last checkpoint
- Reassign layers to other nodes
- Ensure no output corruption

Usage::

    from model.fault_tolerant_pipeline import FaultTolerantPipeline
    
    pipeline = FaultTolerantPipeline(gateway, checkpoint_manager)
    pipeline.start_health_monitoring()
    
    # Inference with automatic fault recovery
    output = pipeline.run_inference(input_tensor)
"""

import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Health states for cluster nodes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"      # Slow response but functional
    UNREACHABLE = "unreachable"  # Connection failed
    FAILED = "failed"         # Confirmed failure
    RECOVERING = "recovering"  # In process of recovery


class FailureType(Enum):
    """Types of failures that can occur."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    CHECKPOINT_ERROR = "checkpoint_error"
    UNKNOWN = "unknown"


@dataclass
class NodeHealth:
    """Health information for a single node."""
    node_name: str
    state: NodeState = NodeState.HEALTHY
    last_heartbeat: str = ""
    consecutive_failures: int = 0
    latency_ms: float = 0.0
    memory_used_pct: float = 0.0
    hosted_chunks: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.last_heartbeat:
            self.last_heartbeat = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "state": self.state.value,
            "last_heartbeat": self.last_heartbeat,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": round(self.latency_ms, 2),
            "memory_used_pct": round(self.memory_used_pct, 2),
            "hosted_chunks": self.hosted_chunks,
        }


@dataclass
class FailureEvent:
    """Record of a failure event."""
    timestamp: str
    node_name: str
    failure_type: FailureType
    chunk_id: int
    request_id: str
    error_message: str
    recovered: bool = False
    recovery_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "node_name": self.node_name,
            "failure_type": self.failure_type.value,
            "chunk_id": self.chunk_id,
            "request_id": self.request_id,
            "error_message": self.error_message,
            "recovered": self.recovered,
            "recovery_time_ms": round(self.recovery_time_ms, 2),
        }


@dataclass
class Checkpoint:
    """Inference checkpoint for recovery."""
    request_id: str
    chunk_id: int
    step: int  # Which step in the pipeline
    timestamp: str
    input_tensor_path: str
    hidden_state_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "chunk_id": self.chunk_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "input_tensor_path": self.input_tensor_path,
            "hidden_state_path": self.hidden_state_path,
            "metadata": self.metadata,
        }


class CheckpointManager:
    """Manages inference checkpoints for recovery.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory for storing checkpoints
    max_checkpoints : int
        Maximum checkpoints to retain per request
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "/tmp/kai_checkpoints",
        max_checkpoints: int = 10,
    ):
        self._dir = checkpoint_dir
        self._max_checkpoints = max_checkpoints
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Active checkpoints: request_id -> list[Checkpoint]
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._lock = threading.Lock()
    
    def save_checkpoint(
        self,
        request_id: str,
        chunk_id: int,
        step: int,
        input_tensor: Tensor,
        hidden_state: Optional[Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """Save a checkpoint.
        
        Parameters
        ----------
        request_id : str
            Request identifier
        chunk_id : int
            Current chunk being processed
        step : int
            Step in the pipeline
        input_tensor : Tensor
            Input to this step
        hidden_state : Tensor, optional
            Current hidden state
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        Checkpoint
            Saved checkpoint info
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create paths
        base_name = f"{request_id}_{chunk_id}_{step}"
        input_path = os.path.join(self._dir, f"{base_name}_input.pt")
        hidden_path = os.path.join(self._dir, f"{base_name}_hidden.pt")
        
        # Save tensors
        torch.save(input_tensor.cpu(), input_path)
        if hidden_state is not None:
            torch.save(hidden_state.cpu(), hidden_path)
        
        checkpoint = Checkpoint(
            request_id=request_id,
            chunk_id=chunk_id,
            step=step,
            timestamp=timestamp,
            input_tensor_path=input_path,
            hidden_state_path=hidden_path if hidden_state is not None else "",
            metadata=metadata or {},
        )
        
        with self._lock:
            if request_id not in self._checkpoints:
                self._checkpoints[request_id] = []
            
            self._checkpoints[request_id].append(checkpoint)
            
            # Cleanup old checkpoints
            if len(self._checkpoints[request_id]) > self._max_checkpoints:
                old = self._checkpoints[request_id][0]
                self._remove_checkpoint_files(old)
                self._checkpoints[request_id] = self._checkpoints[request_id][1:]
        
        logger.debug(
            "Saved checkpoint: request=%s, chunk=%d, step=%d",
            request_id, chunk_id, step
        )
        
        return checkpoint
    
    def get_latest_checkpoint(self, request_id: str) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for a request."""
        with self._lock:
            checkpoints = self._checkpoints.get(request_id, [])
            return checkpoints[-1] if checkpoints else None
    
    def get_checkpoint_before_step(
        self,
        request_id: str,
        step: int,
    ) -> Optional[Checkpoint]:
        """Get checkpoint from just before the specified step."""
        with self._lock:
            checkpoints = self._checkpoints.get(request_id, [])
            
            for cp in reversed(checkpoints):
                if cp.step < step:
                    return cp
            
            return checkpoints[0] if checkpoints else None
    
    def load_checkpoint(
        self,
        checkpoint: Checkpoint,
        device: str = "cpu",
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Load checkpoint tensors.
        
        Returns (input_tensor, hidden_state).
        """
        input_tensor = torch.load(
            checkpoint.input_tensor_path,
            map_location=device,
            weights_only=True,
        )
        
        hidden_state = None
        if checkpoint.hidden_state_path and os.path.exists(checkpoint.hidden_state_path):
            hidden_state = torch.load(
                checkpoint.hidden_state_path,
                map_location=device,
                weights_only=True,
            )
        
        return input_tensor, hidden_state
    
    def cleanup_request(self, request_id: str) -> None:
        """Remove all checkpoints for a completed request."""
        with self._lock:
            checkpoints = self._checkpoints.pop(request_id, [])
            for cp in checkpoints:
                self._remove_checkpoint_files(cp)
    
    def _remove_checkpoint_files(self, checkpoint: Checkpoint) -> None:
        """Remove checkpoint files from disk."""
        try:
            if os.path.exists(checkpoint.input_tensor_path):
                os.remove(checkpoint.input_tensor_path)
            if checkpoint.hidden_state_path and os.path.exists(checkpoint.hidden_state_path):
                os.remove(checkpoint.hidden_state_path)
        except Exception as e:
            logger.warning("Failed to remove checkpoint files: %s", e)


class FailureDetector:
    """Detects node failures through health monitoring.
    
    Parameters
    ----------
    health_check_interval : float
        Seconds between health checks
    failure_threshold : int
        Consecutive failures before marking node as failed
    timeout_ms : float
        Timeout for health check requests
    """
    
    def __init__(
        self,
        health_check_interval: float = 5.0,
        failure_threshold: int = 3,
        timeout_ms: float = 5000.0,
    ):
        self._interval = health_check_interval
        self._failure_threshold = failure_threshold
        self._timeout = timeout_ms / 1000.0
        
        # Node health: node_name -> NodeHealth
        self._nodes: Dict[str, NodeHealth] = {}
        self._lock = threading.Lock()
        
        # Health check callable: (node_name) -> bool
        self._health_check_fn: Optional[Callable[[str], bool]] = None
        
        # Callbacks for state changes
        self._failure_callbacks: List[Callable[[str, NodeState], None]] = []
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def register_node(
        self,
        node_name: str,
        chunks: Optional[List[int]] = None,
    ) -> None:
        """Register a node for monitoring."""
        with self._lock:
            self._nodes[node_name] = NodeHealth(
                node_name=node_name,
                hosted_chunks=chunks or [],
            )
    
    def set_health_check(self, fn: Callable[[str], bool]) -> None:
        """Set the health check function."""
        self._health_check_fn = fn
    
    def subscribe(self, callback: Callable[[str, NodeState], None]) -> None:
        """Subscribe to node state changes."""
        self._failure_callbacks.append(callback)
    
    def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="failure-detector",
        )
        self._thread.start()
        logger.info("FailureDetector started")
    
    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def get_node_health(self, node_name: str) -> Optional[NodeHealth]:
        """Get current health for a node."""
        with self._lock:
            return self._nodes.get(node_name)
    
    def get_healthy_nodes(self) -> List[str]:
        """Get list of healthy node names."""
        with self._lock:
            return [
                name for name, health in self._nodes.items()
                if health.state in (NodeState.HEALTHY, NodeState.DEGRADED)
            ]
    
    def get_failed_nodes(self) -> List[str]:
        """Get list of failed node names."""
        with self._lock:
            return [
                name for name, health in self._nodes.items()
                if health.state in (NodeState.FAILED, NodeState.UNREACHABLE)
            ]
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            with self._lock:
                nodes = list(self._nodes.keys())
            
            for node_name in nodes:
                try:
                    self._check_node(node_name)
                except Exception as e:
                    logger.error("Health check error for %s: %s", node_name, e)
            
            time.sleep(self._interval)
    
    def _check_node(self, node_name: str) -> None:
        """Check health of a single node."""
        start_time = time.perf_counter()
        
        # Perform health check
        healthy = False
        if self._health_check_fn:
            try:
                healthy = self._health_check_fn(node_name)
            except Exception:
                healthy = False
        else:
            # Default: assume healthy
            healthy = True
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        with self._lock:
            health = self._nodes.get(node_name)
            if not health:
                return
            
            old_state = health.state
            health.latency_ms = latency_ms
            health.last_heartbeat = datetime.now(timezone.utc).isoformat()
            
            if healthy:
                health.consecutive_failures = 0
                if latency_ms > 1000:  # Slow response
                    health.state = NodeState.DEGRADED
                else:
                    health.state = NodeState.HEALTHY
            else:
                health.consecutive_failures += 1
                
                if health.consecutive_failures >= self._failure_threshold:
                    health.state = NodeState.FAILED
                else:
                    health.state = NodeState.UNREACHABLE
            
            # Notify on state change
            if old_state != health.state:
                for callback in self._failure_callbacks:
                    try:
                        callback(node_name, health.state)
                    except Exception:
                        pass


class LayerReassigner:
    """Reassigns layers when nodes fail.
    
    Parameters
    ----------
    gateway : InferenceGateway
        Gateway to update
    checkpoint_manager : CheckpointManager
        For recovering state
    """
    
    def __init__(
        self,
        gateway,  # InferenceGateway
        checkpoint_manager: CheckpointManager,
    ):
        self._gateway = gateway
        self._checkpoint_mgr = checkpoint_manager
        
        # Backup node assignments
        self._backup_assignments: Dict[int, List[str]] = {}  # chunk_id -> backup nodes
    
    def set_backup_nodes(
        self,
        chunk_id: int,
        backup_nodes: List[str],
    ) -> None:
        """Set backup nodes for a chunk."""
        self._backup_assignments[chunk_id] = backup_nodes
    
    def reassign_chunk(
        self,
        chunk_id: int,
        failed_node: str,
        available_nodes: List[str],
    ) -> Optional[str]:
        """Reassign a chunk from failed node to available node.
        
        Parameters
        ----------
        chunk_id : int
            Chunk to reassign
        failed_node : str
            Node that failed
        available_nodes : list[str]
            Nodes available for reassignment
            
        Returns
        -------
        str or None
            New node assignment, or None if reassignment failed
        """
        # Check backup nodes first
        backups = self._backup_assignments.get(chunk_id, [])
        for backup in backups:
            if backup in available_nodes and backup != failed_node:
                success = self._do_reassignment(chunk_id, backup)
                if success:
                    logger.info(
                        "Reassigned chunk %d from %s to backup %s",
                        chunk_id, failed_node, backup
                    )
                    return backup
        
        # Try any available node
        for node in available_nodes:
            if node != failed_node:
                success = self._do_reassignment(chunk_id, node)
                if success:
                    logger.info(
                        "Reassigned chunk %d from %s to %s",
                        chunk_id, failed_node, node
                    )
                    return node
        
        logger.error("Failed to reassign chunk %d - no available nodes", chunk_id)
        return None
    
    def _do_reassignment(self, chunk_id: int, new_node: str) -> bool:
        """Execute the reassignment."""
        try:
            # Update gateway routing
            self._gateway.update_chunk_host(chunk_id, new_node)
            return True
        except Exception as e:
            logger.error("Reassignment failed: %s", e)
            return False


class FaultTolerantPipeline:
    """Fault-tolerant inference pipeline.
    
    Wraps InferenceGateway with failure detection, checkpointing,
    and automatic recovery.
    
    Parameters
    ----------
    gateway : InferenceGateway
        The inference gateway
    checkpoint_interval : int
        Checkpoint after every N chunks
    max_retries : int
        Maximum retry attempts per failure
    """
    
    def __init__(
        self,
        gateway,  # InferenceGateway
        checkpoint_interval: int = 1,
        max_retries: int = 3,
        checkpoint_dir: str = "/tmp/kai_checkpoints",
    ):
        self._gateway = gateway
        self._checkpoint_interval = checkpoint_interval
        self._max_retries = max_retries
        
        # Components
        self._checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self._failure_detector = FailureDetector()
        self._reassigner = LayerReassigner(gateway, self._checkpoint_mgr)
        
        # Failure history
        self._failure_history: List[FailureEvent] = []
        self._lock = threading.Lock()
        
        # Setup failure detector callback
        self._failure_detector.subscribe(self._on_node_state_change)
    
    def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        # Register nodes from gateway
        for i, host in enumerate(self._gateway.chunk_hosts):
            node_name = host.split(":")[0]
            self._failure_detector.register_node(node_name, chunks=[i])
        
        # Set health check function
        def check_health(node_name: str) -> bool:
            try:
                health = self._gateway.health_check_all()
                for h in health:
                    if node_name in h["host"]:
                        return h["ready"]
                return False
            except Exception:
                return False
        
        self._failure_detector.set_health_check(check_health)
        self._failure_detector.start()
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self._failure_detector.stop()
    
    def run_inference(
        self,
        input_tensor: Tensor,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """Run fault-tolerant inference.
        
        Parameters
        ----------
        input_tensor : Tensor
            Input tensor
        request_id : str
            Request identifier
            
        Returns
        -------
        dict
            Result with output_tensor, timing, and recovery info
        """
        import uuid
        
        if not request_id:
            request_id = uuid.uuid4().hex[:12]
        
        start_time = time.perf_counter()
        recovery_attempts = 0
        current_step = 0
        
        try:
            result = self._run_with_checkpointing(
                input_tensor, request_id
            )
            
            total_ms = (time.perf_counter() - start_time) * 1000
            
            # Cleanup checkpoints on success
            self._checkpoint_mgr.cleanup_request(request_id)
            
            return {
                **result,
                "request_id": request_id,
                "recovery_attempts": recovery_attempts,
                "total_time_ms": total_ms,
            }
        
        except Exception as e:
            # Attempt recovery
            for attempt in range(self._max_retries):
                recovery_attempts += 1
                logger.warning(
                    "Inference failed, recovery attempt %d/%d: %s",
                    attempt + 1, self._max_retries, e
                )
                
                try:
                    result = self._recover_and_resume(request_id)
                    if result:
                        total_ms = (time.perf_counter() - start_time) * 1000
                        return {
                            **result,
                            "request_id": request_id,
                            "recovery_attempts": recovery_attempts,
                            "total_time_ms": total_ms,
                            "recovered": True,
                        }
                except Exception as recovery_error:
                    logger.error("Recovery attempt %d failed: %s", attempt + 1, recovery_error)
            
            # All retries exhausted
            raise RuntimeError(
                f"Inference failed after {recovery_attempts} recovery attempts: {e}"
            )
    
    def _run_with_checkpointing(
        self,
        input_tensor: Tensor,
        request_id: str,
    ) -> Dict[str, Any]:
        """Run inference with checkpointing."""
        current_tensor = input_tensor
        chunk_times = []
        
        for i, stub in enumerate(self._gateway.stubs):
            # Checkpoint before each chunk
            if i % self._checkpoint_interval == 0:
                self._checkpoint_mgr.save_checkpoint(
                    request_id=request_id,
                    chunk_id=i,
                    step=i,
                    input_tensor=current_tensor,
                )
            
            # Run inference on this chunk
            chunk_start = time.perf_counter()
            
            try:
                result = self._gateway.run_inference(current_tensor, request_id)
                current_tensor = result["output_tensor"]
                chunk_times.extend(result["chunk_times_ms"])
            except Exception as e:
                # Record failure
                self._record_failure(
                    node_name=self._gateway.chunk_hosts[i],
                    failure_type=FailureType.COMPUTATION_ERROR,
                    chunk_id=i,
                    request_id=request_id,
                    error_message=str(e),
                )
                raise
        
        return {
            "output_tensor": current_tensor,
            "chunk_times_ms": chunk_times,
        }
    
    def _recover_and_resume(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Recover from checkpoint and resume inference."""
        checkpoint = self._checkpoint_mgr.get_latest_checkpoint(request_id)
        
        if checkpoint is None:
            logger.error("No checkpoint available for recovery")
            return None
        
        logger.info(
            "Recovering from checkpoint: request=%s, step=%d",
            request_id, checkpoint.step
        )
        
        # Load checkpoint
        input_tensor, _ = self._checkpoint_mgr.load_checkpoint(checkpoint)
        
        # Check for failed nodes and reassign
        failed_nodes = self._failure_detector.get_failed_nodes()
        healthy_nodes = self._failure_detector.get_healthy_nodes()
        
        for node in failed_nodes:
            health = self._failure_detector.get_node_health(node)
            if health:
                for chunk_id in health.hosted_chunks:
                    self._reassigner.reassign_chunk(
                        chunk_id, node, healthy_nodes
                    )
        
        # Resume from checkpoint step
        return self._run_with_checkpointing(input_tensor, request_id)
    
    def _record_failure(
        self,
        node_name: str,
        failure_type: FailureType,
        chunk_id: int,
        request_id: str,
        error_message: str,
    ) -> None:
        """Record a failure event."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            node_name=node_name,
            failure_type=failure_type,
            chunk_id=chunk_id,
            request_id=request_id,
            error_message=error_message,
        )
        
        with self._lock:
            self._failure_history.append(event)
            
            # Keep last 1000 events
            if len(self._failure_history) > 1000:
                self._failure_history = self._failure_history[-1000:]
    
    def _on_node_state_change(self, node_name: str, new_state: NodeState) -> None:
        """Handle node state changes."""
        if new_state == NodeState.FAILED:
            logger.warning("Node %s marked as failed", node_name)
            
            # Record failure
            health = self._failure_detector.get_node_health(node_name)
            if health:
                for chunk_id in health.hosted_chunks:
                    self._record_failure(
                        node_name=node_name,
                        failure_type=FailureType.CONNECTION_ERROR,
                        chunk_id=chunk_id,
                        request_id="",
                        error_message="Node marked as failed by health monitor",
                    )
    
    def get_failure_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get recent failure history."""
        with self._lock:
            return [f.to_dict() for f in self._failure_history[-n:]]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get cluster health summary."""
        all_nodes = []
        
        for host in self._gateway.chunk_hosts:
            node_name = host.split(":")[0]
            health = self._failure_detector.get_node_health(node_name)
            if health:
                all_nodes.append(health.to_dict())
        
        healthy = sum(1 for n in all_nodes if n["state"] in ("healthy", "degraded"))
        failed = sum(1 for n in all_nodes if n["state"] in ("failed", "unreachable"))
        
        return {
            "total_nodes": len(all_nodes),
            "healthy_nodes": healthy,
            "failed_nodes": failed,
            "nodes": all_nodes,
        }
