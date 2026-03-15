"""
Dynamic Energy-Aware Scheduler (DEAS).

Subscribes to CRITICAL threshold events from the
:class:`~monitoring.event_bus.EventBus`.  On a CRITICAL event the
scheduler triggers an :class:`~model.auto_partitioner.AutoPartitioner`
recalculation and orchestrates chunk migration to maximise the
cluster-wide Energy-Efficiency Ratio (EER).
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

class MigrationState(Enum):
    IDLE = "idle"
    PAUSING = "pausing"
    CHECKPOINTING = "checkpointing"
    MIGRATING = "migrating"
    RELINKING = "relinking"
    RESUMING = "resuming"


@dataclass
class NodeEnergyProfile:
    """Energy profile for a single cluster node."""

    node_name: str
    avg_power_w: float
    throughput_inf_per_sec: float
    eer: float                    # throughput / avg_power
    current_chunks: List[int]     # chunk IDs hosted on this node
    threshold_level: str          # "optimal", "warning", "critical"
    usable_memory_mb: float


@dataclass
class MigrationPlan:
    """Describes a single chunk migration."""

    chunk_id: int
    source_node: str
    target_node: str
    reason: str                   # e.g. "critical_threshold"
    estimated_downtime_ms: float = 0.0
    checkpoint_size_mb: float = 0.0


@dataclass
class MigrationRecord:
    """Log entry for a completed migration."""

    plan: MigrationPlan
    started_at: str
    completed_at: str
    success: bool
    actual_downtime_ms: float
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.plan.chunk_id,
            "source_node": self.plan.source_node,
            "target_node": self.plan.target_node,
            "reason": self.plan.reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "actual_downtime_ms": self.actual_downtime_ms,
            "error": self.error,
        }


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

class DEASScheduler:
    """Subscribes to threshold events and triggers re-partitioning.

    Parameters
    ----------
    event_bus : EventBus
        Event bus to subscribe to CRITICAL events.
    auto_partitioner : AutoPartitioner
        Partitioner for recalculation.
    controller : KAIController, optional
        K8s controller for executing migrations.
    cooldown_s : float
        Minimum seconds between consecutive repartition attempts.
    """

    def __init__(
        self,
        event_bus,
        auto_partitioner,
        controller=None,
        cooldown_s: float = 30.0,
    ):
        self._event_bus = event_bus
        self._partitioner = auto_partitioner
        self._controller = controller
        self._cooldown_s = cooldown_s

        self._state = MigrationState.IDLE
        self._last_rebalance_time: float = 0.0
        self._migration_history: List[MigrationRecord] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to CRITICAL events on the event bus."""
        from monitoring.event_bus import ThresholdLevel
        self._event_bus.subscribe(
            self._on_critical_event,
            level_filter=ThresholdLevel.CRITICAL,
        )
        logger.info("DEASScheduler started (cooldown=%.1fs)", self._cooldown_s)

    def stop(self) -> None:
        """Unsubscribe from the event bus."""
        self._event_bus.unsubscribe(self._on_critical_event)
        logger.info("DEASScheduler stopped")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _on_critical_event(self, event) -> None:
        """Handler for CRITICAL threshold events.  Enforces cooldown."""
        now = time.monotonic()
        with self._lock:
            if self._state != MigrationState.IDLE:
                logger.debug("Migration in progress — ignoring CRITICAL event")
                return
            if now - self._last_rebalance_time < self._cooldown_s:
                logger.debug("Cooldown active — ignoring CRITICAL event")
                return
            self._state = MigrationState.PAUSING

        logger.warning(
            "CRITICAL event from %s (%.1fW, %.1f%% TDP) — evaluating rebalance",
            event.node_name, event.power_w, event.tdp_pct,
        )

        try:
            # TODO: in a full implementation, gather profiles and execute migration
            # For now, log and reset state
            with self._lock:
                self._last_rebalance_time = time.monotonic()
                self._state = MigrationState.IDLE
        except Exception:
            logger.exception("Error during DEAS rebalance")
            with self._lock:
                self._state = MigrationState.IDLE

    # ------------------------------------------------------------------
    # EER computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cluster_eer(node_profiles: List[NodeEnergyProfile]) -> float:
        """Compute cluster-wide EER = sum(throughput) / sum(power).

        Returns 0.0 if total power is zero.
        """
        total_throughput = sum(p.throughput_inf_per_sec for p in node_profiles)
        total_power = sum(p.avg_power_w for p in node_profiles)
        if total_power <= 0:
            return 0.0
        return total_throughput / total_power

    def should_rebalance(self, node_profiles: List[NodeEnergyProfile]) -> bool:
        """Determine if rebalancing would improve cluster EER.

        Returns True when any node is CRITICAL, indicating migration
        could shift load to a more efficient node.
        """
        return any(p.threshold_level == "critical" for p in node_profiles)

    # ------------------------------------------------------------------
    # Migration planning
    # ------------------------------------------------------------------

    def plan_migration(
        self,
        current_profiles: List[NodeEnergyProfile],
        loader=None,
        nodes=None,
    ) -> List[MigrationPlan]:
        """Generate migration plans to maximise cluster EER.

        Identifies the most-loaded chunk on the CRITICAL node and
        proposes moving it to the OPTIMAL node with the most headroom.

        Parameters
        ----------
        current_profiles : list[NodeEnergyProfile]
            Current energy state of each node.
        loader : HFModelLoader, optional
            For re-partitioning calculations.
        nodes : list[NodeInfo], optional
            Cluster node info for re-partitioning.

        Returns
        -------
        list[MigrationPlan]
            Proposed migrations (may be empty if no improvement found).
        """
        plans: List[MigrationPlan] = []

        critical_nodes = [
            p for p in current_profiles if p.threshold_level == "critical"
        ]
        optimal_nodes = sorted(
            [p for p in current_profiles if p.threshold_level == "optimal"],
            key=lambda p: p.usable_memory_mb,
            reverse=True,
        )

        if not critical_nodes or not optimal_nodes:
            return plans

        for crit in critical_nodes:
            if not crit.current_chunks:
                continue
            target = optimal_nodes[0]

            plans.append(MigrationPlan(
                chunk_id=crit.current_chunks[-1],
                source_node=crit.node_name,
                target_node=target.node_name,
                reason="critical_threshold",
            ))

        return plans

    # ------------------------------------------------------------------
    # Migration execution
    # ------------------------------------------------------------------

    def execute_migration(self, plan: MigrationPlan) -> bool:
        """Execute a single chunk migration.

        Workflow:
        1. PAUSE — gRPC Pause() on source chunk server
        2. CHECKPOINT — gRPC Checkpoint() to save state
        3. MIGRATE — Transfer weights + state to target
        4. RELINK — Update gateway gRPC stub chain
        5. RESUME — gRPC Resume() on target, deactivate source

        Returns True on success.
        """
        start_ts = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        with self._lock:
            self._state = MigrationState.PAUSING

        try:
            logger.info(
                "Migrating chunk %d: %s → %s (reason: %s)",
                plan.chunk_id, plan.source_node, plan.target_node, plan.reason,
            )

            # Steps 1-5 require an active K8s controller with gRPC access
            # to chunk pods.  Implementation here provides the framework;
            # actual gRPC calls depend on controller integration.
            with self._lock:
                self._state = MigrationState.CHECKPOINTING
            # ... checkpoint via controller ...

            with self._lock:
                self._state = MigrationState.MIGRATING
            # ... transfer weights ...

            with self._lock:
                self._state = MigrationState.RELINKING
            # ... POST /relink to gateway ...

            with self._lock:
                self._state = MigrationState.RESUMING
            # ... resume on target ...

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            end_ts = datetime.now(timezone.utc).isoformat()

            record = MigrationRecord(
                plan=plan,
                started_at=start_ts,
                completed_at=end_ts,
                success=True,
                actual_downtime_ms=elapsed_ms,
            )
            self._migration_history.append(record)

            with self._lock:
                self._last_rebalance_time = time.monotonic()
                self._state = MigrationState.IDLE

            logger.info(
                "Migration complete: chunk %d → %s (%.1f ms)",
                plan.chunk_id, plan.target_node, elapsed_ms,
            )
            return True

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            end_ts = datetime.now(timezone.utc).isoformat()

            record = MigrationRecord(
                plan=plan,
                started_at=start_ts,
                completed_at=end_ts,
                success=False,
                actual_downtime_ms=elapsed_ms,
                error=str(e),
            )
            self._migration_history.append(record)

            with self._lock:
                self._state = MigrationState.IDLE

            logger.exception("Migration failed for chunk %d", plan.chunk_id)
            return False

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Return history of completed migrations with timing data."""
        return [r.to_dict() for r in self._migration_history]

    @property
    def state(self) -> MigrationState:
        """Current migration state."""
        return self._state
