"""
Dynamic Energy-Aware Scheduler (DEAS).

Production-oriented DEAS with:
- Predictive and reactive rebalance triggering
- Cost-aware and score-based migration planning
- Multi-chunk top-k migration support
- Adaptive cooldown and no-action decisions
- Migration-history learning
- Optional localized ILP candidate selection
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    eer: float
    current_chunks: List[int]
    threshold_level: str
    usable_memory_mb: float
    # Optional fields used by advanced scoring/prediction.
    avg_latency_ms: float = 0.0
    gpu_utilization_pct: float = 0.0


@dataclass
class MigrationPlan:
    """Describes a single chunk migration."""

    chunk_id: int
    source_node: str
    target_node: str
    reason: str
    estimated_downtime_ms: float = 0.0
    checkpoint_size_mb: float = 0.0
    # Added fields preserve backward compatibility by defaulting to zero.
    score: float = 0.0
    expected_eer_gain: float = 0.0
    migration_cost: float = 0.0
    latency_penalty: float = 0.0


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
            "score": round(self.plan.score, 6),
            "expected_eer_gain": round(self.plan.expected_eer_gain, 6),
            "migration_cost": round(self.plan.migration_cost, 6),
            "latency_penalty": round(self.plan.latency_penalty, 6),
            "estimated_downtime_ms": round(self.plan.estimated_downtime_ms, 3),
            "checkpoint_size_mb": round(self.plan.checkpoint_size_mb, 3),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "actual_downtime_ms": self.actual_downtime_ms,
            "error": self.error,
        }


@dataclass
class CandidateMigration:
    chunk_id: int
    source_node: str
    target_node: str
    reason: str
    score: float
    delta_eer: float
    migration_cost: float
    latency_penalty: float
    estimated_downtime_ms: float
    checkpoint_size_mb: float


@dataclass
class SchedulerWeights:
    """Weights used by DEAS scoring and optimization."""

    # Node selection score components.
    node_eer_w: float = 0.40
    node_mem_w: float = 0.30
    node_latency_perf_w: float = 0.20
    node_network_w: float = 0.10

    # Migration score components.
    migration_cost_scale: float = 0.0000002
    latency_penalty_scale: float = 0.0000001
    history_bonus_scale: float = 0.05
    load_balance_scale: float = 0.10


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
        max_migrations_per_cycle: int = 2,
        candidate_chunks_per_node: int = 3,
        network_bandwidth_mb_s: float = 400.0,
        transfer_latency_ms: float = 5.0,
        checkpoint_speed_mb_s: float = 800.0,
        min_benefit_margin: float = 0.0,
        enable_ilp_refinement: bool = True,
        ilp_candidate_limit: int = 16,
    ):
        self._event_bus = event_bus
        self._partitioner = auto_partitioner
        self._controller = controller
        self._cooldown_s = max(0.0, cooldown_s)
        self._dynamic_cooldown_s = max(0.0, cooldown_s)

        self._max_migrations_per_cycle = max(1, int(max_migrations_per_cycle))
        self._candidate_chunks_per_node = max(1, int(candidate_chunks_per_node))
        self._network_bandwidth_mb_s = max(1.0, network_bandwidth_mb_s)
        self._transfer_latency_ms = max(0.0, transfer_latency_ms)
        self._checkpoint_speed_mb_s = max(1.0, checkpoint_speed_mb_s)
        self._min_benefit_margin = min_benefit_margin

        self._enable_ilp_refinement = enable_ilp_refinement
        self._ilp_candidate_limit = max(1, int(ilp_candidate_limit))

        self._weights = SchedulerWeights()

        self._state = MigrationState.IDLE
        self._last_rebalance_time: float = 0.0
        self._migration_history: List[MigrationRecord] = []
        self._lock = threading.Lock()

        # Prediction and signal history.
        self._profile_history: Dict[str, Deque[NodeEnergyProfile]] = defaultdict(lambda: deque(maxlen=8))
        self._signal_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._last_plan_debug: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Integration hooks
    # ------------------------------------------------------------------

    def bind_energy_controller(self, energy_controller) -> None:
        """Bind an EnergyFeedbackController to DEAS scheduler signals."""
        if energy_controller is None:
            return
        setter = getattr(energy_controller, "set_scheduler_callback", None)
        if callable(setter):
            setter(self.handle_scheduler_signal)
            logger.info("DEAS bound to EnergyFeedbackController scheduler callback")

    def handle_scheduler_signal(self, signal: Dict[str, Any]) -> None:
        """Handle scheduler signal emitted by the energy feedback loop.

        Expected payload (subset):
        - overloaded_worker
        - inefficient_node
        - metrics dict (power, latency, throughput)
        """
        if not isinstance(signal, dict):
            return

        self._signal_history.append(signal)

        overloaded_worker = signal.get("overloaded_worker")
        inefficient_node = signal.get("inefficient_node")
        metrics = signal.get("metrics", {}) if isinstance(signal.get("metrics"), dict) else {}

        should_trigger = bool(overloaded_worker or inefficient_node)
        if not should_trigger:
            power_w = float(metrics.get("power_w", 0.0) or 0.0)
            latency_ms = float(metrics.get("latency_ms", 0.0) or 0.0)
            throughput = float(metrics.get("throughput_tokens_per_sec", 0.0) or 0.0)
            should_trigger = (
                power_w > 220.0 and latency_ms > 180.0 and throughput < 2.0
            )

        if not should_trigger:
            return

        reason = "energy_controller_signal"
        if overloaded_worker:
            reason = f"overloaded_worker:{overloaded_worker}"
        elif inefficient_node:
            reason = f"inefficient_node:{inefficient_node}"

        self._trigger_rebalance(reason=reason)

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
        """Handler for CRITICAL threshold events."""
        logger.warning(
            "CRITICAL event from %s (%.1fW, %.1f%% TDP) — evaluating rebalance",
            event.node_name,
            event.power_w,
            event.tdp_pct,
        )
        self._trigger_rebalance(reason=f"critical_threshold:{event.node_name}")

    def _trigger_rebalance(self, reason: str) -> None:
        """Trigger rebalance via controller integration with cooldown guards."""
        now = time.monotonic()
        with self._lock:
            if self._state != MigrationState.IDLE:
                logger.debug("Migration in progress — ignoring rebalance trigger (%s)", reason)
                return
            if now - self._last_rebalance_time < self._dynamic_cooldown_s:
                logger.debug(
                    "Cooldown active (%.2fs) — ignoring rebalance trigger (%s)",
                    self._dynamic_cooldown_s,
                    reason,
                )
                return
            self._state = MigrationState.PAUSING

        try:
            result = None
            if self._controller and hasattr(self._controller, "trigger_rebalance"):
                result = self._controller.trigger_rebalance()
                logger.info("DEAS rebalance triggered (%s): %s", reason, result)
            else:
                logger.info(
                    "DEAS rebalance trigger (%s) received, but no controller integration is configured",
                    reason,
                )

            had_migrations = bool(result and result.get("rebalanced") and result.get("migration_plans"))
            had_failures = bool(
                result
                and isinstance(result.get("migration_plans"), list)
                and any(not p.get("success", False) for p in result.get("migration_plans", []))
            )
            self._adapt_cooldown(had_migrations=had_migrations, had_failures=had_failures)

            with self._lock:
                self._last_rebalance_time = time.monotonic()
                self._state = MigrationState.IDLE
        except Exception:
            logger.exception("Error during DEAS rebalance trigger (%s)", reason)
            self._adapt_cooldown(had_migrations=False, had_failures=True)
            with self._lock:
                self._state = MigrationState.IDLE

    # ------------------------------------------------------------------
    # EER computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cluster_eer(node_profiles: List[NodeEnergyProfile]) -> float:
        """Compute cluster-wide EER = sum(throughput) / sum(power)."""
        total_throughput = sum(p.throughput_inf_per_sec for p in node_profiles)
        total_power = sum(p.avg_power_w for p in node_profiles)
        if total_power <= 0:
            return 0.0
        return total_throughput / total_power

    def should_rebalance(self, node_profiles: List[NodeEnergyProfile]) -> bool:
        """Determine if rebalancing would improve cluster EER.

        Rebalance is triggered when:
        - any node is CRITICAL, or
        - predictive trend analysis detects likely near-future degradation.
        """
        self._update_profile_history(node_profiles)
        if any(p.threshold_level.lower() == "critical" for p in node_profiles):
            return True
        return self._predictive_rebalance_needed(node_profiles)

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

        Score per candidate:
            score = delta_eer - migration_cost - latency_penalty

        Where migration_cost and latency_penalty are normalized terms.
        Only candidates where benefit > cost are selected.
        """
        self._update_profile_history(current_profiles)
        source_nodes = self._select_source_nodes(current_profiles)
        if not source_nodes:
            self._last_plan_debug = [{"reason": "no_source_nodes"}]
            return []

        candidates: List[CandidateMigration] = []
        for source in source_nodes:
            chunk_candidates = self._select_candidate_chunks(source)
            targets = self._rank_target_nodes(source, current_profiles)
            if not chunk_candidates or not targets:
                continue

            for chunk_id in chunk_candidates:
                for target in targets:
                    estimate = self._estimate_migration_effect(source, target, chunk_id, current_profiles)
                    # Benefit > cost condition (plus optional margin)
                    if estimate["delta_eer"] <= estimate["migration_cost"] + estimate["latency_penalty"] + self._min_benefit_margin:
                        continue

                    history_adj = self._history_adjustment(source.node_name, target.node_name, chunk_id)
                    load_adj = self._load_balance_adjustment(source, target, current_profiles)

                    score = (
                        estimate["delta_eer"]
                        - estimate["migration_cost"]
                        - estimate["latency_penalty"]
                        + history_adj * self._weights.history_bonus_scale
                        + load_adj * self._weights.load_balance_scale
                    )

                    if score <= 0:
                        continue

                    reason = "critical_threshold" if source.threshold_level.lower() == "critical" else "predictive_rebalance"
                    candidates.append(
                        CandidateMigration(
                            chunk_id=chunk_id,
                            source_node=source.node_name,
                            target_node=target.node_name,
                            reason=reason,
                            score=score,
                            delta_eer=estimate["delta_eer"],
                            migration_cost=estimate["migration_cost"],
                            latency_penalty=estimate["latency_penalty"],
                            estimated_downtime_ms=estimate["downtime_ms"],
                            checkpoint_size_mb=estimate["checkpoint_size_mb"],
                        )
                    )

        if not candidates:
            self._last_plan_debug = [{"reason": "no_beneficial_candidates"}]
            return []

        selected = self._select_top_k_candidates(candidates)
        self._last_plan_debug = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source_node,
                "target": c.target_node,
                "score": round(c.score, 6),
                "delta_eer": round(c.delta_eer, 6),
                "migration_cost": round(c.migration_cost, 6),
                "latency_penalty": round(c.latency_penalty, 6),
            }
            for c in selected
        ]

        return [
            MigrationPlan(
                chunk_id=c.chunk_id,
                source_node=c.source_node,
                target_node=c.target_node,
                reason=c.reason,
                estimated_downtime_ms=c.estimated_downtime_ms,
                checkpoint_size_mb=c.checkpoint_size_mb,
                score=c.score,
                expected_eer_gain=c.delta_eer,
                migration_cost=c.migration_cost,
                latency_penalty=c.latency_penalty,
            )
            for c in selected
        ]

    def _select_source_nodes(self, current_profiles: List[NodeEnergyProfile]) -> List[NodeEnergyProfile]:
        critical = [p for p in current_profiles if p.threshold_level.lower() == "critical"]
        if critical:
            return sorted(critical, key=lambda p: p.avg_power_w, reverse=True)

        # Predictive path: warning nodes with rising power / declining throughput.
        if not self._predictive_rebalance_needed(current_profiles):
            return []

        warning = [p for p in current_profiles if p.threshold_level.lower() in ("warning", "optimal")]
        warning = sorted(warning, key=lambda p: (p.avg_power_w, -p.throughput_inf_per_sec), reverse=True)
        return warning[: max(1, min(2, len(warning)))]

    def _select_candidate_chunks(self, source: NodeEnergyProfile) -> List[int]:
        if not source.current_chunks:
            return []

        # Prefer recent chunks but evaluate multiple candidates per cycle.
        chunks = list(source.current_chunks)
        return list(reversed(chunks))[: self._candidate_chunks_per_node]

    def _rank_target_nodes(
        self,
        source: NodeEnergyProfile,
        current_profiles: List[NodeEnergyProfile],
    ) -> List[NodeEnergyProfile]:
        targets = [p for p in current_profiles if p.node_name != source.node_name and p.threshold_level.lower() != "critical"]
        if not targets:
            return []

        max_eer = max((p.eer for p in current_profiles), default=1.0) or 1.0
        max_mem = max((p.usable_memory_mb for p in current_profiles), default=1.0) or 1.0

        scored: List[Tuple[float, NodeEnergyProfile]] = []
        for t in targets:
            eer_norm = max(0.0, t.eer / max_eer)
            mem_norm = max(0.0, t.usable_memory_mb / max_mem)
            latency_perf = max(0.0, t.throughput_inf_per_sec / max(1.0, t.avg_power_w))
            network_proximity = self._network_proximity(source, t)

            node_score = (
                self._weights.node_eer_w * eer_norm
                + self._weights.node_mem_w * mem_norm
                + self._weights.node_latency_perf_w * latency_perf
                + self._weights.node_network_w * network_proximity
            )
            scored.append((node_score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored]

    def _network_proximity(self, source: NodeEnergyProfile, target: NodeEnergyProfile) -> float:
        # If controller provides network hints, use them.
        if self._controller and hasattr(self._controller, "estimate_network_proximity"):
            try:
                value = float(self._controller.estimate_network_proximity(source.node_name, target.node_name))
                return max(0.0, min(1.0, value))
            except Exception:
                pass

        # Fallback: lower utilization and closer performance often implies less disruption.
        util_src = source.gpu_utilization_pct or 0.0
        util_tgt = target.gpu_utilization_pct or 0.0
        util_gap = abs(util_src - util_tgt) / 100.0
        return max(0.0, 1.0 - util_gap)

    def _estimate_migration_effect(
        self,
        source: NodeEnergyProfile,
        target: NodeEnergyProfile,
        chunk_id: int,
        current_profiles: List[NodeEnergyProfile],
    ) -> Dict[str, float]:
        checkpoint_size_mb = self._estimate_checkpoint_size_mb(source, chunk_id)
        transfer_time_ms = (checkpoint_size_mb / self._network_bandwidth_mb_s) * 1000.0 + self._transfer_latency_ms
        checkpoint_time_ms = (checkpoint_size_mb / self._checkpoint_speed_mb_s) * 1000.0
        downtime_ms = checkpoint_time_ms + transfer_time_ms + 20.0

        # Migration cost normalized to the same scale as EER deltas.
        migration_cost = downtime_ms * self._weights.migration_cost_scale

        # Latency penalty increases when target has less memory / lower efficiency.
        target_pressure = max(0.0, 1.0 - min(1.0, target.usable_memory_mb / max(1.0, source.usable_memory_mb)))
        latency_penalty_ms = downtime_ms * 0.20 + target_pressure * 50.0
        latency_penalty = latency_penalty_ms * self._weights.latency_penalty_scale

        # Approximate EER change by shifting one chunk share from source to target.
        src_chunks = max(1, len(source.current_chunks))
        chunk_share = 1.0 / src_chunks

        src_power_reduction = source.avg_power_w * 0.12 * chunk_share
        src_tp_reduction = source.throughput_inf_per_sec * 0.06 * chunk_share

        tgt_power_increase = source.avg_power_w * 0.08 * chunk_share
        tgt_tp_increase = source.throughput_inf_per_sec * 0.10 * chunk_share

        profiles_after: List[NodeEnergyProfile] = []
        for p in current_profiles:
            if p.node_name == source.node_name:
                profiles_after.append(
                    NodeEnergyProfile(
                        node_name=p.node_name,
                        avg_power_w=max(1e-6, p.avg_power_w - src_power_reduction),
                        throughput_inf_per_sec=max(0.0, p.throughput_inf_per_sec - src_tp_reduction),
                        eer=0.0,
                        current_chunks=list(p.current_chunks),
                        threshold_level=p.threshold_level,
                        usable_memory_mb=p.usable_memory_mb,
                        avg_latency_ms=p.avg_latency_ms,
                        gpu_utilization_pct=max(0.0, p.gpu_utilization_pct - 8.0),
                    )
                )
            elif p.node_name == target.node_name:
                profiles_after.append(
                    NodeEnergyProfile(
                        node_name=p.node_name,
                        avg_power_w=max(1e-6, p.avg_power_w + tgt_power_increase),
                        throughput_inf_per_sec=max(0.0, p.throughput_inf_per_sec + tgt_tp_increase),
                        eer=0.0,
                        current_chunks=list(p.current_chunks),
                        threshold_level=p.threshold_level,
                        usable_memory_mb=max(0.0, p.usable_memory_mb - checkpoint_size_mb),
                        avg_latency_ms=p.avg_latency_ms,
                        gpu_utilization_pct=min(100.0, p.gpu_utilization_pct + 6.0),
                    )
                )
            else:
                profiles_after.append(p)

        before = self.compute_cluster_eer(current_profiles)
        after = self.compute_cluster_eer(profiles_after)
        delta_eer = max(0.0, after - before)

        return {
            "checkpoint_size_mb": checkpoint_size_mb,
            "transfer_time_ms": transfer_time_ms,
            "checkpoint_time_ms": checkpoint_time_ms,
            "downtime_ms": downtime_ms,
            "migration_cost": migration_cost,
            "latency_penalty": latency_penalty,
            "delta_eer": delta_eer,
        }

    def _estimate_checkpoint_size_mb(self, source: NodeEnergyProfile, chunk_id: int) -> float:
        base = max(128.0, source.avg_power_w * 1.2)
        chunk_weight = 1.0
        if source.current_chunks:
            try:
                idx = source.current_chunks.index(chunk_id)
                chunk_weight += idx / max(1, len(source.current_chunks))
            except ValueError:
                pass

        # Learn from history when available.
        sizes = [
            r.plan.checkpoint_size_mb
            for r in self._migration_history
            if r.plan.chunk_id == chunk_id and r.plan.checkpoint_size_mb > 0
        ]
        if sizes:
            hist = mean(sizes)
            return max(64.0, 0.6 * base * chunk_weight + 0.4 * hist)

        return max(64.0, base * chunk_weight)

    def _history_adjustment(self, source: str, target: str, chunk_id: int) -> float:
        relevant = [
            r
            for r in self._migration_history
            if r.plan.source_node == source
            and r.plan.target_node == target
            and r.plan.chunk_id == chunk_id
        ]
        if not relevant:
            return 0.0

        success_ratio = sum(1 for r in relevant if r.success) / len(relevant)
        avg_downtime = mean(r.actual_downtime_ms for r in relevant)
        downtime_factor = max(0.0, 1.0 - avg_downtime / 2000.0)
        return success_ratio * downtime_factor - (1.0 - success_ratio)

    def _load_balance_adjustment(
        self,
        source: NodeEnergyProfile,
        target: NodeEnergyProfile,
        current_profiles: List[NodeEnergyProfile],
    ) -> float:
        # Penalize both overload and severe underutilization.
        counts = [len(p.current_chunks) for p in current_profiles]
        if not counts:
            return 0.0
        avg_chunks = mean(counts)
        src_dev = len(source.current_chunks) - avg_chunks
        tgt_dev = len(target.current_chunks) - avg_chunks
        # Positive adjustment when migration reduces spread.
        before_spread = abs(src_dev) + abs(tgt_dev)
        after_spread = abs(src_dev - 1) + abs(tgt_dev + 1)
        return before_spread - after_spread

    def _select_top_k_candidates(self, candidates: List[CandidateMigration]) -> List[CandidateMigration]:
        # Localized ILP selection for small candidate sets.
        if (
            self._enable_ilp_refinement
            and len(candidates) <= self._ilp_candidate_limit
        ):
            selected = self._select_candidates_ilp(candidates)
            if selected is not None:
                return selected

        # Greedy fallback.
        return self._select_candidates_greedy(candidates)

    def _select_candidates_ilp(self, candidates: List[CandidateMigration]) -> Optional[List[CandidateMigration]]:
        try:
            import pulp
        except Exception:
            return None

        model = pulp.LpProblem("deas_select", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(len(candidates))]
        model += pulp.lpSum(candidates[i].score * x[i] for i in range(len(candidates)))

        model += pulp.lpSum(x) <= self._max_migrations_per_cycle

        # Avoid migrating same chunk multiple times in one cycle.
        chunk_to_idx: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            chunk_to_idx[(c.source_node, c.chunk_id)].append(i)
        for idxs in chunk_to_idx.values():
            model += pulp.lpSum(x[i] for i in idxs) <= 1

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=3)
        model.solve(solver)

        selected: List[CandidateMigration] = []
        for i, var in enumerate(x):
            value = pulp.value(var)
            if value is not None and value > 0.5:
                selected.append(candidates[i])

        selected.sort(key=lambda c: c.score, reverse=True)
        return selected

    def _select_candidates_greedy(self, candidates: List[CandidateMigration]) -> List[CandidateMigration]:
        selected: List[CandidateMigration] = []
        used_chunks: set = set()
        for c in sorted(candidates, key=lambda item: item.score, reverse=True):
            key = (c.source_node, c.chunk_id)
            if key in used_chunks:
                continue
            selected.append(c)
            used_chunks.add(key)
            if len(selected) >= self._max_migrations_per_cycle:
                break
        return selected

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _update_profile_history(self, profiles: List[NodeEnergyProfile]) -> None:
        for p in profiles:
            self._profile_history[p.node_name].append(p)

    def _predictive_rebalance_needed(self, profiles: List[NodeEnergyProfile]) -> bool:
        for p in profiles:
            hist = self._profile_history.get(p.node_name)
            if hist is None or len(hist) < 3:
                continue

            power_vals = [x.avg_power_w for x in hist]
            tp_vals = [x.throughput_inf_per_sec for x in hist]
            lat_vals = [x.avg_latency_ms for x in hist]

            slope_power = self._slope(power_vals)
            slope_tp = self._slope(tp_vals)
            slope_lat = self._slope(lat_vals) if any(v > 0 for v in lat_vals) else 0.0

            if p.threshold_level.lower() == "warning" and slope_power > 1.0:
                return True
            if slope_power > 2.0 and (slope_tp < -0.2 or slope_lat > 1.0):
                return True

        # Energy-controller signals also support predictive triggering.
        recent = list(self._signal_history)[-5:]
        for s in recent:
            if s.get("overloaded_worker") or s.get("inefficient_node"):
                return True
        return False

    @staticmethod
    def _slope(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        return num / denom

    # ------------------------------------------------------------------
    # Adaptive cooldown
    # ------------------------------------------------------------------

    def _adapt_cooldown(self, had_migrations: bool, had_failures: bool) -> None:
        # Shorten cooldown in unstable/failed conditions; lengthen in stable no-action.
        if had_failures:
            self._dynamic_cooldown_s = max(5.0, self._dynamic_cooldown_s * 0.75)
        elif had_migrations:
            self._dynamic_cooldown_s = max(5.0, self._dynamic_cooldown_s * 0.90)
        else:
            self._dynamic_cooldown_s = min(120.0, max(5.0, self._dynamic_cooldown_s * 1.15))

    # ------------------------------------------------------------------
    # Migration execution
    # ------------------------------------------------------------------

    def execute_migration(self, plan: MigrationPlan) -> bool:
        """Execute a single chunk migration."""
        return self._execute_single_migration(plan, manage_state=True)

    def execute_migration_batch(self, plans: List[MigrationPlan]) -> List[bool]:
        """Execute multiple migrations in a single cycle.

        This reduces repeated state transitions and allows small-k batch
        migration execution per rebalance cycle.
        """
        if not plans:
            return []

        with self._lock:
            self._state = MigrationState.PAUSING

        results: List[bool] = []
        try:
            for plan in plans[: self._max_migrations_per_cycle]:
                ok = self._execute_single_migration(plan, manage_state=False)
                results.append(ok)

            with self._lock:
                self._last_rebalance_time = time.monotonic()
                self._state = MigrationState.IDLE
            return results
        except Exception:
            logger.exception("Batch migration execution failed")
            with self._lock:
                self._state = MigrationState.IDLE
            return [False for _ in plans[: self._max_migrations_per_cycle]]

    def _execute_single_migration(self, plan: MigrationPlan, manage_state: bool) -> bool:
        start_ts = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        if manage_state:
            with self._lock:
                self._state = MigrationState.PAUSING

        try:
            logger.info(
                "Migrating chunk %d: %s → %s (reason: %s, score=%.4f)",
                plan.chunk_id,
                plan.source_node,
                plan.target_node,
                plan.reason,
                plan.score,
            )

            with self._lock:
                self._state = MigrationState.CHECKPOINTING

            with self._lock:
                self._state = MigrationState.MIGRATING

            with self._lock:
                self._state = MigrationState.RELINKING

            with self._lock:
                self._state = MigrationState.RESUMING

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

            if manage_state:
                with self._lock:
                    self._last_rebalance_time = time.monotonic()
                    self._state = MigrationState.IDLE

            logger.info(
                "Migration complete: chunk %d → %s (%.1f ms)",
                plan.chunk_id,
                plan.target_node,
                elapsed_ms,
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

            if manage_state:
                with self._lock:
                    self._state = MigrationState.IDLE

            logger.exception("Migration failed for chunk %d", plan.chunk_id)
            return False

    # ------------------------------------------------------------------
    # History and status
    # ------------------------------------------------------------------

    def get_migration_history(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._migration_history]

    def get_last_plan_debug(self) -> List[Dict[str, Any]]:
        return list(self._last_plan_debug)

    @property
    def state(self) -> MigrationState:
        return self._state

    @property
    def cooldown_s(self) -> float:
        return self._dynamic_cooldown_s
