"""
FCIM - Fair Cost-Efficient Analysis Mechanism
Picks the best possible worker based on cost efficiency and fairness.

Key Features:
- Multi-criteria worker scoring (cost, performance, fairness)
- Jain's Fairness Index for load balancing
- Cost modeling (energy + computational + network)
- Real-time worker health monitoring
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading


class WorkerStatus(Enum):
    """Worker availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"


@dataclass
class WorkerProfile:
    """Profile of a worker node with cost and performance metrics."""
    worker_id: str
    # Hardware specs
    gpu_memory_gb: float
    gpu_flops: float  # TFLOPS
    cpu_cores: int
    ram_gb: float
    network_bandwidth_gbps: float
    
    # Cost metrics (per hour)
    energy_cost_per_kwh: float = 0.12
    power_consumption_watts: float = 300.0
    
    # Performance metrics (updated in real-time)
    current_load: float = 0.0  # 0.0 to 1.0
    avg_latency_ms: float = 10.0
    tasks_completed: int = 0
    total_compute_time_s: float = 0.0
    
    # Fairness tracking
    allocation_share: float = 0.0  # Share of total allocations
    utilization_history: List[float] = field(default_factory=list)
    
    # Status
    status: WorkerStatus = WorkerStatus.AVAILABLE
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def hourly_energy_cost(self) -> float:
        """Calculate hourly energy cost."""
        kwh = self.power_consumption_watts / 1000.0
        return kwh * self.energy_cost_per_kwh
    
    @property
    def cost_per_tflop(self) -> float:
        """Cost efficiency: energy cost per TFLOP."""
        if self.gpu_flops <= 0:
            return float('inf')
        return self.hourly_energy_cost / self.gpu_flops
    
    @property
    def effective_capacity(self) -> float:
        """Remaining capacity considering current load."""
        return max(0.0, 1.0 - self.current_load) * self.gpu_flops


@dataclass
class TaskRequirement:
    """Requirements for a task to be scheduled."""
    task_id: str
    min_memory_gb: float
    estimated_flops: float
    priority: int = 1  # 1 (low) to 5 (high)
    deadline_ms: Optional[float] = None
    data_locality_node: Optional[str] = None  # Preferred node for data locality


@dataclass
class AllocationDecision:
    """Result of FCIM allocation decision."""
    task_id: str
    worker_id: str
    score: float
    cost_component: float
    efficiency_component: float
    fairness_component: float
    latency_estimate_ms: float


class FCIMWorkerSelector:
    """
    Fair Cost-Efficient Analysis Mechanism for worker selection.
    
    Balances three objectives:
    1. Cost Efficiency - Minimize energy/compute cost
    2. Performance Efficiency - Maximize throughput, minimize latency
    3. Fairness - Ensure balanced utilization across workers
    """
    
    def __init__(
        self,
        cost_weight: float = 0.35,
        efficiency_weight: float = 0.35,
        fairness_weight: float = 0.30,
        fairness_threshold: float = 0.8,  # Minimum Jain's Fairness Index
        overload_threshold: float = 0.9,
    ):
        """
        Initialize FCIM selector.
        
        Args:
            cost_weight: Weight for cost efficiency in scoring
            efficiency_weight: Weight for performance efficiency
            fairness_weight: Weight for fairness consideration
            fairness_threshold: Minimum acceptable Jain's Fairness Index
            overload_threshold: Load threshold to mark worker as overloaded
        """
        self.cost_weight = cost_weight
        self.efficiency_weight = efficiency_weight
        self.fairness_weight = fairness_weight
        self.fairness_threshold = fairness_threshold
        self.overload_threshold = overload_threshold
        
        self.workers: Dict[str, WorkerProfile] = {}
        self.allocation_history: List[AllocationDecision] = []
        self.total_allocations: int = 0
        self._lock = threading.Lock()
    
    def register_worker(self, worker: WorkerProfile) -> None:
        """Register a new worker with FCIM."""
        with self._lock:
            self.workers[worker.worker_id] = worker
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from FCIM."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
    
    def update_worker_metrics(
        self,
        worker_id: str,
        current_load: Optional[float] = None,
        avg_latency_ms: Optional[float] = None,
        tasks_completed: Optional[int] = None,
        power_consumption: Optional[float] = None,
    ) -> None:
        """Update real-time metrics for a worker."""
        with self._lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            if current_load is not None:
                worker.current_load = current_load
                worker.utilization_history.append(current_load)
                # Keep last 100 samples
                if len(worker.utilization_history) > 100:
                    worker.utilization_history = worker.utilization_history[-100:]
                
                # Update status based on load
                if current_load >= self.overload_threshold:
                    worker.status = WorkerStatus.OVERLOADED
                elif current_load > 0:
                    worker.status = WorkerStatus.BUSY
                else:
                    worker.status = WorkerStatus.AVAILABLE
            
            if avg_latency_ms is not None:
                worker.avg_latency_ms = avg_latency_ms
            if tasks_completed is not None:
                worker.tasks_completed = tasks_completed
            if power_consumption is not None:
                worker.power_consumption_watts = power_consumption
            
            worker.last_heartbeat = time.time()
    
    def select_worker(self, task: TaskRequirement) -> Optional[AllocationDecision]:
        """
        Select the best worker for a task using FCIM algorithm.
        
        The algorithm scores each worker based on:
        1. Cost Efficiency Score (CES) = 1 / (cost_per_tflop * required_flops)
        2. Performance Efficiency Score (PES) = effective_capacity / avg_latency
        3. Fairness Adjustment Score (FAS) = based on allocation share deviation
        
        Final Score = w1*CES + w2*PES + w3*FAS
        
        Args:
            task: Task requirements
            
        Returns:
            AllocationDecision with selected worker and scoring details
        """
        with self._lock:
            eligible_workers = self._get_eligible_workers(task)
            
            if not eligible_workers:
                return None
            
            # Calculate fairness index before selection
            current_fairness = self._calculate_jains_fairness_index()
            
            best_decision: Optional[AllocationDecision] = None
            best_score = float('-inf')
            
            for worker in eligible_workers:
                # Calculate component scores
                cost_score = self._calculate_cost_score(worker, task)
                efficiency_score = self._calculate_efficiency_score(worker, task)
                fairness_score = self._calculate_fairness_score(worker, current_fairness)
                
                # Weighted combination
                total_score = (
                    self.cost_weight * cost_score +
                    self.efficiency_weight * efficiency_score +
                    self.fairness_weight * fairness_score
                )
                
                # Apply deadline penalty if task has deadline
                if task.deadline_ms is not None:
                    latency_estimate = self._estimate_latency(worker, task)
                    if latency_estimate > task.deadline_ms:
                        total_score *= 0.1  # Heavy penalty for missing deadline
                
                # Apply locality bonus
                if task.data_locality_node == worker.worker_id:
                    total_score *= 1.2  # 20% bonus for data locality
                
                if total_score > best_score:
                    best_score = total_score
                    best_decision = AllocationDecision(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        score=total_score,
                        cost_component=cost_score,
                        efficiency_component=efficiency_score,
                        fairness_component=fairness_score,
                        latency_estimate_ms=self._estimate_latency(worker, task),
                    )
            
            if best_decision:
                self._record_allocation(best_decision)
            
            return best_decision
    
    def _get_eligible_workers(self, task: TaskRequirement) -> List[WorkerProfile]:
        """Get workers that can handle the task."""
        eligible = []
        for worker in self.workers.values():
            # Check availability
            if worker.status == WorkerStatus.OFFLINE:
                continue
            
            # Check memory requirement
            available_memory = worker.gpu_memory_gb * (1 - worker.current_load)
            if available_memory < task.min_memory_gb:
                continue
            
            # Check if worker has capacity
            if worker.effective_capacity < task.estimated_flops * 0.1:
                continue
            
            eligible.append(worker)
        
        return eligible
    
    def _calculate_cost_score(self, worker: WorkerProfile, task: TaskRequirement) -> float:
        """
        Calculate cost efficiency score.
        Lower cost per computation = higher score.
        """
        if worker.cost_per_tflop <= 0:
            return 0.0
        
        # Estimated cost for this task
        estimated_hours = (task.estimated_flops / worker.gpu_flops) / 3600.0
        estimated_cost = estimated_hours * worker.hourly_energy_cost
        
        # Normalize: higher is better (inverse of cost)
        if estimated_cost <= 0:
            return 1.0
        
        # Score from 0 to 1 using exponential decay
        return math.exp(-estimated_cost * 10)
    
    def _calculate_efficiency_score(self, worker: WorkerProfile, task: TaskRequirement) -> float:
        """
        Calculate performance efficiency score.
        Higher capacity + lower latency = higher score.
        """
        # Capacity component (0 to 1)
        max_flops = max(w.gpu_flops for w in self.workers.values()) if self.workers else 1.0
        capacity_score = worker.effective_capacity / max_flops
        
        # Latency component (0 to 1)
        max_latency = max(w.avg_latency_ms for w in self.workers.values()) if self.workers else 1.0
        if max_latency <= 0:
            latency_score = 1.0
        else:
            latency_score = 1.0 - (worker.avg_latency_ms / max_latency)
        
        # Combined score
        return 0.6 * capacity_score + 0.4 * latency_score
    
    def _calculate_fairness_score(self, worker: WorkerProfile, current_fairness: float) -> float:
        """
        Calculate fairness adjustment score.
        Workers with lower allocation share get bonus.
        """
        if self.total_allocations == 0:
            return 1.0  # All workers equal at start
        
        # Calculate expected fair share
        num_workers = len(self.workers)
        if num_workers == 0:
            return 1.0
        
        fair_share = 1.0 / num_workers
        
        # Deviation from fair share
        deviation = fair_share - worker.allocation_share
        
        # Convert to score (positive deviation = underutilized = higher score)
        fairness_score = 0.5 + (deviation * 2.0)
        fairness_score = max(0.0, min(1.0, fairness_score))
        
        # If system fairness is below threshold, increase fairness importance
        if current_fairness < self.fairness_threshold:
            fairness_score *= 1.5
        
        return fairness_score
    
    def _calculate_jains_fairness_index(self) -> float:
        """
        Calculate Jain's Fairness Index for current allocation distribution.
        
        J(x) = (sum(xi))^2 / (n * sum(xi^2))
        
        Returns value between 0 (completely unfair) and 1 (perfectly fair).
        """
        if not self.workers or self.total_allocations == 0:
            return 1.0  # Perfect fairness when no allocations
        
        shares = [w.allocation_share for w in self.workers.values()]
        n = len(shares)
        
        sum_x = sum(shares)
        sum_x_squared = sum(x * x for x in shares)
        
        if sum_x_squared == 0:
            return 1.0
        
        fairness_index = (sum_x * sum_x) / (n * sum_x_squared)
        return fairness_index
    
    def _estimate_latency(self, worker: WorkerProfile, task: TaskRequirement) -> float:
        """Estimate task completion latency on worker."""
        # Base latency from historical average
        base_latency = worker.avg_latency_ms
        
        # Compute time estimate
        if worker.effective_capacity > 0:
            compute_time_ms = (task.estimated_flops / worker.effective_capacity) * 1000
        else:
            compute_time_ms = float('inf')
        
        # Queue delay estimate based on load
        queue_delay_ms = worker.current_load * base_latency * 2
        
        return base_latency + compute_time_ms + queue_delay_ms
    
    def _record_allocation(self, decision: AllocationDecision) -> None:
        """Record allocation for fairness tracking."""
        self.total_allocations += 1
        self.allocation_history.append(decision)
        
        # Keep last 1000 allocations
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]
        
        # Update allocation shares
        for worker in self.workers.values():
            count = sum(1 for d in self.allocation_history if d.worker_id == worker.worker_id)
            worker.allocation_share = count / len(self.allocation_history)
    
    def get_fairness_report(self) -> Dict:
        """Get fairness metrics report."""
        with self._lock:
            fairness_index = self._calculate_jains_fairness_index()
            
            worker_stats = []
            for worker in self.workers.values():
                avg_utilization = (
                    sum(worker.utilization_history) / len(worker.utilization_history)
                    if worker.utilization_history else 0.0
                )
                worker_stats.append({
                    "worker_id": worker.worker_id,
                    "allocation_share": worker.allocation_share,
                    "avg_utilization": avg_utilization,
                    "tasks_completed": worker.tasks_completed,
                    "cost_per_tflop": worker.cost_per_tflop,
                })
            
            return {
                "jains_fairness_index": fairness_index,
                "total_allocations": self.total_allocations,
                "num_workers": len(self.workers),
                "fairness_threshold": self.fairness_threshold,
                "is_fair": fairness_index >= self.fairness_threshold,
                "workers": worker_stats,
            }
    
    def rebalance_for_fairness(self) -> List[Tuple[str, str]]:
        """
        Suggest task migrations to improve fairness.
        
        Returns list of (task_id, target_worker_id) migrations.
        """
        with self._lock:
            current_fairness = self._calculate_jains_fairness_index()
            
            if current_fairness >= self.fairness_threshold:
                return []  # System is fair enough
            
            migrations = []
            
            # Find overloaded and underloaded workers
            avg_share = 1.0 / len(self.workers) if self.workers else 0
            overloaded = [w for w in self.workers.values() if w.allocation_share > avg_share * 1.2]
            underloaded = [w for w in self.workers.values() if w.allocation_share < avg_share * 0.8]
            
            # Suggest migrations from overloaded to underloaded
            for over_worker in overloaded:
                if not underloaded:
                    break
                
                under_worker = min(underloaded, key=lambda w: w.allocation_share)
                
                # Find a recent task on overloaded worker
                recent_tasks = [
                    d for d in reversed(self.allocation_history)
                    if d.worker_id == over_worker.worker_id
                ]
                
                if recent_tasks:
                    migrations.append((recent_tasks[0].task_id, under_worker.worker_id))
            
            return migrations


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("worker_selector", "fcim")
    class FCIMPlugin:
        """FCIM Worker Selector Plugin."""
        
        def __init__(self, **kwargs):
            self.selector = FCIMWorkerSelector(**kwargs)
        
        def select(self, task: TaskRequirement) -> Optional[AllocationDecision]:
            return self.selector.select_worker(task)
        
        def register_worker(self, worker: WorkerProfile) -> None:
            self.selector.register_worker(worker)

except ImportError:
    pass  # Plugin architecture not available
