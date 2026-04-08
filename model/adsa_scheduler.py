"""
ADSA - Adaptive Dynamic Scheduling Algorithm
Reorders tasks dynamically based on arrival time, size, and system state.

Key Features:
- Dynamic priority queue with arrival-time awareness
- Task size-based scheduling (Shortest Job First variants)
- Aging mechanism to prevent starvation
- Adaptive threshold tuning based on system load
"""

import time
import heapq
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import math


class SchedulingPolicy(Enum):
    """Scheduling policy types."""
    FIFO = "fifo"  # First In, First Out
    SJF = "sjf"  # Shortest Job First
    SRPT = "srpt"  # Shortest Remaining Processing Time
    WEIGHTED = "weighted"  # Weighted combination
    ADAPTIVE = "adaptive"  # Dynamically selected


class TaskState(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ADSATask:
    """Task representation for ADSA scheduler."""
    task_id: str
    arrival_time: float  # Timestamp when task arrived
    estimated_size: float  # Estimated computation units (FLOPs, tokens, etc.)
    priority: int = 1  # Base priority (1-10)
    deadline: Optional[float] = None  # Optional deadline timestamp
    
    # Runtime tracking
    state: TaskState = TaskState.PENDING
    remaining_size: float = field(default=0.0)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_worker: Optional[str] = None
    
    # Aging counter (increases over time to prevent starvation)
    age_bonus: float = 0.0
    wait_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.remaining_size == 0.0:
            self.remaining_size = self.estimated_size
    
    @property
    def effective_priority(self) -> float:
        """Calculate priority including age bonus."""
        return self.priority + self.age_bonus
    
    @property
    def urgency(self) -> float:
        """Calculate urgency based on deadline proximity."""
        if self.deadline is None:
            return 0.0
        remaining_time = self.deadline - time.time()
        if remaining_time <= 0:
            return float('inf')  # Overdue
        return 1.0 / remaining_time


@dataclass 
class SchedulerMetrics:
    """Metrics for scheduler performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    avg_wait_time: float = 0.0
    avg_turnaround_time: float = 0.0
    avg_response_time: float = 0.0
    throughput: float = 0.0  # Tasks per second
    fairness_index: float = 1.0
    deadline_miss_rate: float = 0.0


class ADSAScheduler:
    """
    Adaptive Dynamic Scheduling Algorithm.
    
    Key algorithms:
    1. Multi-level Feedback Queue: Tasks move between priority levels
    2. Aging: Long-waiting tasks get priority boost
    3. Size-based Ordering: Prefer smaller tasks to minimize avg wait
    4. Deadline Awareness: Urgent tasks get priority
    5. Adaptive Policy: Switch policies based on workload
    """
    
    def __init__(
        self,
        initial_policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE,
        aging_rate: float = 0.1,  # Priority boost per second of waiting
        size_weight: float = 0.3,
        arrival_weight: float = 0.3,
        priority_weight: float = 0.2,
        deadline_weight: float = 0.2,
        reorder_interval_ms: float = 100.0,  # How often to reorder queue
        starvation_threshold_s: float = 30.0,  # Seconds before starvation boost
    ):
        """
        Initialize ADSA scheduler.
        
        Args:
            initial_policy: Starting scheduling policy
            aging_rate: Priority increase per second of waiting
            size_weight: Weight for task size in scoring
            arrival_weight: Weight for arrival time in scoring
            priority_weight: Weight for base priority in scoring
            deadline_weight: Weight for deadline urgency in scoring
            reorder_interval_ms: Interval for queue reordering
            starvation_threshold_s: Time threshold for starvation prevention
        """
        self.policy = initial_policy
        self.aging_rate = aging_rate
        self.size_weight = size_weight
        self.arrival_weight = arrival_weight
        self.priority_weight = priority_weight
        self.deadline_weight = deadline_weight
        self.reorder_interval_ms = reorder_interval_ms
        self.starvation_threshold_s = starvation_threshold_s
        
        # Task storage
        self.pending_tasks: List[ADSATask] = []
        self.running_tasks: Dict[str, ADSATask] = {}
        self.completed_tasks: List[ADSATask] = []
        
        # Priority queue for scheduling (min-heap by score)
        self._task_heap: List[tuple] = []
        self._task_map: Dict[str, ADSATask] = {}
        
        # Metrics
        self.metrics = SchedulerMetrics()
        self._metrics_history: List[SchedulerMetrics] = []
        
        # Synchronization
        self._lock = threading.RLock()
        self._last_reorder_time = time.time()
        
        # Callbacks
        self._on_task_ready: Optional[Callable[[ADSATask], None]] = None
        self._on_task_complete: Optional[Callable[[ADSATask], None]] = None
    
    def submit_task(self, task: ADSATask) -> str:
        """
        Submit a new task to the scheduler.
        
        Returns task_id for tracking.
        """
        with self._lock:
            task.arrival_time = time.time() if task.arrival_time == 0 else task.arrival_time
            task.state = TaskState.PENDING
            
            self._task_map[task.task_id] = task
            self.pending_tasks.append(task)
            self.metrics.total_tasks += 1
            
            # Add to priority queue
            self._insert_to_heap(task)
            
            # Check if reordering is needed
            self._maybe_reorder()
            
            return task.task_id
    
    def get_next_task(self) -> Optional[ADSATask]:
        """
        Get the next task to execute based on current policy.
        
        Returns None if no tasks are ready.
        """
        with self._lock:
            self._maybe_reorder()
            self._apply_aging()
            
            while self._task_heap:
                _, _, task_id = heapq.heappop(self._task_heap)
                
                if task_id not in self._task_map:
                    continue  # Task was cancelled
                
                task = self._task_map[task_id]
                
                if task.state != TaskState.PENDING:
                    continue  # Task already running or completed
                
                # Mark as ready
                task.state = TaskState.READY
                task.wait_time = time.time() - task.arrival_time
                
                if self._on_task_ready:
                    self._on_task_ready(task)
                
                return task
            
            return None
    
    def start_task(self, task_id: str, worker_id: str) -> bool:
        """Mark task as started on a worker."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            if task.state not in (TaskState.PENDING, TaskState.READY):
                return False
            
            task.state = TaskState.RUNNING
            task.start_time = time.time()
            task.assigned_worker = worker_id
            
            if task in self.pending_tasks:
                self.pending_tasks.remove(task)
            self.running_tasks[task_id] = task
            
            return True
    
    def complete_task(self, task_id: str, success: bool = True) -> bool:
        """Mark task as completed."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            task.completion_time = time.time()
            task.state = TaskState.COMPLETED if success else TaskState.FAILED
            
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            self.completed_tasks.append(task)
            
            # Update metrics
            self._update_metrics(task)
            
            if self._on_task_complete:
                self._on_task_complete(task)
            
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            if task.state == TaskState.RUNNING:
                return False  # Cannot cancel running task
            
            task.state = TaskState.CANCELLED
            
            if task in self.pending_tasks:
                self.pending_tasks.remove(task)
            
            del self._task_map[task_id]
            return True
    
    def update_task_progress(self, task_id: str, remaining_size: float) -> None:
        """Update remaining work for a running task (for SRPT)."""
        with self._lock:
            if task_id in self._task_map:
                self._task_map[task_id].remaining_size = remaining_size
    
    def reorder_queue(self) -> None:
        """Force reorder of the task queue based on current policy."""
        with self._lock:
            self._reorder_queue()
    
    def set_policy(self, policy: SchedulingPolicy) -> None:
        """Change scheduling policy."""
        with self._lock:
            self.policy = policy
            self._reorder_queue()
    
    def _insert_to_heap(self, task: ADSATask) -> None:
        """Insert task into priority heap."""
        score = self._calculate_score(task)
        heapq.heappush(self._task_heap, (score, task.arrival_time, task.task_id))
    
    def _calculate_score(self, task: ADSATask) -> float:
        """
        Calculate scheduling score for a task.
        Lower score = higher priority (min-heap).
        """
        if self.policy == SchedulingPolicy.FIFO:
            return task.arrival_time
        
        elif self.policy == SchedulingPolicy.SJF:
            return task.estimated_size
        
        elif self.policy == SchedulingPolicy.SRPT:
            return task.remaining_size
        
        elif self.policy == SchedulingPolicy.WEIGHTED:
            return self._weighted_score(task)
        
        elif self.policy == SchedulingPolicy.ADAPTIVE:
            return self._adaptive_score(task)
        
        return task.arrival_time  # Default to FIFO
    
    def _weighted_score(self, task: ADSATask) -> float:
        """Calculate weighted combination score."""
        # Normalize components to 0-1 range
        now = time.time()
        
        # Arrival time component (older = lower score)
        max_wait = max((now - t.arrival_time) for t in self.pending_tasks) if self.pending_tasks else 1.0
        arrival_score = (now - task.arrival_time) / max_wait if max_wait > 0 else 0
        
        # Size component (smaller = lower score)
        max_size = max(t.estimated_size for t in self.pending_tasks) if self.pending_tasks else 1.0
        size_score = task.estimated_size / max_size if max_size > 0 else 0
        
        # Priority component (higher priority = lower score)
        priority_score = 1.0 - (task.effective_priority / 10.0)
        
        # Deadline component
        deadline_score = 0.0
        if task.deadline:
            remaining = task.deadline - now
            if remaining <= 0:
                deadline_score = -1.0  # Already overdue, highest priority
            else:
                deadline_score = remaining / 3600.0  # Normalize to hours
        
        # Weighted sum (lower is better)
        score = (
            -self.arrival_weight * arrival_score +
            self.size_weight * size_score +
            -self.priority_weight * (1 - priority_score) +
            self.deadline_weight * deadline_score
        )
        
        return score
    
    def _adaptive_score(self, task: ADSATask) -> float:
        """
        Adaptively select scoring based on system state.
        
        Rules:
        - High load: Favor smaller tasks (SJF-like)
        - Many deadlines: Favor urgent tasks
        - Long waits detected: Favor older tasks (FIFO-like)
        """
        now = time.time()
        
        # Analyze current state
        num_running = len(self.running_tasks)
        num_pending = len(self.pending_tasks)
        total_load = num_running + num_pending
        
        # Check for deadline pressure
        urgent_tasks = sum(
            1 for t in self.pending_tasks
            if t.deadline and (t.deadline - now) < 60
        )
        deadline_pressure = urgent_tasks / num_pending if num_pending > 0 else 0
        
        # Check for starvation risk
        max_wait = max(
            (now - t.arrival_time for t in self.pending_tasks),
            default=0
        )
        starvation_risk = max_wait > self.starvation_threshold_s
        
        # Adaptive weights based on state
        if deadline_pressure > 0.3:
            # Many urgent tasks: prioritize deadlines
            return self._deadline_aware_score(task)
        elif starvation_risk:
            # Starvation detected: favor older tasks
            return task.arrival_time - task.age_bonus
        elif total_load > 10:
            # High load: favor smaller tasks
            return task.remaining_size
        else:
            # Normal: use weighted combination
            return self._weighted_score(task)
    
    def _deadline_aware_score(self, task: ADSATask) -> float:
        """Score heavily weighted by deadline."""
        now = time.time()
        
        if task.deadline is None:
            return task.arrival_time + 3600  # Push non-deadline tasks back
        
        slack = task.deadline - now - task.remaining_size
        return -slack if slack > 0 else slack * 10  # Overdue gets massive priority
    
    def _apply_aging(self) -> None:
        """Apply aging bonus to waiting tasks."""
        now = time.time()
        
        for task in self.pending_tasks:
            wait_time = now - task.arrival_time
            task.age_bonus = wait_time * self.aging_rate
            
            # Boost severely starving tasks
            if wait_time > self.starvation_threshold_s:
                task.age_bonus *= 2.0
    
    def _maybe_reorder(self) -> None:
        """Reorder queue if interval has passed."""
        now = time.time()
        if (now - self._last_reorder_time) * 1000 >= self.reorder_interval_ms:
            self._reorder_queue()
            self._last_reorder_time = now
    
    def _reorder_queue(self) -> None:
        """Rebuild priority queue with current scores."""
        self._task_heap = []
        for task in self.pending_tasks:
            if task.state == TaskState.PENDING:
                self._insert_to_heap(task)
    
    def _update_metrics(self, task: ADSATask) -> None:
        """Update scheduler metrics after task completion."""
        if task.state == TaskState.COMPLETED and task.start_time and task.completion_time:
            self.metrics.completed_tasks += 1
            
            # Calculate times
            wait_time = task.start_time - task.arrival_time
            turnaround_time = task.completion_time - task.arrival_time
            response_time = task.start_time - task.arrival_time
            
            # Update running averages
            n = self.metrics.completed_tasks
            self.metrics.avg_wait_time = (
                (self.metrics.avg_wait_time * (n - 1) + wait_time) / n
            )
            self.metrics.avg_turnaround_time = (
                (self.metrics.avg_turnaround_time * (n - 1) + turnaround_time) / n
            )
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (n - 1) + response_time) / n
            )
            
            # Check deadline
            if task.deadline and task.completion_time > task.deadline:
                misses = self.metrics.deadline_miss_rate * (n - 1)
                self.metrics.deadline_miss_rate = (misses + 1) / n
    
    def get_metrics(self) -> SchedulerMetrics:
        """Get current scheduler metrics."""
        with self._lock:
            return SchedulerMetrics(
                total_tasks=self.metrics.total_tasks,
                completed_tasks=self.metrics.completed_tasks,
                avg_wait_time=self.metrics.avg_wait_time,
                avg_turnaround_time=self.metrics.avg_turnaround_time,
                avg_response_time=self.metrics.avg_response_time,
                throughput=self._calculate_throughput(),
                fairness_index=self._calculate_fairness(),
                deadline_miss_rate=self.metrics.deadline_miss_rate,
            )
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks completed per second."""
        if not self.completed_tasks:
            return 0.0
        
        first_arrival = min(t.arrival_time for t in self.completed_tasks)
        last_completion = max(t.completion_time for t in self.completed_tasks if t.completion_time)
        
        if last_completion and last_completion > first_arrival:
            duration = last_completion - first_arrival
            return len(self.completed_tasks) / duration
        
        return 0.0
    
    def _calculate_fairness(self) -> float:
        """Calculate Jain's fairness index based on wait times."""
        if len(self.completed_tasks) < 2:
            return 1.0
        
        wait_times = [
            t.wait_time for t in self.completed_tasks
            if t.wait_time is not None
        ]
        
        if not wait_times:
            return 1.0
        
        n = len(wait_times)
        sum_x = sum(wait_times)
        sum_x2 = sum(w * w for w in wait_times)
        
        if sum_x2 == 0:
            return 1.0
        
        return (sum_x * sum_x) / (n * sum_x2)
    
    def get_queue_status(self) -> Dict:
        """Get current queue status."""
        with self._lock:
            return {
                "policy": self.policy.value,
                "pending_count": len(self.pending_tasks),
                "running_count": len(self.running_tasks),
                "completed_count": len(self.completed_tasks),
                "next_task_id": self._task_heap[0][2] if self._task_heap else None,
            }
    
    def set_callbacks(
        self,
        on_task_ready: Optional[Callable[[ADSATask], None]] = None,
        on_task_complete: Optional[Callable[[ADSATask], None]] = None,
    ) -> None:
        """Set callback functions for task events."""
        self._on_task_ready = on_task_ready
        self._on_task_complete = on_task_complete


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("scheduler", "adsa")
    class ADSAPlugin:
        """ADSA Scheduler Plugin."""
        
        def __init__(self, **kwargs):
            self.scheduler = ADSAScheduler(**kwargs)
        
        def submit(self, task: ADSATask) -> str:
            return self.scheduler.submit_task(task)
        
        def next(self) -> Optional[ADSATask]:
            return self.scheduler.get_next_task()
        
        def complete(self, task_id: str) -> bool:
            return self.scheduler.complete_task(task_id)

except ImportError:
    pass
