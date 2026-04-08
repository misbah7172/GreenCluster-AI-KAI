"""
DFS Scheduler - Depth-First Tree Search with Pruning
Explores scheduling/allocation search space using DFS with pruning.

Key Features:
- State-space exploration for task scheduling
- Alpha-beta style pruning for efficiency
- Branch-and-bound for resource allocation
- Configurable pruning heuristics
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from enum import Enum
import heapq


class PruningStrategy(Enum):
    """Pruning strategy types."""
    NONE = "none"
    ALPHA_BETA = "alpha_beta"
    BOUND = "bound"  # Branch and bound
    BEAM = "beam"  # Beam search (limited width)
    HEURISTIC = "heuristic"  # Custom heuristic pruning


@dataclass
class ScheduleState:
    """State in the scheduling search tree."""
    # Current assignments: task_id -> worker_id
    assignments: Dict[str, str] = field(default_factory=dict)
    
    # Remaining unassigned tasks
    unassigned_tasks: List[str] = field(default_factory=list)
    
    # Resource usage per worker: worker_id -> (memory_used, compute_used)
    resource_usage: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Cost metrics
    total_cost: float = 0.0
    estimated_remaining_cost: float = 0.0
    
    # Search metadata
    depth: int = 0
    parent_state: Optional['ScheduleState'] = None
    action_taken: Optional[str] = None  # Description of action that led here
    
    @property
    def is_complete(self) -> bool:
        """Check if all tasks are assigned."""
        return len(self.unassigned_tasks) == 0
    
    @property
    def lower_bound(self) -> float:
        """Lower bound on total cost from this state."""
        return self.total_cost + self.estimated_remaining_cost
    
    def copy(self) -> 'ScheduleState':
        """Create a copy of this state."""
        return ScheduleState(
            assignments=dict(self.assignments),
            unassigned_tasks=list(self.unassigned_tasks),
            resource_usage={k: (v[0], v[1]) for k, v in self.resource_usage.items()},
            total_cost=self.total_cost,
            estimated_remaining_cost=self.estimated_remaining_cost,
            depth=self.depth,
            parent_state=self,
            action_taken=None,
        )


@dataclass
class TaskSpec:
    """Task specification for scheduling."""
    task_id: str
    memory_required: float  # GB
    compute_required: float  # FLOPs
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    
    # Affinity constraints
    preferred_workers: List[str] = field(default_factory=list)
    excluded_workers: List[str] = field(default_factory=list)


@dataclass
class WorkerSpec:
    """Worker specification for allocation."""
    worker_id: str
    memory_capacity: float  # GB
    compute_capacity: float  # FLOPs
    cost_per_unit: float = 1.0  # Cost multiplier
    current_load: float = 0.0


@dataclass
class SearchResult:
    """Result of DFS search."""
    best_state: Optional[ScheduleState]
    best_cost: float
    nodes_explored: int
    nodes_pruned: int
    search_time_s: float
    is_optimal: bool


class DFSScheduler:
    """
    Depth-First Search scheduler with pruning for task scheduling
    and resource allocation.
    
    Algorithms:
    - DFS with alpha-beta pruning for min-cost scheduling
    - Branch and bound for resource allocation
    - Beam search for large search spaces
    """
    
    def __init__(
        self,
        pruning_strategy: PruningStrategy = PruningStrategy.BOUND,
        max_depth: int = 100,
        time_limit_s: float = 10.0,
        beam_width: int = 10,
        heuristic_fn: Optional[Callable[[ScheduleState], float]] = None,
    ):
        """
        Initialize DFS scheduler.
        
        Args:
            pruning_strategy: Strategy for pruning search tree
            max_depth: Maximum search depth
            time_limit_s: Time limit for search
            beam_width: Width for beam search
            heuristic_fn: Custom heuristic function for pruning
        """
        self.pruning_strategy = pruning_strategy
        self.max_depth = max_depth
        self.time_limit_s = time_limit_s
        self.beam_width = beam_width
        self.heuristic_fn = heuristic_fn or self._default_heuristic
        
        self.tasks: Dict[str, TaskSpec] = {}
        self.workers: Dict[str, WorkerSpec] = {}
        
        # Search statistics
        self._nodes_explored = 0
        self._nodes_pruned = 0
        self._best_cost = float('inf')
        self._best_state: Optional[ScheduleState] = None
        self._start_time = 0.0
    
    def add_task(self, task: TaskSpec) -> None:
        """Add a task to schedule."""
        self.tasks[task.task_id] = task
    
    def add_worker(self, worker: WorkerSpec) -> None:
        """Add a worker for allocation."""
        self.workers[worker.worker_id] = worker
    
    def solve(self) -> SearchResult:
        """
        Find optimal or near-optimal schedule using DFS with pruning.
        
        Returns SearchResult with best found schedule.
        """
        if not self.tasks or not self.workers:
            return SearchResult(
                best_state=None,
                best_cost=float('inf'),
                nodes_explored=0,
                nodes_pruned=0,
                search_time_s=0.0,
                is_optimal=False,
            )
        
        # Initialize search
        self._nodes_explored = 0
        self._nodes_pruned = 0
        self._best_cost = float('inf')
        self._best_state = None
        self._start_time = time.time()
        
        # Create initial state
        initial_state = ScheduleState(
            unassigned_tasks=self._get_task_order(),
            resource_usage={w: (0.0, 0.0) for w in self.workers},
        )
        initial_state.estimated_remaining_cost = self._estimate_remaining_cost(initial_state)
        
        # Run search based on strategy
        if self.pruning_strategy == PruningStrategy.BEAM:
            self._beam_search(initial_state)
        else:
            self._dfs_search(initial_state, float('-inf'), float('inf'))
        
        search_time = time.time() - self._start_time
        
        return SearchResult(
            best_state=self._best_state,
            best_cost=self._best_cost,
            nodes_explored=self._nodes_explored,
            nodes_pruned=self._nodes_pruned,
            search_time_s=search_time,
            is_optimal=self._best_cost < float('inf') and not self._time_exceeded(),
        )
    
    def _get_task_order(self) -> List[str]:
        """
        Get task ordering for search (affects efficiency).
        Orders by: dependencies satisfied, priority, resource requirements.
        """
        # Topological sort considering dependencies
        in_degree = {t: 0 for t in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1
        
        # Sort by in-degree, then priority (descending), then resource (descending)
        tasks = sorted(
            self.tasks.values(),
            key=lambda t: (
                in_degree[t.task_id],
                -t.priority,
                -(t.memory_required + t.compute_required)
            )
        )
        
        return [t.task_id for t in tasks]
    
    def _dfs_search(
        self,
        state: ScheduleState,
        alpha: float,
        beta: float
    ) -> float:
        """
        DFS with alpha-beta style pruning.
        
        Returns the best cost found from this state.
        """
        # Check termination conditions
        if self._time_exceeded():
            return self._best_cost
        
        self._nodes_explored += 1
        
        # Complete state - update best if better
        if state.is_complete:
            if state.total_cost < self._best_cost:
                self._best_cost = state.total_cost
                self._best_state = state
            return state.total_cost
        
        # Check depth limit
        if state.depth >= self.max_depth:
            return float('inf')
        
        # Pruning checks
        if self._should_prune(state, alpha, beta):
            self._nodes_pruned += 1
            return float('inf')
        
        # Get next task to assign
        task_id = state.unassigned_tasks[0]
        task = self.tasks[task_id]
        
        # Generate and explore child states
        best_cost = float('inf')
        children = self._generate_children(state, task)
        
        # Sort children by heuristic (best first)
        children.sort(key=lambda s: s.lower_bound)
        
        for child in children:
            if self._should_prune(child, alpha, beta):
                self._nodes_pruned += 1
                continue
            
            cost = self._dfs_search(child, alpha, beta)
            
            if cost < best_cost:
                best_cost = cost
                
                # Update alpha for minimization
                if cost < beta:
                    beta = cost
                
                # Alpha-beta cutoff
                if self.pruning_strategy == PruningStrategy.ALPHA_BETA:
                    if alpha >= beta:
                        break
        
        return best_cost
    
    def _beam_search(self, initial_state: ScheduleState) -> None:
        """
        Beam search - maintains top-k states at each level.
        """
        current_beam = [initial_state]
        
        while current_beam and not self._time_exceeded():
            # Check for complete states
            for state in current_beam:
                if state.is_complete and state.total_cost < self._best_cost:
                    self._best_cost = state.total_cost
                    self._best_state = state
            
            # Filter complete states from beam
            current_beam = [s for s in current_beam if not s.is_complete]
            
            if not current_beam:
                break
            
            # Generate all children
            all_children = []
            for state in current_beam:
                self._nodes_explored += 1
                
                if state.unassigned_tasks:
                    task_id = state.unassigned_tasks[0]
                    task = self.tasks[task_id]
                    children = self._generate_children(state, task)
                    all_children.extend(children)
            
            if not all_children:
                break
            
            # Select top-k by lower bound
            all_children.sort(key=lambda s: s.lower_bound)
            current_beam = all_children[:self.beam_width]
            
            # Prune states worse than current best
            current_beam = [
                s for s in current_beam
                if s.lower_bound < self._best_cost
            ]
            self._nodes_pruned += len(all_children) - len(current_beam)
    
    def _generate_children(
        self,
        state: ScheduleState,
        task: TaskSpec
    ) -> List[ScheduleState]:
        """Generate child states by assigning task to each valid worker."""
        children = []
        
        for worker_id, worker in self.workers.items():
            # Check exclusion constraints
            if worker_id in task.excluded_workers:
                continue
            
            # Check resource availability
            used_mem, used_compute = state.resource_usage.get(worker_id, (0.0, 0.0))
            available_mem = worker.memory_capacity - used_mem
            available_compute = worker.compute_capacity - used_compute
            
            if available_mem < task.memory_required:
                continue
            if available_compute < task.compute_required:
                continue
            
            # Check dependency constraints
            if not self._dependencies_satisfied(state, task, worker_id):
                continue
            
            # Create child state
            child = state.copy()
            child.assignments[task.task_id] = worker_id
            child.unassigned_tasks = state.unassigned_tasks[1:]
            child.resource_usage[worker_id] = (
                used_mem + task.memory_required,
                used_compute + task.compute_required
            )
            child.depth = state.depth + 1
            child.action_taken = f"Assign {task.task_id} to {worker_id}"
            
            # Calculate cost
            assignment_cost = self._calculate_assignment_cost(task, worker)
            child.total_cost = state.total_cost + assignment_cost
            child.estimated_remaining_cost = self._estimate_remaining_cost(child)
            
            children.append(child)
        
        return children
    
    def _dependencies_satisfied(
        self,
        state: ScheduleState,
        task: TaskSpec,
        worker_id: str
    ) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in state.assignments:
                return False  # Dependency not yet assigned
        return True
    
    def _calculate_assignment_cost(
        self,
        task: TaskSpec,
        worker: WorkerSpec
    ) -> float:
        """Calculate cost of assigning task to worker."""
        # Base cost from worker cost multiplier
        base_cost = worker.cost_per_unit * (task.memory_required + task.compute_required)
        
        # Affinity bonus/penalty
        if task.preferred_workers and worker.worker_id in task.preferred_workers:
            base_cost *= 0.8  # 20% discount for preferred
        elif task.preferred_workers:
            base_cost *= 1.2  # 20% penalty for non-preferred
        
        # Load balancing cost
        load_factor = 1.0 + worker.current_load * 0.5
        
        return base_cost * load_factor
    
    def _estimate_remaining_cost(self, state: ScheduleState) -> float:
        """
        Estimate lower bound on cost for remaining tasks.
        Used for branch-and-bound pruning.
        """
        if not state.unassigned_tasks:
            return 0.0
        
        remaining_cost = 0.0
        
        for task_id in state.unassigned_tasks:
            task = self.tasks[task_id]
            
            # Find minimum cost worker for this task
            min_cost = float('inf')
            for worker_id, worker in self.workers.items():
                if worker_id in task.excluded_workers:
                    continue
                
                cost = self._calculate_assignment_cost(task, worker)
                min_cost = min(min_cost, cost)
            
            remaining_cost += min_cost if min_cost < float('inf') else 0
        
        return remaining_cost
    
    def _should_prune(
        self,
        state: ScheduleState,
        alpha: float,
        beta: float
    ) -> bool:
        """Determine if state should be pruned."""
        if self.pruning_strategy == PruningStrategy.NONE:
            return False
        
        # Bound-based pruning
        if self.pruning_strategy in (PruningStrategy.BOUND, PruningStrategy.ALPHA_BETA):
            if state.lower_bound >= self._best_cost:
                return True
        
        # Heuristic-based pruning
        if self.pruning_strategy == PruningStrategy.HEURISTIC:
            heuristic_value = self.heuristic_fn(state)
            if heuristic_value > self._best_cost * 1.5:  # 50% tolerance
                return True
        
        return False
    
    def _default_heuristic(self, state: ScheduleState) -> float:
        """Default heuristic for pruning."""
        return state.lower_bound
    
    def _time_exceeded(self) -> bool:
        """Check if time limit exceeded."""
        return (time.time() - self._start_time) > self.time_limit_s
    
    def get_schedule_from_result(self, result: SearchResult) -> Dict[str, str]:
        """Extract task-to-worker mapping from search result."""
        if result.best_state:
            return dict(result.best_state.assignments)
        return {}


class ResourceAllocator:
    """
    Resource allocation using DFS with branch-and-bound.
    Optimizes resource distribution across nodes.
    """
    
    def __init__(self, time_limit_s: float = 5.0):
        self.time_limit_s = time_limit_s
        self.scheduler = DFSScheduler(
            pruning_strategy=PruningStrategy.BOUND,
            time_limit_s=time_limit_s,
        )
    
    def allocate_layers(
        self,
        layers: List[Dict[str, float]],  # List of {memory, compute}
        nodes: List[Dict[str, float]],  # List of {memory_cap, compute_cap, cost}
    ) -> Tuple[Dict[int, int], float]:
        """
        Allocate layers to nodes minimizing cost.
        
        Returns (layer_to_node mapping, total_cost).
        """
        # Convert to scheduler format
        for i, layer in enumerate(layers):
            task = TaskSpec(
                task_id=f"layer_{i}",
                memory_required=layer.get("memory", 0),
                compute_required=layer.get("compute", 0),
            )
            self.scheduler.add_task(task)
        
        for j, node in enumerate(nodes):
            worker = WorkerSpec(
                worker_id=f"node_{j}",
                memory_capacity=node.get("memory_cap", 1e9),
                compute_capacity=node.get("compute_cap", 1e15),
                cost_per_unit=node.get("cost", 1.0),
            )
            self.scheduler.add_worker(worker)
        
        result = self.scheduler.solve()
        
        # Convert result to layer->node mapping
        layer_to_node = {}
        if result.best_state:
            for task_id, worker_id in result.best_state.assignments.items():
                layer_idx = int(task_id.split("_")[1])
                node_idx = int(worker_id.split("_")[1])
                layer_to_node[layer_idx] = node_idx
        
        return layer_to_node, result.best_cost


class TaskDependencyScheduler:
    """
    Scheduler for tasks with dependencies using DFS.
    Ensures dependency constraints are satisfied.
    """
    
    def __init__(self, time_limit_s: float = 10.0):
        self.scheduler = DFSScheduler(
            pruning_strategy=PruningStrategy.ALPHA_BETA,
            time_limit_s=time_limit_s,
        )
    
    def schedule(
        self,
        tasks: List[TaskSpec],
        workers: List[WorkerSpec]
    ) -> SearchResult:
        """
        Schedule tasks respecting dependencies.
        """
        for task in tasks:
            self.scheduler.add_task(task)
        
        for worker in workers:
            self.scheduler.add_worker(worker)
        
        return self.scheduler.solve()


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("scheduler", "dfs")
    class DFSSchedulerPlugin:
        """DFS Scheduler Plugin."""
        
        def __init__(self, **kwargs):
            self.scheduler = DFSScheduler(**kwargs)
        
        def add_task(self, task: TaskSpec):
            self.scheduler.add_task(task)
        
        def add_worker(self, worker: WorkerSpec):
            self.scheduler.add_worker(worker)
        
        def solve(self) -> SearchResult:
            return self.scheduler.solve()

except ImportError:
    pass
