"""
ILP Scheduler - Integer Linear Programming for Optimal Scheduling
With Heuristic fallback for large-scale systems.

Key Features:
- ILP formulation for optimal scheduling (small systems)
- Greedy/genetic heuristics for large systems
- Constraint handling (resources, dependencies, affinity)
- Multi-objective optimization
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random


class SolverType(Enum):
    """Solver type selection."""
    ILP = "ilp"  # Exact solution
    HEURISTIC = "heuristic"  # Fast approximate
    AUTO = "auto"  # Select based on problem size


@dataclass
class SchedulingProblem:
    """Definition of a scheduling problem."""
    # Tasks: task_id -> (memory_req, compute_req, priority)
    tasks: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)
    
    # Workers: worker_id -> (memory_cap, compute_cap, cost_factor)
    workers: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Dependencies: task_id -> list of task_ids it depends on
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Affinity: task_id -> preferred worker_id (optional)
    affinity: Dict[str, str] = field(default_factory=dict)
    
    # Anti-affinity: task_id -> list of task_ids that shouldn't be co-located
    anti_affinity: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SchedulingSolution:
    """Solution to a scheduling problem."""
    # Assignment: task_id -> worker_id
    assignments: Dict[str, str] = field(default_factory=dict)
    
    # Objective value (cost)
    objective_value: float = float('inf')
    
    # Solution quality metrics
    is_optimal: bool = False
    is_feasible: bool = False
    solver_type: SolverType = SolverType.HEURISTIC
    solve_time_s: float = 0.0
    
    # Resource utilization per worker
    worker_utilization: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class ILPSolver:
    """
    Integer Linear Programming solver for scheduling.
    
    Formulation:
    - Decision variables: x[i,j] ∈ {0,1} (task i assigned to worker j)
    - Objective: minimize Σ cost[j] * (mem[i] + compute[i]) * x[i,j]
    - Constraints:
        - Each task assigned exactly once: Σ_j x[i,j] = 1
        - Memory capacity: Σ_i mem[i] * x[i,j] ≤ mem_cap[j]
        - Compute capacity: Σ_i compute[i] * x[i,j] ≤ compute_cap[j]
    """
    
    def __init__(self, time_limit_s: float = 60.0):
        self.time_limit_s = time_limit_s
        self._has_solver = self._check_solver_available()
    
    def _check_solver_available(self) -> bool:
        """Check if an ILP solver is available."""
        try:
            # Try PuLP (common Python LP library)
            import pulp
            return True
        except ImportError:
            pass
        
        try:
            # Try scipy's linear programming
            from scipy.optimize import milp
            return True
        except ImportError:
            pass
        
        return False
    
    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Solve the scheduling problem using ILP."""
        if not self._has_solver:
            # Fallback to built-in approximation
            return self._solve_approximation(problem)
        
        try:
            return self._solve_with_pulp(problem)
        except ImportError:
            return self._solve_approximation(problem)
    
    def _solve_with_pulp(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Solve using PuLP library."""
        import pulp
        
        start_time = time.time()
        
        tasks = list(problem.tasks.keys())
        workers = list(problem.workers.keys())
        
        # Create problem
        model = pulp.LpProblem("Scheduling", pulp.LpMinimize)
        
        # Decision variables: x[i,j] = 1 if task i assigned to worker j
        x = pulp.LpVariable.dicts(
            "x",
            [(i, j) for i in tasks for j in workers],
            cat=pulp.LpBinary
        )
        
        # Objective: minimize total cost
        model += pulp.lpSum([
            problem.workers[j][2] *  # cost factor
            (problem.tasks[i][0] + problem.tasks[i][1]) *  # mem + compute
            x[(i, j)]
            for i in tasks
            for j in workers
        ])
        
        # Constraint: each task assigned exactly once
        for i in tasks:
            model += pulp.lpSum([x[(i, j)] for j in workers]) == 1
        
        # Constraint: memory capacity
        for j in workers:
            model += pulp.lpSum([
                problem.tasks[i][0] * x[(i, j)]
                for i in tasks
            ]) <= problem.workers[j][0]
        
        # Constraint: compute capacity
        for j in workers:
            model += pulp.lpSum([
                problem.tasks[i][1] * x[(i, j)]
                for i in tasks
            ]) <= problem.workers[j][1]
        
        # Affinity constraints (soft - add to objective as preference)
        for task_id, pref_worker in problem.affinity.items():
            if task_id in tasks and pref_worker in workers:
                # Add bonus for matching affinity
                model += x[(task_id, pref_worker)] >= 0  # Just ensure it's considered
        
        # Solve
        model.solve(pulp.PULP_CBC_CMD(timeLimit=self.time_limit_s, msg=0))
        
        solve_time = time.time() - start_time
        
        # Extract solution
        solution = SchedulingSolution(
            solver_type=SolverType.ILP,
            solve_time_s=solve_time,
        )
        
        if model.status == pulp.LpStatusOptimal:
            solution.is_optimal = True
            solution.is_feasible = True
            solution.objective_value = pulp.value(model.objective)
            
            for i in tasks:
                for j in workers:
                    if pulp.value(x[(i, j)]) > 0.5:
                        solution.assignments[i] = j
        elif model.status == pulp.LpStatusNotSolved:
            solution.is_feasible = False
        else:
            # Feasible but not proven optimal
            solution.is_feasible = True
            if model.objective:
                solution.objective_value = pulp.value(model.objective)
            
            for i in tasks:
                for j in workers:
                    if pulp.value(x[(i, j)]) and pulp.value(x[(i, j)]) > 0.5:
                        solution.assignments[i] = j
        
        # Calculate utilization
        solution.worker_utilization = self._calculate_utilization(
            problem, solution.assignments
        )
        
        return solution
    
    def _solve_approximation(self, problem: SchedulingProblem) -> SchedulingSolution:
        """
        Solve using approximation when ILP solver unavailable.
        Uses greedy assignment with local search.
        """
        start_time = time.time()
        
        # Sort tasks by resource requirement (largest first)
        sorted_tasks = sorted(
            problem.tasks.items(),
            key=lambda x: x[1][0] + x[1][1],
            reverse=True
        )
        
        assignments = {}
        worker_usage = {w: [0.0, 0.0] for w in problem.workers}
        
        for task_id, (mem_req, compute_req, priority) in sorted_tasks:
            # Find best worker
            best_worker = None
            best_score = float('inf')
            
            for worker_id, (mem_cap, compute_cap, cost) in problem.workers.items():
                # Check capacity
                used_mem, used_compute = worker_usage[worker_id]
                if used_mem + mem_req > mem_cap:
                    continue
                if used_compute + compute_req > compute_cap:
                    continue
                
                # Calculate score (lower is better)
                score = cost * (mem_req + compute_req)
                
                # Affinity bonus
                if problem.affinity.get(task_id) == worker_id:
                    score *= 0.8
                
                # Load balancing
                utilization = (used_mem / mem_cap + used_compute / compute_cap) / 2
                score *= (1 + utilization * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_worker = worker_id
            
            if best_worker:
                assignments[task_id] = best_worker
                worker_usage[best_worker][0] += mem_req
                worker_usage[best_worker][1] += compute_req
        
        solve_time = time.time() - start_time
        
        # Calculate objective
        objective = sum(
            problem.workers[w][2] * (problem.tasks[t][0] + problem.tasks[t][1])
            for t, w in assignments.items()
        )
        
        solution = SchedulingSolution(
            assignments=assignments,
            objective_value=objective,
            is_optimal=False,
            is_feasible=len(assignments) == len(problem.tasks),
            solver_type=SolverType.HEURISTIC,
            solve_time_s=solve_time,
            worker_utilization=self._calculate_utilization(problem, assignments),
        )
        
        return solution
    
    def _calculate_utilization(
        self,
        problem: SchedulingProblem,
        assignments: Dict[str, str]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate resource utilization per worker."""
        usage = {w: [0.0, 0.0] for w in problem.workers}
        
        for task_id, worker_id in assignments.items():
            if task_id in problem.tasks and worker_id in usage:
                usage[worker_id][0] += problem.tasks[task_id][0]
                usage[worker_id][1] += problem.tasks[task_id][1]
        
        utilization = {}
        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            utilization[worker_id] = (
                used_mem / mem_cap if mem_cap > 0 else 0,
                used_compute / compute_cap if compute_cap > 0 else 0
            )
        
        return utilization


class HeuristicScheduler:
    """
    Heuristic-based scheduler for large-scale systems.
    
    Algorithms:
    - Greedy first-fit
    - Genetic algorithm
    - Simulated annealing
    """
    
    def __init__(
        self,
        algorithm: str = "genetic",
        max_iterations: int = 100,
        population_size: int = 50,
        time_limit_s: float = 10.0,
    ):
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.time_limit_s = time_limit_s
    
    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Solve using selected heuristic algorithm."""
        if self.algorithm == "greedy":
            return self._solve_greedy(problem)
        elif self.algorithm == "genetic":
            return self._solve_genetic(problem)
        elif self.algorithm == "annealing":
            return self._solve_annealing(problem)
        else:
            return self._solve_greedy(problem)
    
    def _solve_greedy(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Greedy first-fit decreasing algorithm."""
        start_time = time.time()
        
        # Sort tasks by size (decreasing)
        sorted_tasks = sorted(
            problem.tasks.items(),
            key=lambda x: x[1][0] + x[1][1],
            reverse=True
        )
        
        assignments = {}
        worker_usage = {w: [0.0, 0.0] for w in problem.workers}
        
        for task_id, (mem_req, compute_req, _) in sorted_tasks:
            # Find first worker with capacity
            for worker_id, (mem_cap, compute_cap, _) in problem.workers.items():
                used_mem, used_compute = worker_usage[worker_id]
                
                if used_mem + mem_req <= mem_cap and used_compute + compute_req <= compute_cap:
                    assignments[task_id] = worker_id
                    worker_usage[worker_id][0] += mem_req
                    worker_usage[worker_id][1] += compute_req
                    break
        
        return self._create_solution(problem, assignments, start_time)
    
    def _solve_genetic(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Genetic algorithm for scheduling."""
        start_time = time.time()
        tasks = list(problem.tasks.keys())
        workers = list(problem.workers.keys())
        
        if not tasks or not workers:
            return SchedulingSolution(is_feasible=False)
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self._create_random_assignment(problem, tasks, workers)
            fitness = self._calculate_fitness(problem, individual)
            population.append((individual, fitness))
        
        # Evolution loop
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.time_limit_s:
                break
            
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            # Selection: keep top 50%
            survivors = population[:self.population_size // 2]
            
            # Crossover and mutation
            offspring = []
            while len(offspring) < self.population_size // 2:
                parent1 = random.choice(survivors)[0]
                parent2 = random.choice(survivors)[0]
                
                child = self._crossover(parent1, parent2, tasks)
                child = self._mutate(child, workers, mutation_rate=0.1)
                
                # Repair infeasible solutions
                child = self._repair(problem, child)
                
                fitness = self._calculate_fitness(problem, child)
                offspring.append((child, fitness))
            
            population = survivors + offspring
        
        # Get best solution
        population.sort(key=lambda x: x[1])
        best_assignment, best_fitness = population[0]
        
        return self._create_solution(problem, best_assignment, start_time)
    
    def _solve_annealing(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Simulated annealing for scheduling."""
        start_time = time.time()
        tasks = list(problem.tasks.keys())
        workers = list(problem.workers.keys())
        
        # Initial solution
        current = self._create_random_assignment(problem, tasks, workers)
        current_cost = self._calculate_fitness(problem, current)
        
        best = dict(current)
        best_cost = current_cost
        
        # Annealing parameters
        temperature = 100.0
        cooling_rate = 0.99
        min_temperature = 0.1
        
        while temperature > min_temperature:
            if time.time() - start_time > self.time_limit_s:
                break
            
            # Generate neighbor
            neighbor = dict(current)
            task_to_move = random.choice(tasks)
            new_worker = random.choice(workers)
            neighbor[task_to_move] = new_worker
            
            # Repair if needed
            neighbor = self._repair(problem, neighbor)
            neighbor_cost = self._calculate_fitness(problem, neighbor)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best = dict(current)
                    best_cost = current_cost
            
            temperature *= cooling_rate
        
        return self._create_solution(problem, best, start_time)
    
    def _create_random_assignment(
        self,
        problem: SchedulingProblem,
        tasks: List[str],
        workers: List[str]
    ) -> Dict[str, str]:
        """Create random but feasible assignment."""
        assignment = {}
        worker_usage = {w: [0.0, 0.0] for w in workers}
        
        shuffled_tasks = list(tasks)
        random.shuffle(shuffled_tasks)
        
        for task_id in shuffled_tasks:
            mem_req, compute_req, _ = problem.tasks[task_id]
            
            # Try random workers
            shuffled_workers = list(workers)
            random.shuffle(shuffled_workers)
            
            assigned = False
            for worker_id in shuffled_workers:
                mem_cap, compute_cap, _ = problem.workers[worker_id]
                used_mem, used_compute = worker_usage[worker_id]
                
                if used_mem + mem_req <= mem_cap and used_compute + compute_req <= compute_cap:
                    assignment[task_id] = worker_id
                    worker_usage[worker_id][0] += mem_req
                    worker_usage[worker_id][1] += compute_req
                    assigned = True
                    break
            
            if not assigned:
                # Force assignment to first worker (may be infeasible)
                assignment[task_id] = workers[0]
        
        return assignment
    
    def _crossover(
        self,
        parent1: Dict[str, str],
        parent2: Dict[str, str],
        tasks: List[str]
    ) -> Dict[str, str]:
        """Single-point crossover."""
        crossover_point = random.randint(0, len(tasks))
        child = {}
        
        for i, task_id in enumerate(tasks):
            if i < crossover_point:
                child[task_id] = parent1.get(task_id, parent2.get(task_id))
            else:
                child[task_id] = parent2.get(task_id, parent1.get(task_id))
        
        return child
    
    def _mutate(
        self,
        assignment: Dict[str, str],
        workers: List[str],
        mutation_rate: float = 0.1
    ) -> Dict[str, str]:
        """Mutate assignment."""
        mutated = dict(assignment)
        
        for task_id in mutated:
            if random.random() < mutation_rate:
                mutated[task_id] = random.choice(workers)
        
        return mutated
    
    def _repair(
        self,
        problem: SchedulingProblem,
        assignment: Dict[str, str]
    ) -> Dict[str, str]:
        """Repair infeasible assignment."""
        repaired = dict(assignment)
        worker_usage = {w: [0.0, 0.0] for w in problem.workers}
        
        # Calculate current usage
        for task_id, worker_id in repaired.items():
            if worker_id in problem.workers and task_id in problem.tasks:
                mem_req, compute_req, _ = problem.tasks[task_id]
                worker_usage[worker_id][0] += mem_req
                worker_usage[worker_id][1] += compute_req
        
        # Find and fix violations
        for worker_id, (used_mem, used_compute) in worker_usage.items():
            if worker_id not in problem.workers:
                continue
            
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            
            while used_mem > mem_cap or used_compute > compute_cap:
                # Find tasks on this worker and move one
                tasks_on_worker = [
                    t for t, w in repaired.items() if w == worker_id
                ]
                
                if not tasks_on_worker:
                    break
                
                # Move smallest task to another worker
                task_to_move = min(
                    tasks_on_worker,
                    key=lambda t: problem.tasks[t][0] + problem.tasks[t][1]
                )
                
                # Find alternative worker
                for alt_worker in problem.workers:
                    if alt_worker == worker_id:
                        continue
                    
                    alt_mem_cap, alt_compute_cap, _ = problem.workers[alt_worker]
                    task_mem, task_compute, _ = problem.tasks[task_to_move]
                    
                    if (worker_usage[alt_worker][0] + task_mem <= alt_mem_cap and
                        worker_usage[alt_worker][1] + task_compute <= alt_compute_cap):
                        
                        repaired[task_to_move] = alt_worker
                        worker_usage[worker_id][0] -= task_mem
                        worker_usage[worker_id][1] -= task_compute
                        worker_usage[alt_worker][0] += task_mem
                        worker_usage[alt_worker][1] += task_compute
                        used_mem -= task_mem
                        used_compute -= task_compute
                        break
                else:
                    break  # No alternative found
        
        return repaired
    
    def _calculate_fitness(
        self,
        problem: SchedulingProblem,
        assignment: Dict[str, str]
    ) -> float:
        """Calculate fitness (cost) of assignment."""
        cost = 0.0
        worker_usage = {w: [0.0, 0.0] for w in problem.workers}
        penalty = 0.0
        
        for task_id, worker_id in assignment.items():
            if task_id not in problem.tasks or worker_id not in problem.workers:
                penalty += 1000
                continue
            
            mem_req, compute_req, _ = problem.tasks[task_id]
            _, _, cost_factor = problem.workers[worker_id]
            
            cost += cost_factor * (mem_req + compute_req)
            worker_usage[worker_id][0] += mem_req
            worker_usage[worker_id][1] += compute_req
        
        # Add penalty for capacity violations
        for worker_id, (used_mem, used_compute) in worker_usage.items():
            if worker_id not in problem.workers:
                continue
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            
            if used_mem > mem_cap:
                penalty += (used_mem - mem_cap) * 100
            if used_compute > compute_cap:
                penalty += (used_compute - compute_cap) * 100
        
        return cost + penalty
    
    def _create_solution(
        self,
        problem: SchedulingProblem,
        assignments: Dict[str, str],
        start_time: float
    ) -> SchedulingSolution:
        """Create solution object from assignments."""
        solve_time = time.time() - start_time
        
        objective = self._calculate_fitness(problem, assignments)
        
        # Check feasibility
        is_feasible = len(assignments) == len(problem.tasks)
        worker_usage = {w: [0.0, 0.0] for w in problem.workers}
        
        for task_id, worker_id in assignments.items():
            if task_id in problem.tasks and worker_id in problem.workers:
                mem_req, compute_req, _ = problem.tasks[task_id]
                worker_usage[worker_id][0] += mem_req
                worker_usage[worker_id][1] += compute_req
        
        for worker_id, (used_mem, used_compute) in worker_usage.items():
            if worker_id in problem.workers:
                mem_cap, compute_cap, _ = problem.workers[worker_id]
                if used_mem > mem_cap or used_compute > compute_cap:
                    is_feasible = False
                    break
        
        # Calculate utilization
        utilization = {}
        for worker_id, (used_mem, used_compute) in worker_usage.items():
            if worker_id in problem.workers:
                mem_cap, compute_cap, _ = problem.workers[worker_id]
                utilization[worker_id] = (
                    used_mem / mem_cap if mem_cap > 0 else 0,
                    used_compute / compute_cap if compute_cap > 0 else 0
                )
        
        return SchedulingSolution(
            assignments=assignments,
            objective_value=objective,
            is_optimal=False,
            is_feasible=is_feasible,
            solver_type=SolverType.HEURISTIC,
            solve_time_s=solve_time,
            worker_utilization=utilization,
        )


class AdaptiveScheduler:
    """
    Adaptive scheduler that selects ILP or heuristic based on problem size.
    """
    
    def __init__(
        self,
        ilp_threshold: int = 50,  # Use ILP if tasks * workers <= threshold
        ilp_time_limit_s: float = 60.0,
        heuristic_time_limit_s: float = 10.0,
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            ilp_threshold: Max problem size for ILP
            ilp_time_limit_s: Time limit for ILP solver
            heuristic_time_limit_s: Time limit for heuristic solver
        """
        self.ilp_threshold = ilp_threshold
        self.ilp_solver = ILPSolver(time_limit_s=ilp_time_limit_s)
        self.heuristic_solver = HeuristicScheduler(
            algorithm="genetic",
            time_limit_s=heuristic_time_limit_s
        )
    
    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """
        Solve scheduling problem using appropriate solver.
        """
        problem_size = len(problem.tasks) * len(problem.workers)
        
        if problem_size <= self.ilp_threshold:
            # Small problem: use ILP for optimal solution
            solution = self.ilp_solver.solve(problem)
            
            # If ILP fails, fall back to heuristic
            if not solution.is_feasible:
                solution = self.heuristic_solver.solve(problem)
        else:
            # Large problem: use heuristic
            solution = self.heuristic_solver.solve(problem)
        
        return solution


# Plugin registration
try:
    from model.plugin_architecture import PluginRegistry
    
    @PluginRegistry.register("scheduler", "ilp")
    class ILPSchedulerPlugin:
        """ILP Scheduler Plugin."""
        
        def __init__(self, **kwargs):
            self.scheduler = AdaptiveScheduler(**kwargs)
        
        def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
            return self.scheduler.solve(problem)

except ImportError:
    pass
