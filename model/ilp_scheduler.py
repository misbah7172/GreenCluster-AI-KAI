"""
ILP Scheduler - Integer Linear Programming for Optimal Scheduling.

This module provides:
- ILP formulation for task-to-worker assignment (small/medium systems)
- Heuristic schedulers for larger systems
- Adaptive/hybrid strategy that combines ILP for critical tasks and
  heuristics for the rest
- Diagnostics and monitoring hooks for production operation
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class SolverType(Enum):
    """Solver type selection."""

    ILP = "ilp"
    HEURISTIC = "heuristic"
    AUTO = "auto"


@dataclass
class SchedulingProblem:
    """Definition of a scheduling problem."""

    # Tasks: task_id -> (memory_req, compute_req, priority)
    tasks: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)

    # Workers: worker_id -> (memory_cap, compute_cap, cost_factor)
    workers: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    # Dependencies: task_id -> list of task_ids it depends on
    dependencies: Dict[str, List[str]] = field(default_factory=dict)

    # Affinity: task_id -> preferred worker_id
    affinity: Dict[str, str] = field(default_factory=dict)

    # Tasks in this set enforce affinity as hard constraints.
    hard_affinity_tasks: Set[str] = field(default_factory=set)

    # How to treat dependencies when no full time model is used.
    # - "none": ignore dependencies at assignment stage
    # - "colocate": dependent tasks must be assigned to same worker
    # - "precedence": enable simple start-time precedence constraints
    dependency_mode: str = "colocate"

    # Anti-affinity: task_id -> list of task_ids that must not be co-located
    anti_affinity: Dict[str, List[str]] = field(default_factory=dict)

    # Optional latency penalty for assigning a task to a worker.
    # Key: (task_id, worker_id) -> penalty
    latency_penalty: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Optional per-task durations (seconds), used by precedence mode.
    task_durations_s: Dict[str, float] = field(default_factory=dict)

    # Objective weights: alpha*resource + beta*imbalance + gamma*latency + delta*soft_affinity + epsilon*time
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "resource": 1.0,
            "imbalance": 10.0,
            "latency": 0.0,
            "soft_affinity": 5.0,
            "time": 0.0,
        }
    )

    # Whether hard constraints may be relaxed in fallback attempts.
    allow_constraint_relaxation: bool = True

    def clone(self) -> "SchedulingProblem":
        """Return a deep-ish clone suitable for solver retries/relaxation."""
        return SchedulingProblem(
            tasks=dict(self.tasks),
            workers=dict(self.workers),
            dependencies={k: list(v) for k, v in self.dependencies.items()},
            affinity=dict(self.affinity),
            hard_affinity_tasks=set(self.hard_affinity_tasks),
            dependency_mode=self.dependency_mode,
            anti_affinity={k: list(v) for k, v in self.anti_affinity.items()},
            latency_penalty=dict(self.latency_penalty),
            task_durations_s=dict(self.task_durations_s),
            objective_weights=dict(self.objective_weights),
            allow_constraint_relaxation=self.allow_constraint_relaxation,
        )


@dataclass
class SchedulingSolution:
    """Solution to a scheduling problem."""

    # Assignment: task_id -> worker_id
    assignments: Dict[str, str] = field(default_factory=dict)

    # Objective value (lower is better)
    objective_value: float = float("inf")

    # Solution quality metrics
    is_optimal: bool = False
    is_feasible: bool = False
    solver_type: SolverType = SolverType.HEURISTIC
    solve_time_s: float = 0.0

    # Resource utilization per worker: worker -> (mem_util, compute_util)
    worker_utilization: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Diagnostics for failures/relaxation decisions
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ILPSolver:
    """ILP solver with PuLP/CBC backend and production diagnostics."""

    def __init__(self, time_limit_s: float = 60.0):
        self.time_limit_s = time_limit_s
        self._has_solver = self._check_solver_available()

    def _check_solver_available(self) -> bool:
        """Check whether PuLP is available."""
        try:
            import pulp  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Solve the scheduling problem using ILP with heuristic warm start."""
        if not problem.tasks:
            return SchedulingSolution(
                assignments={},
                objective_value=0.0,
                is_optimal=True,
                is_feasible=True,
                solver_type=SolverType.ILP,
                solve_time_s=0.0,
                worker_utilization={w: (0.0, 0.0) for w in problem.workers},
                diagnostics={"message": "No tasks to schedule."},
            )

        if not self._has_solver:
            sol = self._solve_approximation(problem)
            sol.diagnostics["solver_backend"] = "pulp_missing"
            return sol

        try:
            return self._solve_with_pulp(problem, allow_relaxation=problem.allow_constraint_relaxation)
        except ImportError:
            sol = self._solve_approximation(problem)
            sol.diagnostics["solver_backend"] = "import_error"
            return sol

    def _iter_anti_affinity_pairs(self, problem: SchedulingProblem) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for task_id, blocked in problem.anti_affinity.items():
            for other in blocked:
                if task_id in problem.tasks and other in problem.tasks and task_id != other:
                    a, b = sorted((task_id, other))
                    pairs.add((a, b))
        return pairs

    def _is_hard_affinity(self, problem: SchedulingProblem, task_id: str) -> bool:
        return task_id in problem.affinity and task_id in problem.hard_affinity_tasks

    def _duration(self, problem: SchedulingProblem, task_id: str) -> float:
        return max(0.0, problem.task_durations_s.get(task_id, 1.0))

    def _solve_with_pulp(
        self,
        problem: SchedulingProblem,
        allow_relaxation: bool,
    ) -> SchedulingSolution:
        """Solve using PuLP/CBC with weighted objective and extra constraints."""
        import pulp

        start_time = time.time()
        tasks = list(problem.tasks.keys())
        workers = list(problem.workers.keys())

        model = pulp.LpProblem("Scheduling", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "x",
            [(i, j) for i in tasks for j in workers],
            cat=pulp.LpBinary,
        )

        # Optional variables for advanced terms.
        u_max = pulp.LpVariable("u_max", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)

        soft_affinity_violation = {}
        for task_id, pref_worker in problem.affinity.items():
            if task_id in problem.tasks and pref_worker in problem.workers and not self._is_hard_affinity(problem, task_id):
                soft_affinity_violation[task_id] = pulp.LpVariable(
                    f"soft_aff_violate_{task_id}", lowBound=0.0, upBound=1.0, cat=pulp.LpBinary
                )

        start_vars = {}
        c_max = None
        if problem.dependency_mode == "precedence":
            start_vars = {
                task_id: pulp.LpVariable(f"start_{task_id}", lowBound=0.0, cat=pulp.LpContinuous)
                for task_id in tasks
            }
            c_max = pulp.LpVariable("c_max", lowBound=0.0, cat=pulp.LpContinuous)

        # Assignment constraints.
        for task_id in tasks:
            model += pulp.lpSum([x[(task_id, worker_id)] for worker_id in workers]) == 1

        # Resource capacity constraints.
        for worker_id in workers:
            model += (
                pulp.lpSum([problem.tasks[task_id][0] * x[(task_id, worker_id)] for task_id in tasks])
                <= problem.workers[worker_id][0]
            )
            model += (
                pulp.lpSum([problem.tasks[task_id][1] * x[(task_id, worker_id)] for task_id in tasks])
                <= problem.workers[worker_id][1]
            )

        # Load balancing term: bound max utilization across workers.
        for worker_id in workers:
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            if mem_cap > 0:
                model += (
                    pulp.lpSum([(problem.tasks[t][0] / mem_cap) * x[(t, worker_id)] for t in tasks]) <= u_max
                )
            if compute_cap > 0:
                model += (
                    pulp.lpSum([(problem.tasks[t][1] / compute_cap) * x[(t, worker_id)] for t in tasks]) <= u_max
                )

        # Affinity constraints.
        for task_id, pref_worker in problem.affinity.items():
            if task_id not in tasks or pref_worker not in workers:
                continue

            if self._is_hard_affinity(problem, task_id):
                model += x[(task_id, pref_worker)] == 1
            elif task_id in soft_affinity_violation:
                # violation >= 1 - x(task, preferred)
                model += soft_affinity_violation[task_id] >= 1 - x[(task_id, pref_worker)]

        # Dependency constraints.
        if problem.dependency_mode == "colocate":
            for task_id, deps in problem.dependencies.items():
                if task_id not in problem.tasks:
                    continue
                for dep in deps:
                    if dep not in problem.tasks:
                        continue
                    for worker_id in workers:
                        model += x[(task_id, worker_id)] == x[(dep, worker_id)]

        elif problem.dependency_mode == "precedence" and start_vars:
            for task_id, deps in problem.dependencies.items():
                if task_id not in start_vars:
                    continue
                for dep in deps:
                    if dep in start_vars:
                        model += start_vars[task_id] >= start_vars[dep] + self._duration(problem, dep)

            if c_max is not None:
                for task_id in tasks:
                    model += c_max >= start_vars[task_id] + self._duration(problem, task_id)

        # Anti-affinity constraints.
        for task_a, task_b in self._iter_anti_affinity_pairs(problem):
            for worker_id in workers:
                model += x[(task_a, worker_id)] + x[(task_b, worker_id)] <= 1

        # Weighted multi-objective.
        alpha = problem.objective_weights.get("resource", 1.0)
        beta = problem.objective_weights.get("imbalance", 10.0)
        gamma = problem.objective_weights.get("latency", 0.0)
        delta = problem.objective_weights.get("soft_affinity", 5.0)
        epsilon = problem.objective_weights.get("time", 0.0)

        resource_cost = pulp.lpSum(
            [
                problem.workers[w][2] * (problem.tasks[t][0] + problem.tasks[t][1]) * x[(t, w)]
                for t in tasks
                for w in workers
            ]
        )

        latency_cost = pulp.lpSum(
            [problem.latency_penalty.get((t, w), 0.0) * x[(t, w)] for t in tasks for w in workers]
        )

        affinity_penalty = pulp.lpSum([var for var in soft_affinity_violation.values()]) if soft_affinity_violation else 0

        time_cost = c_max if c_max is not None else 0

        model += alpha * resource_cost + beta * u_max + gamma * latency_cost + delta * affinity_penalty + epsilon * time_cost

        # Warm start from greedy heuristic.
        warm = self._solve_approximation(problem).assignments
        for task_id in tasks:
            for worker_id in workers:
                x[(task_id, worker_id)].setInitialValue(1 if warm.get(task_id) == worker_id else 0)

        solver = None
        try:
            solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit_s, msg=0, warmStart=True)
        except TypeError:
            solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit_s, msg=0)

        model.solve(solver)
        solve_time = time.time() - start_time
        status_name = pulp.LpStatus.get(model.status, "Unknown")

        solution = SchedulingSolution(
            solver_type=SolverType.ILP,
            solve_time_s=solve_time,
            diagnostics={
                "status": status_name,
                "warm_start_used": True,
                "objective_weights": dict(problem.objective_weights),
            },
        )

        if model.status == pulp.LpStatusOptimal:
            solution.is_optimal = True
            solution.is_feasible = True
        elif model.status in (pulp.LpStatusNotSolved, pulp.LpStatusUndefined):
            solution.is_feasible = False
        elif model.status == pulp.LpStatusInfeasible:
            solution.is_feasible = False
        else:
            # Feasible but not proven optimal.
            solution.is_feasible = True

        if solution.is_feasible:
            if model.objective is not None:
                solution.objective_value = float(pulp.value(model.objective))

            for task_id in tasks:
                for worker_id in workers:
                    val = pulp.value(x[(task_id, worker_id)])
                    if val is not None and val > 0.5:
                        solution.assignments[task_id] = worker_id
                        break

            solution.worker_utilization = self._calculate_utilization(problem, solution.assignments)
            return solution

        # Infeasible handling + diagnostics.
        solution.diagnostics.update(self._diagnose_infeasibility(problem))

        if allow_relaxation and problem.allow_constraint_relaxation:
            relaxed = problem.clone()
            relaxed.hard_affinity_tasks = set()
            relaxed.dependency_mode = "none" if problem.dependency_mode == "colocate" else problem.dependency_mode
            relaxed.allow_constraint_relaxation = False
            retry = self._solve_with_pulp(relaxed, allow_relaxation=False)
            retry.diagnostics["relaxed_from_original"] = True
            retry.diagnostics["original_status"] = status_name
            if retry.is_feasible:
                return retry

        # Final fallback to approximation if ILP cannot produce feasible result.
        fallback = self._solve_approximation(problem)
        fallback.diagnostics["fallback_reason"] = "ilp_infeasible_or_unsolved"
        fallback.diagnostics["ilp_status"] = status_name
        fallback.diagnostics.update(solution.diagnostics)
        return fallback

    def _diagnose_infeasibility(self, problem: SchedulingProblem) -> Dict[str, Any]:
        """Best-effort diagnostics for infeasible models."""
        details: Dict[str, Any] = {}

        total_mem_demand = sum(v[0] for v in problem.tasks.values())
        total_compute_demand = sum(v[1] for v in problem.tasks.values())
        total_mem_cap = sum(v[0] for v in problem.workers.values())
        total_compute_cap = sum(v[1] for v in problem.workers.values())

        details["total_mem_demand"] = total_mem_demand
        details["total_mem_capacity"] = total_mem_cap
        details["total_compute_demand"] = total_compute_demand
        details["total_compute_capacity"] = total_compute_cap

        oversized_tasks = []
        max_mem = max((w[0] for w in problem.workers.values()), default=0.0)
        max_compute = max((w[1] for w in problem.workers.values()), default=0.0)
        for task_id, (mem_req, compute_req, _) in problem.tasks.items():
            if mem_req > max_mem or compute_req > max_compute:
                oversized_tasks.append(task_id)
        if oversized_tasks:
            details["oversized_tasks"] = oversized_tasks

        bad_hard_affinity = []
        for task_id in problem.hard_affinity_tasks:
            worker_id = problem.affinity.get(task_id)
            if worker_id not in problem.workers:
                bad_hard_affinity.append(task_id)
                continue
            mem_req, compute_req, _ = problem.tasks.get(task_id, (0.0, 0.0, 0))
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            if mem_req > mem_cap or compute_req > compute_cap:
                bad_hard_affinity.append(task_id)
        if bad_hard_affinity:
            details["invalid_hard_affinity_tasks"] = bad_hard_affinity

        details["anti_affinity_pairs"] = len(self._iter_anti_affinity_pairs(problem))
        return details

    def _solve_approximation(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Fast approximation fallback."""
        heuristic = HeuristicScheduler(algorithm="greedy", time_limit_s=max(1.0, self.time_limit_s / 6.0))
        solution = heuristic.solve(problem)
        solution.solver_type = SolverType.HEURISTIC
        solution.diagnostics.setdefault("approximation", True)
        return solution

    def _calculate_utilization(
        self,
        problem: SchedulingProblem,
        assignments: Dict[str, str],
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
                used_mem / mem_cap if mem_cap > 0 else 0.0,
                used_compute / compute_cap if compute_cap > 0 else 0.0,
            )

        return utilization


class HeuristicScheduler:
    """Heuristic scheduler for large-scale systems."""

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
        if not problem.tasks:
            return SchedulingSolution(
                assignments={},
                objective_value=0.0,
                is_optimal=False,
                is_feasible=True,
                solver_type=SolverType.HEURISTIC,
                solve_time_s=0.0,
                worker_utilization={w: (0.0, 0.0) for w in problem.workers},
            )

        if self.algorithm == "greedy":
            return self._solve_greedy(problem)
        if self.algorithm == "genetic":
            return self._solve_genetic(problem)
        if self.algorithm == "annealing":
            return self._solve_annealing(problem)
        return self._solve_greedy(problem)

    def _solve_greedy(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Greedy weighted assignment with constraint-aware scoring."""
        start_time = time.time()

        sorted_tasks = sorted(
            problem.tasks.items(),
            key=lambda x: (-x[1][2], -(x[1][0] + x[1][1])),
        )

        assignments: Dict[str, str] = {}
        worker_usage = {w: [0.0, 0.0] for w in problem.workers}

        for task_id, (mem_req, compute_req, _) in sorted_tasks:
            pref = problem.affinity.get(task_id)
            hard_aff = task_id in problem.hard_affinity_tasks and pref in problem.workers
            candidate_workers = [pref] if hard_aff else list(problem.workers.keys())

            best_worker = None
            best_score = float("inf")

            for worker_id in candidate_workers:
                if worker_id not in problem.workers:
                    continue

                mem_cap, compute_cap, cost = problem.workers[worker_id]
                used_mem, used_compute = worker_usage[worker_id]

                if used_mem + mem_req > mem_cap or used_compute + compute_req > compute_cap:
                    continue

                # Respect anti-affinity with already assigned tasks.
                conflict = False
                for other in problem.anti_affinity.get(task_id, []):
                    if assignments.get(other) == worker_id:
                        conflict = True
                        break
                if conflict:
                    continue

                # Dependency co-location preference/constraint.
                if problem.dependency_mode == "colocate":
                    dep_conflict = False
                    for dep in problem.dependencies.get(task_id, []):
                        if dep in assignments and assignments[dep] != worker_id:
                            dep_conflict = True
                            break
                    if dep_conflict:
                        continue

                imbalance = 0.0
                if mem_cap > 0 and compute_cap > 0:
                    new_mem = (used_mem + mem_req) / mem_cap
                    new_compute = (used_compute + compute_req) / compute_cap
                    imbalance = max(new_mem, new_compute)

                latency = problem.latency_penalty.get((task_id, worker_id), 0.0)
                affinity_pen = 0.0
                if pref and worker_id != pref:
                    affinity_pen = 1.0

                alpha = problem.objective_weights.get("resource", 1.0)
                beta = problem.objective_weights.get("imbalance", 10.0)
                gamma = problem.objective_weights.get("latency", 0.0)
                delta = problem.objective_weights.get("soft_affinity", 5.0)

                score = alpha * cost * (mem_req + compute_req) + beta * imbalance + gamma * latency + delta * affinity_pen
                if score < best_score:
                    best_score = score
                    best_worker = worker_id

            if best_worker is not None:
                assignments[task_id] = best_worker
                worker_usage[best_worker][0] += mem_req
                worker_usage[best_worker][1] += compute_req

        assignments = self._repair(problem, assignments)
        return self._create_solution(problem, assignments, start_time)

    def _solve_genetic(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Genetic algorithm for scheduling."""
        start_time = time.time()
        tasks = list(problem.tasks.keys())
        workers = list(problem.workers.keys())

        if not tasks or not workers:
            return SchedulingSolution(is_feasible=False)

        population = []
        for _ in range(self.population_size):
            individual = self._create_random_assignment(problem, tasks, workers)
            fitness = self._calculate_fitness(problem, individual)
            population.append((individual, fitness))

        for _ in range(self.max_iterations):
            if time.time() - start_time > self.time_limit_s:
                break

            population.sort(key=lambda x: x[1])
            survivors = population[: max(2, self.population_size // 2)]

            offspring = []
            while len(offspring) < max(1, self.population_size // 2):
                parent1 = random.choice(survivors)[0]
                parent2 = random.choice(survivors)[0]

                child = self._crossover(parent1, parent2, tasks)
                child = self._mutate(child, workers, mutation_rate=0.1)
                child = self._repair(problem, child)

                fitness = self._calculate_fitness(problem, child)
                offspring.append((child, fitness))

            population = survivors + offspring

        population.sort(key=lambda x: x[1])
        best_assignment = population[0][0]
        best_assignment = self._repair(problem, best_assignment)
        return self._create_solution(problem, best_assignment, start_time)

    def _solve_annealing(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Simulated annealing for scheduling."""
        start_time = time.time()
        tasks = list(problem.tasks.keys())
        workers = list(problem.workers.keys())

        current = self._create_random_assignment(problem, tasks, workers)
        current = self._repair(problem, current)
        current_cost = self._calculate_fitness(problem, current)

        best = dict(current)
        best_cost = current_cost

        temperature = 100.0
        cooling_rate = 0.99
        min_temperature = 0.1

        while temperature > min_temperature:
            if time.time() - start_time > self.time_limit_s:
                break

            neighbor = dict(current)
            task_to_move = random.choice(tasks)
            neighbor[task_to_move] = random.choice(workers)
            neighbor = self._repair(problem, neighbor)
            neighbor_cost = self._calculate_fitness(problem, neighbor)

            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-9)):
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
        workers: List[str],
    ) -> Dict[str, str]:
        """Create random assignment that tries to respect capacity."""
        assignment: Dict[str, str] = {}
        worker_usage = {w: [0.0, 0.0] for w in workers}

        shuffled_tasks = list(tasks)
        random.shuffle(shuffled_tasks)

        for task_id in shuffled_tasks:
            mem_req, compute_req, _ = problem.tasks[task_id]
            pref = problem.affinity.get(task_id)
            hard_aff = task_id in problem.hard_affinity_tasks and pref in problem.workers

            choices = [pref] if hard_aff else list(workers)
            random.shuffle(choices)

            assigned = False
            for worker_id in choices:
                if worker_id not in problem.workers:
                    continue
                mem_cap, compute_cap, _ = problem.workers[worker_id]
                used_mem, used_compute = worker_usage[worker_id]

                if used_mem + mem_req <= mem_cap and used_compute + compute_req <= compute_cap:
                    assignment[task_id] = worker_id
                    worker_usage[worker_id][0] += mem_req
                    worker_usage[worker_id][1] += compute_req
                    assigned = True
                    break

            if not assigned:
                assignment[task_id] = workers[0]

        return assignment

    def _crossover(self, parent1: Dict[str, str], parent2: Dict[str, str], tasks: List[str]) -> Dict[str, str]:
        """Single-point crossover."""
        crossover_point = random.randint(0, len(tasks)) if tasks else 0
        child = {}
        for i, task_id in enumerate(tasks):
            child[task_id] = parent1.get(task_id) if i < crossover_point else parent2.get(task_id)
            if child[task_id] is None:
                child[task_id] = parent2.get(task_id, parent1.get(task_id))
        return child

    def _mutate(
        self,
        assignment: Dict[str, str],
        workers: List[str],
        mutation_rate: float = 0.1,
    ) -> Dict[str, str]:
        """Mutate assignment."""
        mutated = dict(assignment)
        for task_id in mutated:
            if random.random() < mutation_rate:
                mutated[task_id] = random.choice(workers)
        return mutated

    def _repair(self, problem: SchedulingProblem, assignment: Dict[str, str]) -> Dict[str, str]:
        """Repair assignment to improve feasibility."""
        repaired = dict(assignment)
        workers = list(problem.workers.keys())

        # Enforce hard affinity first.
        for task_id in problem.hard_affinity_tasks:
            pref = problem.affinity.get(task_id)
            if task_id in problem.tasks and pref in problem.workers:
                repaired[task_id] = pref

        # Enforce colocate dependencies if requested.
        if problem.dependency_mode == "colocate":
            for task_id, deps in problem.dependencies.items():
                if task_id not in problem.tasks:
                    continue
                for dep in deps:
                    if dep in repaired and dep in problem.tasks:
                        repaired[task_id] = repaired[dep]
                        break

        # Resolve anti-affinity conflicts by moving lower-priority task.
        changed = True
        while changed:
            changed = False
            for task_id, blocked in problem.anti_affinity.items():
                for other in blocked:
                    if task_id in repaired and other in repaired and repaired[task_id] == repaired[other]:
                        # Move lower-priority task if possible.
                        p_task = problem.tasks.get(task_id, (0, 0, 0))[2]
                        p_other = problem.tasks.get(other, (0, 0, 0))[2]
                        move_task = other if p_task >= p_other else task_id
                        current_worker = repaired[move_task]

                        for cand in workers:
                            if cand == current_worker:
                                continue
                            repaired[move_task] = cand
                            if self._is_feasible_fast(problem, repaired):
                                changed = True
                                break
                        else:
                            repaired[move_task] = current_worker

        # Capacity repair loop.
        for _ in range(max(1, len(problem.tasks) * 2)):
            if self._is_feasible_fast(problem, repaired):
                break

            over_worker = self._find_overloaded_worker(problem, repaired)
            if over_worker is None:
                break

            tasks_on_worker = [t for t, w in repaired.items() if w == over_worker]
            if not tasks_on_worker:
                break

            # Move smallest non-hard-affinity task first.
            movable = [t for t in tasks_on_worker if t not in problem.hard_affinity_tasks]
            if not movable:
                break

            task_to_move = min(movable, key=lambda t: problem.tasks[t][0] + problem.tasks[t][1])
            moved = False
            for cand in workers:
                if cand == over_worker:
                    continue
                repaired[task_to_move] = cand
                if self._is_feasible_fast(problem, repaired):
                    moved = True
                    break
            if not moved:
                repaired[task_to_move] = over_worker
                break

        return repaired

    def _find_overloaded_worker(self, problem: SchedulingProblem, assignment: Dict[str, str]) -> Optional[str]:
        usage = {w: [0.0, 0.0] for w in problem.workers}
        for task_id, worker_id in assignment.items():
            if task_id in problem.tasks and worker_id in usage:
                usage[worker_id][0] += problem.tasks[task_id][0]
                usage[worker_id][1] += problem.tasks[task_id][1]

        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            if used_mem > mem_cap or used_compute > compute_cap:
                return worker_id
        return None

    def _is_feasible_fast(self, problem: SchedulingProblem, assignment: Dict[str, str]) -> bool:
        if len(assignment) < len(problem.tasks):
            return False

        usage = {w: [0.0, 0.0] for w in problem.workers}
        for task_id, worker_id in assignment.items():
            if task_id not in problem.tasks or worker_id not in problem.workers:
                return False

            # Hard affinity.
            if task_id in problem.hard_affinity_tasks and problem.affinity.get(task_id) != worker_id:
                return False

            usage[worker_id][0] += problem.tasks[task_id][0]
            usage[worker_id][1] += problem.tasks[task_id][1]

        # Capacity.
        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            if used_mem > mem_cap or used_compute > compute_cap:
                return False

        # Dependency colocate.
        if problem.dependency_mode == "colocate":
            for task_id, deps in problem.dependencies.items():
                if task_id not in assignment:
                    continue
                for dep in deps:
                    if dep in assignment and assignment[dep] != assignment[task_id]:
                        return False

        # Anti-affinity.
        for task_id, blocked in problem.anti_affinity.items():
            for other in blocked:
                if task_id in assignment and other in assignment and assignment[task_id] == assignment[other]:
                    return False

        return True

    def _calculate_fitness(self, problem: SchedulingProblem, assignment: Dict[str, str]) -> float:
        """Weighted fitness (lower is better)."""
        alpha = problem.objective_weights.get("resource", 1.0)
        beta = problem.objective_weights.get("imbalance", 10.0)
        gamma = problem.objective_weights.get("latency", 0.0)
        delta = problem.objective_weights.get("soft_affinity", 5.0)

        resource_cost = 0.0
        latency_cost = 0.0
        affinity_penalty = 0.0
        penalty = 0.0
        usage = {w: [0.0, 0.0] for w in problem.workers}

        for task_id, worker_id in assignment.items():
            if task_id not in problem.tasks or worker_id not in problem.workers:
                penalty += 1000.0
                continue

            mem_req, compute_req, _ = problem.tasks[task_id]
            _, _, cost_factor = problem.workers[worker_id]

            resource_cost += cost_factor * (mem_req + compute_req)
            latency_cost += problem.latency_penalty.get((task_id, worker_id), 0.0)
            if task_id in problem.affinity and worker_id != problem.affinity[task_id]:
                affinity_penalty += 1.0

            usage[worker_id][0] += mem_req
            usage[worker_id][1] += compute_req

        # Capacity violations.
        max_util = 0.0
        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            mem_util = used_mem / mem_cap if mem_cap > 0 else 1.0
            compute_util = used_compute / compute_cap if compute_cap > 0 else 1.0
            max_util = max(max_util, mem_util, compute_util)

            if used_mem > mem_cap:
                penalty += (used_mem - mem_cap) * 100.0
            if used_compute > compute_cap:
                penalty += (used_compute - compute_cap) * 100.0

        # Dependency/anti-affinity violations.
        if problem.dependency_mode == "colocate":
            for task_id, deps in problem.dependencies.items():
                for dep in deps:
                    if task_id in assignment and dep in assignment and assignment[task_id] != assignment[dep]:
                        penalty += 250.0

        for task_id, blocked in problem.anti_affinity.items():
            for other in blocked:
                if task_id in assignment and other in assignment and assignment[task_id] == assignment[other]:
                    penalty += 250.0

        imbalance_cost = max_util
        return alpha * resource_cost + beta * imbalance_cost + gamma * latency_cost + delta * affinity_penalty + penalty

    def _create_solution(
        self,
        problem: SchedulingProblem,
        assignments: Dict[str, str],
        start_time: float,
    ) -> SchedulingSolution:
        """Create solution object from assignments."""
        solve_time = time.time() - start_time
        objective = self._calculate_fitness(problem, assignments)
        is_feasible = self._is_feasible_fast(problem, assignments)

        usage = {w: [0.0, 0.0] for w in problem.workers}
        for task_id, worker_id in assignments.items():
            if task_id in problem.tasks and worker_id in usage:
                usage[worker_id][0] += problem.tasks[task_id][0]
                usage[worker_id][1] += problem.tasks[task_id][1]

        utilization = {}
        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            utilization[worker_id] = (
                used_mem / mem_cap if mem_cap > 0 else 0.0,
                used_compute / compute_cap if compute_cap > 0 else 0.0,
            )

        return SchedulingSolution(
            assignments=assignments,
            objective_value=objective,
            is_optimal=False,
            is_feasible=is_feasible,
            solver_type=SolverType.HEURISTIC,
            solve_time_s=solve_time,
            worker_utilization=utilization,
            diagnostics={"algorithm": self.algorithm},
        )


class AdaptiveScheduler:
    """Adaptive scheduler with ILP/heuristic hybrid mode for scalability."""

    def __init__(
        self,
        ilp_threshold: int = 50,
        ilp_time_limit_s: float = 60.0,
        heuristic_time_limit_s: float = 10.0,
        enable_hybrid: bool = True,
        hybrid_priority_cutoff: int = 4,
        hybrid_ratio: float = 0.35,
    ):
        self.ilp_threshold = ilp_threshold
        self.enable_hybrid = enable_hybrid
        self.hybrid_priority_cutoff = hybrid_priority_cutoff
        self.hybrid_ratio = max(0.05, min(0.95, hybrid_ratio))

        self.ilp_solver = ILPSolver(time_limit_s=ilp_time_limit_s)
        self.heuristic_solver = HeuristicScheduler(
            algorithm="genetic",
            time_limit_s=heuristic_time_limit_s,
        )

        self._metrics_history: List[Dict[str, Any]] = []
        self._metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_metrics_hook(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for per-solve metrics."""
        self._metrics_hook = callback

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Return recent scheduler metrics."""
        return list(self._metrics_history)

    def solve(self, problem: SchedulingProblem) -> SchedulingSolution:
        """Solve scheduling problem with adaptive/hybrid strategy."""
        problem_size = len(problem.tasks) * len(problem.workers)

        if problem_size <= self.ilp_threshold:
            solution = self.ilp_solver.solve(problem)
            if not solution.is_feasible:
                solution = self.heuristic_solver.solve(problem)
        else:
            if self.enable_hybrid and len(problem.tasks) >= 8 and len(problem.workers) >= 2:
                solution = self._solve_hybrid(problem)
            else:
                solution = self.heuristic_solver.solve(problem)

        self._record_metrics(problem, solution)
        return solution

    def _solve_hybrid(self, problem: SchedulingProblem) -> SchedulingSolution:
        """
        Hybrid strategy:
        - ILP on high-priority subset (+ dependency closure)
        - Heuristic on the remaining tasks with residual capacities
        """
        tasks_sorted = sorted(
            problem.tasks.items(),
            key=lambda item: (-item[1][2], -(item[1][0] + item[1][1])),
        )

        candidate = [t for t, (_, _, prio) in tasks_sorted if prio >= self.hybrid_priority_cutoff]
        if not candidate:
            target_n = max(1, int(len(problem.tasks) * self.hybrid_ratio))
            candidate = [t for t, _ in tasks_sorted[:target_n]]

        target_n = max(1, int(len(problem.tasks) * self.hybrid_ratio))
        high_set: Set[str] = set(candidate[:target_n])

        # Expand by dependency closure so colocation/precedence relations remain meaningful.
        stack = list(high_set)
        while stack:
            task_id = stack.pop()
            for dep in problem.dependencies.get(task_id, []):
                if dep in problem.tasks and dep not in high_set:
                    high_set.add(dep)
                    stack.append(dep)

        if len(high_set) >= len(problem.tasks):
            return self.ilp_solver.solve(problem)

        low_set = [t for t in problem.tasks if t not in high_set]

        high_problem = self._sub_problem(problem, list(high_set), problem.workers)
        high_solution = self.ilp_solver.solve(high_problem)
        if not high_solution.is_feasible:
            high_solution = HeuristicScheduler(algorithm="greedy", time_limit_s=self.heuristic_solver.time_limit_s).solve(high_problem)

        residual_workers = self._residual_workers(problem, high_solution.assignments)
        if not residual_workers:
            return self._merge_solutions(problem, high_solution, SchedulingSolution(assignments={}), "hybrid_no_residual")

        low_problem = self._sub_problem(problem, low_set, residual_workers)

        # If low task depends on high-assigned task and colocate mode is active,
        # force low task to follow the dependency worker.
        if low_problem.dependency_mode == "colocate":
            for task_id in low_set:
                for dep in problem.dependencies.get(task_id, []):
                    if dep in high_solution.assignments:
                        low_problem.affinity[task_id] = high_solution.assignments[dep]
                        low_problem.hard_affinity_tasks.add(task_id)
                        break

        low_solution = self.heuristic_solver.solve(low_problem)

        merged = self._merge_solutions(problem, high_solution, low_solution, "hybrid")
        return merged

    def _sub_problem(
        self,
        problem: SchedulingProblem,
        selected_tasks: List[str],
        workers: Dict[str, Tuple[float, float, float]],
    ) -> SchedulingProblem:
        """Create filtered problem retaining only selected tasks and relevant constraints."""
        selected = set(selected_tasks)

        sub = SchedulingProblem(
            tasks={t: problem.tasks[t] for t in selected_tasks if t in problem.tasks},
            workers=dict(workers),
            dependencies={
                t: [d for d in problem.dependencies.get(t, []) if d in selected]
                for t in selected_tasks
                if t in problem.tasks
            },
            affinity={t: w for t, w in problem.affinity.items() if t in selected_tasks},
            hard_affinity_tasks={t for t in problem.hard_affinity_tasks if t in selected_tasks},
            dependency_mode=problem.dependency_mode,
            anti_affinity={
                t: [o for o in problem.anti_affinity.get(t, []) if o in selected]
                for t in selected_tasks
                if t in problem.tasks
            },
            latency_penalty={
                (t, w): v
                for (t, w), v in problem.latency_penalty.items()
                if t in selected_tasks and w in workers
            },
            task_durations_s={t: d for t, d in problem.task_durations_s.items() if t in selected_tasks},
            objective_weights=dict(problem.objective_weights),
            allow_constraint_relaxation=problem.allow_constraint_relaxation,
        )

        return sub

    def _residual_workers(
        self,
        problem: SchedulingProblem,
        assignments: Dict[str, str],
    ) -> Dict[str, Tuple[float, float, float]]:
        """Compute remaining worker capacities after fixed assignments."""
        usage = {w: [0.0, 0.0] for w in problem.workers}
        for task_id, worker_id in assignments.items():
            if task_id in problem.tasks and worker_id in usage:
                usage[worker_id][0] += problem.tasks[task_id][0]
                usage[worker_id][1] += problem.tasks[task_id][1]

        residual = {}
        for worker_id, (mem_cap, compute_cap, cost_factor) in problem.workers.items():
            rem_mem = max(0.0, mem_cap - usage[worker_id][0])
            rem_compute = max(0.0, compute_cap - usage[worker_id][1])
            if rem_mem > 0 and rem_compute > 0:
                residual[worker_id] = (rem_mem, rem_compute, cost_factor)

        return residual

    def _merge_solutions(
        self,
        problem: SchedulingProblem,
        high: SchedulingSolution,
        low: SchedulingSolution,
        mode: str,
    ) -> SchedulingSolution:
        """Merge two partial solutions into one full solution."""
        assignments = dict(high.assignments)
        assignments.update(low.assignments)

        is_feasible = len(assignments) == len(problem.tasks)
        usage = {w: [0.0, 0.0] for w in problem.workers}

        for task_id, worker_id in assignments.items():
            if task_id not in problem.tasks or worker_id not in problem.workers:
                is_feasible = False
                continue
            usage[worker_id][0] += problem.tasks[task_id][0]
            usage[worker_id][1] += problem.tasks[task_id][1]

        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            if used_mem > mem_cap or used_compute > compute_cap:
                is_feasible = False
                break

        worker_utilization = {}
        for worker_id, (used_mem, used_compute) in usage.items():
            mem_cap, compute_cap, _ = problem.workers[worker_id]
            worker_utilization[worker_id] = (
                used_mem / mem_cap if mem_cap > 0 else 0.0,
                used_compute / compute_cap if compute_cap > 0 else 0.0,
            )

        objective_value = high.objective_value + low.objective_value
        return SchedulingSolution(
            assignments=assignments,
            objective_value=objective_value,
            is_optimal=False,
            is_feasible=is_feasible,
            solver_type=SolverType.AUTO,
            solve_time_s=high.solve_time_s + low.solve_time_s,
            worker_utilization=worker_utilization,
            diagnostics={
                "mode": mode,
                "high_solver": high.solver_type.value,
                "low_solver": low.solver_type.value,
                "high_feasible": high.is_feasible,
                "low_feasible": low.is_feasible,
            },
        )

    def _record_metrics(self, problem: SchedulingProblem, solution: SchedulingSolution) -> None:
        """Store monitoring metrics for feedback loops and offline analysis."""
        max_util = 0.0
        avg_mem = 0.0
        avg_compute = 0.0
        if solution.worker_utilization:
            mem_vals = [v[0] for v in solution.worker_utilization.values()]
            compute_vals = [v[1] for v in solution.worker_utilization.values()]
            max_util = max(mem_vals + compute_vals)
            avg_mem = sum(mem_vals) / len(mem_vals)
            avg_compute = sum(compute_vals) / len(compute_vals)

        metric = {
            "timestamp": time.time(),
            "tasks": len(problem.tasks),
            "workers": len(problem.workers),
            "solver_type": solution.solver_type.value,
            "is_feasible": solution.is_feasible,
            "is_optimal": solution.is_optimal,
            "solve_time_s": solution.solve_time_s,
            "objective": solution.objective_value,
            "max_utilization": max_util,
            "avg_mem_utilization": avg_mem,
            "avg_compute_utilization": avg_compute,
        }

        self._metrics_history.append(metric)
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]

        if self._metrics_hook:
            self._metrics_hook(metric)


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
