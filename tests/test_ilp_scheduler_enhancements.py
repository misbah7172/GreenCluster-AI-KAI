import pytest

from model.ilp_scheduler import AdaptiveScheduler, ILPSolver, SchedulingProblem, SolverType


@pytest.fixture
def small_problem() -> SchedulingProblem:
    problem = SchedulingProblem()
    problem.tasks = {
        "t1": (2.0, 10.0, 5),
        "t2": (2.0, 10.0, 4),
    }
    problem.workers = {
        "w1": (8.0, 100.0, 1.0),
        "w2": (8.0, 100.0, 1.0),
    }
    return problem


def test_ilp_enforces_anti_affinity(small_problem: SchedulingProblem):
    pytest.importorskip("pulp")

    small_problem.anti_affinity = {"t1": ["t2"], "t2": ["t1"]}
    solver = ILPSolver(time_limit_s=10)
    solution = solver.solve(small_problem)

    assert solution.is_feasible
    assert solution.assignments["t1"] != solution.assignments["t2"]


def test_ilp_enforces_hard_affinity(small_problem: SchedulingProblem):
    pytest.importorskip("pulp")

    small_problem.affinity = {"t1": "w2"}
    small_problem.hard_affinity_tasks = {"t1"}

    solver = ILPSolver(time_limit_s=10)
    solution = solver.solve(small_problem)

    assert solution.is_feasible
    assert solution.assignments["t1"] == "w2"


def test_ilp_enforces_dependency_colocation(small_problem: SchedulingProblem):
    pytest.importorskip("pulp")

    small_problem.dependency_mode = "colocate"
    small_problem.dependencies = {"t2": ["t1"]}

    solver = ILPSolver(time_limit_s=10)
    solution = solver.solve(small_problem)

    assert solution.is_feasible
    assert solution.assignments["t1"] == solution.assignments["t2"]


def test_ilp_reports_infeasibility_diagnostics():
    pytest.importorskip("pulp")

    problem = SchedulingProblem(
        tasks={"big": (100.0, 10.0, 1)},
        workers={"w1": (8.0, 100.0, 1.0)},
    )

    solver = ILPSolver(time_limit_s=5)
    solution = solver.solve(problem)

    assert not solution.is_feasible
    assert "oversized_tasks" in solution.diagnostics
    assert "big" in solution.diagnostics["oversized_tasks"]


def test_adaptive_hybrid_mode_large_problem():
    problem = SchedulingProblem()
    for i in range(20):
        # Mix priorities to trigger high-priority ILP subset selection.
        priority = 5 if i < 6 else 2
        problem.tasks[f"t{i}"] = (1.0, 5.0, priority)

    problem.workers = {
        "w1": (20.0, 200.0, 1.0),
        "w2": (20.0, 200.0, 1.0),
        "w3": (20.0, 200.0, 1.1),
    }

    scheduler = AdaptiveScheduler(
        ilp_threshold=10,
        enable_hybrid=True,
        hybrid_priority_cutoff=4,
        hybrid_ratio=0.4,
    )

    solution = scheduler.solve(problem)

    assert len(solution.assignments) == len(problem.tasks)
    assert solution.is_feasible
    assert solution.solver_type in (SolverType.AUTO, SolverType.ILP, SolverType.HEURISTIC)

    history = scheduler.get_metrics_history()
    assert history
    assert history[-1]["tasks"] == 20
