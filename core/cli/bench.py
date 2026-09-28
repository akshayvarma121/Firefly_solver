"""
core/cli/bench.py
-----------------
Shared benchmark and batch-solve logic consumed by:
  - firefly benchmark <folder>  (CLI)
  - firefly solve-batch <folder>  (CLI)
  - api/main.py  /benchmark endpoint  (FastAPI)

Import from here; never duplicate the loop logic.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    """Single-problem solve outcome, serialisable to JSON / tabular display."""
    problem: str                          # filename (basename)
    path: str                             # absolute path
    status: str                           # OPTIMAL / INFEASIBLE / ERROR / ...
    solver_used: Optional[str] = None     # Which engine produced the result
    objective: Optional[float] = None
    reference: Optional[float] = None     # known-good value if provided
    reference_time_ms: float = 0.0        # time taken by HiGHS
    difference: Optional[float] = None    # abs(objective - reference)
    passed: Optional[bool] = None         # difference < tol
    wall_time_ms: float = 0.0
    iterations: int = 0
    error_message: Optional[str] = None
    mock: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BatchSummary:
    """Aggregate result for a folder run."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: List[ProblemResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.errors == 0 and self.failed == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_mps_files(folder: str) -> List[str]:
    """Return sorted list of .mps file paths inside *folder* (non-recursive)."""
    try:
        names = sorted(os.listdir(folder))
    except OSError as e:
        raise FileNotFoundError(f"Cannot list folder {folder!r}: {e}") from e
    return [os.path.join(folder, n) for n in names if n.lower().endswith(".mps")]


def _run_highspy(filepath: str) -> tuple[float, float]:
    import highspy
    import time
    
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(filepath)
    
    t0 = time.perf_counter()
    h.run()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    
    info = h.getInfo()
    return info.objective_function_value, wall_ms

def solve_one(
    filepath: str,
    *,
    method: str = "auto",
    gpu: bool = True,
    tol: float = 1e-5,
    iteration_callback: Optional[Callable] = None,
    firefly_solver=None,
    raise_errors: bool = False,
) -> ProblemResult:
    """
    Parse and solve a single MPS file.

    Parameters
    ----------
    filepath        : absolute path to the .mps file
    method          : solver method string ("auto"|"simplex"|"pdlp"|"milp")
    gpu             : attempt GPU acceleration
    reference       : known-good objective for pass/fail comparison
    tol             : tolerance for pass/fail check
    iteration_callback : forwarded to firefly_solver.solve()
    firefly_solver  : the already-imported firefly_solver module (or None for
                      mock fallback).  Callers pass the module they imported
                      so this module never imports it directly.
    raise_errors    : if True, do not suppress exceptions
    """
    basename = os.path.basename(filepath)
    result = ProblemResult(problem=basename, path=filepath, status="ERROR")
    
    try:
        ref_obj, ref_time = _run_highspy(filepath)
    except Exception as exc:
        if raise_errors:
            raise
        result.status = "ERROR"
        result.error_message = f"HiGHS reference solve failed: {exc}"
        return result

    if firefly_solver is None:
        result.status = "OPTIMAL"
        result.objective = -99.99
        result.wall_time_ms = 0.0
        result.iterations = 0
        result.mock = True
        diff = abs(result.objective - ref_obj)
        result.reference = ref_obj
        result.reference_time_ms = ref_time
        result.difference = diff
        result.passed = diff < tol
        return result

    try:
        t0 = time.perf_counter()
        prob = firefly_solver.parse_mps(filepath)
        kwargs: dict = dict(method=method, gpu=gpu)
        if iteration_callback is not None:
            kwargs["iteration_callback"] = iteration_callback
        res = firefly_solver.solve(prob, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result.status = res.status
        if hasattr(res, "solver_used"):
            result.solver_used = res.solver_used
        result.objective = res.objective
        result.wall_time_ms = res.wall_time_ms if res.wall_time_ms else elapsed_ms
        result.iterations = res.iterations
        result.mock = False

        diff = abs(res.objective - ref_obj)
        result.reference = ref_obj
        result.reference_time_ms = ref_time
        result.difference = diff
        result.passed = diff < tol

    except Exception as exc:
        if raise_errors:
            raise
        result.status = "ERROR"
        result.error_message = str(exc)

    return result


def run_batch(
    folder: str,
    *,
    method: str = "auto",
    gpu: bool = True,
    tol: float = 1e-5,
    on_result: Optional[Callable[[ProblemResult], None]] = None,
    firefly_solver=None,
) -> BatchSummary:
    """
    Solve every .mps file in *folder* and return a BatchSummary.

    Parameters
    ----------
    folder      : directory containing .mps files
    references  : map from basename -> known-good objective value
    on_result   : optional callback called after each file finishes
    firefly_solver : injected module handle (see solve_one)
    """
    files = collect_mps_files(folder)
    summary = BatchSummary(total=len(files))

    for fpath in files:
        pr = solve_one(
            fpath,
            method=method,
            gpu=gpu,
            tol=tol,
            firefly_solver=firefly_solver,
        )
        summary.results.append(pr)

        if pr.status == "ERROR":
            summary.errors += 1
        elif pr.passed is True:
            summary.passed += 1
        elif pr.passed is False:
            summary.failed += 1

        if on_result:
            on_result(pr)

    return summary


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def result_to_row(pr: ProblemResult) -> str:
    """One-line tabular representation of a ProblemResult."""
    obj_str  = f"{pr.objective:>14.6f}" if pr.objective is not None else f"{'N/A':>14}"
    ref_str  = f"{pr.reference:>14.6f}" if pr.reference  is not None else f"{'':>14}"
    diff_str = f"{pr.difference:>10.2e}" if pr.difference is not None else f"{'':>10}"
    pass_str = ("PASS" if pr.passed else "FAIL") if pr.passed is not None else "----"
    time_str = f"{pr.wall_time_ms:>8.1f} ms"
    iter_str = f"{pr.iterations:>7}"
    solver_str = f"{pr.solver_used or 'pdlp':>16}"
    mock_tag = " [MOCK]" if pr.mock else ""
    return (
        f"  {pr.problem:<35s}  {pr.status:<12s}  {solver_str}"
        f"{obj_str}  {ref_str}  {diff_str}  {pass_str}"
        f"  {time_str}  {iter_str}{mock_tag}"
    )


def summary_header() -> str:
    return (
        "  " + f"{'PROBLEM':<35s}  {'STATUS':<12s}  {'ENGINE':>16}"
        f"{'OBJECTIVE':>14}  {'REFERENCE':>14}  {'DIFF':>10}  PASS"
        f"  {'TIME':>10}  {'ITERS':>7}"
        "\n  " + "-" * 133
    )
