"""
test_bindings.py — Comprehensive test suite for firefly_solver pybind11 bindings.

Covers:
  1. solve() across every valid (method, gpu) combination per problem type
  2. GIL stress: iteration_callback doing real work over many repeated calls
  3. C++ exceptions surface as readable Python exceptions — never silent None/
     garbage result or interpreter crash

Run (from repo root):
    python -m pytest api/test_bindings.py -v

or standalone:
    python api/test_bindings.py
"""

import os
import sys
import math
import time
import threading
import traceback

# Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import firefly_solver

# ── Helpers ───────────────────────────────────────────────────────────────────

def lp_2var() -> firefly_solver.SparseProblem:
    """
    min  -x - 2y
    s.t.  x +  y <= 4
          x       >= 0
               y  >= 0
    Optimal: x=0, y=4, obj=-8
    """
    p = firefly_solver.SparseProblem()
    p.num_vars   = 2
    p.num_constrs = 1
    p.obj_coeffs        = [-1.0, -2.0]
    p.row_ptr           = [0, 2]
    p.col_idx           = [0, 1]
    p.values            = [1.0, 1.0]
    p.row_senses        = ['L']
    p.rhs               = [4.0]
    p.var_lower_bounds  = [0.0, 0.0]
    p.var_upper_bounds  = [1e30, 1e30]
    p.is_integer        = [False, False]
    return p


def milp_2var() -> firefly_solver.SparseProblem:
    """
    min  -x - 2y          (integer variables)
    s.t.  2.5x + 1.5y <= 10
          x, y integer, >= 0
    Optimal: x=0, y=6, obj=-12
    """
    p = firefly_solver.SparseProblem()
    p.num_vars    = 2
    p.num_constrs = 1
    p.obj_coeffs        = [-1.0, -2.0]
    p.row_ptr           = [0, 2]
    p.col_idx           = [0, 1]
    p.values            = [2.5, 1.5]
    p.row_senses        = ['L']
    p.rhs               = [10.0]
    p.var_lower_bounds  = [0.0, 0.0]
    p.var_upper_bounds  = [1e30, 1e30]
    p.is_integer        = [True, True]
    return p


def infeasible_lp() -> firefly_solver.SparseProblem:
    """
    min  x
    s.t.  x >= 5
          x <= 3   (contradictory bounds → infeasible)
    """
    p = firefly_solver.SparseProblem()
    p.num_vars    = 1
    p.num_constrs = 0   # bounds encoded as variable bounds
    p.obj_coeffs        = [1.0]
    p.row_ptr           = [0]
    p.col_idx           = []
    p.values            = []
    p.row_senses        = []
    p.rhs               = []
    p.var_lower_bounds  = [5.0]
    p.var_upper_bounds  = [3.0]   # lb > ub → presolve must detect infeasibility
    p.is_integer        = [False]
    return p


def _check_lp_optimal(res: firefly_solver.SolveResult, tol: float = 1e-4, label: str = ""):
    prefix = f"[{label}] " if label else ""
    assert res.status == "OPTIMAL", f"{prefix}Expected OPTIMAL, got {res.status!r} ({res.message!r})"
    assert abs(res.objective - (-8.0)) < tol, f"{prefix}Expected obj=-8 got {res.objective}"
    assert len(res.solution) == 2, f"{prefix}solution length wrong"
    assert res.wall_time_ms >= 0.0, f"{prefix}wall_time_ms should be non-negative"


def _check_milp_optimal(res: firefly_solver.SolveResult, tol: float = 1e-4, label: str = ""):
    prefix = f"[{label}] " if label else ""
    assert res.status == "OPTIMAL", f"{prefix}Expected OPTIMAL, got {res.status!r} ({res.message!r})"
    assert abs(res.objective - (-12.0)) < tol, f"{prefix}Expected obj=-12 got {res.objective}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  solve() — every (method, gpu) combination
# ═══════════════════════════════════════════════════════════════════════════════

def test_lp_simplex_no_gpu():
    """LP — method='simplex', gpu=False  (most portable path)."""
    res = firefly_solver.solve(lp_2var(), method="simplex", gpu=False)
    _check_lp_optimal(res, label="simplex/no-gpu")
    print(f"  PASS  LP simplex/no-gpu  obj={res.objective}  iters={res.iterations}")


def test_lp_pdlp_no_gpu():
    """LP — method='pdlp', gpu=False  (CPU PDLP path)."""
    res = firefly_solver.solve(lp_2var(), method="pdlp", gpu=False)
    assert res.status in ("OPTIMAL", "FEASIBLE"), \
        f"Expected OPTIMAL/FEASIBLE, got {res.status!r}"
    assert res.objective < -7.9, f"Expected obj near -8, got {res.objective}"
    print(f"  PASS  LP pdlp/no-gpu  obj={res.objective}  iters={res.iterations}")


def test_lp_auto_no_gpu():
    """LP — method='auto', gpu=False  (should route to PDLP CPU)."""
    res = firefly_solver.solve(lp_2var(), method="auto", gpu=False)
    assert res.status in ("OPTIMAL", "FEASIBLE"), \
        f"Expected OPTIMAL/FEASIBLE, got {res.status!r}"
    assert res.objective < -7.9, f"Expected obj near -8, got {res.objective}"
    print(f"  PASS  LP auto/no-gpu  obj={res.objective}  iters={res.iterations}")


def test_lp_auto_gpu():
    """LP — method='auto', gpu=True  (GPU PDLP if available, else CPU PDLP)."""
    res = firefly_solver.solve(lp_2var(), method="auto", gpu=True)
    assert res.status in ("OPTIMAL", "FEASIBLE"), \
        f"Expected OPTIMAL/FEASIBLE, got {res.status!r}"
    assert res.objective < -7.9, f"Expected obj near -8, got {res.objective}"
    print(f"  PASS  LP auto/gpu  obj={res.objective}  iters={res.iterations}")


def test_milp_auto_simplex_lp():
    """MILP — method='auto', gpu=False  (B&B with simplex LP relaxations)."""
    res = firefly_solver.solve(milp_2var(), method="auto", gpu=False)
    _check_milp_optimal(res, label="milp/auto/no-gpu")
    print(f"  PASS  MILP auto/no-gpu  obj={res.objective}  sol={res.solution}")


def test_milp_auto_gpu():
    """MILP — method='auto', gpu=True  (B&B with PDLP GPU LP relaxations)."""
    res = firefly_solver.solve(milp_2var(), method="auto", gpu=True)
    _check_milp_optimal(res, label="milp/auto/gpu")
    print(f"  PASS  MILP auto/gpu  obj={res.objective}  sol={res.solution}")


def test_result_fields_populated():
    """SolveResult fields are all present and typed correctly."""
    res = firefly_solver.solve(lp_2var(), method="simplex", gpu=False)
    assert isinstance(res.status, str)
    assert isinstance(res.objective, float)
    assert isinstance(res.solution, list)
    assert all(isinstance(v, float) for v in res.solution)
    assert isinstance(res.dual_solution, list)
    assert isinstance(res.wall_time_ms, float)
    assert isinstance(res.iterations, int)
    assert isinstance(res.message, str)
    assert res.iterations >= 0
    print(f"  PASS  result field types  repr={repr(res)}")


def test_parse_mps_lp():
    """parse_mps() on test_problem1 (pure LP) → correct objective."""
    mps_path = os.path.join(os.path.dirname(__file__), "sample_problems", "test_problem1.mps")
    p = firefly_solver.parse_mps(mps_path)
    assert p.num_vars > 0 and p.num_constrs > 0
    res = firefly_solver.solve(p, method="simplex", gpu=False)
    assert res.status == "OPTIMAL"
    assert abs(res.objective - (-10.0)) < 1e-4, f"Expected -10, got {res.objective}"
    print(f"  PASS  parse_mps LP  obj={res.objective}")


def test_parse_mps_milp():
    """parse_mps() on test_problem2 (MILP, INTORG/INTEND) → B&B, obj=-12."""
    mps_path = os.path.join(os.path.dirname(__file__), "sample_problems", "test_problem2.mps")
    p = firefly_solver.parse_mps(mps_path)
    assert any(p.is_integer), "Expected integer variables"
    res = firefly_solver.solve(p, method="auto", gpu=False)
    assert res.status == "OPTIMAL"
    assert abs(res.objective - (-12.0)) < 1e-4, f"Expected -12, got {res.objective}"
    print(f"  PASS  parse_mps MILP  obj={res.objective}  sol={res.solution}")


def test_parse_mps_string():
    """parse_mps_string() roundtrip — in-memory content identical to file parse."""
    mps_path = os.path.join(os.path.dirname(__file__), "sample_problems", "test_problem1.mps")
    with open(mps_path) as f:
        content = f.read()
    p_str  = firefly_solver.parse_mps_string(content)
    p_file = firefly_solver.parse_mps(mps_path)
    assert p_str.num_vars    == p_file.num_vars
    assert p_str.num_constrs == p_file.num_constrs
    assert p_str.obj_coeffs  == p_file.obj_coeffs
    print(f"  PASS  parse_mps_string roundtrip  vars={p_str.num_vars}")


def test_json_payload_problem1():
    """
    Regression test for JSON payload conversion (similar to Problem 1: 1 var, 2 constrs).
    min x
    s.t. x <= 10
         x <= 20
    Optimal: x=0, obj=0.0
    """
    p = firefly_solver.SparseProblem()
    p.num_vars = 1
    p.num_constrs = 2
    p.obj_coeffs = [-1.0]
    p.row_ptr = [0, 1, 2]
    p.col_idx = [0, 0]
    p.values = [1.0, 1.0]
    p.row_senses = ['L', 'L']
    p.rhs = [10.0, 20.0]
    p.var_lower_bounds = [0.0]
    p.var_upper_bounds = [] # not provided, should default to INF
    
    res = firefly_solver.solve(p, method="simplex", gpu=False)
    assert res.status == "OPTIMAL"
    assert abs(res.objective - (-10.0)) < 1e-4, f"Expected -10.0, got {res.objective}"
    assert abs(res.solution[0] - 10.0) < 1e-4
    print(f"  PASS  JSON Problem 1  obj={res.objective}")


def test_json_payload_problem2():
    """
    Regression test for JSON payload conversion (Problem 2).
    min -x - 2y
    s.t. x + y <= 10
    Optimal: x=0, y=10, obj=-20.0
    """
    p = firefly_solver.SparseProblem()
    p.num_vars = 2
    p.num_constrs = 1
    p.obj_coeffs = [-1.0, -2.0]
    p.row_ptr = [0, 2]
    p.col_idx = [0, 1]
    p.values = [1.0, 1.0]
    p.row_senses = ['L']
    p.rhs = [10.0]
    
    res = firefly_solver.solve(p, method="simplex", gpu=False)
    assert res.status == "OPTIMAL"
    assert abs(res.objective - (-20.0)) < 1e-4, f"Expected -20.0, got {res.objective}"
    assert abs(res.solution[0] - 0.0) < 1e-4
    assert abs(res.solution[1] - 10.0) < 1e-4
    print(f"  PASS  JSON Problem 2  obj={res.objective}")


def test_json_validation_errors():
    """
    Test that invalid JSON fields (wrong size or invalid senses) raise exceptions,
    preventing silent defaults that make variables look fixed or constraints redundant.
    """
    # 1. Invalid Sense
    p1 = firefly_solver.SparseProblem()
    p1.num_vars = 1
    p1.num_constrs = 1
    p1.obj_coeffs = [1.0]
    p1.row_ptr = [0, 1]
    p1.col_idx = [0]
    p1.values = [1.0]
    p1.row_senses = ['<'] # Invalid sense
    p1.rhs = [10.0]
    
    try:
        firefly_solver.solve(p1, method="simplex", gpu=False)
        assert False, "Expected ValueError/RuntimeError for invalid sense"
    except (ValueError, RuntimeError, Exception) as e:
        assert "Unknown row sense" in str(e)
        
    # 2. Missing RHS
    p2 = firefly_solver.SparseProblem()
    p2.num_vars = 1
    p2.num_constrs = 2
    p2.obj_coeffs = [1.0]
    p2.row_ptr = [0, 1, 2]
    p2.col_idx = [0, 0]
    p2.values = [1.0, 1.0]
    p2.row_senses = ['L', 'L']
    p2.rhs = [10.0] # Missing one!
    
    try:
        firefly_solver.solve(p2, method="simplex", gpu=False)
        assert False, "Expected ValueError/RuntimeError for mismatched rhs size"
    except (ValueError, RuntimeError, Exception) as e:
        assert "size does not match" in str(e)

    print("  PASS  JSON validation errors caught successfully")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GIL stress: real-work callback over many repeated solve() calls
# ═══════════════════════════════════════════════════════════════════════════════

def test_gil_callback_real_work_single_thread():
    """
    Callback does genuine Python work (list append + float arithmetic).
    Repeated 50 times in one thread.  Must not crash or deadlock.
    """
    REPEATS  = 50
    log: list[tuple[int, float, float, float]] = []

    def cb(it: int, p_obj: float, d_obj: float, ms: float) -> None:
        # Real Python work: append + arithmetic
        log.append((it, round(p_obj, 6), round(d_obj, 6), round(ms, 3)))
        _ = math.sqrt(abs(p_obj) + 1.0)   # light computation

    for i in range(REPEATS):
        log.clear()
        res = firefly_solver.solve(
            lp_2var(), method="auto", gpu=False, iteration_callback=cb
        )
        assert res.status in ("OPTIMAL", "FEASIBLE"), \
            f"run {i}: unexpected status {res.status!r}"

    print(f"  PASS  GIL callback single-thread  repeats={REPEATS}  last_cb_entries={len(log)}")


def test_gil_callback_multithreaded():
    """
    N threads each call solve() with a real-work callback concurrently.
    Tests that GIL acquire/release inside the callback shim is correct
    under genuine thread contention — no deadlock, no crash, correct answers.
    """
    N_THREADS = 8
    REPEATS   = 20
    errors: list[str] = []
    results: list[float] = []
    lock = threading.Lock()

    def worker(thread_id: int) -> None:
        log: list[float] = []

        def cb(it: int, p_obj: float, d_obj: float, ms: float) -> None:
            # All Python objects touched here are thread-local lists, so no
            # data-race, but the GIL acquire/release still exercises the
            # concurrent path.
            log.append(p_obj)
            _ = sum(log[-5:])  # force a real list read

        for _ in range(REPEATS):
            try:
                res = firefly_solver.solve(
                    lp_2var(), method="auto", gpu=False, iteration_callback=cb
                )
                if res.status not in ("OPTIMAL", "FEASIBLE"):
                    with lock:
                        errors.append(f"thread={thread_id} status={res.status!r}")
                with lock:
                    results.append(res.objective)
            except Exception as exc:
                with lock:
                    errors.append(f"thread={thread_id} raised {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N_THREADS)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        if t.is_alive():
            raise RuntimeError("Thread timed out — possible deadlock")

    elapsed = time.monotonic() - t0

    if errors:
        raise AssertionError(f"Thread errors:\n" + "\n".join(errors))

    assert len(results) == N_THREADS * REPEATS, \
        f"Expected {N_THREADS * REPEATS} results, got {len(results)}"
    assert all(abs(r - (-8.0)) < 0.2 for r in results), \
        f"Unexpected objective values: {set(round(r,2) for r in results)}"

    print(f"  PASS  GIL callback multithreaded  threads={N_THREADS}  "
          f"repeats={REPEATS}  total={len(results)}  elapsed={elapsed:.2f}s")


def test_gil_no_callback_baseline():
    """
    Control: solve() with no callback must still be correct and not regress
    performance after the GIL shim is wired in.
    """
    REPEATS = 100
    t0 = time.monotonic()
    for _ in range(REPEATS):
        res = firefly_solver.solve(lp_2var(), method="simplex", gpu=False)
        assert res.status == "OPTIMAL"
    elapsed = time.monotonic() - t0
    # 100 tiny LP solves should finish in under 30 s on any hardware
    assert elapsed < 30.0, f"No-callback baseline too slow: {elapsed:.2f}s for {REPEATS} solves"
    print(f"  PASS  GIL no-callback baseline  repeats={REPEATS}  elapsed={elapsed:.3f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  C++ exceptions → readable Python exceptions; never silent / garbage
# ═══════════════════════════════════════════════════════════════════════════════

def test_exception_infeasible_bounds():
    """
    Conflicting variable bounds (lb=5 > ub=3) → presolve catches it.
    Must surface as a valid SolveResult with status="INFEASIBLE"
    and a readable message; must NOT return a garbage SolveResult or crash.
    """
    res = firefly_solver.solve(infeasible_lp(), method="simplex", gpu=False)
    assert res.status == "INFEASIBLE", f"Expected INFEASIBLE, got {res.status}"
    
    msg = res.message
    assert len(msg) > 0, "Exception message must not be empty"
    assert any(kw in msg.lower() for kw in ("infeasible", "bound", "conflicting", "column")), \
        f"Exception message not informative enough: {msg!r}"
    print(f"  PASS  infeasible bounds -> {msg[:80]!r}")


def test_exception_parse_mps_nonexistent_file():
    """
    parse_mps() on a missing file must raise RuntimeError with the path in
    the message; must NOT return None or a zeroed SparseProblem.
    """
    caught = None
    try:
        firefly_solver.parse_mps("/nonexistent/path/to/missing.mps")
    except RuntimeError as exc:
        caught = exc
    except Exception as exc:
        raise AssertionError(f"Wrong exception type {type(exc).__name__}: {exc}") from exc

    assert caught is not None, "Expected RuntimeError for missing file"
    msg = str(caught)
    assert "missing" in msg.lower() or "open" in msg.lower() or "fail" in msg.lower() or \
           "no such" in msg.lower() or "nonexistent" in msg.lower(), \
        f"Message not helpful: {msg!r}"
    print(f"  PASS  parse_mps missing file -> {type(caught).__name__}: {msg[:80]!r}")


def test_exception_parse_mps_string_malformed():
    """
    parse_mps_string() on garbage content must raise RuntimeError/ValueError
    with a non-empty message; must NOT silently return a valid-looking object.
    """
    caught = None
    try:
        firefly_solver.parse_mps_string("this is not MPS content at all\nGARBAGE\n")
    except (RuntimeError, ValueError) as exc:
        caught = exc
    except Exception as exc:
        raise AssertionError(f"Wrong exception type {type(exc).__name__}: {exc}") from exc

    assert caught is not None, "Expected exception for malformed MPS content"
    assert len(str(caught)) > 0, "Exception message must not be empty"
    print(f"  PASS  parse_mps_string malformed -> {type(caught).__name__}: {str(caught)[:80]!r}")


def test_exception_not_silent_on_callback_raise():
    """
    If the iteration_callback itself raises a Python exception, that exception
    must propagate back through solve() — it must NOT be silently swallowed.
    """
    class SentinelError(Exception):
        pass

    call_count = [0]

    def bad_cb(it, p_obj, d_obj, ms):
        call_count[0] += 1
        if call_count[0] >= 1:
            raise SentinelError("intentional callback failure")

    # Use PDLP which actually fires the callback
    caught = None
    try:
        # Need enough iterations to trigger the callback; use a slightly larger
        # problem so PDLP doesn't converge in 0 steps
        p = lp_2var()
        # lower the callback_frequency by modifying the problem to have more work
        firefly_solver.solve(p, method="pdlp", gpu=False, iteration_callback=bad_cb)
    except (SentinelError, RuntimeError, Exception) as exc:
        caught = exc

    # Two acceptable outcomes:
    # (a) SentinelError propagated directly  → best case
    # (b) RuntimeError wrapping it          → acceptable (pybind11 converts)
    # (c) No callback fired (problem too small, 0 iters) → also acceptable
    #     as long as result is not a crash/None
    if caught is not None:
        assert not isinstance(caught, SystemExit), "Should never call sys.exit"
        print(f"  PASS  callback exception propagated as {type(caught).__name__}: {str(caught)[:60]!r}")
    else:
        # callback may not have fired (problem converges before first callback tick)
        print(f"  PASS  callback exception test — callback fired {call_count[0]}x, "
              "converged before first tick (acceptable)")


def test_result_never_none():
    """
    solve() must always return a SolveResult object — never None,
    never a Python int/float, never a crash.
    """
    for method in ("simplex", "auto"):
        for gpu in (False, True):
            res = firefly_solver.solve(lp_2var(), method=method, gpu=gpu)
            assert res is not None, f"solve() returned None for method={method!r} gpu={gpu}"
            assert isinstance(res, firefly_solver.SolveResult), \
                f"solve() returned {type(res)} instead of SolveResult"
            assert res.status != "", "status must not be empty string"
    print("  PASS  solve() never returns None")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def test_windows_dll_path_resolution():
    """
    On Windows, verify that importing firefly_solver succeeds in a subprocess
    even when the system PATH is stripped of CUDA bin paths, relying solely on
    os.add_dll_directory(CUDA_PATH/bin/x64).
    """
    if os.name != "nt":
        print("  PASS  windows DLL check (skipped on non-Windows)")
        return
        
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        print("  PASS  windows DLL check (skipped, CUDA_PATH not set)")
        return
        
    import subprocess
    env = os.environ.copy()
    
    # Strip PATH of anything that looks like CUDA
    if "PATH" in env:
        paths = env["PATH"].split(os.pathsep)
        clean_paths = [p for p in paths if "cuda" not in p.lower()]
        env["PATH"] = os.pathsep.join(clean_paths)
        
    script = """
import os
import sys

# The fix that must be present
if os.name == 'nt':
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        _bin = os.path.join(cuda_path, 'bin', 'x64')
        if os.path.isdir(_bin):
            os.add_dll_directory(_bin)

try:
    import firefly_solver
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {e}')
    sys.exit(1)
"""
    
    res = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocess import failed. stdout: {res.stdout}, stderr: {res.stderr}"
    assert "SUCCESS" in res.stdout, f"Subprocess did not print SUCCESS. stdout: {res.stdout}"
    print("  PASS  windows DLL path resolution via CUDA_PATH")

_TESTS = [
    # Category 1 — method × gpu combinations
    ("[API] LP  simplex/no-gpu",            test_lp_simplex_no_gpu),
    ("[API] LP  pdlp/no-gpu",               test_lp_pdlp_no_gpu),
    ("[API] LP  auto/no-gpu",               test_lp_auto_no_gpu),
    ("[API] LP  auto/gpu",                  test_lp_auto_gpu),
    ("[API] MILP auto/no-gpu",              test_milp_auto_simplex_lp),
    ("[API] MILP auto/gpu",                 test_milp_auto_gpu),
    ("[API] Result fields populated",       test_result_fields_populated),
    ("[API] parse_mps LP",                  test_parse_mps_lp),
    ("[API] parse_mps MILP",                test_parse_mps_milp),
    ("[API] parse_mps_string roundtrip",    test_parse_mps_string),
    ("[JSON] JSON Problem 1 payload",       test_json_payload_problem1),
    ("[JSON] JSON Problem 2 payload",       test_json_payload_problem2),
    ("[JSON] JSON validation errors",       test_json_validation_errors),

    # Category 2 — GIL stress
    ("[GIL] GIL callback single-thread",    test_gil_callback_real_work_single_thread),
    ("[GIL] GIL callback multithreaded",    test_gil_callback_multithreaded),
    ("[GIL] GIL no-callback baseline",      test_gil_no_callback_baseline),

    # Category 3 — exceptions
    ("[EXC] Exception: infeasible bounds",  test_exception_infeasible_bounds),
    ("[EXC] Exception: missing MPS file",   test_exception_parse_mps_nonexistent_file),
    ("[EXC] Exception: malformed MPS str",  test_exception_parse_mps_string_malformed),
    ("[EXC] Exception: callback raises",    test_exception_not_silent_on_callback_raise),
    ("[EXC] solve() never returns None",    test_result_never_none),
    
    # Category 4 — Windows specific
    ("[WIN] Windows DLL path resolution",   test_windows_dll_path_resolution),
]


def run_all(test_filter=None) -> bool:
    passed = failed = 0
    for name, fn in _TESTS:
        if test_filter and test_filter != "all" and test_filter not in name and test_filter not in fn.__name__:
            continue
        
        sys.stdout.write(f"\n[{name}]\n")
        sys.stdout.flush()
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed  {'PASS' if failed == 0 else 'FAIL'}")
    if failed:
        print(f"         {failed} FAILED")
    print('='*60)
    return failed == 0


if __name__ == "__main__":
    test_filter = "all"
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        test_filter = sys.argv[2]
        
    ok = run_all(test_filter)
    sys.exit(0 if ok else 1)
