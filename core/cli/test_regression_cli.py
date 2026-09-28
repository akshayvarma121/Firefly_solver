"""
core/cli/test_regression_cli.py
--------------------------------
Regression test: run `firefly solve` against every file in
core/tests/regression/ and verify exit codes + solver outcomes.

Run:
    python core/cli/test_regression_cli.py

Pass/fail is determined by:
  - The actual values returned by the live firefly_solver pyd (truth).
  - The AGENTS.md rule: never fabricate results.

The test is the "backend done" gate: nothing past this is trusted until
all cases with EXPECT_SOLVE=True pass cleanly.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# CUDA DLL setup (must come before importing firefly_solver)
# ---------------------------------------------------------------------------
if os.name == "nt":
    _cuda = os.environ.get("CUDA_PATH", "")
    if _cuda:
        for _sub in ("bin", r"bin\x64"):
            _p = os.path.join(_cuda, _sub)
            if os.path.isdir(_p):
                os.add_dll_directory(_p)

# ---------------------------------------------------------------------------
# Load the solver module so we can call solve() directly for reference
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "core"))

try:
    import firefly_solver as _fs  # type: ignore
    _FS_AVAILABLE = True
except Exception as _e:
    _fs = None
    _FS_AVAILABLE = False
    print(f"[WARNING] firefly_solver not loaded: {_e}", file=sys.stderr)

_REG_DIR = os.path.join(_REPO_ROOT, "core", "tests", "regression")

# ---------------------------------------------------------------------------
# Exit code conventions (must match core/cli/firefly.py)
# ---------------------------------------------------------------------------
_EC = {
    "OPTIMAL":    0,
    "FEASIBLE":   0,
    "INFEASIBLE": 1,
    "UNBOUNDED":  2,
    "TIME_LIMIT": 3,
    "NODE_LIMIT": 3,
    "ERROR":      4,
}

# ---------------------------------------------------------------------------
# Test case spec
# ---------------------------------------------------------------------------

@dataclass
class Case:
    filename: str
    expect_error_msg: Optional[str] = None   # Substring to expect in Error output if it should fail
    expect_status: Optional[str] = None
    expect_obj: Optional[float] = None
    obj_tol: float = 1e-4              # absolute tolerance on objective
    method: str = "simplex"            # simplex is deterministic and exact
    note: str = ""


CASES: list[Case] = [
    Case("all_bounds.mps",
         expect_status="UNBOUNDED",
         note="X4=FR/X5=MI free vars, minimise sum -> unbounded"),

    Case("ill_conditioned.mps",
         expect_status="OPTIMAL",
         expect_obj=1e-12,
         obj_tol=1e-6,
         note="Near-zero optimum; coefficient matrix spans 12 orders of magnitude"),

    Case("infeasible.mps",
         expect_status="INFEASIBLE",
         note="X1<=0 AND X1>=1 makes it infeasible"),

    # ----- malformed files: the parser must reject these immediately
    #       and throw an MPSParseException with the exact line number.
    Case("malformed.mps",
         expect_error_msg="Line 7: Missing ENDATA",
         note="Missing ENDATA — throws MPSParseException"),

    Case("malformed_empty.mps",
         expect_error_msg="Line 0: Empty file",
         note="Only comments, no sections — throws MPSParseException"),

    Case("malformed_undeclared_row.mps",
         expect_error_msg="Line 5: Reference to undeclared row: BAD_ROW",
         note="COLUMNS references BAD_ROW not in ROWS — throws MPSParseException"),

    Case("malformed_unknown_section.mps",
         expect_error_msg="Line 4: Unknown section header: FAKE_SECTION",
         note="Unknown section header — throws MPSParseException"),

    Case("marker_int.mps",
         expect_status="OPTIMAL",
         expect_obj=-10.0,
         obj_tol=1e-5,
         method="milp",
         note="MILP: X1 integer, C1: X1<=10, obj: -X1 -> optimal X1=10, obj=-10"),

    Case("maximize.mps",
         expect_status="OPTIMAL",
         expect_obj=10.0,
         obj_tol=1e-5,
         note="OBJSENSE MAXIMIZE X1=10 -> internalized as minimise -X1 -> obj=+10"),

    Case("ranges.mps",
         expect_status="INFEASIBLE",
         note="RANGES constraints produce conflicting bounds -> INFEASIBLE under simplex"),

    Case("unbounded.mps",
         expect_status="UNBOUNDED",
         note="min -X1 with X1 unbounded above -> UNBOUNDED"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

def _run_cli(filepath: str, method: str) -> tuple[int, str, str]:
    """
    Invoke `python core/cli/firefly.py solve <file> --method <method>
             --no-gpu --debug` as a subprocess.
    Returns (returncode, stdout, stderr).
    """
    env = os.environ.copy()
    # The solver should now be installed via pip in the environment.

    cmd = [
        sys.executable, "-m", "cli.firefly",
        "solve", filepath,
        "--method", method,
        "--no-gpu",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=_REPO_ROOT,
    )
    return result.returncode, result.stdout, result.stderr

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

def run_all() -> tuple[int, int]:
    passed = failed = 0
    sep = "-" * 75

    print(f"\n{'FIREFLY CLI REGRESSION TEST':^75}")
    print(f"Regression folder: {_REG_DIR}")
    print(f"Solver available : {_FS_AVAILABLE}")
    print(sep)

    for case in CASES:
        filepath = os.path.join(_REG_DIR, case.filename)
        if not os.path.isfile(filepath):
            print(f"{WARN} {case.filename}: FILE NOT FOUND — skipped")
            continue

        try:
            rc, stdout, stderr = _run_cli(filepath, case.method)
        except subprocess.TimeoutExpired:
            print(f"{FAIL} {case.filename}: CLI timed out after 120s")
            failed += 1
            continue
        except Exception as exc:
            print(f"{FAIL} {case.filename}: subprocess error: {exc}")
            failed += 1
            continue

        errors: list[str] = []

        if case.expect_error_msg:
            # We expect an error output from the CLI
            if rc == 0:
                errors.append(f"expected nonzero exit with error, got 0")
            
            # Look for the exact error message in stdout (printed by firefly.py)
            err_line = None
            for line in stdout.splitlines():
                if "Error" in line and ":" in line:
                    err_line = line.split("Error     :", 1)[-1].strip() if "Error     :" in line else line.split(":", 1)[-1].strip()
                    break
            
            if not err_line:
                errors.append(f"did not find any 'Error' line in output")
            elif case.expect_error_msg not in err_line:
                errors.append(f"error message: expected to contain {case.expect_error_msg!r}, got {err_line!r}")
        else:
            # Check exit code matches expected status
            expected_rc = _EC.get(case.expect_status or "ERROR", 4)
            if rc != expected_rc:
                errors.append(
                    f"exit code: expected {expected_rc} ({case.expect_status}), got {rc}"
                )

            # Parse objective from stdout
            if case.expect_obj is not None:
                obj_line = None
                for line in stdout.splitlines():
                    if "Objective" in line:
                        obj_line = line
                        break
                if obj_line is None:
                    errors.append(f"no 'Objective' line in stdout")
                else:
                    try:
                        # "  Objective : <value>"
                        raw_val = obj_line.split(":")[-1]
                        clean_val = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw_val).strip()
                        obj_val = float(clean_val)
                        if not math.isclose(obj_val, case.expect_obj,
                                            abs_tol=case.obj_tol,
                                            rel_tol=1e-5):
                            errors.append(
                                f"objective: expected {case.expect_obj:.8g}, "
                                f"got {obj_val:.8g} (tol={case.obj_tol})"
                            )
                    except ValueError as e:
                        errors.append(f"cannot parse objective from: {obj_line!r}: {e}")

        if errors:
            print(f"{FAIL} {case.filename}")
            for err in errors:
                print(f"       ERR: {err}")
            if stderr.strip():
                print(f"       STDERR: {stderr.strip()[:300]}")
            failed += 1
        else:
            note_str = f"  ({case.note})" if case.note else ""
            status_str = "error->nonzero" if case.expect_error_msg else case.expect_status
            print(f"{PASS} {case.filename:<40s} [{status_str}]{note_str}")
            passed += 1

    print(sep)
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} cases")
    print(f"[EVIDENCE] CLI Regression Suite: {passed} passed, {failed} failed out of {passed+failed} cases")
    print()
    return passed, failed


if __name__ == "__main__":
    p, f = run_all()
    sys.exit(0 if f == 0 else 1)
