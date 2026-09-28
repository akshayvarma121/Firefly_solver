"""
core/cli/firefly.py
-------------------
Firefly command-line interface.

Usage
-----
  firefly solve <file.mps> [--method auto|simplex|pdlp|milp]
                            [--gpu | --no-gpu]
                            [--output <path>]
                            [--quiet | --verbose | --debug]

  firefly benchmark  <folder>   -- solve all .mps + compare against references
  firefly solve-batch <folder>  -- solve all .mps, no reference comparison

Exit codes
----------
  0  OPTIMAL or FEASIBLE
  1  INFEASIBLE
  2  UNBOUNDED
  3  TIME_LIMIT or NODE_LIMIT
  4  ERROR / parse failure / unexpected exception
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from io import StringIO
from typing import Optional

# ---------------------------------------------------------------------------
# TrueColor ANSI Theme (Firefly Brand)
# ---------------------------------------------------------------------------
import ctypes
if os.name == 'nt':
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and "--no-color" not in sys.argv

class Theme:
    PRIMARY = "\x1b[38;2;232;230;222m" if _USE_COLOR else ""
    MUTED   = "\x1b[38;2;140;139;128m" if _USE_COLOR else ""
    ACCENT  = "\x1b[38;2;232;163;61m" if _USE_COLOR else ""
    SECOND  = "\x1b[38;2;107;143;113m" if _USE_COLOR else ""
    ERROR   = "\x1b[31m" if _USE_COLOR else ""
    RESET   = "\x1b[0m" if _USE_COLOR else ""

# ---------------------------------------------------------------------------
# Windows: add CUDA DLL directories before importing the native extension.
# The old pyd (core_old) links against DLLs in %CUDA_PATH%\bin, while the
# new one needs %CUDA_PATH%\bin\x64.  Add both.
# ---------------------------------------------------------------------------
if os.name == "nt":
    _cuda = os.environ.get("CUDA_PATH", "")
    if _cuda:
        for _sub in ("bin", os.path.join("bin", "x64")):
            _p = os.path.join(_cuda, _sub)
            if os.path.isdir(_p):
                os.add_dll_directory(_p)

# ---------------------------------------------------------------------------
# Attempt to load the native extension
# ---------------------------------------------------------------------------
try:
    import firefly_solver as _fs          # type: ignore
    _FS_AVAILABLE = True
except Exception as _import_err:          # noqa: BLE001
    _fs = None                            # type: ignore
    _FS_AVAILABLE = False
    _FS_IMPORT_ERR = _import_err
else:
    _FS_IMPORT_ERR = None

# ---------------------------------------------------------------------------
# Shared bench logic
# ---------------------------------------------------------------------------
# Support both:
#   (a) installed via `pip install -e .` → `from cli.bench import …`
#   (b) run directly:  `python core/cli/firefly.py …`
#                       → inject core/ root so the absolute import resolves.
_CLI_DIR  = os.path.dirname(os.path.abspath(__file__))
_CORE_ROOT = os.path.abspath(os.path.join(_CLI_DIR, ".."))
if _CORE_ROOT not in sys.path:
    sys.path.insert(0, _CORE_ROOT)

from cli.bench import (              # type: ignore
    BatchSummary,
    ProblemResult,
    collect_mps_files,
    result_to_row,
    run_batch,
    solve_one,
    summary_header,
)

# ---------------------------------------------------------------------------
# Exit-code mapping
# ---------------------------------------------------------------------------
_EXIT_CODES: dict[str, int] = {
    "OPTIMAL":    0,
    "FEASIBLE":   0,
    "INFEASIBLE": 1,
    "UNBOUNDED":  2,
    "TIME_LIMIT": 3,
    "NODE_LIMIT": 3,
    "ERROR":      4,
}


def _exit_code_for(status: str) -> int:
    return _EXIT_CODES.get(status.upper(), 4)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_solution(path: str, result, col_names: list[str]) -> None:
    """Write the full solution vector to *path* as CSV or JSON."""
    ext = os.path.splitext(path)[1].lower()
    sol = result.solution if result.solution else []
    names = col_names if col_names else [f"x{i}" for i in range(len(sol))]

    if ext == ".json":
        payload = {
            "status":      result.status,
            "objective":   result.objective,
            "iterations":  result.iterations,
            "wall_time_ms": result.wall_time_ms,
            "solution":    {n: v for n, v in zip(names, sol)},
            "mock":        getattr(result, "mock", False),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    else:
        # Default: CSV (even for unknown extensions)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["variable", "value"])
            for n, v in zip(names, sol):
                writer.writerow([n, v])


def _print_summary(result, filepath: str, quiet: bool, verbose: bool) -> None:
    """Print the human-readable solve summary (unless --quiet)."""
    if quiet:
        # Only the objective value
        if result.objective is not None:
            print(result.objective)
        return

    mock_tag = "  [MOCK — solver not available]" if getattr(result, "mock", False) else ""
    print()
    print(f"  {Theme.MUTED}Problem   :{Theme.RESET} {Theme.PRIMARY}{os.path.basename(filepath)}{Theme.RESET}")
    print(f"  {Theme.MUTED}Status    :{Theme.RESET} {Theme.ACCENT}{result.status}{mock_tag}{Theme.RESET}")
    if result.status == "ERROR" and getattr(result, "error_message", None):
        print(f"  {Theme.MUTED}Error     :{Theme.RESET} {Theme.ERROR}{result.error_message}{Theme.RESET}")
    if result.objective is not None:
        print(f"  {Theme.MUTED}Objective :{Theme.RESET} {Theme.SECOND}{result.objective:.10g}{Theme.RESET}")
    print(f"  {Theme.MUTED}Time      :{Theme.RESET} {Theme.PRIMARY}{result.wall_time_ms:.1f} ms{Theme.RESET}")
    print(f"  {Theme.MUTED}Iterations:{Theme.RESET} {Theme.PRIMARY}{result.iterations}{Theme.RESET}")
    print()


# ---------------------------------------------------------------------------
# Sub-command: solve
# ---------------------------------------------------------------------------

def _cmd_solve(args: argparse.Namespace) -> int:
    filepath = args.file
    if not os.path.isfile(filepath):
        print(f"firefly: error: file not found: {filepath!r}", file=sys.stderr)
        return 4

    # Verbose progress callback — overwrites the same terminal line
    last_line: list[str] = [""]
    def _verbose_cb(iteration: int, primal: float, dual: float, elapsed_ms: float) -> None:
        if args.verbose:
            line = (
                f"\r  iter {iteration:>6d} | primal {primal:>14.6g} "
                f"| dual {dual:>14.6g} | {elapsed_ms/1000.0:.2f} s"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            last_line[0] = line

    cb = _verbose_cb if args.verbose else None

    try:
        result = solve_one(
            filepath,
            method=args.method,
            gpu=args.gpu,
            iteration_callback=cb,
            firefly_solver=_fs,
            raise_errors=args.debug,
        )
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"firefly: error: {exc}", file=sys.stderr)
        return 4

    if args.verbose and last_line[0]:
        # Move to a fresh line after the live progress
        print()

    _print_summary(result, filepath, quiet=args.quiet, verbose=args.verbose)

    if args.output:
        try:
            col_names: list[str] = []
            # Try to extract column names from the SparseProblem if available
            if _fs is not None:
                try:
                    prob = _fs.parse_mps(filepath)
                    # SparseProblem exposes no col_names in the Python layer;
                    # generate canonical names matching the C++ default.
                    col_names = [f"x{i}" for i in range(prob.num_vars)]
                except Exception:
                    pass
            _write_solution(args.output, result, col_names)
            if not args.quiet:
                print(f"  Solution written to: {args.output}")
        except Exception as exc:
            if args.debug:
                traceback.print_exc()
            else:
                print(f"firefly: warning: could not write output: {exc}", file=sys.stderr)

    _add_recent_solve(args.file, result.objective, result.status, result.wall_time_ms)
    return _exit_code_for(result.status)


# ---------------------------------------------------------------------------
# Sub-command: compare
# ---------------------------------------------------------------------------

def _cmd_compare(args: argparse.Namespace) -> int:
    f1 = args.file1
    f2 = args.file2
    
    if not os.path.isfile(f1):
        print(f"firefly: error: file not found: {f1!r}", file=sys.stderr)
        return 4
    if not os.path.isfile(f2):
        print(f"firefly: error: file not found: {f2!r}", file=sys.stderr)
        return 4
        
    print(f"\n{Theme.ACCENT}▶ SOLVER COMPARISON{Theme.RESET}")
    print(f"  {Theme.PRIMARY}Model A:{Theme.RESET} {os.path.basename(f1)}")
    print(f"  {Theme.PRIMARY}Model B:{Theme.RESET} {os.path.basename(f2)}\n")
    
    print(f"  Solving Model A...")
    from .firefly import _fs, _solve_mock, _FS_AVAILABLE
    
    def solve_file(f):
        if not _FS_AVAILABLE:
            return _solve_mock(f, args.method, args.gpu, False)
        else:
            return _fs.solve_mps(f, args.method, args.gpu, False)
            
    res1 = solve_file(f1)
    
    print(f"  Solving Model B...")
    res2 = solve_file(f2)
    
    print(f"\n{Theme.ACCENT}▶ RESULTS{Theme.RESET}")
    
    def fmt_obj(o): return f"{o:.6g}" if o is not None else "N/A"
    def fmt_time(t): return f"{t:.1f}ms"
    
    print(f"  {'Metric':<15} | {'Model A':<20} | {'Model B':<20}")
    print(f"  {'-'*15}-+-{'-'*20}-+-{'-'*20}")
    print(f"  {'Status':<15} | {res1.status:<20} | {res2.status:<20}")
    print(f"  {'Objective':<15} | {fmt_obj(res1.objective):<20} | {fmt_obj(res2.objective):<20}")
    print(f"  {'Solve Time':<15} | {fmt_time(res1.wall_time_ms):<20} | {fmt_time(res2.wall_time_ms):<20}")
    print()
    return _exit_code_for(res1.status) if res1.status != "OPTIMAL" else _exit_code_for(res2.status)

# ---------------------------------------------------------------------------
# Sub-command: benchmark
# ---------------------------------------------------------------------------

# Known reference values shipped with the repo.
_BUILTIN_REFERENCES: dict[str, float] = {
    "test_problem1.mps": -10.0,
    "test_problem2.mps": -12.0,
}

def _print_batch_result(pr: ProblemResult, quiet: bool) -> None:
    if not quiet:
        print(result_to_row(pr))


def _cmd_benchmark(args: argparse.Namespace) -> int:
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"firefly: error: folder not found: {folder!r}", file=sys.stderr)
        return 4

    files = collect_mps_files(folder)
    if not files:
        print(f"firefly: error: no .mps files found in {folder!r}", file=sys.stderr)
        return 4

    if not args.quiet:
        print(f"\nBenchmark: {len(files)} problem(s) in {folder!r}\n")
        print(summary_header())

    summary = run_batch(
        folder,
        method=args.method,
        gpu=args.gpu,
        references=_BUILTIN_REFERENCES,
        on_result=lambda pr: _print_batch_result(pr, args.quiet),
        firefly_solver=_fs,
    )

    if not args.quiet:
        print()
        print(f"  Total: {summary.total}  |  Passed: {summary.passed}  "
              f"|  Failed: {summary.failed}  |  Errors: {summary.errors}")
        print()

    return 0 if summary.all_passed else 4


# ---------------------------------------------------------------------------
# Sub-command: solve-batch
# ---------------------------------------------------------------------------

def _cmd_solve_batch(args: argparse.Namespace) -> int:
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"firefly: error: folder not found: {folder!r}", file=sys.stderr)
        return 4

    files = collect_mps_files(folder)
    if not files:
        print(f"firefly: error: no .mps files found in {folder!r}", file=sys.stderr)
        return 4

    if not args.quiet:
        print(f"\nSolve-batch: {len(files)} problem(s) in {folder!r}\n")
        print(summary_header())

    summary = run_batch(
        folder,
        method=args.method,
        gpu=args.gpu,
        references=None,          # no reference comparison
        on_result=lambda pr: _print_batch_result(pr, args.quiet),
        firefly_solver=_fs,
    )

    if not args.quiet:
        print()
        print(f"  Total: {summary.total}  |  Errors: {summary.errors}")
        print()

    return 0 if summary.errors == 0 else 4


# ---------------------------------------------------------------------------
# Sub-command: test-standard
# ---------------------------------------------------------------------------

def _cmd_test_standard(args: argparse.Namespace) -> int:
    import tempfile
    import urllib.request
    
    # Standard small netlib problems
    files = ["afiro.mps", "adlittle.mps", "israel.mps"]
    
    print(f"\n{Theme.ACCENT}Downloading standard test problems (Netlib)...{Theme.RESET}")
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in files:
            url = f"https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/{name}"
            out_path = os.path.join(tmpdir, name)
            print(f"  Fetching {name}...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    with open(out_path, 'wb') as f:
                        f.write(response.read())
            except Exception as e:
                print(f"{Theme.MUTED}Failed to download {name}: {e}{Theme.RESET}")
                return 4
                
        print(f"\n{Theme.ACCENT}Running Solver on Standard Problems...{Theme.RESET}\n")
        print(summary_header())
        
        netlib_refs = {
            "afiro.mps": -464.75314286,
            "adlittle.mps": 225494.96316,
            "israel.mps": -896644.82186,
        }

        summary = run_batch(
            tmpdir,
            method=args.method,
            gpu=args.gpu,
            references=netlib_refs,
            on_result=lambda pr: _print_batch_result(pr, args.quiet),
            firefly_solver=_fs,
        )
        
        print()
        print(f"  Total: {summary.total}  |  Passed: {summary.passed}  "
              f"|  Failed: {summary.failed}  |  Errors: {summary.errors}")
        if summary.all_passed:
            print(f"\n  {Theme.PRIMARY}All standard tests completed successfully!{Theme.RESET}")
        else:
            print(f"\n  {Theme.MUTED}Some tests encountered errors or missed the optimum.{Theme.RESET}")
        print()
        
    return 0 if summary.all_passed else 4

# ---------------------------------------------------------------------------
# Sub-command: update
# ---------------------------------------------------------------------------

def _cmd_update(args: argparse.Namespace) -> int:
    import subprocess
    import tempfile
    
    print(f"\n{Theme.ACCENT}Checking for updates...{Theme.RESET}")

    exe_path = os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__)
    exe_old = exe_path + ".old"
    
    print(f"\n  Downloading latest Firefly...\n")
    
    if os.name == "nt":
        release_url = "https://github.com/akshayvarma121/Firefly_solver/releases/download/v0.2.0/firefly.exe"
        ps_lines = [
            f'if (Test-Path "{exe_path}") {{',
            f'    try {{ Move-Item -Path "{exe_path}" -Destination "{exe_old}" -Force -ErrorAction SilentlyContinue }} catch {{}}',
            f'}}',
            f'$req = [System.Net.HttpWebRequest]::Create("{release_url}")',
            f'$req.UserAgent = "firefly-updater"',
            f'$req.AllowAutoRedirect = $true',
            f'try {{',
            f'    $response = $req.GetResponse()',
            f'    $totalBytes = $response.ContentLength',
            f'    $stream = $response.GetResponseStream()',
            f'    $outStream = [System.IO.File]::Create("{exe_path}")',
            f'    $buffer = New-Object byte[] 65536',
            f'    $received = 0',
            f'    $startTime = [DateTime]::Now',
            f'    while ($true) {{',
            f'        $read = $stream.Read($buffer, 0, $buffer.Length)',
            f'        if ($read -le 0) {{ break }}',
            f'        $outStream.Write($buffer, 0, $read)',
            f'        $received += $read',
            f'        $elapsed = ([DateTime]::Now - $startTime).TotalSeconds',
            f'        $speedMBs  = if ($elapsed -gt 0.1) {{ [math]::Round(($received / 1MB) / $elapsed, 1) }} else {{ 0 }}',
            f'        $recvMB    = [math]::Round($received / 1MB, 1)',
            f'        $totalMB   = [math]::Round($totalBytes / 1MB, 1)',
            f'        $pct       = [math]::Round(($received / $totalBytes) * 100, 0)',
            f'        Write-Host -NoNewline "`r  $pct% - $recvMB MB / $totalMB MB  |  $speedMBs MB/s   "',
            f'    }}',
            f'    $outStream.Close()',
            f'    $stream.Close()',
            f'    Write-Host "`n`n  [OK] Firefly updated successfully!`n" -ForegroundColor Green',
            f'}} catch {{',
            f'    Write-Host "`n`n  [FAILED] Update failed: $_`n" -ForegroundColor Red',
            f'}}',
        ]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8-sig')
        tmp.write('\n'.join(ps_lines))
        tmp.close()

        subprocess.call(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", tmp.name])
    else:
        if sys.platform == "darwin":
            release_url = "https://github.com/akshayvarma121/Firefly_solver/releases/download/v0.2.0/firefly-macos"
        else:
            release_url = "https://github.com/akshayvarma121/Firefly_solver/releases/download/v0.2.0/firefly-linux"
            
        try:
            if os.path.exists(exe_old):
                os.remove(exe_old)
            if os.path.exists(exe_path):
                os.rename(exe_path, exe_old)
        except OSError:
            pass

        bash_lines = [
            f'#!/bin/bash',
            f'echo ""',
            f'curl -L --progress-bar "{release_url}" -o "{exe_path}"',
            f'if [ $? -eq 0 ]; then',
            f'  chmod +x "{exe_path}"',
            f'  echo -e "\\n  \\033[32m[OK] Firefly updated successfully!\\033[0m\\n"',
            f'else',
            f'  echo -e "\\n  \\033[31m[FAILED] Update failed.\\033[0m\\n"',
            f'fi',
            f'rm -- "$0"',
        ]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, encoding='utf-8')
        tmp.write('\n'.join(bash_lines))
        tmp.close()
        os.chmod(tmp.name, 0o755)
        
        subprocess.call(["bash", tmp.name])
        
    return 0

# ---------------------------------------------------------------------------
# Sub-command: test
# ---------------------------------------------------------------------------

def _cmd_test(args: argparse.Namespace) -> int:
    import subprocess
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.abspath(os.path.join(cli_dir, ".."))
    root_dir = os.path.abspath(os.path.join(core_dir, ".."))
    
    if os.name == "nt":
        script_path = os.path.join(root_dir, "run_audit.bat")
        cwd = root_dir
    else:
        script_path = os.path.join(core_dir, "audit.sh")
        cwd = core_dir
    
    if not os.path.isfile(script_path):
        print(f"\n{Theme.WARNING}Developer Command Unavailable{Theme.RESET}")
        print("The 'test' command runs the C++ source-code audit suite.")
        print("Because you are running the compiled binary, the source code and C++ tests are not present.")
        print(f"\nTip: Use '{Theme.ACCENT}firefly test-standard{Theme.RESET}' to run the Netlib mathematical test suite instead!\n")
        return 4
        
    try:
        print(f"\n{Theme.ACCENT}Running Firefly full audit suite...{Theme.RESET}")
        
        if os.name == "nt":
            return subprocess.call([script_path], cwd=cwd)
        else:
            return subprocess.call(["bash", script_path], cwd=cwd)
    except Exception as exc:
        if getattr(args, "debug", False):
            import traceback
            traceback.print_exc()
        print(f"firefly: error running tests: {exc}", file=sys.stderr)
        return 4

# ---------------------------------------------------------------------------
# Utility sub-commands (Version, Recent, Audit)
# ---------------------------------------------------------------------------

def _cmd_version(args: argparse.Namespace) -> int:
    print(f"{Theme.ACCENT}Firefly Solver Engine{Theme.RESET} v0.2.0")
    print(f"{Theme.MUTED}Build: sm_89 CUDA-accelerated{Theme.RESET}")
    return 0

def _cmd_top(args: argparse.Namespace) -> int:
    import psutil
    import subprocess
    import time
    
    print(f"\n{Theme.MUTED}Starting Firefly Resource Monitor (Press Ctrl+C to stop)...{Theme.RESET}")
    time.sleep(1)
    
    def _bar(pct: float, length: int = 30) -> str:
        filled = int((pct / 100.0) * length)
        empty = length - filled
        return f"{Theme.ACCENT}{'=' * filled}{Theme.MUTED}{'-' * empty}{Theme.RESET}"
    
    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")
            _print_firefly_logo()
            print(f"\n{Theme.ACCENT}▶ LIVE RESOURCE MONITOR{Theme.RESET}\n")
            
            cpu_pct = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            
            print(f"  {Theme.PRIMARY}CPU Usage:{Theme.RESET} [{_bar(cpu_pct)}] {cpu_pct:5.1f}%")
            print(f"  {Theme.PRIMARY}RAM Usage:{Theme.RESET} [{_bar(ram.percent)}] {ram.percent:5.1f}%  ({ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB)")
            print()
            
            try:
                smi = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL
                ).strip().split('\n')
                for i, line in enumerate(smi):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 4:
                        name, util, mem_used, mem_total = parts
                        util_float = float(util)
                        print(f"  {Theme.PRIMARY}GPU {i} ({name}):{Theme.RESET}")
                        print(f"    Usage: [{_bar(util_float)}] {util_float:5.1f}%")
                        mem_pct = (float(mem_used) / float(mem_total)) * 100.0
                        print(f"    VRAM:  [{_bar(mem_pct)}] {mem_pct:5.1f}%  ({mem_used}MB / {mem_total}MB)")
            except Exception:
                if sys.platform == "darwin":
                    # Mac fallback info
                    try:
                        import platform
                        mac_cpu = platform.processor()
                        print(f"  {Theme.PRIMARY}Apple Silicon ({mac_cpu}):{Theme.RESET} [OK] Metal framework available (CPU-mode compiled)")
                    except:
                        print(f"  {Theme.MUTED}GPU: No NVIDIA GPU detected.{Theme.RESET}")
                else:
                    print(f"  {Theme.MUTED}GPU: No NVIDIA GPU detected.{Theme.RESET}")
                
            print(f"\n{Theme.MUTED}Press Ctrl+C to exit{Theme.RESET}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\033[2J\033[H", end="")
        return 0

def _get_recent_file() -> str:
    app_data = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Firefly")
    os.makedirs(app_data, exist_ok=True)
    return os.path.join(app_data, "recent_solves.json")

def _add_recent_solve(filepath: str, objective: float, status: str, time_ms: float) -> None:
    history_file = _get_recent_file()
    try:
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
    except Exception:
        history = []
        
    entry = {
        "file": os.path.abspath(filepath),
        "objective": objective,
        "status": status,
        "time_ms": time_ms,
        "timestamp": __import__("time").time()
    }
    history.insert(0, entry)
    history = history[:5]  # Keep last 5
    
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f)
    except Exception:
        pass

def _cmd_recent(args: argparse.Namespace) -> int:
    history_file = _get_recent_file()
    if not os.path.exists(history_file):
        print(f"\n  {Theme.MUTED}No recent solves found.{Theme.RESET}\n")
        return 0
        
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        print(f"\n  {Theme.MUTED}Failed to read recent history.{Theme.RESET}\n")
        return 4
        
    print(f"\n{Theme.ACCENT}▶ RECENT SOLVES{Theme.RESET}")
    for i, entry in enumerate(history):
        fname = os.path.basename(entry["file"])
        obj = f"{entry['objective']:.6g}" if entry["objective"] is not None else "N/A"
        time_struct = __import__("time").localtime(entry["timestamp"])
        time_str = __import__("time").strftime("%Y-%m-%d %H:%M", time_struct)
        
        print(f"  {Theme.PRIMARY}{i+1}. {fname}{Theme.RESET}")
        print(f"     Status: {entry['status']} | Objective: {obj} | Time: {entry['time_ms']:.1f}ms | {time_str}")
        print()
    return 0

def _cmd_audit(args: argparse.Namespace) -> int:
    import subprocess
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.abspath(os.path.join(cli_dir, ".."))
    audit_dir = os.path.join(core_dir, "audit_reports")
    
    if not os.path.exists(audit_dir):
        try:
            os.makedirs(audit_dir)
        except Exception:
            print(f"firefly: error: could not create directory {audit_dir}", file=sys.stderr)
            return 4
            
    print(f"\n{Theme.ACCENT}Opening audit reports directory...{Theme.RESET}")
    if os.name == "nt":
        os.startfile(audit_dir)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", audit_dir])
    else:
        subprocess.Popen(["xdg-open", audit_dir])
    return 0

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="firefly",
        description="Firefly LP/MILP/QP solver — command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes
----------
  0   OPTIMAL or FEASIBLE
  1   INFEASIBLE
  2   UNBOUNDED
  3   TIME_LIMIT or NODE_LIMIT
  4   ERROR / unexpected failure

Examples
--------
  firefly solve problem.mps
  firefly solve problem.mps --method simplex --no-gpu --output solution.csv
  firefly solve problem.mps --verbose --output result.json
  firefly benchmark  ./problems/
  firefly solve-batch ./problems/ --method pdlp
  firefly test-standard
  firefly test (Developer Only)
""",
    )
    
    root.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    sub = root.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ------------------------------------------------------------------
    # Shared method/gpu flags (reused across sub-commands)
    # ------------------------------------------------------------------
    def _add_method_gpu(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--method",
            choices=["auto", "simplex", "pdlp", "milp"],
            default="auto",
            metavar="METHOD",
            help="Solver method: auto (default), simplex, pdlp, milp",
        )
        gpu_grp = p.add_mutually_exclusive_group()
        gpu_grp.add_argument(
            "--gpu",
            dest="gpu",
            action="store_true",
            default=True,
            help="Use GPU acceleration (default)",
        )
        gpu_grp.add_argument(
            "--no-gpu",
            dest="gpu",
            action="store_false",
            help="Disable GPU; use CPU solver",
        )

    def _add_verbosity(p: argparse.ArgumentParser) -> None:
        vgrp = p.add_mutually_exclusive_group()
        vgrp.add_argument(
            "--quiet", "-q",
            action="store_true",
            default=False,
            help="Print only the final objective value",
        )
        vgrp.add_argument(
            "--verbose", "-v",
            action="store_true",
            default=False,
            help="Print live-updating iteration progress",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Show full traceback on error",
        )

    # ------------------------------------------------------------------
    # firefly solve
    # ------------------------------------------------------------------
    p_solve = sub.add_parser(
        "solve",
        help="Solve a single MPS file",
        description="Solve a single LP/MILP/QP problem in MPS format.",
    )
    p_solve.add_argument("file", metavar="<file.mps>", help="Path to the MPS file")
    _add_method_gpu(p_solve)
    p_solve.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help=(
            "Write full solution vector to PATH (.csv) or (.json); "
            "format is inferred from the file extension"
        ),
    )
    _add_verbosity(p_solve)
    
    # ------------------------------------------------------------------
    # firefly compare
    # ------------------------------------------------------------------
    p_compare = sub.add_parser(
        "compare",
        help="Compare solver performance on two .mps files",
        description="Solve two .mps files and compare their objectives and times side-by-side.",
    )
    p_compare.add_argument("file1", metavar="<model_A.mps>", help="Path to the first MPS file")
    p_compare.add_argument("file2", metavar="<model_B.mps>", help="Path to the second MPS file")
    _add_method_gpu(p_compare)

    # ------------------------------------------------------------------
    # firefly benchmark
    # ------------------------------------------------------------------
    p_bench = sub.add_parser(
        "benchmark",
        help="Solve all .mps files in a folder and compare against known references",
        description=(
            "Run every .mps file in FOLDER through the solver and compare "
            "results against built-in or provided reference objectives."
        ),
    )
    p_bench.add_argument("folder", metavar="<folder>", help="Directory containing .mps files")
    _add_method_gpu(p_bench)
    p_bench.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress table output",
    )
    p_bench.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show full traceback on error",
    )

    # ------------------------------------------------------------------
    # firefly solve-batch
    # ------------------------------------------------------------------
    p_batch = sub.add_parser(
        "solve-batch",
        help="Solve all .mps files in a folder (no reference comparison)",
        description="Solve every .mps file in FOLDER. No reference comparison.",
    )
    p_batch.add_argument("folder", metavar="<folder>", help="Directory containing .mps files")
    _add_method_gpu(p_batch)
    p_batch.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress table output",
    )
    p_batch.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show full traceback on error",
    )

    # ------------------------------------------------------------------
    # firefly test
    # ------------------------------------------------------------------
    p_test = sub.add_parser(
        "test",
        help="Run the internal solver test suite",
        description="Run all internal C++ unit tests for the Firefly solver.",
    )
    p_test.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show full traceback on error",
    )

    # ------------------------------------------------------------------
    # firefly test-standard
    # ------------------------------------------------------------------
    p_test_std = sub.add_parser(
        "test-standard",
        help="Test the solver against real, standard MPS problems (Netlib)",
        description="Downloads a few standard Netlib MPS files and solves them visibly.",
    )
    _add_method_gpu(p_test_std)
    p_test_std.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress table output",
    )

    # ------------------------------------------------------------------
    # firefly update
    # ------------------------------------------------------------------
    p_update = sub.add_parser(
        "update",
        help="Update Firefly CLI to the latest version from GitHub",
        description="Downloads the latest executable and replaces the current one.",
    )

    # ------------------------------------------------------------------
    # firefly home
    p_home = sub.add_parser(
        "home",
        help="Show this menu",
    )
    
    # ------------------------------------------------------------------
    # firefly audit
    # ------------------------------------------------------------------
    p_audit = sub.add_parser(
        "audit",
        help="Open the audit reports directory",
    )
    
    # ------------------------------------------------------------------
    # firefly recent
    # ------------------------------------------------------------------
    p_recent = sub.add_parser(
        "recent",
        help="Show the last 5 solved files and their objectives",
    )
    
    # ------------------------------------------------------------------
    # firefly version
    # ------------------------------------------------------------------
    p_version = sub.add_parser(
        "version",
        help="Show current engine version",
    )

    # ------------------------------------------------------------------
    # firefly top
    # ------------------------------------------------------------------
    p_top = sub.add_parser(
        "top",
        help="Open live system resource monitor (CPU/RAM/GPU)",
    )

    return root


# ---------------------------------------------------------------------------
# ASCII Logo
# ---------------------------------------------------------------------------
def _print_firefly_logo() -> None:
    print(f"""
{Theme.ACCENT}      \\ /       {Theme.PRIMARY}  ___ _            __ _       
{Theme.ACCENT}======= ======= {Theme.PRIMARY} | __|(_) _ _  ___ / _|| | _  _ 
{Theme.ACCENT}  ====   ====   {Theme.PRIMARY} | _| | || '_|/ -_)|  _|| || || |
{Theme.ACCENT}   /  | |  \\    {Theme.PRIMARY} |_|  |_||_|  \\___||_|  |_| \\_, |
{Theme.ACCENT}  /   | |   \\   {Theme.PRIMARY}                            |__/ 
{Theme.ACCENT}      | |       
{Theme.ACCENT}      | |       {Theme.MUTED}LP/MILP/QP Solver Engine - SIH 2026{Theme.RESET}""")

def _print_homepage() -> None:
    _print_firefly_logo()
    print(f"\n{Theme.PRIMARY}Firefly Solver Engine v0.2.0{Theme.RESET}\n")
    
    print(f"{Theme.ACCENT}▶ CORE COMMANDS{Theme.RESET}")
    print(f"  {Theme.PRIMARY}firefly solve <file.mps>{Theme.RESET}    Solve a single LP/MILP/QP file")
    print(f"  {Theme.PRIMARY}firefly solve-batch <folder>{Theme.RESET} Solve all .mps files in a directory")
    print(f"  {Theme.PRIMARY}firefly compare <f1> <f2>{Theme.RESET}    Compare time and objective between two models")
    print()
    
    print(f"{Theme.ACCENT}▶ TESTING & BENCHMARKING{Theme.RESET}")
    print(f"  {Theme.PRIMARY}firefly benchmark <folder>{Theme.RESET}   Run the benchmark suite against reference optimums")
    print(f"  {Theme.PRIMARY}firefly test-standard{Theme.RESET}        Download and solve standard Netlib problems")
    print(f"  {Theme.PRIMARY}firefly test{Theme.RESET}                 Run the internal C++ test suite {Theme.MUTED}(Developer Only){Theme.RESET}")
    print()
    
    print(f"{Theme.ACCENT}▶ UTILITIES{Theme.RESET}")
    print(f"  {Theme.PRIMARY}firefly top{Theme.RESET}                  Open live system resource monitor (CPU/RAM/GPU)")
    print(f"  {Theme.PRIMARY}firefly audit{Theme.RESET}                Open the audit reports directory")
    print(f"  {Theme.PRIMARY}firefly recent{Theme.RESET}               Show recent solve results")
    print(f"  {Theme.PRIMARY}firefly update{Theme.RESET}               Update Firefly to the latest version")
    print(f"  {Theme.PRIMARY}firefly version{Theme.RESET}              Show current version info")
    print(f"  {Theme.PRIMARY}firefly home{Theme.RESET}                 Show this menu")
    print()
    
    print(f"{Theme.ACCENT}▶ LINKS{Theme.RESET}")
    print(f"  {Theme.PRIMARY}GitHub Repository{Theme.RESET}            https://github.com/akshayvarma121/Firefly_solver")
    print(f"  {Theme.PRIMARY}Official Website{Theme.RESET}             https://firefly-solver.vercel.app")
    print()
    
    print(f"{Theme.MUTED}Tip: Use `firefly solve -h` to see all available flags.{Theme.RESET}\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Strip --no-color from argv so argparse doesn't complain if it's passed after a sub-command
    while "--no-color" in sys.argv:
        sys.argv.remove("--no-color")

    # Auto-cleanup: if there's a leftover .old file from a previous update, delete it
    try:
        exe_path = os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__)
        old_path = exe_path + ".old"
        if os.path.exists(old_path):
            os.remove(old_path)
    except Exception:
        pass

    # Quick solve shortcut: if `firefly path/to/file.mps` is used, insert "solve"
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-") and sys.argv[1].lower().endswith(".mps"):
        sys.argv.insert(1, "solve")

    parser = _build_parser()
    interactive_mode = False
    
    # If run without arguments (e.g. double-clicked in Windows Explorer)
    if len(sys.argv) == 1:
        _print_firefly_logo()
        print()
        print("Firefly is running in interactive mode.")
        while True:
            try:
                user_input = input(f"\n{Theme.PRIMARY}Provide a path to a .mps file to solve{Theme.RESET} (or press Enter for menu): ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                user_input = ""
                break
                
            if not user_input:
                break
                
            if os.path.isfile(user_input):
                print(f"\nSolving {os.path.basename(user_input)}...\n")
                interactive_mode = True
                sys.argv.extend(["solve", user_input])
                break
            else:
                print(f"  {Theme.ERROR}[Error] File not found: {user_input}{Theme.RESET}")
        
        if not interactive_mode:
            _print_homepage()
            print("\n[Tip: You can also use Firefly directly from the command prompt.]")
            if os.name == "nt":
                try:
                    input("Press Enter to exit...")
                except (EOFError, KeyboardInterrupt):
                    pass
            sys.exit(0)
        else:
            _print_homepage()
            sys.exit(0)


    args = parser.parse_args()

    quiet = getattr(args, "quiet", False)
    
    # Print the logo for normal CLI runs — skip for home/help since _print_homepage already prints it
    if not quiet and len(sys.argv) > 1 and not interactive_mode and args.command not in ["home", "help"]:
        _print_firefly_logo()

    # Warn if the native solver is unavailable (unless --quiet)
    if not _FS_AVAILABLE and not quiet:
        print(
            f"[WARNING] firefly_solver native extension not loaded: {_FS_IMPORT_ERR}\n"
            "          Results are mock values and are NOT real solver output.",
            file=sys.stderr,
        )

    debug = getattr(args, "debug", False)

    try:
        if args.command == "solve":
            code = _cmd_solve(args)
        elif args.command == "compare":
            code = _cmd_compare(args)
        elif args.command == "benchmark":
            code = _cmd_benchmark(args)
        elif args.command == "solve-batch":
            code = _cmd_solve_batch(args)
        elif args.command == "test":
            code = _cmd_test(args)
        elif args.command == "test-standard":
            code = _cmd_test_standard(args)
        elif args.command == "update":
            code = _cmd_update(args)
        elif args.command == "audit":
            code = _cmd_audit(args)
        elif args.command == "recent":
            code = _cmd_recent(args)
        elif args.command == "top":
            code = _cmd_top(args)
        elif args.command == "version":
            code = _cmd_version(args)
        elif args.command in ["home", "help"]:
            _print_homepage()
            code = 0
        else:
            _print_homepage()
            code = 4
    except Exception as exc:
        if debug:
            traceback.print_exc()
        else:
            print(f"firefly: unexpected error: {exc}", file=sys.stderr)
        code = 4

    if interactive_mode and os.name == "nt":
        print("\n[Done]")
        try:
            input("Press Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass

    sys.exit(code)


if __name__ == "__main__":
    main()
