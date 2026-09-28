#!/bin/bash

# Define paths and create directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="${ROOT_DIR}/audit_reports"
mkdir -p "$REPORT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${REPORT_DIR}/audit_${TIMESTAMP}.md"
TEMP_FILE=$(mktemp)

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Helper Functions
record_pass() {
    local test_name="$1"
    echo "[PASS] $test_name" >> "$TEMP_FILE"
    ((TOTAL_TESTS++))
    ((PASSED_TESTS++))
}

record_fail() {
    local test_name="$1"
    local output="$2"
    echo "[FAIL] $test_name" >> "$TEMP_FILE"
    if [ -n "$output" ]; then
        # Indent output by 4 spaces
        printf "%s\n" "$output" | sed 's/^/    /' >> "$TEMP_FILE"
    fi
    ((TOTAL_TESTS++))
    ((FAILED_TESTS++))
}

record_skip() {
    local test_name="$1"
    local reason="$2"
    echo "[SKIP] $test_name ($reason)" >> "$TEMP_FILE"
    ((TOTAL_TESTS++))
    ((SKIPPED_TESTS++))
}

add_section() {
    echo ""
    echo "--- $1 ---"
    echo "" >> "$TEMP_FILE"
    echo "## $1" >> "$TEMP_FILE"
}

run_test() {
    local test_name="$1"
    shift
    local cmd="$*"
    
    echo -n "  Running ${test_name}... "
    
    local output
    # eval is used to allow complex commands (like pipes) if passed as a string
    output=$(eval "$cmd" 2>&1)
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "PASS"
        record_pass "$test_name"
        
        # Check if the test explicitly logged an evidence line
        local evidence
        evidence=$(echo "$output" | grep -E '\[EVIDENCE\]' || true)
        if [ -n "$evidence" ]; then
            # Print it out to the console and append to the report
            echo "$evidence" | sed 's/^[[:space:]]*\[EVIDENCE\]/    [EVIDENCE]/'
            echo "$evidence" | sed 's/^[[:space:]]*\[EVIDENCE\]/    - EVIDENCE:/' >> "$TEMP_FILE"
        fi
    else
        echo "FAIL"
        record_fail "$test_name" "$output"
    fi
}

get_test_bin() {
    local base_name="$1"
    if [ -f "${BUILD_DIR}/tests/${base_name}" ]; then
        echo "${BUILD_DIR}/tests/${base_name}"
    elif [ -f "${BUILD_DIR}/tests/${base_name}.exe" ]; then
        echo "${BUILD_DIR}/tests/${base_name}.exe"
    elif [ -f "${BUILD_DIR}/tests/Debug/${base_name}.exe" ]; then
        echo "${BUILD_DIR}/tests/Debug/${base_name}.exe"
    elif [ -f "${BUILD_DIR}/tests/Release/${base_name}.exe" ]; then
        echo "${BUILD_DIR}/tests/Release/${base_name}.exe"
    else
        # Fallback to standard Linux path so errors reflect the missing file cleanly
        echo "${BUILD_DIR}/tests/${base_name}"
    fi
}

# --- 1. Gather Header Information ---

# Git commit hash or dirty state
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git status --porcelain | grep -q .; then
        COMMIT_HASH="uncommitted changes present"
    else
        COMMIT_HASH=$(git rev-parse HEAD)
    fi
else
    COMMIT_HASH="Unknown (Not a git repository)"
fi

if [ -z "$WITH_CUDA" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        WITH_CUDA="ON"
    else
        WITH_CUDA="OFF"
    fi
fi

GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
if [ -z "$GPU_INFO" ]; then
    GPU_INFO="none detected"
fi

OS_INFO=$(uname -s 2>/dev/null || echo "Unknown")
COMPILER_VERSION=$(g++ --version 2>/dev/null | head -n 1 || echo "Unknown")
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -o "release [0-9.]*" | cut -d' ' -f2 || echo "Unknown")
if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="Unknown"
fi
PYTHON_VERSION=$(python --version 2>/dev/null || echo "Unknown")

# --- 2. Write Header Block to Temp File ---
{
    echo "## Environment"
    echo "- **Commit Hash:** \`$COMMIT_HASH\`"
    echo "- **WITH_CUDA:** \`$WITH_CUDA\`"
    echo "- **GPU:** \`$GPU_INFO\`"
    echo "- **OS:** \`$OS_INFO\`"
    echo "- **Compiler:** \`$COMPILER_VERSION\`"
    echo "- **CUDA Toolkit:** \`$CUDA_VERSION\`"
    echo "- **Python:** \`$PYTHON_VERSION\`"
} > "$TEMP_FILE"

export TEST_VENV_DIR="${SCRIPT_DIR}/.audit_venv"
rm -rf "$TEST_VENV_DIR"
${PYTHON:-python} -m venv "$TEST_VENV_DIR" || { echo "Failed to create TEST_VENV_DIR"; exit 1; }

# --- 3. Clean Build ---
BUILD_DIR="${SCRIPT_DIR}/build"
echo "Cleaning and building core..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Capture both stdout and stderr of the build process
# Using cmake as standard for C++20 + CUDA, adjust if using a different build system
BUILD_OUTPUT=$(cd "$BUILD_DIR" && cmake .. -DWITH_CUDA="${WITH_CUDA}" 2>&1 && cmake --build . -j 2>&1)
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    # Write build failed report
    {
        echo "# Audit Report"
        echo ""
        echo "## BUILD FAILED"
        echo "The clean build failed. All test phases skipped."
        echo ""
        echo '```text'
        echo "$BUILD_OUTPUT"
        echo '```'
        echo ""
        cat "$TEMP_FILE"
    } > "$REPORT_FILE"
    rm -f "$TEMP_FILE"

    echo "======================================"
    echo "BUILD FAILED"
    echo "======================================"
    echo "Report written to: $REPORT_FILE"
    exit 1
fi

# --- 4. Pre-flight Keyword Guard ---
# Ensure every C++ test binary correctly reports its valid tests on an invalid input,
# and check that the keywords we're about to use actually exist in those lists.
echo "Running pre-flight test keyword guard..."
check_binary_keywords() {
    local bin_name=$1
    shift
    local expected_kws=("$@")
    local bin_path=$(get_test_bin "$bin_name")
    
    if [ ! -f "$bin_path" ]; then
        echo "Pre-flight failed: Binary $bin_name not found at $bin_path"
        exit 1
    fi
    
    local guard_out=$("$bin_path" --test invalid_guard_check_123 2>&1 || true)
    if ! echo "$guard_out" | grep -q "Valid tests:"; then
        echo "Pre-flight failed: $bin_name did not report 'Valid tests:' on invalid input."
        echo "Output was: $guard_out"
        exit 1
    fi
    
    for kw in "${expected_kws[@]}"; do
        if ! echo "$guard_out" | grep -q "\b$kw\b"; then
            echo "Pre-flight failed: Keyword '$kw' not found in $bin_name valid tests list."
            echo "Binary reported: $guard_out"
            exit 1
        fi
    done
}

check_binary_keywords "firefly_tests" marker_int ranges all_bounds malformed_missing_endata malformed_unknown_section malformed_undeclared_row malformed_empty ill_conditioned whitespace
check_binary_keywords "test_presolve" redundant_row fixed_variable bound_tightening scaling
check_binary_keywords "test_presolve_rules" feasibility_invariance isolation_and_postsolve
check_binary_keywords "test_pdlp_ruiz" ruiz_equilibration
check_binary_keywords "test_simplex" netlib_reference infeasible unbounded bland_cycling feasibility_self_check
check_binary_keywords "test_pdlp_cross_validation" cross_validation
check_binary_keywords "test_pdlp_stability" ill_conditioned_stability
check_binary_keywords "test_pdlp" convergence_check cpu_gpu_parity
check_binary_keywords "test_branch_and_bound" end_to_end_parser callback status_outcomes pruning_infeasible incumbent_feasibility miplib_crosscheck
echo "Pre-flight guard passed."

# --- 5. Run Test Phases ---
# The phases must be in this order:
# Parser, Presolve, Simplex, PDLP, MILP, Bindings, Install, CLI, API, Benchmark.

add_section "Parser"
run_test "MARKER-line integer parsing" "$(get_test_bin firefly_tests) --test marker_int"
run_test "RANGES on all three row-sense cases" "$(get_test_bin firefly_tests) --test ranges"
run_test "Every BOUNDS type (UP/LO/FX/FR/MI/PL/BV)" "$(get_test_bin firefly_tests) --test all_bounds"
run_test "Malformed input: Missing ENDATA" "$(get_test_bin firefly_tests) --test malformed_missing_endata"
run_test "Malformed input: Unknown section header" "$(get_test_bin firefly_tests) --test malformed_unknown_section"
run_test "Malformed input: Undeclared row reference" "$(get_test_bin firefly_tests) --test malformed_undeclared_row"
run_test "Malformed input: Empty file" "$(get_test_bin firefly_tests) --test malformed_empty"
run_test "Scientific-notation parsing" "$(get_test_bin firefly_tests) --test ill_conditioned"
run_test "Whitespace tolerance" "$(get_test_bin firefly_tests) --test whitespace"
run_test "Real Netlib/MIPLIB parse sweep" "$(get_test_bin run_real_files)"

add_section "Presolve"
run_test "Redundant-row removal" "$(get_test_bin test_presolve) --test redundant_row"
run_test "Fixed-variable substitution" "$(get_test_bin test_presolve) --test fixed_variable"
run_test "Bound tightening" "$(get_test_bin test_presolve) --test bound_tightening"
run_test "Scaling" "$(get_test_bin test_presolve) --test scaling"
run_test "Feasibility-invariant check" "$(get_test_bin test_presolve_rules) --test feasibility_invariance"
run_test "Un-presolve mapping for eliminated fixed variable" "$(get_test_bin test_presolve_rules) --test isolation_and_postsolve"
run_test "Ruiz equilibration: row/col norm and bad-scale un-scaling" "$(get_test_bin test_pdlp_ruiz) --test ruiz_equilibration"

add_section "Simplex"
run_test "Netlib correctness (within 1e-6)" "$(get_test_bin test_simplex) --test netlib_reference"
run_test "Explicit INFEASIBLE detection" "$(get_test_bin test_simplex) --test infeasible"
run_test "Explicit UNBOUNDED detection" "$(get_test_bin test_simplex) --test unbounded"
run_test "Anti-cycling (Beale's LP)" "$(get_test_bin test_simplex) --test bland_cycling"
run_test "Pre-return solution-feasibility self-check" "$(get_test_bin test_simplex) --test feasibility_self_check"

add_section "PDLP"
run_test "Cross-validation against Simplex (Netlib Tier 1)" "$(get_test_bin test_pdlp_cross_validation) --test cross_validation"
run_test "Ill-conditioned stress stability" "$(get_test_bin test_pdlp_stability) --test ill_conditioned_stability"
run_test "Explicit convergence check (1e-4 gap strict limits)" "$(get_test_bin test_pdlp) --test convergence_check"

if [ "$WITH_CUDA" = "ON" ]; then
    run_test "Concurrency (CudaContext singleton/mutex)" "$(get_test_bin test_pdlp_concurrency)"
    run_test "Graceful GPU-unavailability handling" "$(get_test_bin test_gpu_unavailability)"
    
    run_test "CPU-fallback parity (GPU vs CPU results)" "$(get_test_bin test_pdlp) --test cpu_gpu_parity"
else
    record_skip "Concurrency (CudaContext singleton/mutex)" "WITH_CUDA=OFF"
    record_skip "Graceful GPU-unavailability handling" "WITH_CUDA=OFF"
    record_skip "CPU-fallback parity (GPU vs CPU results)" "WITH_CUDA=OFF"
fi

add_section "MILP"
run_test "End-to-end solve (real parser, no SparseProblem bypass)" "$(get_test_bin test_branch_and_bound) --test end_to_end_parser"
run_test "Node-based progress callback" "$(get_test_bin test_branch_and_bound) --test callback"
run_test "Status outcomes (OPTIMAL, FEASIBLE, TIME_LIMIT, NODE_LIMIT)" "$(get_test_bin test_branch_and_bound) --test status_outcomes"
run_test "Correct pruning on INFEASIBLE relaxation" "$(get_test_bin test_branch_and_bound) --test pruning_infeasible"
run_test "Incumbent feasibility self-check" "$(get_test_bin test_branch_and_bound) --test incumbent_feasibility"
run_test "Cross-check against published MIPLIB optimal (flugpl)" "$(get_test_bin test_branch_and_bound) --test miplib_crosscheck"


add_section "Install"
INSTALL_TEST_CMD="
if [ -z \"\$TEST_VENV_DIR\" ]; then
    export TEST_VENV_DIR=\$(mktemp -d)
    \${PYTHON:-python} -m venv \"\$TEST_VENV_DIR\" || { echo \"Failed to create venv\"; rm -rf \"\$TEST_VENV_DIR\"; exit 1; }
fi

if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then
    source \"\$TEST_VENV_DIR/Scripts/activate\"
else
    source \"\$TEST_VENV_DIR/bin/activate\"
fi

# Run pip install -e . from core/
OUTPUT=\$(pip install -e . 2>&1)
if [ \$? -ne 0 ]; then
    echo \"pip install failed! Full pip output:\"
    echo \"\$OUTPUT\"
    rm -rf \"\$TEST_VENV_DIR\"
    exit 1
fi

# Move to a neutral directory to prevent local path shadowing, then verify import
pushd \"\$TEST_VENV_DIR\" >/dev/null
IMPORT_OUTPUT=\$(python -c 'import firefly_solver; print(firefly_solver.__name__)' 2>&1)
if [ \$? -ne 0 ] || [ \"\$IMPORT_OUTPUT\" != \"firefly_solver\" ]; then
    echo \"Import failed! Output:\"
    echo \"\$IMPORT_OUTPUT\"
    popd >/dev/null
    rm -rf \"\$TEST_VENV_DIR\"
    exit 1
fi

popd >/dev/null
echo \"Install and clean import successful in pristine temporary virtual environment.\"
"
run_test "Clean virtual environment pip install and import" "$INSTALL_TEST_CMD"

add_section "Bindings"
run_test "Full solve() API surface (methods & GPU combinations)" "
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then source \"\$TEST_VENV_DIR/Scripts/activate\"; else source \"\$TEST_VENV_DIR/bin/activate\"; fi
python ../api/test_bindings.py --test \"[API]\""

run_test "GIL-safety stress test (concurrent callbacks)" "
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then source \"\$TEST_VENV_DIR/Scripts/activate\"; else source \"\$TEST_VENV_DIR/bin/activate\"; fi
python ../api/test_bindings.py --test \"[GIL]\""

run_test "C++ exception propagation to Python" "
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then source \"\$TEST_VENV_DIR/Scripts/activate\"; else source \"\$TEST_VENV_DIR/bin/activate\"; fi
python ../api/test_bindings.py --test \"[EXC]\""

run_test "Windows CUDA_PATH DLL directory resolution" "
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then source \"\$TEST_VENV_DIR/Scripts/activate\"; else source \"\$TEST_VENV_DIR/bin/activate\"; fi
python ../api/test_bindings.py --test \"[WIN]\""

add_section "JSON Problem Definitions"
run_test "JSON payloads (Problem 1 & 2) and validation errors" "
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then source \"\$TEST_VENV_DIR/Scripts/activate\"; else source \"\$TEST_VENV_DIR/bin/activate\"; fi
python ../api/test_bindings.py --test \"[JSON]\""

add_section "CLI"
CLI_VENV_CMD="
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then
    source \"\$TEST_VENV_DIR/Scripts/activate\"
else
    source \"\$TEST_VENV_DIR/bin/activate\"
fi

# Run the comprehensive regression wrapper
python cli/test_regression_cli.py
RC=\$?
if [ \$RC -ne 0 ]; then
    echo \"Regression CLI wrapper failed.\"
    exit 1
fi

# Verify --verbose produces live progress output
OUT=\$(python -m cli.firefly solve tests/netlib_miplib/afiro.mps --method pdlp --no-gpu --verbose 2>&1)
if ! echo \"\$OUT\" | grep -q 'iter '; then
    echo \"--verbose flag failed to produce live progress output.\"
    echo \"Output: \$OUT\"
    exit 1
fi

# Verify --quiet produces only the final objective
OUT=\$(python -m cli.firefly solve tests/regression/marker_int.mps --method milp --no-gpu --quiet 2>&1)
if ! echo \"\$OUT\" | grep -q '^-10'; then
    echo \"--quiet flag failed to produce only the objective (-10).\"
    echo \"Output: \$OUT\"
    exit 1
fi
if echo \"\$OUT\" | grep -q 'Status'; then
    echo \"--quiet flag produced extra text instead of just objective.\"
    echo \"Output: \$OUT\"
    exit 1
fi

# Verify --debug shows full traceback on deliberately malformed file
# The firefly CLI normally intercepts errors, --debug allows them through.
OUT=\$(python -m cli.firefly solve tests/regression/malformed.mps --no-gpu --debug 2>&1)
if ! echo \"\$OUT\" | grep -q 'Traceback'; then
    echo \"--debug flag failed to show traceback.\"
    echo \"Output: \$OUT\"
    exit 1
fi

echo \"CLI tools tested successfully.\"
"
run_test "firefly CLI solve against regression suite + verbosity flags" "$CLI_VENV_CMD"

add_section "API"
API_TEST_CMD="
if [ -f \"\$TEST_VENV_DIR/Scripts/activate\" ]; then
    source \"\$TEST_VENV_DIR/Scripts/activate\"
else
    source \"\$TEST_VENV_DIR/bin/activate\"
fi

pip install -q -r ../api/requirements.txt || { echo \"Failed to install API requirements\"; exit 1; }

# Start server in background from the API directory
pushd ../api >/dev/null
python -m uvicorn main:app --host 127.0.0.1 --port 8080 > server.log 2>&1 &
SERVER_PID=\$!
popd >/dev/null

# Clean shutdown trap: guaranteed death to background job on subshell exit/interrupt
trap 'kill \$SERVER_PID 2>/dev/null; wait \$SERVER_PID 2>/dev/null' EXIT SIGINT SIGTERM

echo \"Waiting for FastAPI server to start on port 8080...\"
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s -f http://127.0.0.1:8080/docs >/dev/null 2>&1; do
    RETRY_COUNT=\$((RETRY_COUNT + 1))
    if [ \$RETRY_COUNT -ge \$MAX_RETRIES ]; then
        echo \"Server failed to start within 30 seconds. Logs:\"
        cat ../api/server.log
        exit 1
    fi
    sleep 1
done

echo \"Server is up! Running API regression sweep...\"
python ../api/test_regression_api.py
RC=\$?

if [ \$RC -ne 0 ]; then
    echo \"API regression tests failed! Server logs:\"
    cat ../api/server.log
    exit 1
fi

echo \"API tests passed successfully. Now testing MOCK fallback...\"
# Kill the healthy server
kill \$SERVER_PID 2>/dev/null
wait \$SERVER_PID 2>/dev/null

# Uninstall the native extension to break the import and trigger mock mode
pip uninstall -y firefly-solver >/dev/null 2>&1

# Start server in broken state
pushd ../api >/dev/null
python -m uvicorn main:app --host 127.0.0.1 --port 8080 > server.log 2>&1 &
SERVER_PID=\$!
popd >/dev/null


echo \"Waiting for broken FastAPI server to start...\"
RETRY_COUNT=0
while ! curl -s -f http://127.0.0.1:8080/docs >/dev/null 2>&1; do
    RETRY_COUNT=\$((RETRY_COUNT + 1))
    if [ \$RETRY_COUNT -ge 30 ]; then exit 1; fi
    sleep 1
done

# Assert mock:true on a solve request
MOCK_PROB='{\"num_vars\": 2, \"num_constrs\": 1, \"obj_coeffs\": [1.0, 1.0], \"row_ptr\": [0, 2], \"col_idx\": [0, 1], \"values\": [1.0, 1.0], \"row_senses\": \"L\", \"rhs\": [10.0]}'
MOCK_RES=\$(curl -s -X POST http://127.0.0.1:8080/solve -d \"problem_def=\$MOCK_PROB\" 2>/dev/null)
if ! echo \"\$MOCK_RES\" | grep -q '\"mock\":true'; then
    echo \"Failed to activate mock mode! Response: \$MOCK_RES\"
    exit 1
fi
echo \"Mock mode properly engaged.\"

# Restore the solver and restart
kill \$SERVER_PID 2>/dev/null
wait \$SERVER_PID 2>/dev/null

pushd ../api >/dev/null
pip install -q -e ../core || { echo "Failed to reinstall firefly solver"; exit 1; }

# Start server in restored state
uvicorn main:app --host 127.0.0.1 --port 8080 > server.log 2>&1 &
SERVER_PID=\$!
popd >/dev/null

echo \"Waiting for restored FastAPI server to start...\"
RETRY_COUNT=0
while ! curl -s -f http://127.0.0.1:8080/docs >/dev/null 2>&1; do
    RETRY_COUNT=\$((RETRY_COUNT + 1))
    if [ \$RETRY_COUNT -ge 30 ]; then exit 1; fi
    sleep 1
done

# Assert mock:true is gone
RESTORED_RES=\$(curl -s -X POST http://127.0.0.1:8080/solve -d \"problem_def=\$MOCK_PROB\" 2>/dev/null)
if echo \"\$RESTORED_RES\" | grep -q '\"mock\":true'; then
    echo \"Failed to restore native solver! Response: \$RESTORED_RES\"
    exit 1
fi

echo \"Native solver successfully resumed.\"
"
run_test "Live FastAPI server background lifecycle + HTTP regression sweep" "$API_TEST_CMD"

add_section "Benchmark (Informational)"
# Benchmark is explicitly informational. We do not use run_test so it cannot trigger audit failure.

run_benchmark_suite() {
    local suite_name="\$1"
    local folder_name="\$2"
    
    # Try a few common paths
    local suite_path=\"\"
    for d in \"../data/\$folder_name\" \"data/\$folder_name\" \"../core/data/\$folder_name\" \"../tests/\$folder_name\" \"tests/\$folder_name\"; do
        if [ -d \"\$d\" ]; then
            suite_path=\"\$d\"
            break
        fi
    done
    
    if [ -n \"\$suite_path\" ] && ls \"\$suite_path\"/*.mps >/dev/null 2>&1; then
        echo \"Running \$suite_name benchmarks from \$suite_path...\"
        echo \"### \$suite_name\" >> \"\$TEMP_FILE\"
        echo '```text' >> \"\$TEMP_FILE\"
        # Use || true so it never crashes the script
        PYTHONPATH=\"../api:\${BUILD_DIR}\" \${PYTHON:-python} ../core/cli/firefly.py benchmark \"\$suite_path\" >> \"\$TEMP_FILE\" 2>&1 || true
        echo '```' >> \"\$TEMP_FILE\"
        echo \"\" >> \"\$TEMP_FILE\"
    else
        echo \"[SKIP] \$suite_name benchmark data not found\" >> \"\$TEMP_FILE\"
        echo \"\" >> \"\$TEMP_FILE\"
    fi
}

run_benchmark_suite \"Netlib\" \"netlib\"
run_benchmark_suite \"MIPLIB\" \"miplib\"
run_benchmark_suite \"QPLIB\" \"qplib\"



# --- 5. Finalize Report ---
EFFECTIVE_TOTAL=$((TOTAL_TESTS - SKIPPED_TESTS))
if [ "$EFFECTIVE_TOTAL" -gt 0 ]; then
    PASS_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS / $EFFECTIVE_TOTAL) * 100}")
else
    PASS_PERCENT="0.0"
fi

# Write the summary block first to the actual report file
{
    echo "# Audit Report"
    echo ""
    echo "## Summary"
    echo "### Passed: ${PASSED_TESTS}/${EFFECTIVE_TOTAL} (${PASS_PERCENT}%)"
    echo ""
    echo "- **Total Tests:** $TOTAL_TESTS"
    echo "- **Passed:** $PASSED_TESTS"
    echo "- **Failed:** $FAILED_TESTS"
    echo "- **Skipped:** $SKIPPED_TESTS"
    echo ""
} > "$REPORT_FILE"

# Append the rest of the report (header + test sections) from the temp file
cat "$TEMP_FILE" >> "$REPORT_FILE"
rm -f "$TEMP_FILE"

# --- 6. Terminal Output ---
echo "======================================"
echo "Audit Summary"
echo "======================================"
echo "Passed:      ${PASSED_TESTS}/${EFFECTIVE_TOTAL} (${PASS_PERCENT}%)"
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo "Passed:      $PASSED_TESTS"
echo "Failed:      $FAILED_TESTS"
echo "Skipped:     $SKIPPED_TESTS"
echo "======================================"
echo "Report written to: $REPORT_FILE"

# Exit nonzero if any tests failed
if [ "$FAILED_TESTS" -gt 0 ]; then
    rm -rf "$TEST_VENV_DIR" 2>/dev/null
    exit 1
fi

rm -rf "$TEST_VENV_DIR" 2>/dev/null
exit 0
