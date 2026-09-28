// core/bindings/bindings.cpp
//
// pybind11 module exposing the Firefly solver core to Python.
//
// Exposed API:
//   class SparseProblem  — wraps firefly::Problem (CSR + metadata)
//   class SolveResult    — wraps firefly::SolveResult
//   solve(problem, method="auto", gpu=True, iteration_callback=None) -> SolveResult
//   parse_mps(filepath: str) -> SparseProblem
//   parse_mps_string(content: str) -> SparseProblem
//
// GIL contract:
//   - The CUDA path in PDLPSolver::solve() may release the GIL internally via
//     a py::gil_scoped_release guard in the calling thread.
//   - Every iteration_callback invocation re-acquires the GIL with
//     py::gil_scoped_acquire before crossing the C++->Python boundary.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "firefly/mps_parser.h"
#include "firefly/presolve.h"
#include "firefly/simplex.h"
#include "firefly/pdlp.h"
#include "firefly/branch_and_bound.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Convert firefly::SolveStatus -> string for Python consumers
static std::string status_to_str(firefly::SolveStatus s) {
    switch (s) {
        case firefly::SolveStatus::OPTIMAL:    return "OPTIMAL";
        case firefly::SolveStatus::INFEASIBLE: return "INFEASIBLE";
        case firefly::SolveStatus::UNBOUNDED:  return "UNBOUNDED";
        case firefly::SolveStatus::FEASIBLE:   return "FEASIBLE";
        case firefly::SolveStatus::TIME_LIMIT: return "TIME_LIMIT";
        case firefly::SolveStatus::NODE_LIMIT: return "NODE_LIMIT";
        case firefly::SolveStatus::ERROR:      return "ERROR";
    }
    return "UNKNOWN";
}

// Build a firefly::Problem from the Python-facing SparseProblem fields.
// The Python class mirrors the JSON schema already used by api/main.py.
struct SparseProblem {
    std::string name;
    int num_vars    = 0;
    int num_constrs = 0;

    // Objective
    std::vector<double> obj_coeffs;

    // Constraint matrix (CSR)
    std::vector<int>    row_ptr;
    std::vector<int>    col_idx;
    std::vector<double> values;

    // Row metadata
    std::vector<char>   row_senses;    // 'L', 'G', 'E', 'N'
    std::vector<double> rhs;

    // Column bounds
    std::vector<double> var_lower_bounds;
    std::vector<double> var_upper_bounds;

    // Integrality
    std::vector<bool>   is_integer;
};

// Convert SparseProblem -> firefly::Problem for solver consumption
static firefly::Problem to_firefly_problem(const SparseProblem& sp) {
    firefly::Problem p;
    p.name     = sp.name;
    p.minimize = true;  // API always passes minimisation form

    int nv = sp.num_vars;
    int nc = sp.num_constrs;

    // Objective
    p.objective.assign(sp.obj_coeffs.begin(), sp.obj_coeffs.end());
    if ((int)p.objective.size() < nv)
        p.objective.resize(nv, 0.0);

    // Column names, bounds, integrality
    p.col_names.resize(nv);
    p.col_lower_bounds.resize(nv, 0.0);
    p.col_upper_bounds.resize(nv, firefly::INF);
    p.is_integer.resize(nv, false);

    if (!sp.var_lower_bounds.empty() && (int)sp.var_lower_bounds.size() != nv) {
        throw std::invalid_argument("var_lower_bounds size does not match num_vars");
    }
    if (!sp.var_upper_bounds.empty() && (int)sp.var_upper_bounds.size() != nv) {
        throw std::invalid_argument("var_upper_bounds size does not match num_vars");
    }
    if (!sp.is_integer.empty() && (int)sp.is_integer.size() != nv) {
        throw std::invalid_argument("is_integer size does not match num_vars");
    }

    for (int j = 0; j < nv; ++j) {
        p.col_names[j] = "x" + std::to_string(j);
        p.col_to_index[p.col_names[j]] = j;
        if (j < (int)sp.var_lower_bounds.size()) p.col_lower_bounds[j] = sp.var_lower_bounds[j];
        if (j < (int)sp.var_upper_bounds.size()) p.col_upper_bounds[j] = sp.var_upper_bounds[j];
        if (j < (int)sp.is_integer.size())       p.is_integer[j]       = sp.is_integer[j];
    }

    // Row metadata: senses and bounds derived from RHS
    if ((int)sp.row_senses.size() != nc) {
        throw std::invalid_argument("row_senses size does not match num_constrs");
    }
    if ((int)sp.rhs.size() != nc) {
        throw std::invalid_argument("rhs size does not match num_constrs");
    }

    p.row_names.resize(nc);
    p.row_senses.resize(nc, 'L');
    p.row_lower_bounds.resize(nc, -firefly::INF);
    p.row_upper_bounds.resize(nc,  firefly::INF);

    for (int i = 0; i < nc; ++i) {
        p.row_names[i] = "c" + std::to_string(i);
        p.row_to_index[p.row_names[i]] = i;

        char sense = sp.row_senses[i];
        double rhs  = sp.rhs[i];
        p.row_senses[i] = sense;

        // Convert sense + rhs to explicit bounds used by Presolver
        switch (sense) {
            case 'L':  // <= rhs
                p.row_lower_bounds[i] = -firefly::INF;
                p.row_upper_bounds[i] = rhs;
                break;
            case 'G':  // >= rhs
                p.row_lower_bounds[i] = rhs;
                p.row_upper_bounds[i] = firefly::INF;
                break;
            case 'E':  // == rhs
                p.row_lower_bounds[i] = rhs;
                p.row_upper_bounds[i] = rhs;
                break;
            case 'N':  // free row (objective); set no bound
                p.row_lower_bounds[i] = -firefly::INF;
                p.row_upper_bounds[i] =  firefly::INF;
                break;
            default:
                throw std::invalid_argument(std::string("Unknown row sense: ") + sense);
        }
    }

    // Constraint matrix: expand CSR triplets
    if (!sp.row_ptr.empty()) {
        for (int i = 0; i < nc; ++i) {
            if (i + 1 >= (int)sp.row_ptr.size()) break;
            for (int k = sp.row_ptr[i]; k < sp.row_ptr[i + 1]; ++k) {
                firefly::Triplet t;
                t.row   = static_cast<size_t>(i);
                t.col   = static_cast<size_t>(sp.col_idx[k]);
                t.value = sp.values[k];
                p.matrix.push_back(t);
            }
        }
    }

    return p;
}

struct PyPresolveStats {
    int original_rows = 0;
    int original_cols = 0;
    int reduced_rows = 0;
    int reduced_cols = 0;
    int rows_removed = 0;
    int bounds_tightened = 0;
    int variables_fixed = 0;
    int scaling_applied = 0;
};

// Convert firefly::SolveResult -> Python-friendly struct
struct PySolveResult {
    std::string         status;
    std::string         solver_used;
    double              objective   = 0.0;
    std::vector<double> solution;
    std::vector<double> dual_solution;
    double              primal_dual_gap = 0.0;
    double              wall_time_ms    = 0.0;
    int                 iterations      = 0;
    std::string         message;
    PyPresolveStats     presolve_stats;
};

static PySolveResult to_py_result(const firefly::SolveResult& r) {
    PySolveResult pr;
    pr.status         = status_to_str(r.status);
    pr.solver_used    = r.solver_used;
    pr.objective      = r.objective_value;
    pr.solution       = r.primal_solution;
    pr.dual_solution  = r.dual_solution;
    pr.primal_dual_gap = r.primal_dual_gap;
    pr.wall_time_ms   = static_cast<double>(r.solve_time_ms);
    pr.iterations     = static_cast<int>(r.phase1_iterations + r.phase2_iterations);
    pr.message        = r.message;
    return pr;
}

// ---------------------------------------------------------------------------
// SparseProblem from a parsed firefly::Problem (used by parse_mps/parse_mps_string)
// ---------------------------------------------------------------------------
static SparseProblem from_firefly_problem(const firefly::Problem& fp) {
    SparseProblem sp;
    sp.name        = fp.name;
    sp.num_vars    = static_cast<int>(fp.col_names.size());
    sp.num_constrs = static_cast<int>(fp.row_names.size());

    sp.obj_coeffs       = fp.objective;
    sp.var_lower_bounds = fp.col_lower_bounds;
    sp.var_upper_bounds = fp.col_upper_bounds;
    sp.is_integer       = std::vector<bool>(fp.is_integer.begin(), fp.is_integer.end());
    sp.row_senses       = fp.row_senses;

    // RHS: take the tightest finite bound for each sense
    sp.rhs.resize(sp.num_constrs, 0.0);
    for (int i = 0; i < sp.num_constrs; ++i) {
        char s = (i < (int)fp.row_senses.size()) ? fp.row_senses[i] : 'L';
        if (s == 'L' || s == 'N')
            sp.rhs[i] = fp.row_upper_bounds[i];
        else
            sp.rhs[i] = fp.row_lower_bounds[i];
    }

    // Build CSR from triplets
    sp.row_ptr.assign(sp.num_constrs + 1, 0);
    // Count entries per row
    for (const auto& t : fp.matrix)
        if ((int)t.row < sp.num_constrs)
            sp.row_ptr[t.row + 1]++;
    // Prefix sum
    for (int i = 1; i <= sp.num_constrs; ++i)
        sp.row_ptr[i] += sp.row_ptr[i - 1];

    sp.col_idx.resize(fp.matrix.size());
    sp.values.resize(fp.matrix.size());
    std::vector<int> pos(sp.num_constrs, 0);
    for (const auto& t : fp.matrix) {
        int r = static_cast<int>(t.row);
        if (r >= sp.num_constrs) continue;
        int dst = sp.row_ptr[r] + pos[r]++;
        sp.col_idx[dst] = static_cast<int>(t.col);
        sp.values[dst]  = t.value;
    }

    return sp;
}

// ---------------------------------------------------------------------------
// solve() — main dispatch
// ---------------------------------------------------------------------------
static PySolveResult py_solve(
    const SparseProblem& sp,
    const std::string&   method,
    bool                 gpu,
    py::object           iteration_callback)
{
    firefly::Problem fp = to_firefly_problem(sp);

    // Detect MILP: any integer variable
    bool has_integers = false;
    for (bool b : fp.is_integer) if (b) { has_integers = true; break; }

    // Presolve (shared by all LP paths)
    firefly::PresolvedProblem pre;
    try {
        pre = firefly::Presolver::presolve(fp);
    } catch (const firefly::InfeasibleProblemException& e) {
        std::string msg = e.what();
        firefly::SolveResult result;
        if (msg.find("Unbounded") != std::string::npos) {
            result.status = firefly::SolveStatus::UNBOUNDED;
        } else {
            result.status = firefly::SolveStatus::INFEASIBLE;
        }
        result.message = msg;
        return to_py_result(result);
    } catch (const std::exception& e) {
        std::string msg = e.what();
        firefly::SolveResult result;
        if (msg.find("Infeasible") != std::string::npos) {
            result.status = firefly::SolveStatus::INFEASIBLE;
        } else if (msg.find("Unbounded") != std::string::npos) {
            result.status = firefly::SolveStatus::UNBOUNDED;
        } else {
            result.status = firefly::SolveStatus::ERROR;
        }
        result.message = msg;
        return to_py_result(result);
    }

    auto make_iteration_cb = [&](auto& opts_cb) {
        if (!iteration_callback.is_none()) {
            opts_cb = [&iteration_callback](size_t iter, double p, double d, double ms) {
                // Re-acquire the GIL — this lambda may be called from a thread
                // that released it for the CUDA call.
                py::gil_scoped_acquire acquire;
                iteration_callback(
                    static_cast<int>(iter), p, d, ms
                );
            };
        }
    };

    firefly::SolveResult result;

    try {
        if (has_integers) {
            // MILP: branch-and-bound
            firefly::BBSolverOptions bbo;
            bbo.lp_solver = gpu ? firefly::LPSolverChoice::PDLP
                                 : firefly::LPSolverChoice::SIMPLEX;

        if (!iteration_callback.is_none()) {
            bbo.progress_callback = [&iteration_callback](
                    size_t node, double incumbent, double bound, double ms) {
                py::gil_scoped_acquire acquire;
                // Adapt B&B signature to the 4-arg Python callback convention
                iteration_callback(
                    static_cast<int>(node), incumbent, bound, ms
                );
            };
        }

        {
            py::gil_scoped_release release;  // release while solving
            auto bb_res = firefly::BranchAndBoundSolver::solve(fp, bbo);
            // Upcast BBSolverResult -> SolveResult
            result = static_cast<firefly::SolveResult>(bb_res);
        }

    } else if (method == "simplex" || (!gpu && method == "auto")) {
        // Pure LP, simplex
        firefly::SimplexOptions so;

        {
            py::gil_scoped_release release;
            result = firefly::SimplexSolver::solve(pre, so);
        }
        result.solver_used = "simplex";

        // Simplex doesn't natively support per-iteration callbacks here.
        // (The simplex implementation would need to be extended; for now
        //  the callback is a no-op for the simplex path — document this.)
        (void)iteration_callback;

    } else {
        // Default: PDLP (GPU if available and requested)
        firefly::PDLPOptions po;

#if !defined(FIREFLY_WITH_CUDA)
        (void)gpu;   // CUDA not compiled in; silently fall back to CPU PDLP
#endif

        make_iteration_cb(po.iteration_callback);

        {
            py::gil_scoped_release release;  // release while in CUDA
            result = firefly::PDLPSolver::solve(pre, po);
        }
        result.solver_used = "pdlp";

        // FALLBACK LOGIC
        if (method == "auto" && (result.status == firefly::SolveStatus::ITERATION_LIMIT || result.status == firefly::SolveStatus::TIME_LIMIT)) {
            firefly::SimplexOptions so;
            {
                py::gil_scoped_release release;
                result = firefly::SimplexSolver::solve(pre, so);
            }
            result.solver_used = "simplex_fallback";
        }
    }

    } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg.find("Infeasible") != std::string::npos) {
            result.status = firefly::SolveStatus::INFEASIBLE;
        } else if (msg.find("Unbounded") != std::string::npos) {
            result.status = firefly::SolveStatus::UNBOUNDED;
        } else {
            result.status = firefly::SolveStatus::ERROR;
        }
        result.message = msg;
    }
    
    PySolveResult py_res = to_py_result(result);
    py_res.presolve_stats.original_rows = static_cast<int>(pre.stats.original_rows);
    py_res.presolve_stats.original_cols = static_cast<int>(pre.stats.original_cols);
    py_res.presolve_stats.reduced_rows = static_cast<int>(pre.stats.reduced_rows);
    py_res.presolve_stats.reduced_cols = static_cast<int>(pre.stats.reduced_cols);
    py_res.presolve_stats.rows_removed = static_cast<int>(pre.stats.rows_removed);
    py_res.presolve_stats.bounds_tightened = static_cast<int>(pre.stats.bounds_tightened);
    py_res.presolve_stats.variables_fixed = static_cast<int>(pre.stats.variables_fixed);
    py_res.presolve_stats.scaling_applied = static_cast<int>(pre.stats.scaling_applied);

    return py_res;
}

// ---------------------------------------------------------------------------
// parse_mps / parse_mps_string
// ---------------------------------------------------------------------------
static SparseProblem py_parse_mps(const std::string& filepath) {
    firefly::Problem fp = firefly::MPSParser::parse(filepath);
    return from_firefly_problem(fp);
}

static SparseProblem py_parse_mps_string(const std::string& content) {
    // Write to a temp file, parse, delete
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "firefly_tmp.mps";
    {
        FILE* f = fopen(tmp.string().c_str(), "w");
        if (!f) throw std::runtime_error("Cannot open temp file for MPS parsing");
        fwrite(content.data(), 1, content.size(), f);
        fclose(f);
    }
    firefly::Problem fp = firefly::MPSParser::parse(tmp.string());
    fs::remove(tmp);
    return from_firefly_problem(fp);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_firefly_solver, m) {
    m.doc() = "Firefly LP/MILP/QP solver — pybind11 bindings";

    // -- SparseProblem -------------------------------------------------------
    py::class_<SparseProblem>(m, "SparseProblem",
        "Optimization problem in CSR (Compressed Sparse Row) format.\n\n"
        "Constraint matrix row i covers row_ptr[i]..row_ptr[i+1]-1 entries in\n"
        "col_idx / values.  Row senses: 'L' (<=), 'G' (>=), 'E' (==), 'N' (free).")
        .def(py::init<>())
        .def_readwrite("name",              &SparseProblem::name)
        .def_readwrite("num_vars",          &SparseProblem::num_vars)
        .def_readwrite("num_constrs",       &SparseProblem::num_constrs)
        .def_readwrite("obj_coeffs",        &SparseProblem::obj_coeffs)
        .def_readwrite("row_ptr",           &SparseProblem::row_ptr)
        .def_readwrite("col_idx",           &SparseProblem::col_idx)
        .def_readwrite("values",            &SparseProblem::values)
        .def_readwrite("row_senses",        &SparseProblem::row_senses)
        .def_readwrite("rhs",               &SparseProblem::rhs)
        .def_readwrite("var_lower_bounds",  &SparseProblem::var_lower_bounds)
        .def_readwrite("var_upper_bounds",  &SparseProblem::var_upper_bounds)
        .def_readwrite("is_integer",        &SparseProblem::is_integer)
        .def("__repr__", [](const SparseProblem& sp) {
            return "<SparseProblem name='" + sp.name
                 + "' vars=" + std::to_string(sp.num_vars)
                 + " constrs=" + std::to_string(sp.num_constrs) + ">";
        });

    // -- PresolveStats ---------------------------------------------------------
    py::class_<PyPresolveStats>(m, "PresolveStats",
        "Presolve statistics returned as part of SolveResult.")
        .def(py::init<>())
        .def_readwrite("original_rows",    &PyPresolveStats::original_rows)
        .def_readwrite("original_cols",    &PyPresolveStats::original_cols)
        .def_readwrite("reduced_rows",     &PyPresolveStats::reduced_rows)
        .def_readwrite("reduced_cols",     &PyPresolveStats::reduced_cols)
        .def_readwrite("rows_removed",     &PyPresolveStats::rows_removed)
        .def_readwrite("bounds_tightened", &PyPresolveStats::bounds_tightened)
        .def_readwrite("variables_fixed",  &PyPresolveStats::variables_fixed)
        .def_readwrite("scaling_applied",  &PyPresolveStats::scaling_applied);

    // -- SolveResult ---------------------------------------------------------
    py::class_<PySolveResult>(m, "SolveResult",
        "Result returned by solve().")
        .def(py::init<>())
        .def_readwrite("status",          &PySolveResult::status,
            "One of: OPTIMAL, INFEASIBLE, UNBOUNDED, FEASIBLE, TIME_LIMIT, NODE_LIMIT, ERROR")
        .def_readwrite("solver_used",     &PySolveResult::solver_used,
            "The engine that produced this result (e.g., pdlp, simplex, simplex_fallback)")
        .def_readwrite("objective",       &PySolveResult::objective,
            "Optimal objective value (valid when status is OPTIMAL or FEASIBLE)")
        .def_readwrite("solution",        &PySolveResult::solution,
            "Primal solution vector (original variable order)")
        .def_readwrite("dual_solution",   &PySolveResult::dual_solution,
            "Dual solution vector (original row order)")
        .def_readwrite("primal_dual_gap", &PySolveResult::primal_dual_gap)
        .def_readwrite("wall_time_ms",    &PySolveResult::wall_time_ms,
            "Total solver wall time in milliseconds")
        .def_readwrite("iterations",      &PySolveResult::iterations,
            "Total iterations (phase1 + phase2 for simplex, PDLP iterations, or B&B nodes)")
        .def_readwrite("presolve_stats",  &PySolveResult::presolve_stats,
            "Presolve statistics")
        .def_readwrite("message",         &PySolveResult::message,
            "Detailed message (populated on ERROR)")
        .def("__repr__", [](const PySolveResult& r) {
            return "<SolveResult status=" + r.status
                 + " obj=" + std::to_string(r.objective)
                 + " iters=" + std::to_string(r.iterations) + ">";
        });

    // -- solve() -------------------------------------------------------------
    m.def(
        "solve",
        &py_solve,
        py::arg("problem"),
        py::arg("method")             = "auto",
        py::arg("gpu")                = true,
        py::arg("iteration_callback") = py::none(),
        R"doc(
Solve an optimization problem.

Parameters
----------
problem : SparseProblem
    The problem to solve.
method : str, optional
    "auto"    — PDLP with GPU if compiled in, simplex otherwise
    "simplex" — force revised simplex (LP only)
    "pdlp"    — force PDLP (primal-dual LP / first-order)
    Ignored for MILP (branch-and-bound is always used).
gpu : bool, optional
    Use CUDA-accelerated PDLP when True (ignored if CUDA not compiled in).
iteration_callback : callable or None, optional
    Called periodically during solving with signature:
        callback(iteration: int, primal_obj: float, dual_obj: float, elapsed_ms: float)
    The GIL is re-acquired before each call, so it is safe to update Python
    data structures or asyncio queues from within the callback.

Returns
-------
SolveResult
)doc");

    // -- parse_mps() ---------------------------------------------------------
    m.def(
        "parse_mps",
        &py_parse_mps,
        py::arg("filepath"),
        "Parse an MPS file from disk and return a SparseProblem.");

    // -- parse_mps_string() --------------------------------------------------
    m.def(
        "parse_mps_string",
        &py_parse_mps_string,
        py::arg("content"),
        "Parse an MPS file given as a string (written to a temp file internally).");
}
