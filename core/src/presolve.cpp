#include "firefly/presolve.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace firefly {

namespace {

const double PRESOLVE_TOL = 1e-9;

bool is_zero(double val) {
    return std::abs(val) < PRESOLVE_TOL;
}

bool is_inf(double val) {
    return std::isinf(val) || std::abs(val) >= 1e19;
}

bool is_eq(double a, double b) {
    if (is_inf(a) && is_inf(b) && (a > 0) == (b > 0)) return true;
    if (is_inf(a) || is_inf(b)) return false;
    return std::abs(a - b) < PRESOLVE_TOL;
}

} // namespace

PresolvedProblem Presolver::presolve(const Problem& orig, const PresolveOptions& options) {
    PresolvedProblem result;
    result.stats.original_rows = orig.row_names.size();
    result.stats.original_cols = orig.col_names.size();
    
    size_t num_rows = orig.row_names.size();
    size_t num_cols = orig.col_names.size();

    std::vector<bool> active_row(num_rows, true);
    std::vector<bool> active_col(num_cols, true);

    std::vector<std::unordered_map<size_t, double>> row_nz(num_rows);
    std::vector<std::unordered_map<size_t, double>> col_nz(num_cols);

    for (const auto& trip : orig.matrix) {
        if (!is_zero(trip.value)) {
            row_nz[trip.row][trip.col] += trip.value;
            col_nz[trip.col][trip.row] += trip.value;
        }
    }
    
    // Clean initial perfect cancellations (rare but possible)
    for(size_t i=0; i<num_rows; ++i) {
        for(auto it = row_nz[i].begin(); it != row_nz[i].end(); ) {
            if(is_zero(it->second)) {
                col_nz[it->first].erase(i);
                it = row_nz[i].erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<double> row_lower = orig.row_lower_bounds;
    std::vector<double> row_upper = orig.row_upper_bounds;
    std::vector<double> col_lower = orig.col_lower_bounds;
    std::vector<double> col_upper = orig.col_upper_bounds;
    std::vector<double> obj = orig.objective;
    
    auto count_active = [&]() {
        int ar = 0, ac = 0;
        for(bool b : active_row) if(b) ar++;
        for(bool b : active_col) if(b) ac++;
        return std::make_pair(ar, ac);
    };

    bool changed = true;
    size_t presolve_iters = 0;
    while (changed) {
        presolve_iters++;
        if (presolve_iters > 100) {
            throw std::runtime_error("Presolve infinite loop");
        }
        changed = false;

        auto before_fixed = count_active();


        // 1. Fixed variables substitution
        for (size_t j = 0; j < num_cols; ++j) {
            if (!active_col[j]) continue;
            
            if (col_lower[j] > col_upper[j] + PRESOLVE_TOL) {
                throw InfeasibleProblemException("Column " + orig.col_names[j] + " has conflicting bounds: L=" + std::to_string(col_lower[j]) + ", U=" + std::to_string(col_upper[j]));
            }
            
            if (options.enable_fixed_variable_substitution && is_eq(col_lower[j], col_upper[j])) {
                double val = col_lower[j];
                
                if (orig.is_integer[j]) {
                    if (std::abs(std::round(val) - val) > PRESOLVE_TOL) {
                        throw InfeasibleProblemException("Integer column " + orig.col_names[j] + " fixed to non-integer value " + std::to_string(val));
                    }
                    val = std::round(val);
                }
                
                result.postsolve.fixed_variables[j] = val;
                result.postsolve.objective_offset += obj[j] * val;
                result.stats.variables_fixed++;
                active_col[j] = false;
                changed = true;
                
                for (const auto& [r, coef] : col_nz[j]) {
                    if (!active_row[r]) continue;
                    double shift = coef * val;
                    if (!is_inf(row_lower[r])) row_lower[r] -= shift;
                    if (!is_inf(row_upper[r])) row_upper[r] -= shift;
                    
                    row_nz[r].erase(j);
                }
                col_nz[j].clear();
            }
        }
        auto after_fixed = count_active();
        std::cout << "[DEBUG] Fixed variables substitution: BEFORE (" << before_fixed.first << " rows, " << before_fixed.second << " cols) -> AFTER (" << after_fixed.first << " rows, " << after_fixed.second << " cols)\n";
        
        auto before_empty_single = count_active();

        // 2. Empty rows and Single-variable rows
        for (size_t i = 0; i < num_rows; ++i) {
            if (!active_row[i]) continue;
            
            if (row_lower[i] > row_upper[i] + PRESOLVE_TOL) {
                throw InfeasibleProblemException("Row " + orig.row_names[i] + " has conflicting bounds");
            }

            if (options.enable_empty_row_removal && row_nz[i].empty()) {
                if (row_lower[i] > PRESOLVE_TOL || row_upper[i] < -PRESOLVE_TOL) {
                    throw InfeasibleProblemException("Row " + orig.row_names[i] + " is empty but 0.0 is not within bounds [" + std::to_string(row_lower[i]) + ", " + std::to_string(row_upper[i]) + "]");
                }
                active_row[i] = false;
                result.stats.rows_removed++;
                changed = true;
            } else if (options.enable_singleton_row_tightening && row_nz[i].size() == 1) {
                auto it = row_nz[i].begin();
                size_t j = it->first;
                double coef = it->second;
                
                double nb_lower = -INF;
                double nb_upper = INF;
                
                if (coef > 0) {
                    if (!is_inf(row_lower[i])) nb_lower = row_lower[i] / coef;
                    if (!is_inf(row_upper[i])) nb_upper = row_upper[i] / coef;
                } else {
                    if (!is_inf(row_upper[i])) nb_lower = row_upper[i] / coef;
                    if (!is_inf(row_lower[i])) nb_upper = row_lower[i] / coef;
                }
                
                if (nb_lower > col_lower[j] + PRESOLVE_TOL) {
                    col_lower[j] = nb_lower;
                    result.stats.bounds_tightened++;
                    changed = true;
                }
                if (nb_upper < col_upper[j] - PRESOLVE_TOL) {
                    col_upper[j] = nb_upper;
                    result.stats.bounds_tightened++;
                    changed = true;
                }
                
                if (col_lower[j] > col_upper[j] + PRESOLVE_TOL) {
                    if (orig.row_names[i] == "DWASE") {
                        std::cout << "[DEBUG] Row DWASE has " << row_nz[i].size() << " variables:\n";
                        for (const auto& kv : row_nz[i]) {
                            std::cout << "  Var: " << orig.col_names[kv.first] << " Coef: " << kv.second << "\n";
                        }
                    }
                    throw InfeasibleProblemException("Row " + orig.row_names[i] + " causes conflicting bounds for " + orig.col_names[j] + ": L=" + std::to_string(col_lower[j]) + ", U=" + std::to_string(col_upper[j]));
                }
                
                active_row[i] = false;
                result.stats.rows_removed++;
                changed = true;
                
                col_nz[j].erase(i);
                row_nz[i].clear();
            }
        }
        auto after_empty_single = count_active();
        std::cout << "[DEBUG] Empty/Single-variable rows: BEFORE (" << before_empty_single.first << " rows, " << before_empty_single.second << " cols) -> AFTER (" << after_empty_single.first << " rows, " << after_empty_single.second << " cols)\n";
        
        auto before_empty_vars = count_active();
        
        // 3. Remove empty variables
        for (size_t j = 0; j < num_cols; ++j) {
            if (!active_col[j]) continue;
            
            if (options.enable_empty_col_removal && col_nz[j].empty()) {
                if (is_zero(obj[j])) {
                    double fix_val = 0.0;
                    if (0.0 < col_lower[j]) fix_val = col_lower[j];
                    if (0.0 > col_upper[j]) fix_val = col_upper[j];
                    
                    if (orig.is_integer[j]) fix_val = std::round(fix_val);
                    
                    result.postsolve.fixed_variables[j] = fix_val;
                    result.postsolve.objective_offset += obj[j] * fix_val;
                    result.stats.variables_fixed++;
                    active_col[j] = false;
                    changed = true;
                } else {
                    bool min_dir = orig.minimize ? (obj[j] > 0) : (obj[j] < 0);
                    if (min_dir) {
                        if (is_inf(col_lower[j])) throw InfeasibleProblemException("Unbounded: Cost pulls to -INF but no lower bound");
                        result.postsolve.fixed_variables[j] = col_lower[j];
                    } else {
                        if (is_inf(col_upper[j])) throw InfeasibleProblemException("Unbounded: Cost pulls to +INF but no upper bound");
                        result.postsolve.fixed_variables[j] = col_upper[j];
                    }
                    result.postsolve.objective_offset += obj[j] * result.postsolve.fixed_variables[j];
                    result.stats.variables_fixed++;
                    active_col[j] = false;
                    changed = true;
                }
            }
        }
        auto after_empty_vars = count_active();
        std::cout << "[DEBUG] Remove empty variables: BEFORE (" << before_empty_vars.first << " rows, " << before_empty_vars.second << " cols) -> AFTER (" << after_empty_vars.first << " rows, " << after_empty_vars.second << " cols)\n";
    }
    
    auto before_scaling = count_active();
    
    // Coefficient Scaling (Equilibration)
    std::vector<double> r_scale(num_rows, 1.0);
    std::vector<double> c_scale(num_cols, 1.0);
    if (options.enable_scaling) {
    
    // Row scaling
    for (size_t i = 0; i < num_rows; ++i) {
        if (!active_row[i]) continue;
        double max_abs = 0.0;
        for (const auto& [j, v] : row_nz[i]) {
            max_abs = std::max(max_abs, std::abs(v));
        }
        if (max_abs > PRESOLVE_TOL && !is_eq(max_abs, 1.0)) {
            r_scale[i] = max_abs;
            if (!is_inf(row_lower[i])) row_lower[i] /= max_abs;
            if (!is_inf(row_upper[i])) row_upper[i] /= max_abs;
            for (auto& [j, v] : row_nz[i]) {
                v /= max_abs;
                col_nz[j][i] = v; // Update dual representation
            }
            result.stats.scaling_applied++;
        }
    }
    
    // Column scaling
    for (size_t j = 0; j < num_cols; ++j) {
        if (!active_col[j]) continue;
        
        // Don't scale integer variables - scaling destroys integrality
        if (orig.is_integer[j]) continue;
        
        double max_abs = 0.0;
        for (const auto& [i, v] : col_nz[j]) {
            max_abs = std::max(max_abs, std::abs(v));
        }
        if (max_abs > PRESOLVE_TOL && !is_eq(max_abs, 1.0)) {
            c_scale[j] = max_abs;
            if (!is_inf(col_lower[j])) col_lower[j] *= max_abs; // x' = x / scale => L' = L / scale? Wait!
            // If we divide x by scale: x_new = x_orig / scale => x_orig = x_new * scale.
            // Bounds on x_orig: L <= x_orig <= U  =>  L <= x_new * scale <= U  =>  L/scale <= x_new <= U/scale
            // So col_lower[j] /= max_abs
            // Wait! The previous line col_lower[j] *= max_abs was WRONG.
            // Let's fix this mathematically:
            // L <= x <= U
            // We want new variable y = x * a. Then y/a = x.
            // But standard scaling multiplies coefficients: matrix column is divided by scale, so variable is multiplied by scale.
            // A_j * x_j = (A_j / scale) * (x_j * scale) = A'_j * x'_j
            // So x'_j = x_j * scale.
            // Then L' = L * scale, U' = U * scale.
            // Postsolve: x_j = x'_j / scale.
            // Let's implement A'_j = A_j / scale. Then x'_j = x_j * scale.
            
            if (!is_inf(col_lower[j])) col_lower[j] *= max_abs;
            if (!is_inf(col_upper[j])) col_upper[j] *= max_abs;
            obj[j] /= max_abs; // cost * x_j = (cost / scale) * (x_j * scale)
            
            for (auto& [i, v] : col_nz[j]) {
                v /= max_abs;
                row_nz[i][j] = v; // Update dual representation
            }
            result.stats.scaling_applied++;
        }
    }
    }
    auto after_scaling = count_active();
    std::cout << "[DEBUG] Coefficient Scaling: BEFORE (" << before_scaling.first << " rows, " << before_scaling.second << " cols) -> AFTER (" << after_scaling.first << " rows, " << after_scaling.second << " cols)\n";
    
    // Build Reduced Problem
    Problem reduced;
    reduced.name = orig.name + "_presolved";
    reduced.minimize = orig.minimize;
    reduced.objective_name = orig.objective_name;
    
    result.postsolve.col_scale_factors.assign(num_cols, 1.0);
    for (size_t j = 0; j < num_cols; ++j) {
        if (!orig.is_integer[j]) {
            result.postsolve.col_scale_factors[j] = 1.0 / c_scale[j]; // x = x' / c_scale
        }
    }
    result.postsolve.row_scale_factors = r_scale;
    
    std::vector<size_t> r_map(num_rows, static_cast<size_t>(-1));
    std::vector<size_t> c_map(num_cols, static_cast<size_t>(-1));
    
    size_t new_r = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        if (active_row[i]) {
            r_map[i] = new_r++;
            reduced.row_names.push_back(orig.row_names[i]);
            reduced.row_to_index[orig.row_names[i]] = r_map[i];
            reduced.row_lower_bounds.push_back(row_lower[i]);
            reduced.row_upper_bounds.push_back(row_upper[i]);
            if (i < orig.row_senses.size()) {
                reduced.row_senses.push_back(orig.row_senses[i]); // Original sense is kept for record, though L/U bounds rule now
            } else {
                reduced.row_senses.push_back('E'); // fallback
            }
            result.postsolve.reduced_row_to_orig_row.push_back(i);
        }
    }
    
    size_t new_c = 0;
    for (size_t j = 0; j < num_cols; ++j) {
        if (active_col[j]) {
            c_map[j] = new_c++;
            reduced.col_names.push_back(orig.col_names[j]);
            reduced.col_to_index[orig.col_names[j]] = c_map[j];
            reduced.col_lower_bounds.push_back(col_lower[j]);
            reduced.col_upper_bounds.push_back(col_upper[j]);
            reduced.is_integer.push_back(orig.is_integer[j]);
            reduced.objective.push_back(obj[j]);
            result.postsolve.reduced_col_to_orig_col.push_back(j);
        }
    }
    
    for (size_t i = 0; i < num_rows; ++i) {
        if (!active_row[i]) continue;
        for (const auto& [j, v] : row_nz[i]) {
            reduced.matrix.push_back({r_map[i], c_map[j], v});
        }
    }
    
    result.problem = std::move(reduced);
    result.stats.reduced_rows = new_r;
    result.stats.reduced_cols = new_c;
    
    return result;
}

std::vector<double> Presolver::postsolve_primal(const PresolvedProblem& pre, const std::vector<double>& reduced_primal) {
    if (reduced_primal.size() != pre.stats.reduced_cols) {
        throw std::invalid_argument("Reduced primal solution size mismatch");
    }
    
    std::vector<double> orig_primal(pre.stats.original_cols, 0.0);
    
    // 1. Map reduced variables back and apply col_scale_factors
    for (size_t rj = 0; rj < reduced_primal.size(); ++rj) {
        size_t orig_j = pre.postsolve.reduced_col_to_orig_col[rj];
        double scale = pre.postsolve.col_scale_factors[orig_j];
        orig_primal[orig_j] = reduced_primal[rj] * scale;
    }
    
    // 2. Fill in fixed variables
    for (const auto& [orig_j, val] : pre.postsolve.fixed_variables) {
        orig_primal[orig_j] = val;
    }
    
    return orig_primal;
}

std::vector<double> Presolver::postsolve_dual(const PresolvedProblem& pre, const std::vector<double>& reduced_dual) {
    if (reduced_dual.size() != pre.stats.reduced_rows) {
        throw std::invalid_argument("Reduced dual solution size mismatch");
    }
    
    std::vector<double> orig_dual(pre.stats.original_rows, 0.0);
    
    // Map reduced rows back and apply row_scale_factors (divide)
    for (size_t ri = 0; ri < reduced_dual.size(); ++ri) {
        size_t orig_i = pre.postsolve.reduced_row_to_orig_row[ri];
        double scale = pre.postsolve.row_scale_factors[orig_i];
        orig_dual[orig_i] = reduced_dual[ri] / scale;
    }
    
    return orig_dual;
}

} // namespace firefly
