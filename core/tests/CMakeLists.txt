add_executable(firefly_tests main.cpp test_mps_parser.cpp)

# Link against the core library
target_link_libraries(firefly_tests PRIVATE firefly_core)

# Register the test runner
add_test(NAME FireflyTests COMMAND firefly_tests)

add_executable(run_real_files run_real_files.cpp)
target_link_libraries(run_real_files PRIVATE firefly_core)

add_executable(test_presolve test_presolve.cpp)
target_link_libraries(test_presolve PRIVATE firefly_core)
add_test(NAME PresolveTests COMMAND test_presolve)

add_executable(test_presolve_rules test_presolve_rules.cpp)
target_link_libraries(test_presolve_rules PRIVATE firefly_core)
add_test(NAME PresolveRulesTests COMMAND test_presolve_rules)

add_executable(test_simplex test_simplex.cpp)
target_link_libraries(test_simplex PRIVATE firefly_core)
add_test(NAME SimplexTests COMMAND test_simplex)

add_executable(test_pdlp test_pdlp.cpp)
target_link_libraries(test_pdlp PRIVATE firefly_core)
add_test(NAME PDLPTests COMMAND test_pdlp)

add_executable(test_pdlp_concurrency test_pdlp_concurrency.cpp)
target_link_libraries(test_pdlp_concurrency PRIVATE firefly_core)
add_test(NAME test_pdlp_concurrency COMMAND test_pdlp_concurrency)

add_executable(test_pdlp_cross_validation test_pdlp_cross_validation.cpp)
target_link_libraries(test_pdlp_cross_validation PRIVATE firefly_core)
add_test(NAME test_pdlp_cross_validation COMMAND test_pdlp_cross_validation)

add_executable(test_pdlp_stability test_pdlp_stability.cpp)
target_link_libraries(test_pdlp_stability PRIVATE firefly_core)
add_test(NAME test_pdlp_stability COMMAND test_pdlp_stability)

add_executable(test_gpu_unavailability test_gpu_unavailability.cpp)
target_link_libraries(test_gpu_unavailability PRIVATE firefly_core)
add_test(NAME test_gpu_unavailability COMMAND test_gpu_unavailability)

add_executable(test_pdlp_ruiz test_pdlp_ruiz.cpp)
target_link_libraries(test_pdlp_ruiz PRIVATE firefly_core)
add_test(NAME test_pdlp_ruiz COMMAND test_pdlp_ruiz)

add_executable(report_ruiz report_ruiz.cpp)
target_link_libraries(report_ruiz PRIVATE firefly_core)

add_executable(benchmark_gpu benchmark_gpu.cpp)
target_link_libraries(benchmark_gpu PRIVATE firefly_core)

add_executable(test_branch_and_bound test_branch_and_bound.cpp)
target_link_libraries(test_branch_and_bound PRIVATE firefly_core)
add_test(NAME test_branch_and_bound COMMAND test_branch_and_bound)

add_executable(test_repro test_repro.cpp)
target_link_libraries(test_repro PRIVATE firefly_core)
