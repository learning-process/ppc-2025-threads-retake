# tasks/seq/yasakova_t_sort/CMakeLists.txt
set(PROJ yasakova_t_sort_seq)

add_library(${PROJ}
    src/radix_double_seq.cpp
)
target_include_directories(${PROJ} PUBLIC include)

add_executable(${PROJ}_func_tests
    func_tests/radix_double_seq_gtest.cpp
)
target_link_libraries(${PROJ}_func_tests
    ${PROJ} gtest gtest_main
)
gtest_discover_tests(${PROJ}_func_tests)

add_executable(${PROJ}_perf_tests
    perf_tests/perf_seq.cpp
)
target_link_libraries(${PROJ}_perf_tests
    ${PROJ} gtest gtest_main
)
gtest_discover_tests(${PROJ}_perf_tests)
