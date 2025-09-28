# tasks/omp/yasakova_t_sort/CMakeLists.txt
set(PROJ yasakova_t_sort_omp)

add_library(${PROJ}
    src/radix_double_omp.cpp
)
target_include_directories(${PROJ} PUBLIC include)

target_sources(omp_tests PRIVATE
    func_tests/radix_double_omp_gtest.cpp
    perf_tests/perf_omp.cpp
)

target_link_libraries(omp_tests PRIVATE ${PROJ})
