# tasks/tbb/yasakova_t_sort/CMakeLists.txt
set(PROJ yasakova_t_sort_tbb)

add_library(${PROJ}
    src/radix_double_tbb.cpp
)
target_include_directories(${PROJ} PUBLIC include)

target_sources(tbb_tests PRIVATE
    func_tests/radix_double_tbb_gtest.cpp
    perf_tests/perf_tbb.cpp
)

target_link_libraries(tbb_tests PRIVATE ${PROJ})
