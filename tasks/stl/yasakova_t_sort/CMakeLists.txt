# tasks/stl/yasakova_t_sort/CMakeLists.txt
set(PROJ yasakova_t_sort_stl)

add_library(${PROJ}
    src/radix_double_stl.cpp
)
target_include_directories(${PROJ} PUBLIC include)

target_sources(stl_tests PRIVATE
    func_tests/radix_double_stl_gtest.cpp
    perf_tests/perf_stl.cpp
)

target_link_libraries(stl_tests PRIVATE ${PROJ})
