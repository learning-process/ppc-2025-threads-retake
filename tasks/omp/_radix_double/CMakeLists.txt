
cmake_minimum_required(VERSION 3.16)

# Библиотека реализации
add_library(bobylev_m_radix_double_omp STATIC
    src/radix_sort_double.cpp
)
target_include_directories(bobylev_m_radix_double_omp PUBLIC include)
target_compile_features(bobylev_m_radix_double_omp PUBLIC cxx_std_17)

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(bobylev_m_radix_double_omp PUBLIC OpenMP::OpenMP_CXX)

# Функциональные тесты
add_executable(bobylev_m_radix_double_omp_func_tests
    func_tests/test_radix_sort_double.cpp
)
target_link_libraries(bobylev_m_radix_double_omp_func_tests
    PRIVATE bobylev_m_radix_double_omp GTest::gtest_main
)
set_target_properties(bobylev_m_radix_double_omp_func_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Тесты на производительность (ровно 2)
add_executable(bobylev_m_radix_double_omp_perf_tests
    perf_tests/perf_radix_sort_double.cpp
)
target_link_libraries(bobylev_m_radix_double_omp_perf_tests
    PRIVATE bobylev_m_radix_double_omp GTest::gtest_main
)
set_target_properties(bobylev_m_radix_double_omp_perf_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Регистрация тестов
include(GoogleTest)
gtest_discover_tests(bobylev_m_radix_double_omp_func_tests)
gtest_discover_tests(bobylev_m_radix_double_omp_perf_tests)
