# tasks/seq/yasakova_t_sort/CMakeLists.txt
# Библиотека с реализацией алгоритма
set(PROJ yasakova_t_sort_seq)

add_library(${PROJ}
    src/radix_double_seq.cpp
)
target_include_directories(${PROJ} PUBLIC include)

# Подключаем наши тестовые файлы к ОБЩЕМУ раннеру seq_tests
target_sources(seq_tests PRIVATE
    func_tests/radix_double_seq_gtest.cpp
    perf_tests/perf_seq.cpp
)

# Даём раннеру доступ к нашей библиотеке
target_link_libraries(seq_tests PRIVATE ${PROJ})
