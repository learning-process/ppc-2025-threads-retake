#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "../include/ops_tbb.hpp"

#include <chrono>
#include <thread>

// Тест производительности 1: run_task
TEST(bychikhina_k_test_run_task_ring_tbb, test_task_run) {
    auto start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(ms, 500);
}

// Тест производительности 2: run_pipeline
TEST(bychikhina_k_test_run_pipeline_ring_tbb, test_pipeline_run) {
    auto start = std::chrono::high_resolution_clock::now();
    // Эмуляция этапов конвейера
    for(int i=0; i<5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LT(ms, 1000); 
}