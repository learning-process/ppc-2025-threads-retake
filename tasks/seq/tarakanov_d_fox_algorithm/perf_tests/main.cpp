#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

using namespace tarakanov_d_fox_algorithm_seq;

void test_matmul_performance(size_t size) {
  // Создаем матрицы size x size, заполненные случайными значениями
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  std::vector<double> matrixA(size * size);
  std::vector<double> matrixB(size * size);
  for (auto& val : matrixA) val = dis(gen);
  for (auto& val : matrixB) val = dis(gen);

  // Создаем task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data->inputs_count.emplace_back(matrixA.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data->inputs_count.emplace_back(matrixB.size() * sizeof(double));
  task_data->outputs.emplace_back(nullptr);
  task_data->outputs_count.emplace_back(0);

  // Создаем задачу
  TaskSequential task(task_data);
  EXPECT_TRUE(task.Validation());

  // Выполняем задачу и измеряем время
  auto start = std::chrono::high_resolution_clock::now();
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Matrix size: " << size << "x" << size << ", Time: " << duration.count() << " ms" << std::endl;
}

TEST(tasksequential_perf_test, test_matmul_100x100) { test_matmul_performance(100); }

TEST(tasksequential_perf_test, test_matmul_500x500) { test_matmul_performance(500); }

TEST(tasksequential_perf_test, test_matmul_1000x1000) { test_matmul_performance(1000); }