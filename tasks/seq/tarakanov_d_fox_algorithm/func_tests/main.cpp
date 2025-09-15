#include <gtest/gtest.h>
#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"
#include "core/task/include/task.hpp"
#include <vector>
#include <algorithm>

using namespace tarakanov_d_fox_algorithm_seq;

TEST(tasksequential_test, test_matmul_2x2) {
  // Создаем матрицы 2x2
  std::vector<double> matrixA = {1, 2, 3, 4};
  std::vector<double> matrixB = {5, 6, 7, 8};
  
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
  
  // Выполняем задачу
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  
  // Извлекаем результат
  double* result = reinterpret_cast<double*>(task_data->outputs[0]);
  std::vector<double> expected = {19, 22, 43, 50};
  std::vector<double> actual(result, result + expected.size());
  
  EXPECT_EQ(actual, expected);
}

TEST(tasksequential_test, test_matmul_3x3) {
  // Создаем матрицы 3x3
  std::vector<double> matrixA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> matrixB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  
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
  
  // Выполняем задачу
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  
  // Извлекаем результат
  double* result = reinterpret_cast<double*>(task_data->outputs[0]);
  std::vector<double> expected = {
    12, 11, 10,
    30, 29, 28,
    48, 47, 46
  };
  std::vector<double> actual(result, result + expected.size());
  
  EXPECT_EQ(actual, expected);
}