#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

using namespace tarakanov_d_fox_algorithm_seq;

TEST(tarakanov_d_fox_algorithm_test_seq, test_matmul_2x2) {
  std::vector<double> matrixA = {1, 2, 3, 4};
  std::vector<double> matrixB = {5, 6, 7, 8};
  std::vector<double> matrixC(4, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data->inputs_count.emplace_back(matrixA.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data->inputs_count.emplace_back(matrixB.size() * sizeof(double));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrixC.data()));
  task_data->outputs_count.emplace_back(matrixC.size() * sizeof(double));

  TaskSequential task(task_data);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  std::vector<double> expected = {19, 22, 43, 50};
  EXPECT_EQ(matrixC, expected);
}

TEST(tarakanov_d_fox_algorithm_test_seq, test_matmul_3x3) {
  std::vector<double> matrixA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> matrixB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> matrixC(9, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data->inputs_count.emplace_back(matrixA.size() * sizeof(double));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data->inputs_count.emplace_back(matrixB.size() * sizeof(double));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrixC.data()));
  task_data->outputs_count.emplace_back(matrixC.size() * sizeof(double));

  TaskSequential task(task_data);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  std::vector<double> expected = {30, 24, 18, 84, 69, 54, 138, 114, 90};
  EXPECT_EQ(matrixC, expected);
}