#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

namespace {
struct LimitsStruct {
  double min_val;
  double max_val;
};

std::vector<double> GenerateRandomMatrix(size_t n, LimitsStruct limits) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(limits.min_val, limits.max_val);

  std::vector<double> matrix(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}
}  // namespace

TEST(tarakanov_d_fox_algorithm_seq, test_random_5x5) {
  constexpr size_t kN = 5;

  LimitsStruct limits{.min_val = -10.0, .max_val = 10.0};
  std::vector<double> a = GenerateRandomMatrix(kN, limits);
  std::vector<double> b = GenerateRandomMatrix(kN, limits);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      for (size_t k = 0; k < kN; ++k) {
        expected[(i * kN) + j] += a[(i * kN) + k] * b[(k * kN) + j];
      }
    }
  }

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  tarakanov_d_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
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