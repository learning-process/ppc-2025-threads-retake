#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/agafeev_s_matmul_fox_algo/include/ops_seq.hpp"

namespace {
std::vector<double> MatrixMultiply(const std::vector<double>& a, const std::vector<double>& b, int row_col_size) {
  std::vector<double> c(row_col_size * row_col_size, 0);

  for (int i = 0; i < row_col_size; ++i) {
    for (int j = 0; j < row_col_size; ++j) {
      for (int k = 0; k < row_col_size; ++k) {
        c[(i * row_col_size) + j] += a[(i * row_col_size) + k] * b[(k * row_col_size) + j];
      }
    }
  }

  return c;
}

std::vector<double> CreateRandomMatrix(int size) {
  auto rand_gen = std::mt19937(time(nullptr));
  std::uniform_real_distribution<double> dist((double)-1e3, (double)1e3);
  std::vector<double> matrix(size);
  for (unsigned int i = 0; i < matrix.size(); i++) {
    matrix[i] = dist(rand_gen);
  }

  return matrix;
}
}  // namespace

TEST(agafeev_s_matmul_fox_algo_seq, matmul_9x9) {
  const int n = 9;
  size_t block_size = 3;
  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(n * n);
  std::vector<double> in_matrix2 = CreateRandomMatrix(n * n);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(n);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(block_size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  auto right_answer = MatrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_matmul_fox_algo_seq, matmul_3x3) {
  const int n = 3;
  size_t block_size = 1;
  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(n * n);
  std::vector<double> in_matrix2 = CreateRandomMatrix(n * n);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(n);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(block_size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  auto right_answer = MatrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_matmul_fox_algo_seq, matmul_16x16) {
  const int n = 16;
  size_t block_size = 4;
  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(n * n);
  std::vector<double> in_matrix2 = CreateRandomMatrix(n * n);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(n);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(block_size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  auto right_answer = MatrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_matmul_fox_algo_seq, wrong_input) {
  const int n = 3;
  size_t block_size = 5;
  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(n * n);
  std::vector<double> in_matrix2 = CreateRandomMatrix(n * n);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(n);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(block_size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, false);
}