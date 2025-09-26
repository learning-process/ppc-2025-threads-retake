#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/guseynov_e_sparse_matrix_multiply_crs/include/ops_omp.hpp"

using guseynov_e_sparse_matrix_multiply_crs_omp::CRSMatrix;
using guseynov_e_sparse_matrix_multiply_crs_omp::SparseMatMultOMP;

struct MultiplyParams {
  const CRSMatrix& left;
  const CRSMatrix& right;
};

namespace {
CRSMatrix RunTask(const MultiplyParams& params) {
  const auto& a = params.left;
  const auto& b = params.right;
  CRSMatrix result;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<CRSMatrix*>(&a)));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<CRSMatrix*>(&b)));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  SparseMatMultOMP task(task_data);
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
  return result;
}

void CompareDimensions(const CRSMatrix& result, const CRSMatrix& expected) {
  ASSERT_EQ(result.n_rows, expected.n_rows);
  ASSERT_EQ(result.n_cols, expected.n_cols);
}

void CompareStructure(const CRSMatrix& result, const CRSMatrix& expected) {
  ASSERT_EQ(result.pointer, expected.pointer);
  ASSERT_EQ(result.col_indexes, expected.col_indexes);
}

void CompareValues(const CRSMatrix& result, const CRSMatrix& expected) {
  ASSERT_EQ(result.non_zero_values.size(), expected.non_zero_values.size());
  for (size_t i = 0; i < expected.non_zero_values.size(); i++) {
    double diff = expected.non_zero_values[i] - result.non_zero_values[i];
    ASSERT_NEAR(0.0, diff, 1e-3);
  }
}

void CompareMatrices(const CRSMatrix& result, const CRSMatrix& expected) {
  CompareDimensions(result, expected);
  CompareStructure(result, expected);
  CompareValues(result, expected);
}
}  // namespace

TEST(guseynov_e_sparse_matrix_multiply_crs_omp, test_square_matrix_by_itself) {
  CRSMatrix a{.n_rows = 5,
              .n_cols = 5,
              .non_zero_values = {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5},
              .pointer = {0, 3, 5, 8, 11, 13},
              .col_indexes = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4}};
  CRSMatrix b{.n_rows = 5,
              .n_cols = 5,
              .non_zero_values = {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5},
              .pointer = {0, 3, 5, 8, 11, 13},
              .col_indexes = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4}};

  CRSMatrix expected{.n_rows = 5,
                     .n_cols = 5,
                     .non_zero_values = {15, -6, -6, -24, -12, 27, 6, -24, 32, 28, 66, -4, -32, 4, 22, 73, 8, -16, 25},
                     .pointer = {0, 4, 7, 12, 17, 19},
                     .col_indexes = {0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 4}};

  CompareMatrices(RunTask({.left = a, .right = b}), expected);
}

TEST(guseynov_e_sparse_matrix_multiply_crs_omp, test_square_matrix) {
  CRSMatrix a{.n_rows = 4,
              .n_cols = 4,
              .non_zero_values = {1, 5, 2, 3, 4, 1, 2},
              .pointer = {0, 2, 4, 6, 7},
              .col_indexes = {0, 2, 0, 1, 0, 3, 2}};
  CRSMatrix b{.n_rows = 4,
              .n_cols = 5,
              .non_zero_values = {5, 3, 7, 6, 8, 3, 2},
              .pointer = {0, 2, 4, 5, 7},
              .col_indexes = {0, 1, 0, 2, 3, 0, 2}};

  CRSMatrix expected{.n_rows = 4,
                     .n_cols = 5,
                     .non_zero_values = {5, 3, 40, 31, 6, 18, 23, 12, 2, 16},
                     .pointer = {0, 3, 6, 9, 10},
                     .col_indexes = {0, 1, 3, 0, 1, 2, 0, 1, 2, 3}};

  CompareMatrices(RunTask({.left = a, .right = b}), expected);
}

TEST(guseynov_e_sparse_matrix_multiply_crs_omp, test_non_square_matrix) {
  CRSMatrix a{.n_rows = 2, .n_cols = 3, .non_zero_values = {1, 2, 3}, .pointer = {0, 2, 3}, .col_indexes = {0, 2, 1}};
  CRSMatrix b{
      .n_rows = 3, .n_cols = 3, .non_zero_values = {4, 7, 6, 8}, .pointer = {0, 2, 3, 4}, .col_indexes = {0, 2, 1, 1}};

  CRSMatrix expected{
      .n_rows = 2, .n_cols = 3, .non_zero_values = {4, 16, 7, 18}, .pointer = {0, 3, 4}, .col_indexes = {0, 1, 2, 1}};

  CompareMatrices(RunTask({.left = a, .right = b}), expected);
}