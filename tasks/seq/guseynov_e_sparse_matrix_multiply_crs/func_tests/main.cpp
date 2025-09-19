#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/guseynov_e_sparse_matrix_multiply_crs/include/ops_seq.hpp"

using guseynov_e_sparse_matrix_multiply_crs::CRSMatrix;
using guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial;

static CRSMatrix runTask(const CRSMatrix& a, const CRSMatrix& b) {
  CRSMatrix Result;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<CRSMatrix*>(&a)));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<CRSMatrix*>(&b)));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&Result));

  SparseMatMultSequantial task(taskData);
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
  return Result;
}

static void compareMatrices(const CRSMatrix& result, const CRSMatrix& expected) {
  ASSERT_EQ(result.n_rows, expected.n_rows);
  ASSERT_EQ(result.n_cols, expected.n_cols);
  ASSERT_EQ(result.pointer, expected.pointer);
  ASSERT_EQ(result.col_indexes, expected.col_indexes);
  ASSERT_EQ(result.non_zero_values.size(), expected.non_zero_values.size());
  for (size_t i = 0; i < expected.non_zero_values.size(); i++) {
    double diff = expected.non_zero_values[i] - result.non_zero_values[i];
    ASSERT_NEAR(0.0F, diff, 1e-3);
  }
}

TEST(guseynov_e_sparse_matrix_multiply_crs, test_square_matrix_by_itself) {
  CRSMatrix a{5,
              5,
              {0, 3, 5, 8, 11, 13},
              {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4},
              {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5}};
  CRSMatrix b = a;

  CRSMatrix expected{5,
                     5,
                     {0, 4, 7, 12, 17, 19},
                     {0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 4},
                     {15, -6, -6, -24, -12, 27, 6, -24, 32, 28, 66, -4, -32, 4, 22, 73, 8, -16, 25}};

  compareMatrices(runTask(a, b), expected);
}

TEST(guseynov_e_sparse_matrix_multiply_crs, test_square_matrix) {
  CRSMatrix a{4, 4, {0, 2, 4, 6, 7}, {0, 2, 0, 1, 0, 3, 2}, {1, 5, 2, 3, 4, 1, 2}};
  CRSMatrix b{4, 5, {0, 2, 4, 5, 7}, {0, 1, 0, 2, 3, 0, 2}, {5, 3, 7, 6, 8, 3, 2}};

  CRSMatrix expected{4, 5, {0, 3, 6, 9, 10}, {0, 1, 3, 0, 1, 2, 0, 1, 2, 3}, {5, 3, 40, 31, 6, 18, 23, 12, 2, 16}};

  compareMatrices(runTask(a, b), expected);
}

TEST(guseynov_e_sparse_matrix_multiply_crs, test_non_square_matrix) {
  CRSMatrix A{2, 3, {0, 2, 3}, {0, 2, 1}, {1, 2, 3}};
  CRSMatrix B{3, 3, {0, 2, 3, 4}, {0, 2, 1, 1}, {4, 7, 6, 8}};

  CRSMatrix expected{2, 3, {0, 3, 4}, {0, 1, 2, 1}, {4, 16, 7, 18}};

  compareMatrices(runTask(A, B), expected);
}
