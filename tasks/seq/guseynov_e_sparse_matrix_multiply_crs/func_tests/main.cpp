#include <gtest/gtest.h>

#include <vector>

#include "seq/guseynov_e_sparse_matrix_multiply_crs/include/ops_seq.hpp"

TEST(guseynov_e_sparse_matrix_multiply_crs, test_square_matrix_by_itself) {
  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix A;
  A.n_rows = 5;
  A.n_cols = 5;
  A.pointer = {0, 3, 5, 8, 11, 13};
  A.col_indexes = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
  A.non_zero_values = {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5};

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix B;
  B.n_rows = 5;
  B.n_cols = 5;
  B.pointer = {0, 3, 5, 8, 11, 13};
  B.col_indexes = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
  B.non_zero_values = {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5};

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix Result;

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix Expected;
  Expected.n_rows = 5;
  Expected.n_cols = 5;
  Expected.pointer = {0, 4, 7, 12, 17, 19};
  Expected.col_indexes = {0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 4};
  Expected.non_zero_values = {15, -6, -6, -24, -12, 27, 6, -24, 32, 28, 66, -4, -32, 4, 22, 73, 8, -16, 25};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.ValidationImpl(), true);
  ASSERT_EQ(taskSequential.PreProcessingImpl(), true);
  ASSERT_EQ(taskSequential.RunImpl(), true);
  ASSERT_EQ(taskSequential.PostProcessingImpl(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    double t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t, 1e-3);
  }
}

TEST(guseynov_e_sparse_matrix_multiply_crs, test_square_matrix) {
  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix A;
  A.n_rows = 4;
  A.n_cols = 4;
  A.pointer = {0, 2, 4, 6, 7};
  A.col_indexes = {0, 2, 0, 1, 0, 3, 2};
  A.non_zero_values = {1, 5, 2, 3, 4, 1, 2};

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix B;
  B.n_rows = 4;
  B.n_cols = 5;
  B.pointer = {0, 2, 4, 5, 7};
  B.col_indexes = {0, 1, 0, 2, 3, 0, 2};
  B.non_zero_values = {5, 3, 7, 6, 8, 3, 2};

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix Result;

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix Expected;
  Expected.n_rows = 4;
  Expected.n_cols = 5;
  Expected.pointer = {0, 3, 6, 9, 10};
  Expected.col_indexes = {0, 1, 3, 0, 1, 2, 0, 1, 2, 3};
  Expected.non_zero_values = {5, 3, 40, 31, 6, 18, 23, 12, 2, 16};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create 1
  guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.ValidationImpl(), true);
  ASSERT_EQ(taskSequential.PreProcessingImpl(), true);
  ASSERT_EQ(taskSequential.RunImpl(), true);
  ASSERT_EQ(taskSequential.PostProcessingImpl(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    std::complex<double> t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-3);
    ASSERT_NEAR(0.0f, t.real(), 1e-3);
  }
}

TEST(guseynov_e_sparse_matrix_multiply_crs, test_non_square_matrix) {
  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix A;
  A.n_rows = 2;
  A.n_cols = 3;
  A.pointer = {0, 2, 3};
  A.col_indexes = {0, 2, 1};
  A.non_zero_values = {1, 2, 3};

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix B;
  B.n_rows = 3;
  B.n_cols = 3;
  B.pointer = {0, 2, 3, 4};
  B.col_indexes = {0, 2, 1, 1};
  B.non_zero_values = {4, 7, 6, 8};

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix Result;

  guseynov_e_sparse_matrix_multiply_crs::CRSMatrix Expected;
  Expected.n_rows = 2;
  Expected.n_cols = 3;
  Expected.pointer = {0, 3, 4};
  Expected.col_indexes = {0, 1, 2, 1};
  Expected.non_zero_values = {4, 16, 7, 18};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.ValidationImpl(), true);
  ASSERT_EQ(taskSequential.PreProcessingImpl(), true);
  ASSERT_EQ(taskSequential.RunImpl(), true);
  ASSERT_EQ(taskSequential.PostProcessingImpl(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    std::complex<double> t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-3);
    ASSERT_NEAR(0.0f, t.real(), 1e-3);
  }
}