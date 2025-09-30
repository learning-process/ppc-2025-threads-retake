#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/polyakov_a_mult_complex_matrix_CRS/include/ops_tbb.hpp"

TEST(polyakov_a_mult_complex_matrix_CRS_tbb, test_mul_identity_matrix) {
  constexpr size_t n = 1000;

  // Create data
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS A =
      polyakov_a_mult_complex_matrix_CRS_tbb::GetRandomMatrixCRS(n, n, 5);

  std::vector<std::complex<double>> values(n, 1.0);
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.push_back(0);

  for (size_t i = 0; i < n; i++) {
    col_ind.push_back(i);
    row_ptr.push_back(i + 1);
  }
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS B(n, n, values, col_ind, row_ptr);
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS C(n, n);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  polyakov_a_mult_complex_matrix_CRS_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(A, C);
}

TEST(polyakov_a_mult_complex_matrix_CRS_tbb, test_mul_negative_identity_matrix) {
  constexpr size_t n = 1000;
  const std::complex<double> k = -1.0;

  // Create data
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS A =
      polyakov_a_mult_complex_matrix_CRS_tbb::GetRandomMatrixCRS(n, n, 5);

  std::vector<std::complex<double>> b_values(n, k);
  std::vector<size_t> b_col_ind;
  std::vector<size_t> b_row_ptr;
  b_row_ptr.push_back(0);

  for (size_t i = 0; i < n; i++) {
    b_col_ind.push_back(i);
    b_row_ptr.push_back(i + 1);
  }
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS B(n, n, b_values, b_col_ind, b_row_ptr);

  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS C(n, n);

  std::vector<std::complex<double>> exp_values;
  for (size_t i = 0; i < A.values.size(); i++) {
    exp_values.push_back(A.values[i] * k);
  }
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS expect(n, n, exp_values, A.col_ind, A.row_ptr);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  polyakov_a_mult_complex_matrix_CRS_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(expect, C);
}

TEST(polyakov_a_mult_complex_matrix_CRS_tbb, test_mul_none_square_matrix) {
  constexpr size_t n = 5;
  constexpr size_t m = 4;
  constexpr size_t k = 3;

  // Create data
  std::vector<std::complex<double>> a_values = {1, 2, 3, 4, 5, 10, 6};
  std::vector<size_t> a_col_ind = {0, 2, 1, 3, 0, 2, 1};
  std::vector<size_t> a_row_ptr = {0, 2, 4, 6, 6, 7};

  std::vector<std::complex<double>> b_values = {5, 4, 9, 2, 1};
  std::vector<size_t> b_col_ind = {0, 2, 0, 2, 1};
  std::vector<size_t> b_row_ptr = {0, 2, 3, 4, 5};

  std::vector<std::complex<double>> exp_values = {5, 8, 27, 4, 25, 40, 54};
  std::vector<size_t> exp_col_ind = {0, 2, 0, 1, 0, 2, 0};
  std::vector<size_t> exp_row_ptr = {0, 2, 4, 6, 6, 7};

  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS C(n, k);

  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS A(n, m, a_values, a_col_ind, a_row_ptr);
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS B(m, k, b_values, b_col_ind, b_row_ptr);
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS expect(n, k, exp_values, exp_col_ind, exp_row_ptr);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  polyakov_a_mult_complex_matrix_CRS_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(expect, C);
}

TEST(polyakov_a_mult_complex_matrix_CRS_tbb, test_none_valid) {
  constexpr size_t n = 5;
  constexpr size_t m = 4;
  constexpr size_t k = 3;
  constexpr size_t t = 10;

  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS A(n, m);
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS B(k, t);
  polyakov_a_mult_complex_matrix_CRS_tbb::MatrixCRS C;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  polyakov_a_mult_complex_matrix_CRS_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}