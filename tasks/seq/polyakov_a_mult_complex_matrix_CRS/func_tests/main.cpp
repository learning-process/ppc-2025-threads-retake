#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/polyakov_a_mult_complex_matrix_CRS/include/ops_seq.hpp"

TEST(polyakov_a_mult_complex_matrix_crs_seq, test_mul_identity_matrix) {
  constexpr size_t kN = 1000;

  // Create data
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS a =
      polyakov_a_mult_complex_matrix_crs_seq::GetRandomMatrixCRS(kN, kN, 5);

  std::vector<std::complex<double>> values(kN, 1.0);
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.push_back(0);

  for (size_t i = 0; i < kN; i++) {
    col_ind.push_back(i);
    row_ptr.push_back(i + 1);
  }
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS b(kN, kN, values, col_ind, row_ptr);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS c(kN, kN);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(a, c);
}

TEST(polyakov_a_mult_complex_matrix_crs_seq, test_mul_negative_identity_matrix) {
  constexpr size_t kN = 1000;
  const std::complex<double> minus_one = -1.0;

  // Create data
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS a =
      polyakov_a_mult_complex_matrix_crs_seq::GetRandomMatrixCRS(kN, kN, 5);

  std::vector<std::complex<double>> b_values(kN, minus_one);
  std::vector<size_t> b_col_ind;
  std::vector<size_t> b_row_ptr;
  b_row_ptr.push_back(0);

  for (size_t i = 0; i < kN; i++) {
    b_col_ind.push_back(i);
    b_row_ptr.push_back(i + 1);
  }
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS b(kN, kN, b_values, b_col_ind, b_row_ptr);

  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS c(kN, kN);

  std::vector<std::complex<double>> exp_values;
  for (size_t i = 0; i < a.values.size(); i++) {
    exp_values.push_back(a.values[i] * k_);
  }
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS expect(kN, kN, exp_values, a.col_ind, a.row_ptr);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expect, c);
}

TEST(polyakov_a_mult_complex_matrix_crs_seq, test_mul_none_square_matrix) {
  constexpr size_t kN = 5;
  constexpr size_t kM = 4;
  constexpr size_t k_ = 3;

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

  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS c(kN, k_);

  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS a(kN, kM, a_values, a_col_ind, a_row_ptr);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS b(kM, k_, b_values, b_col_ind, b_row_ptr);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS expect(kN, k_, exp_values, exp_col_ind, exp_row_ptr);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expect, c);
}

TEST(polyakov_a_mult_complex_matrix_crs_seq, test_none_valid) {
  constexpr size_t kN = 5;
  constexpr size_t kM = 4;
  constexpr size_t k_ = 3;
  constexpr size_t t = 10;

  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS a(kN, kM);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS b(k_, t);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS c;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}