#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/polyakov_a_mult_complex_matrix_CRS/include/ops_tbb.hpp"

namespace pcrs = polyakov_a_mult_complex_matrix_crs_tbb;

TEST(polyakov_a_mult_complex_matrix_crs_tbb, test_mul_identity_matrix) {
  constexpr size_t kN = 1000;

  // Create data
  pcrs::MatrixCRS a = pcrs::GetRandomMatrixCRS(pcrs::Rows{kN}, pcrs::Cols{kN}, 5);

  std::vector<std::complex<double>> values(kN, 1.0);
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.push_back(0);

  for (size_t i = 0; i < kN; i++) {
    col_ind.push_back(i);
    row_ptr.push_back(i + 1);
  }
  pcrs::MatrixCRS b(pcrs::Rows{kN}, pcrs::Cols{kN}, std::move(values), std::move(col_ind), std::move(row_ptr));
  pcrs::MatrixCRS c(pcrs::Rows{kN}, pcrs::Cols{kN});

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(a, c);
}

TEST(polyakov_a_mult_complex_matrix_crs_tbb, test_mul_negative_identity_matrix) {
  constexpr size_t kN = 1000;
  const std::complex<double> minus_one = -1.0;

  // Create data
  pcrs::MatrixCRS a = pcrs::GetRandomMatrixCRS(pcrs::Rows{kN}, pcrs::Cols{kN}, 5);

  std::vector<std::complex<double>> b_values(kN, minus_one);
  std::vector<size_t> b_col_ind;
  std::vector<size_t> b_row_ptr;
  b_row_ptr.push_back(0);

  for (size_t i = 0; i < kN; i++) {
    b_col_ind.push_back(i);
    b_row_ptr.push_back(i + 1);
  }
  pcrs::MatrixCRS b(pcrs::Rows{kN}, pcrs::Cols{kN}, std::move(b_values), std::move(b_col_ind), std::move(b_row_ptr));

  pcrs::MatrixCRS c(pcrs::Rows{kN}, pcrs::Cols{kN});

  std::vector<std::complex<double>> exp_values;
  exp_values.reserve(a.values.size());
  for (size_t i = 0; i < a.values.size(); i++) {
    exp_values.push_back(a.values[i] * minus_one);
  }
  pcrs::MatrixCRS expect(pcrs::Rows{kN}, pcrs::Cols{kN}, std::move(exp_values), a.col_ind, a.row_ptr);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(expect, c);
}

TEST(polyakov_a_mult_complex_matrix_crs_tbb, test_mul_none_square_matrix) {
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

  pcrs::MatrixCRS c(pcrs::Rows{kN}, pcrs::Cols{k_});

  pcrs::MatrixCRS a(pcrs::Rows{kN}, pcrs::Cols{kM}, std::move(a_values), std::move(a_col_ind), std::move(a_row_ptr));
  pcrs::MatrixCRS b(pcrs::Rows{kM}, pcrs::Cols{k_}, std::move(b_values), std::move(b_col_ind), std::move(b_row_ptr));
  pcrs::MatrixCRS expect(pcrs::Rows{kN}, pcrs::Cols{k_}, std::move(exp_values), std::move(exp_col_ind),
                         std::move(exp_row_ptr));

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  EXPECT_EQ(expect, c);
}

TEST(polyakov_a_mult_complex_matrix_crs_tbb, test_none_valid) {
  constexpr size_t kN = 5;
  constexpr size_t kM = 4;
  constexpr size_t k_ = 3;
  constexpr size_t kT = 10;

  polyakov_a_mult_complex_matrix_crs_tbb::MatrixCRS a(pcrs::Rows{kN}, pcrs::Cols{kM});
  polyakov_a_mult_complex_matrix_crs_tbb::MatrixCRS b(pcrs::Rows{k_}, pcrs::Cols{kT});
  polyakov_a_mult_complex_matrix_crs_tbb::MatrixCRS c;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}