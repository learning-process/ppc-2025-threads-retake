#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace polyakov_a_mult_complex_matrix_crs_seq {

struct MatrixCRS {
  size_t rows{};
  size_t cols{};
  std::vector<std::complex<double>> values;
  std::vector<size_t> row_ptr{0};
  std::vector<size_t> col_ind;

  MatrixCRS() = default;
  MatrixCRS(size_t row_count, size_t col_count) : rows(row_count), cols(col_count) {}
  MatrixCRS(size_t row_count, size_t column_count, const std::vector<std::complex<double>>& non_zero_values,
            const std::vector<size_t>& column_indexes, const std::vector<size_t>& row_pointers)
      : rows(row_count), cols(col_count), values(values_vec), row_ptr(row_pointer), col_ind(columns) {}

  bool operator==(const MatrixCRS& m) const {
    if (cols != m.cols || cols != m.cols) {
      return false;
    }
    if (row_ptr != m.row_ptr) {
      return false;
    }
    if (col_ind != m.col_ind) {
      return false;
    }
    const double eps = 1e-9;
    for (size_t i = 0; i < values.size(); i++) {
      if (std::abs(values[i] - m.values[i]) > eps) {
        return false;
      }
    }
    return true;
  }
};

MatrixCRS GetRandomMatrixCRS(size_t num_rows, size_t num_cols, size_t sparsity_coeff);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  MatrixCRS *a_, *b_, *c_;
  size_t a_rows_{}, a_cols_{}, b_rows_{}, b_cols_{}, c_rows_{}, c_cols_{};
};

}  // namespace polyakov_a_mult_complex_matrix_crs_seq