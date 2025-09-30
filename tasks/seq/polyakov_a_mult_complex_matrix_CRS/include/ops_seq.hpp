#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace polyakov_a_mult_complex_matrix_crs_seq {

struct Rows {
  size_t value;
  constexpr explicit Rows(size_t v) : value(v) {}
};

struct Cols {
  size_t value;
  constexpr explicit Cols(size_t v) : value(v) {}
};

struct MatrixCRS {
  size_t rows{};
  size_t cols{};
  std::vector<std::complex<double>> values;
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr{0};

  MatrixCRS() = default;
  MatrixCRS(Rows row_count, Cols col_count) : rows(row_count.value), cols(col_count.value) {}
  MatrixCRS(Rows row_count, Cols column_count, std::vector<std::complex<double>> non_zero_values,
            std::vector<size_t> column_indexes, std::vector<size_t> row_pointers)
      : rows(row_count.value),
        cols(column_count.value),
        values(std::move(non_zero_values)),
        col_ind(std::move(column_indexes)),
        row_ptr(std::move(row_pointers)) {}

  bool operator==(const MatrixCRS& m) const {
    if (rows != m.rows || cols != m.cols) {
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

MatrixCRS GetRandomMatrixCRS(Rows num_rows, Cols num_cols, size_t sparsity_coeff);

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