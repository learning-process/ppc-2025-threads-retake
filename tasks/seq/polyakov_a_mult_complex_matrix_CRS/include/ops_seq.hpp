#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace polyakov_a_mult_complex_matrix_crs_seq {

using Rows = size_t;
using Cols = size_t;
using ColIndices = std::vector<size_t>;
using RowPointers = std::vector<size_t>;

struct MatrixCRS {
  Rows rows{};
  Cols cols{};
  std::vector<std::complex<double>> values;
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr{0};

  MatrixCRS() = default;
  MatrixCRS(Rows row_count, Cols col_count) : rows(row_count), cols(col_count) {}
  MatrixCRS(Rowst row_count, Cols column_count, const std::vector<std::complex<double>>& non_zero_values,
            const ColIndices& column_indexes, const RowPointers& row_pointers)
      : rows(row_count), cols(column_count), values(non_zero_values), col_ind(column_indexes), row_ptr(row_pointers) {}

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