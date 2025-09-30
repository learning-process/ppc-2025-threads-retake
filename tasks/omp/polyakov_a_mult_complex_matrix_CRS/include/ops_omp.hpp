#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace polyakov_a_mult_complex_matrix_CRS_omp {

struct MatrixCRS {
  size_t rows_{};
  size_t cols_{};
  std::vector<std::complex<double>> values;
  std::vector<size_t> row_ptr;
  std::vector<size_t> col_ind;

  MatrixCRS() : rows_(0), cols_(0), values{}, row_ptr{0}, col_ind{} {}
  MatrixCRS(size_t rows, size_t cols) : rows_(rows), cols_(cols), values{}, row_ptr{0}, col_ind{} {}
  MatrixCRS(size_t rows, size_t cols, const std::vector<std::complex<double>>& v, const std::vector<size_t>& c,
            const std::vector<size_t>& rptr)
      : rows_(rows), cols_(cols), values(v), row_ptr(rptr), col_ind(c) {}

  bool operator==(const MatrixCRS& m) const {
    if (rows_ != m.rows_ || cols_ != m.cols_) return false;
    if (row_ptr != m.row_ptr) return false;
    if (col_ind != m.col_ind) return false;
    const double eps = 1e-9;
    for (size_t i = 0; i < values.size(); i++) {
      if (std::abs(values[i] - m.values[i]) > eps) return false;
    }
    return true;
  }
};

MatrixCRS GetRandomMatrixCRS(size_t n, size_t m, size_t sparsity_coeff);

class TestTaskOMP : public ppc::core::Task {
 public:
  explicit TestTaskOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  MatrixCRS *A, *B, *C;
  size_t a_rows{}, a_cols{}, b_rows{}, b_cols{}, c_rows{}, c_cols{};
};

}  // namespace polyakov_a_mult_complex_matrix_CRS_omp