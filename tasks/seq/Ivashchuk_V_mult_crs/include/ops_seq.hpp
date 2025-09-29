#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Ivashchuk_V_sparse_matrix_seq {

class SparseMatrixMultiplier : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiplier(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  struct SparseMatrixCRS {
    std::vector<std::complex<double>> values_;
    std::vector<int> col_indices_;
    std::vector<int> row_pointers_;
    int rows_ = 0;
    int cols_ = 0;
  };

  SparseMatrixCRS matrix_a_;
  SparseMatrixCRS matrix_b_;
  SparseMatrixCRS result_matrix_;

  static void ConvertToCRS(const std::vector<std::complex<double>>& dense, int rows, int cols, SparseMatrixCRS* sparse);
  static void MultiplySparseMatrices(const SparseMatrixCRS& a, const SparseMatrixCRS& b, SparseMatrixCRS* c);
  static void ConvertFromCRS(const SparseMatrixCRS& sparse, std::vector<std::complex<double>>* dense);
};

}  // namespace Ivashchuk_V_sparse_matrix_seq