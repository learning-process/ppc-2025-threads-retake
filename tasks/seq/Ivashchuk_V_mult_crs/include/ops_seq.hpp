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
    std::vector<std::complex<double>> values;
    std::vector<int> col_indices;
    std::vector<int> row_pointers;
    int rows = 0;
    int cols = 0;
  };

  SparseMatrixCRS matrixA_, matrixB_, resultMatrix_;

  static void ConvertToCRS(const std::vector<std::complex<double>>& dense, int rows, int cols, SparseMatrixCRS* sparse);
  static void MultiplySparseMatrices(const SparseMatrixCRS& A, const SparseMatrixCRS& B, SparseMatrixCRS* C);
  static void ConvertFromCRS(const SparseMatrixCRS& sparse, std::vector<std::complex<double>>* dense);
};

}  // namespace Ivashchuk_V_sparse_matrix_seq