#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivashchuk_v_tbb {

class SparseMatrixComplexCRS : public ppc::core::Task {
 public:
  explicit SparseMatrixComplexCRS(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  struct CRSMatrix {
    std::vector<std::complex<double>> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int rows = 0;
    int cols = 0;
  };

  CRSMatrix matrixA, matrixB, result;

  void ConvertToCRS(const std::vector<std::complex<double>>& dense, int rows, int cols, CRSMatrix& crs);
  void SparseMatMul(const CRSMatrix& A, const CRSMatrix& B, CRSMatrix& C);
};

}  // namespace ivashchuk_v_tbb