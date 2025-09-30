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
    std::vector<std::complex<double>> values_;
    std::vector<int> col_indices_;
    std::vector<int> row_ptr_;
    int rows_ = 0;
    int cols_ = 0;
  };

  CRSMatrix matrix_a_, matrix_b_, result_;

  void ConvertToCRS(const std::vector<std::complex<double>>& dense, int rows, int cols, CRSMatrix& crs);
  void SparseMatMul(const CRSMatrix& a, const CRSMatrix& b, CRSMatrix& c);
};

}  // namespace ivashchuk_v_tbb