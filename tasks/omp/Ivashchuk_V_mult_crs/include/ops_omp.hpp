#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Ivashchuk_V_mult_crs {

struct SparseMatrix {
  std::vector<std::complex<double>> values;
  std::vector<int> col_indices;
  std::vector<int> row_pointers;
  int rows;
  int cols;
};

class SparseMatrixMultiplierOMP : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiplierOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrix matrix1;
  SparseMatrix matrix2;
  SparseMatrix result;
  void multiplySparseMatrices();
};

}  // namespace Ivashchuk_V_mult_crs