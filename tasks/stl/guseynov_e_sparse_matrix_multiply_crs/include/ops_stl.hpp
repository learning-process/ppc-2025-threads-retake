#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace guseynov_e_sparse_matrix_multiply_crs_stl {

struct CRSMatrix {
  int n_rows{};
  int n_cols{};
  std::vector<double> non_zero_values;
  std::vector<int> pointer;
  std::vector<int> col_indexes;
};

class SparseMatMultSTL : public ppc::core::Task {
 public:
  explicit SparseMatMultSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  CRSMatrix *A_mat_{}, *B_mat_{}, *Result_{};
};

}  // namespace guseynov_e_sparse_matrix_multiply_crs_stl