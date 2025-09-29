#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_matmul_fox_algo_omp {
void BlockMultiply(const std::vector<double> &matr_a, unsigned long row, const std::vector<double> &matr_b,
                   unsigned long col, std::vector<double> &matr_res, size_t block_index, size_t block_size, size_t n);

class MultiplMatrixOpenMP : public ppc::core::Task {
 public:
  explicit MultiplMatrixOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> first_input_;
  std::vector<double> second_input_;
  std::vector<double> result_;
  size_t size_;
  size_t block_size_;
};

}  // namespace agafeev_s_matmul_fox_algo_omp