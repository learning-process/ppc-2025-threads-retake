#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_matmul_fox_algo_seq {
void BlockMultiply(const std::vector<double> &matr_a, const std::vector<double> &matr_b, std::vector<double> &matr_res,
                   unsigned long row_block, unsigned long col_block, size_t block_index, size_t block_size, size_t n);

class MultiplMatrixSequental : public ppc::core::Task {
 public:
  explicit MultiplMatrixSequental(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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

}  // namespace agafeev_s_matmul_fox_algo_seq