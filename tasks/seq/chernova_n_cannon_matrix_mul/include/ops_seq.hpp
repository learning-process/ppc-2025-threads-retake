#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_cannon_matrix_mul_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> res_;
  int n_ = 0;
};

std::vector<double> MultiplyMatrix(const std::vector<double>& mat_a, const std::vector<double>& mat_b, int n);
std::vector<double> CannonMatrixMultiplication(const std::vector<double>& mat_a, const std::vector<double>& mat_b, int n);

}  // namespace chernova_n_cannon_matrix_mul_seq