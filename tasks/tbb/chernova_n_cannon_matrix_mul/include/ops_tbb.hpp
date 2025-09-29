#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_cannon_matrix_mul_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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

std::vector<double> MultiplyMatrixTBB(const std::vector<double> &mat_a, const std::vector<double> &mat_b, int n);
std::vector<double> CannonMatrixMultiplicationTBB(const std::vector<double> &mat_a, const std::vector<double> &mat_b,
                                                  int n);

}  // namespace chernova_n_cannon_matrix_mul_tbb