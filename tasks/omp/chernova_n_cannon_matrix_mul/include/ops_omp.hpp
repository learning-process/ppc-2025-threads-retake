#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_cannon_matrix_mul_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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

std::vector<double> MultiplyMatrixOMP(const std::vector<double>& a, const std::vector<double>& b, int n);
std::vector<double> CannonMatrixMultiplicationOMP(const std::vector<double>& a, const std::vector<double>& b, int n);

}  // namespace chernova_n_cannon_matrix_mul_omp