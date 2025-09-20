#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sdobnov_v_simpson {

using Func = double (*)(std::vector<double>);
double TestFunction2d(std::vector<double> point);
double LinearFunction1d(std::vector<double> point);
double Gaussian2d(std::vector<double> point);

class SimpsonIntegralSequential : public ppc::core::Task {
 public:
  explicit SimpsonIntegralSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int dimensions_{};
  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  std::vector<int> n_points_;
  double result_{};

  Func integrand_function_{nullptr};

  double SimpsonRecursive(int dim_index, const std::vector<double>& current_point);
};

}  // namespace sdobnov_v_simpson