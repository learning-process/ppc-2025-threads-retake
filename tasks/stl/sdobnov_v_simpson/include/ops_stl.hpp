#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <future>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sdobnov_v_simpson_stl {

using Func = double (*)(std::vector<double>);

double Polynomial3d(std::vector<double> point);
double Trigonometric4d(std::vector<double> point);
double Mixed5d(std::vector<double> point);

class SimpsonIntegralStl : public ppc::core::Task {
 public:
  explicit SimpsonIntegralStl(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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
  double ParallelSimpsonAsync();
  double ParallelSimpsonThreads();
  double ProcessPoint(int i);
};

}  // namespace sdobnov_v_simpson_stl