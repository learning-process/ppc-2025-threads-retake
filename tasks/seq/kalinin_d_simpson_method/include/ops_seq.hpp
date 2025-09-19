#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_simpson_method_seq {

// Sequential multidimensional Simpson integration task
class SimpsonNDSequential : public ppc::core::Task {
 public:
  explicit SimpsonNDSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // Parameters
  int dimension_{};         // number of dimensions
  int segments_per_dim_{};  // must be even
  int function_id_{};       // which integrand to use

  // Bounds
  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;

  // Result
  double result_{};

  // Helpers
  [[nodiscard]] double EvaluateFunction(const std::vector<double>& point) const;
};

}  // namespace kalinin_d_simpson_method_seq