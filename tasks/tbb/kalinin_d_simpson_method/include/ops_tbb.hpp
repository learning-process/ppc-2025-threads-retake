#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_simpson_method_tbb {

class SimpsonNDTBB : public ppc::core::Task {
 public:
  explicit SimpsonNDTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int dimension_{};
  int segments_per_dim_{};
  int function_id_{};

  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;

  double result_{};

  [[nodiscard]] double EvaluateFunction(const std::vector<double>& point) const;
};

}  // namespace kalinin_d_simpson_method_tbb