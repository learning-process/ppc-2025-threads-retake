#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vragov_i_gaussian_filter_vertical_omp {

class GaussianFilterTask : public ppc::core::Task {
 public:
  explicit GaussianFilterTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  unsigned int x_{};
  unsigned int y_{};
};

}  // namespace vragov_i_gaussian_filter_vertical_omp