#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vragov_i_gaussian_filter_vertical_tbb {

class GaussianFilterTask : public ppc::core::Task {
 public:
  explicit GaussianFilterTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  size_t x_{};
  size_t y_ {}
};

}  // namespace vragov_i_gaussian_filter_vertical_tbb