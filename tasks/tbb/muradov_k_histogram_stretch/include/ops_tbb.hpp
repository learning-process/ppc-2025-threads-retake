#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_k_histogram_stretch {

class HistogramStretchTBBTask : public ppc::core::Task {
 public:
  explicit HistogramStretchTBBTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_image_, output_image_;
  int min_val_{0};
  int max_val_{0};
};

}  // namespace muradov_k_histogram_stretch
