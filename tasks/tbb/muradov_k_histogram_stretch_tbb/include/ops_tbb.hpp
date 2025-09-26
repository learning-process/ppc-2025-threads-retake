#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_k_histogram_stretch_tbb {

class HistogramStretchTBBTask : public ppc::core::Task {
 public:
  explicit HistogramStretchTBBTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  uint8_t min_val_{};
  uint8_t max_val_{};
};

}  // namespace muradov_k_histogram_stretch_tbb
