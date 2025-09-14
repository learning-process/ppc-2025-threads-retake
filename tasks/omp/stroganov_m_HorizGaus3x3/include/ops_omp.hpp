#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp.h"

namespace stroganov_m_horiz_gaus3x3_omp {

class ImageFilterOmp : public ppc::core::Task {
 public:
  explicit ImageFilterOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;
  int width_;
  int height_;
  std::vector<int> kernel_;
};

}  // namespace stroganov_m_horiz_gaus3x3_omp
