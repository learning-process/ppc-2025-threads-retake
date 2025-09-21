#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_vertical_gauss_3x3_omp {

class VerticalGauss : public ppc::core::Task {
 public:
  explicit VerticalGauss(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int32_t img_width_, img_height_;
  std::vector<uint8_t> source_img_, filtered_img_;
  std::vector<float> filter_kernel_;
  static constexpr int kCHANNELS = 3;
};

}  // namespace vasenkov_a_vertical_gauss_3x3_omp