#include "tbb/stroganov_m_HorizGaus3x3/include/ops_tbb.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

bool stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);
  return true;
}

bool stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  return (task_data->inputs_count[0] == task_data->outputs_count[0]) && (kernel_.size() == 3) &&
         (sqrt_size * sqrt_size == size);
}

bool stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb::RunImpl() {
  double sum = kernel_[0] + kernel_[1] + kernel_[2];
  double inv_sum = (sum == 0.0) ? 1.0 : 1.0 / sum;
  double k0_inv = kernel_[0] * inv_sum;
  double k1_inv = kernel_[1] * inv_sum;
  double k2_inv = kernel_[2] * inv_sum;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, height_), [this, k0_inv, k1_inv, k2_inv](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
          const int row_offset = i * width_;
          output_[row_offset] = (k1_inv * input_[row_offset]) + (k2_inv * input_[row_offset + 1]);
          for (int j = 1; j < width_ - 1; ++j) {
            const int idx = row_offset + j;
            output_[idx] = (k0_inv * input_[idx - 1]) + (k1_inv * input_[idx]) + (k2_inv * input_[idx + 1]);
          }
          const int last_idx = row_offset + width_ - 1;
          output_[last_idx] = (k0_inv * input_[last_idx - 1]) + (k1_inv * input_[last_idx]);
        }
      });
  return true;
}

bool stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }
  return true;
}
