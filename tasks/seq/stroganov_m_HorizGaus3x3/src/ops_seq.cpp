#include "seq/stroganov_m_HorizGaus3x3/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);
  return true;
}

bool stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  return (task_data->inputs_count[0] == task_data->outputs_count[0]) && (kernel_.size() == 3) &&
         (sqrt_size * sqrt_size == size);
}

bool stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential::RunImpl() {
  double sum = kernel_[0] + kernel_[1] + kernel_[2];
  if (sum == 0.0) {
    sum = 1.0;
  }
  for (int i = 0; i < height_; ++i) {
    output_[i * width_] = (kernel_[1] * input_[i * width_] + kernel_[2] * input_[(i * width_) + 1]) / sum;
    for (int j = 1; j < width_ - 1; ++j) {
      output_[(i * width_) + j] = (kernel_[0] * input_[(i * width_) + j - 1] + kernel_[1] * input_[(i * width_) + j] +
                                   kernel_[2] * input_[(i * width_) + j + 1]) /
                                  sum;
    }
    output_[(i * width_) + width_ - 1] =
        (kernel_[0] * input_[(i * width_) + width_ - 2] + kernel_[1] * input_[(i * width_) + width_ - 1]) / sum;
  }
  return true;
}

bool stroganov_m_horiz_gaus3x3_seq::ImageFilterSequential::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }
  return true;
}
