#include "omp/vragov_i_gaussian_filter_vertical/include/filter.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  x_ = static_cast<int>(task_data->inputs_count[1]);
  y_ = static_cast<int>(task_data->inputs_count[2]);
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask::ValidationImpl() {
  // Verify image dimensions
  return (task_data->inputs_count[1] * task_data->inputs_count[2] == task_data->inputs_count[0]);
}

bool vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask::RunImpl() {
  // Gaussian filter vertical 1x3
  const std::vector<double> kernel = {0.015, 0.8, 0.015};
  const double pi = std::acos(-1.0);
  // If empty image
  if (input_.empty()) {
    return true;
  }
#pragma omp parallel for
  for (int i = 0; i < x_; i++) {
    for (int j = 0; j < y_; j++) {
      double sum = 0.0;
      for (int k = -1; k <= 1; k++) {
        int idx = j + k;
        if (idx < 0) {
          idx = 0;
        } else if (idx >= y_) {
          idx = y_ - 1;
        }
        sum += input_[(i * y_) + idx] * kernel[k + 1];
      }
      sum /= (sqrt(2.0 * pi) * 0.5);
      output_[(i * y_) + j] = static_cast<int>(std::round(sum));
    }
  }
  return true;
}

bool vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
