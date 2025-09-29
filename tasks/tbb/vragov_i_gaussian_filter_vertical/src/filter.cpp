#include "tbb/vragov_i_gaussian_filter_vertical/include/filter.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

bool vragov_i_gaussian_filter_vertical_tbb::GaussianFilterTask::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  x_ = static_cast<size_t>(task_data->inputs_count[1]);
  y_ = static_cast<size_t>(task_data->inputs_count[2]);
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool vragov_i_gaussian_filter_vertical_tbb::GaussianFilterTask::ValidationImpl() {
  // Verify image dimensions
  return (task_data->inputs_count[1] * task_data->inputs_count[2] == task_data->inputs_count[0]);
}

bool vragov_i_gaussian_filter_vertical_tbb::GaussianFilterTask::RunImpl() {
  // Gaussian filter vertical 1x3
  const std::vector<double> kernel = {0.015, 0.8, 0.015};
  const double pi = std::acos(-1.0);
  // If empty image
  if (input_.empty()) {
    return true;
  }
  tbb::parallel_for(static_cast<size_t>(0), x_, [this, &kernel, pi](size_t i) {
    for (size_t j = 0; j < y_; j++) {
      double sum = 0.0;
      for (int k = -1; k <= 1; k++) {
        int idx = static_cast<int>(j) + k;
        if (idx < 0) {
          idx = 0;
        } else if (idx >= static_cast<int>(y_)) {
          idx = static_cast<int>(y_) - 1;
        }
        sum += input_[(i * y_) + static_cast<size_t>(idx)] * kernel[k + 1];
      }
      sum /= (sqrt(2.0 * pi) * 0.5);
      output_[(i * y_) + j] = static_cast<int>(std::round(sum));
    }
  });
  return true;
}

bool vragov_i_gaussian_filter_vertical_tbb::GaussianFilterTask::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}