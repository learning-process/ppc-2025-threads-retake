#include "omp/muradov_k_histogram_stretch/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>

namespace muradov_k_histogram_stretch_omp {

bool HistogramStretchOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_image_.assign(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_image_.assign(output_size, 0);
  return true;
}

bool HistogramStretchOpenMP::ValidationImpl() {
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; ++i) {
    if (in_ptr[i] < 0 || in_ptr[i] > 255) {
      return false;
    }
  }
  return true;
}

bool HistogramStretchOpenMP::RunImpl() {
  if (input_image_.empty()) {
    return false;
  }

  int global_min = 255;
  int global_max = 0;
#pragma omp parallel
  {
    int local_min = 255;
    int local_max = 0;
#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(input_image_.size()); ++i) {
      int v = input_image_[i];
      local_min = std::min(v, local_min);
      local_max = std::max(v, local_max);
    }
#pragma omp critical
    {
      global_min = std::min(local_min, global_min);
      global_max = std::max(local_max, global_max);
    }
  }
  min_val_ = global_min;
  max_val_ = global_max;

  if (min_val_ == max_val_) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(output_image_.size()); ++i) {
      output_image_[i] = 0;
    }
    return true;
  }

  const int range = max_val_ - min_val_;
  static constexpr int kRepeat = 800;  // идентично seq версии
  for (int r = 0; r < kRepeat; ++r) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input_image_.size()); ++i) {
      int stretched = (input_image_[i] - min_val_) * 255 / range;
      stretched = std::clamp(stretched, 0, 255);
      output_image_[i] = stretched;
    }
  }
  return true;
}

bool HistogramStretchOpenMP::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(output_image_.size()); ++i) {
    out_ptr[i] = output_image_[i];
  }
  return true;
}

}  // namespace muradov_k_histogram_stretch_omp
