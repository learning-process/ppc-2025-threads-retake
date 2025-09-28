#include "tbb/muradov_k_histogram_stretch/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace muradov_k_histogram_stretch {

bool HistogramStretchTBBTask::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_image_.assign(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_image_.assign(output_size, 0);
  return true;
}

bool HistogramStretchTBBTask::ValidationImpl() {
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

bool HistogramStretchTBBTask::RunImpl() {
  if (input_image_.empty()) {
    return false;
  }
  struct MinMax {
    int min_v;
    int max_v;
  };
  MinMax initial{.min_v = 255, .max_v = 0};
  MinMax res = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, input_image_.size()), initial,
      [&](const tbb::blocked_range<size_t>& r, MinMax local) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          int v = input_image_[i];
          local.min_v = std::min(v, local.min_v);
          local.max_v = std::max(v, local.max_v);
        }
        return local;
      },
      [](MinMax a, MinMax b) {
        a.min_v = std::min(b.min_v, a.min_v);
        a.max_v = std::max(b.max_v, a.max_v);
        return a;
      });
  min_val_ = res.min_v;
  max_val_ = res.max_v;
  if (min_val_ == max_val_) {
    std::ranges::fill(output_image_, 0);
    return true;
  }
  const int range = max_val_ - min_val_;
  static constexpr int kRepeat = 800;  // как в seq/omp
  for (int r = 0; r < kRepeat; ++r) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, input_image_.size()), [&](const tbb::blocked_range<size_t>& br) {
      for (size_t i = br.begin(); i != br.end(); ++i) {
        int stretched = (input_image_[i] - min_val_) * 255 / range;
        stretched = std::clamp(stretched, 0, 255);
        output_image_[i] = stretched;
      }
    });
  }
  return true;
}

bool HistogramStretchTBBTask::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_image_.size(); ++i) {
    out_ptr[i] = output_image_[i];
  }
  return true;
}

}  // namespace muradov_k_histogram_stretch
