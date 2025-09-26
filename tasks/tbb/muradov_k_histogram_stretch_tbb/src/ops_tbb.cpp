#include "tbb/muradov_k_histogram_stretch_tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace muradov_k_histogram_stretch_tbb {

bool HistogramStretchTBBTask::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  input_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<uint8_t>(output_size, 0);
  return true;
}

bool HistogramStretchTBBTask::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->outputs.empty() && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool HistogramStretchTBBTask::RunImpl() {
  if (input_.empty()) {
    return true;
  }
  struct MinMax {
    uint8_t min_v;
    uint8_t max_v;
  };
  MinMax res = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, input_.size()), MinMax{.min_v = 255, .max_v = 0},
      [&](const tbb::blocked_range<size_t> &r, MinMax local) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          uint8_t v = input_[i];
          local.min_v = static_cast<uint8_t>(std::min<uint8_t>(v, local.min_v));
          local.max_v = static_cast<uint8_t>(std::max<uint8_t>(v, local.max_v));
        }
        return local;
      },
      [](MinMax a, MinMax b) {
        a.min_v = static_cast<uint8_t>(std::min<uint8_t>(b.min_v, a.min_v));
        a.max_v = static_cast<uint8_t>(std::max<uint8_t>(b.max_v, a.max_v));
        return a;
      });
  min_val_ = res.min_v;
  max_val_ = res.max_v;
  if (max_val_ == min_val_) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, output_.size()), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i != r.end(); ++i) {
        output_[i] = 0;
      }
    });
    return true;
  }
  int range = static_cast<int>(max_val_) - static_cast<int>(min_val_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, input_.size()), [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      int val = input_[i];
      int stretched = (val - static_cast<int>(min_val_)) * 255 / range;
      stretched = std::clamp(stretched, 0, 255);
      output_[i] = static_cast<uint8_t>(stretched);
    }
  });
  return true;
}

bool HistogramStretchTBBTask::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<uint8_t *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

}  // namespace muradov_k_histogram_stretch_tbb
