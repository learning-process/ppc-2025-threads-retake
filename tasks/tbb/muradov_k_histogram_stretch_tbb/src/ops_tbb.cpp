#include "tbb/muradov_k_histogram_stretch_tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

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
  if (input_.empty()) return true;
  struct MinMax {
    uint8_t min_v;
    uint8_t max_v;
  };
  MinMax res = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, input_.size()), MinMax{255, 0},
      [&](const tbb::blocked_range<size_t> &r, MinMax local) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          uint8_t v = input_[i];
            if (v < local.min_v) local.min_v = v;
            if (v > local.max_v) local.max_v = v;
        }
        return local;
      },
      [](MinMax a, MinMax b) {
        if (b.min_v < a.min_v) a.min_v = b.min_v;
        if (b.max_v > a.max_v) a.max_v = b.max_v;
        return a;
      });
  min_val_ = res.min_v;
  max_val_ = res.max_v;
  if (max_val_ == min_val_) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, output_.size()), [&](const tbb::blocked_range<size_t> &r) {
      for (size_t i = r.begin(); i != r.end(); ++i) output_[i] = 0;
    });
    return true;
  }
  int range = static_cast<int>(max_val_) - static_cast<int>(min_val_);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, input_.size()), [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      int val = input_[i];
      int stretched = (val - static_cast<int>(min_val_)) * 255 / range;
      if (stretched < 0) stretched = 0;
      if (stretched > 255) stretched = 255;
      output_[i] = static_cast<uint8_t>(stretched);
    }
  });
  return true;
}

bool HistogramStretchTBBTask::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) reinterpret_cast<uint8_t *>(task_data->outputs[0])[i] = output_[i];
  return true;
}

}  // namespace muradov_k_histogram_stretch_tbb

