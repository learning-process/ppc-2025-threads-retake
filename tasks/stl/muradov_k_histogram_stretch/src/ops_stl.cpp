#include "stl/muradov_k_histogram_stretch/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace muradov_k_histogram_stretch_stl {

bool HistogramStretchSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_image_.assign(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_image_.assign(output_size, 0);
  return true;
}

bool HistogramStretchSTL::ValidationImpl() {
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

bool HistogramStretchSTL::RunImpl() {
  if (input_image_.empty()) {
    return false;
  }
  auto mm = std::ranges::minmax_element(input_image_);
  min_val_ = *mm.min;
  max_val_ = *mm.max;
  if (min_val_ == max_val_) {
    std::ranges::fill(output_image_, 0);
    return true;
  }
  const int range = max_val_ - min_val_;
  static constexpr int kRepeat = 800;  // как в seq/omp/tbb
  const int num_threads = std::max(1, ppc::util::GetPPCNumThreads());
  const size_t n = input_image_.size();
  size_t block = (n + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_threads));
  for (int t = 0; t < num_threads; ++t) {
    size_t begin = static_cast<size_t>(t) * block;
    if (begin >= n) {
      break;
    }
    size_t end = std::min(begin + block, n);
    threads.emplace_back([&, begin, end]() {
      for (int r = 0; r < kRepeat; ++r) {
        for (size_t i = begin; i < end; ++i) {
          int stretched = (input_image_[i] - min_val_) * 255 / range;
          stretched = std::clamp(stretched, 0, 255);
          output_image_[i] = stretched;
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  return true;
}

bool HistogramStretchSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_image_.size(); ++i) {
    out_ptr[i] = output_image_[i];
  }
  return true;
}

}  // namespace muradov_k_histogram_stretch_stl
