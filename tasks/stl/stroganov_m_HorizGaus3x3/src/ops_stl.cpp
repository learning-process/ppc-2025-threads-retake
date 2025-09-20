#include "stl/stroganov_m_HorizGaus3x3/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool stroganov_m_horiz_gaus3x3_stl::ImageFilterStl::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);
  return true;
}

bool stroganov_m_horiz_gaus3x3_stl::ImageFilterStl::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  return (task_data->inputs_count[0] == task_data->outputs_count[0]) && (kernel_.size() == 3) &&
         (sqrt_size * sqrt_size == size);
}

bool stroganov_m_horiz_gaus3x3_stl::ImageFilterStl::RunImpl() {
  double sum = kernel_[0] + kernel_[1] + kernel_[2];
  double inv_sum = (sum == 0.0) ? 1.0 : 1.0 / sum;
  double k0_inv = kernel_[0] * inv_sum;
  double k1_inv = kernel_[1] * inv_sum;
  double k2_inv = kernel_[2] * inv_sum;
  const int width = width_;
  const int height = height_;

  const auto num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const int rows_per_thread = height / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    const int start_row = t * rows_per_thread;
    const int end_row = (t == num_threads - 1) ? height : (start_row + rows_per_thread);
    threads.emplace_back([=, &input = input_, &output = output_] {
      for (int i = start_row; i < end_row; ++i) {
        const int row_offset = i * width;
        output[row_offset] = (k1_inv * input[row_offset]) + (k2_inv * input[row_offset + 1]);
        for (int j = 1; j < width - 1; ++j) {
          const int idx = row_offset + j;
          output[idx] = (k0_inv * input[idx - 1]) + (k1_inv * input[idx]) + (k2_inv * input[idx + 1]);
        }
        const int last_idx = row_offset + width - 1;
        output[last_idx] = (k0_inv * input[last_idx - 1]) + (k1_inv * input[last_idx]);
      }
    });
  }
  for (auto &t : threads) {
    t.join();
  }
  return true;
}

bool stroganov_m_horiz_gaus3x3_stl::ImageFilterStl::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }
  return true;
}
