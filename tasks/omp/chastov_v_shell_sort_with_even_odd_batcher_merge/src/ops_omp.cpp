#include "omp/chastov_v_shell_sort_with_even_odd_batcher_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace {

std::vector<size_t> ComputeGapSequence(int n) {
  std::vector<size_t> step_sizes;
  int k = 0;

  while (true) {
    int step_size = 0;
    if (k % 2 == 0) {
      step_size = 9 * (1 << (2 * k)) - 9 * (1 << k) + 1;
    } else {
      step_size = 8 * (1 << k) - 6 * (1 << ((k + 1) / 2)) + 1;
    }

    if (step_size > n / 2) {
      break;
    }

    step_sizes.push_back(static_cast<size_t>(step_size));
    k = k + 1;
  }

  if (step_sizes.empty() || step_sizes.back() != 1) {
    step_sizes.push_back(1);
  }

  std::ranges::reverse(step_sizes.begin(), step_sizes.end());
  return step_sizes;
}

void BatcherMerge(std::vector<int> &data, size_t begin, size_t center, size_t finish) {
  auto begin_iter = std::next(data.begin(), static_cast<std::ptrdiff_t>(begin));
  auto center_iter = std::next(data.begin(), static_cast<std::ptrdiff_t>(center));
  auto finish_iter = std::next(data.begin(), static_cast<std::ptrdiff_t>(finish));

  std::vector<int> left(begin_iter, center_iter);
  std::vector<int> right(center_iter, finish_iter);

  size_t l_pos = 0;
  size_t r_pos = 0;
  size_t position = begin;

  size_t l_length = center - begin;
  size_t r_length = finish - center;

  for (size_t i = begin; i < finish; i++) {
    if (i % 2 == 0) {
      if (l_pos < l_length && (r_pos >= r_length || left[l_pos] <= right[r_pos])) {
        data[position] = left[l_pos];
        position++;
        l_pos++;
      } else {
        data[position] = right[r_pos];
        position++;
        r_pos++;
      }
    } else {
      if (r_pos < r_length && (l_pos >= l_length || right[r_pos] <= left[l_pos])) {
        data[position] = right[r_pos];
        position++;
        r_pos++;
      } else {
        data[position] = left[l_pos];
        position++;
        l_pos++;
      }
    }
  }
}

void EnhancedShellSort(std::vector<int> &data) {
  size_t total_elements = data.size();
  if (total_elements <= 1) {
    return;
  }

  int threads_count = omp_get_max_threads();
  size_t chunk_size = (total_elements + threads_count - 1) / threads_count;

#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();

    size_t chunk_begin = static_cast<size_t>(thread_idx) * chunk_size;
    size_t chunk_end = std::min(chunk_begin + chunk_size, total_elements) - 1;

    if (chunk_begin < total_elements) {
      auto local_size = static_cast<int>(chunk_end - chunk_begin + 1);
      auto step_sizes = ComputeGapSequence(local_size);

      for (size_t step_size : step_sizes) {
        for (size_t i = chunk_begin + step_size; i <= chunk_end; i++) {
          int tmp = data[i];
          size_t j = i;
          while (j >= chunk_begin + step_size && data[j - step_size] > tmp) {
            data[j] = data[j - step_size];
            j -= step_size;
          }
          data[j] = tmp;
        }
      }
    }
  }

  for (size_t merge_size = chunk_size; merge_size < total_elements; merge_size *= 2) {
#pragma omp parallel for schedule(static)
    for (int k = 0; k < static_cast<int>(total_elements); k += static_cast<int>(2 * merge_size)) {
      size_t center = std::min(k + merge_size, total_elements);
      size_t finish = std::min(k + (2 * merge_size), total_elements);
      if (center < finish) {
        BatcherMerge(data, k, center, finish);
      }
    }
  }
}
}  // namespace

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskOpenMP::PreProcessingImpl() {
  size_t data_count = task_data->inputs_count[0];
  uint8_t *input_buffer = task_data->inputs[0];

  input_data_.clear();
  input_data_.reserve(data_count);

  for (size_t i = 0; i < data_count; ++i) {
    int value = *reinterpret_cast<int *>(input_buffer + (i * sizeof(int)));
    input_data_.push_back(value);
  }

  return true;
}

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskOpenMP::RunImpl() {
  EnhancedShellSort(input_data_);
  return true;
}

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskOpenMP::PostProcessingImpl() {
  int *output_destination = reinterpret_cast<int *>(task_data->outputs[0]);
  size_t output_size = task_data->outputs_count[0];

  for (size_t i = 0; i < output_size; ++i) {
    output_destination[i] = input_data_[i];
  }

  return true;
}
