#include "seq/chastov_v_shell_sort_with_even_odd_batcher_merge/include/ops_seq.hpp"

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

  auto step_sizes = ComputeGapSequence(static_cast<int>(total_elements));

  for (size_t step_size : step_sizes) {
    for (size_t i = step_size; i < total_elements; i++) {
      int tmp = data[i];
      size_t j = i;
      while (j >= step_size && data[j - step_size] > tmp) {
        data[j] = data[j - step_size];
        j = j - step_size;
      }
      data[j] = tmp;
    }
  }

  size_t center = total_elements / 2;
  size_t finish = total_elements;
  if (center < finish) {
    BatcherMerge(data, 0, center, finish);
  }
}
}  // namespace

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential::PreProcessingImpl() {
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

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential::RunImpl() {
  EnhancedShellSort(input_data_);
  return true;
}

bool chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential::PostProcessingImpl() {
  int *output_destination = reinterpret_cast<int *>(task_data->outputs[0]);
  size_t output_size = task_data->outputs_count[0];

  for (size_t i = 0; i < output_size; ++i) {
    output_destination[i] = input_data_[i];
  }

  return true;
}
