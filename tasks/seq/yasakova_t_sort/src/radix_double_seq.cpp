#include "seq/yasakova_t_sort/include/radix_double_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <vector>

namespace yasakova_t_sort_seq {

bool SortTaskSequential::ValidationImpl() {
  if (!task_data) {
    return false;
  }
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SortTaskSequential::PreProcessingImpl() {
  const auto count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* input_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskSequential::RunImpl() {
  output_ = input_;
  RadixSortDoubleSeq(output_);
  return true;
}

bool SortTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_, output_ptr);
  return true;
}

void RadixSortDoubleSeq(std::vector<double>& data) {
  std::vector<double> nonnan;
  nonnan.reserve(data.size());
  std::vector<double> nans;
  nans.reserve(16);
  for (double value : data) {
    (IsNan(value) ? nans : nonnan).push_back(value);
  }

  const auto n = nonnan.size();
  if (n <= 1) {
    data = std::move(nonnan);
    data.insert(data.end(), nans.begin(), nans.end());
    return;
  }

  std::vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = ToKey(nonnan[i]);
  }

  std::vector<uint64_t> buffer(n);
  for (int pass = 0; pass < 8; ++pass) {
    std::array<size_t, 256> counts{};
    const int shift = pass * 8;

    for (size_t i = 0; i < n; ++i) {
      ++counts[static_cast<uint8_t>(keys[i] >> shift)];
    }

    size_t sum = 0;
    for (int bucket = 0; bucket < 256; ++bucket) {
      const auto current = counts[bucket];
      counts[bucket] = sum;
      sum += current;
    }

    for (size_t i = 0; i < n; ++i) {
      const auto bucket = static_cast<uint8_t>(keys[i] >> shift);
      buffer[counts[bucket]++] = keys[i];
    }
    keys.swap(buffer);
  }

  for (size_t i = 0; i < n; ++i) {
    nonnan[i] = FromKey(keys[i]);
  }

  data = std::move(nonnan);
  data.insert(data.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_seq
