#include "seq/ersoz_b_hoare_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

using ersoz_b_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential;

bool HoareSortSimpleMergeSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  output_ = std::vector<int>(input_size);
  return true;
}

bool HoareSortSimpleMergeSequential::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void HoareSortSimpleMergeSequential::QuickSortHoare(std::vector<int>& a, long long l, long long r) {
  long long i = l;
  long long j = r;
  long long pivot_index = l + ((r - l) / 2);
  int pivot = a[pivot_index];
  while (i <= j) {
    while (a[i] < pivot) {
      i++;
    }
    while (a[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(a[i], a[j]);
      i++;
      j--;
    }
  }
  if (l < j) {
    QuickSortHoare(a, l, j);
  }
  if (i < r) {
    QuickSortHoare(a, i, r);
  }
}

void HoareSortSimpleMergeSequential::MergeTwo(const std::vector<int>& src, std::pair<std::size_t, std::size_t> left,
                                              std::pair<std::size_t, std::size_t> right, std::vector<int>& dst) {
  std::size_t l = left.first;
  std::size_t m = left.second;
  std::size_t r = right.second;

  std::size_t i = l;
  std::size_t j = m;
  std::size_t k = l;
  while (i < m && j < r) {
    if (src[i] <= src[j]) {
      dst[k++] = src[i++];
    } else {
      dst[k++] = src[j++];
    }
  }
  while (i < m) {
    dst[k++] = src[i++];
  }
  while (j < r) {
    dst[k++] = src[j++];
  }
}

bool HoareSortSimpleMergeSequential::RunImpl() {
  if (input_.empty()) {
    output_.clear();
    return true;
  }
  std::size_t n = input_.size();
  std::size_t mid = n / 2;
  if (mid > 0) {
    QuickSortHoare(input_, 0, static_cast<long long>(mid - 1));
  }
  if (mid < n) {
    QuickSortHoare(input_, static_cast<long long>(mid), static_cast<long long>(n - 1));
  }
  MergeTwo(input_, {0, mid}, {mid, n}, output_);
  return true;
}

bool HoareSortSimpleMergeSequential::PostProcessingImpl() {
  for (std::size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
