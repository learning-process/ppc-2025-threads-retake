#include "omp/ersoz_b_hoare_sort_simple_merge/include/ops_omp.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ersoz_b_hoare_sort_simple_merge_omp {

bool HoareSortSimpleMergeOpenMP::ValidationImpl() {
  if (task_data->inputs.size() != 1U || task_data->outputs.size() != 1U) {
    return false;
  }
  if (task_data->inputs_count.size() != 1U || task_data->outputs_count.size() != 1U) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (task_data->inputs_count[0] == 0U) {
    return false;
  }
  return true;
}

bool HoareSortSimpleMergeOpenMP::PreProcessingImpl() {
  const auto n = task_data->inputs_count[0];
  if (n == 0U) {
    input_.clear();
    output_.clear();
    return true;
  }
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + n);
  output_.assign(n, 0);
  return true;
}

void HoareSortSimpleMergeOpenMP::QuickSortHoare(std::vector<int>& a, long long l, long long r) {
  if (l >= r) {
    return;
  }
  long long i = l;
  long long j = r;
  const long long pivot_index = l + ((r - l) / 2);
  int pivot = a[pivot_index];
  while (i <= j) {
    while (a[i] < pivot) {
      ++i;
    }
    while (a[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(a[i], a[j]);
      ++i;
      --j;
    }
  }
  if (l < j) {
    QuickSortHoare(a, l, j);
  }
  if (i < r) {
    QuickSortHoare(a, i, r);
  }
}

void HoareSortSimpleMergeOpenMP::MergeTwo(const std::vector<int>& src, Segment left, Segment right,
                                          std::vector<int>& dst) {
  if (dst.empty() || src.empty()) {
    return;
  }
  left.end = std::min(left.end, src.size());
  right.end = std::min(right.end, src.size());
  if (left.begin >= left.end && right.begin >= right.end) {
    return;
  }
  std::size_t i = left.begin;
  std::size_t j = right.begin;
  std::size_t k = left.begin;
  while (i < left.end && j < right.end) {
    if (src[i] <= src[j]) {
      dst[k] = src[i];
      ++i;
    } else {
      dst[k] = src[j];
      ++j;
    }
    ++k;
  }
  while (i < left.end) {
    dst[k] = src[i];
    ++i;
    ++k;
  }
  while (j < right.end) {
    dst[k] = src[j];
    ++j;
    ++k;
  }
}

bool HoareSortSimpleMergeOpenMP::RunImpl() {
  const std::size_t n = input_.size();
  if (n <= 1U) {
    output_.resize(n);
    if (n == 1U) {
      output_[0] = input_[0];
    }
    return true;
  }
  output_.resize(n);
  const std::size_t mid = n / 2U;
#pragma omp parallel sections default(none) shared(mid, n, input_)
  {
#pragma omp section
    { QuickSortHoare(input_, 0, static_cast<long long>(mid) - 1); }
#pragma omp section
    { QuickSortHoare(input_, static_cast<long long>(mid), static_cast<long long>(n) - 1); }
  }
  MergeTwo(input_, Segment{.begin = 0, .end = mid}, Segment{.begin = mid, .end = n}, output_);
  return true;
}

bool HoareSortSimpleMergeOpenMP::PostProcessingImpl() {
  if (output_.empty()) {
    return true;
  }
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (std::size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace ersoz_b_hoare_sort_simple_merge_omp
