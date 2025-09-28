#include "stl/ersoz_b_hoare_sort_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ersoz_b_hoare_sort_simple_merge_stl {

long long HoareSortSimpleMergeSTL::Partition(std::vector<int>& a, long long l, long long r) {
  int pivot = a[l + ((r - l) / 2)];
  long long i = l;
  long long j = r;
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
  return i;
}

void HoareSortSimpleMergeSTL::QuickSortHoare(std::vector<int>& a, long long l, long long r) {
  if (l >= r) {
    return;
  }
  long long i = l;
  long long j = r;
  int pivot = a[l + ((r - l) / 2)];
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

void HoareSortSimpleMergeSTL::MergeTwo(const std::vector<int>& src, Segment left, Segment right,
                                       std::vector<int>& dst) {
  std::merge(src.begin() + static_cast<long long>(left.begin), src.begin() + static_cast<long long>(left.end),
             src.begin() + static_cast<long long>(right.begin), src.begin() + static_cast<long long>(right.end),
             dst.begin());
}

bool HoareSortSimpleMergeSTL::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  auto n = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + n);
  output_.assign(n, 0);
  return true;
}

bool HoareSortSimpleMergeSTL::ValidationImpl() {
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool HoareSortSimpleMergeSTL::RunImpl() {
  std::size_t n = input_.size();
  if (n <= 1) {
    output_ = input_;
    return true;
  }
  long long mid = Partition(input_, 0, static_cast<long long>(n - 1));
  if (mid > 0) {
    QuickSortHoare(input_, 0, mid - 1);
  }
  if (mid < static_cast<long long>(n)) {
    QuickSortHoare(input_, mid, static_cast<long long>(n - 1));
  }
  MergeTwo(input_, Segment{.begin = 0, .end = static_cast<std::size_t>(mid)},
           Segment{.begin = static_cast<std::size_t>(mid), .end = n}, output_);
  return true;
}

bool HoareSortSimpleMergeSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (std::size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace ersoz_b_hoare_sort_simple_merge_stl
