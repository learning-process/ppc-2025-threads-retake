#include "tbb/ersoz_b_hoare_sort_simple_merge/include/ops_tbb.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "oneapi/tbb/task_group.h"

namespace ersoz_b_hoare_sort_simple_merge_tbb {
namespace {

void HoarePartition(std::vector<int>& a, long long l, long long r) {
  long long i = l;
  long long j = r;
  const long long pivot_index = l + ((r - l) / 2);
  const int pivot = a[pivot_index];
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
    HoarePartition(a, l, j);
  }
  if (i < r) {
    HoarePartition(a, i, r);
  }
}

}  // namespace

void HoareSortSimpleMergeTBB::QuickSortHoare(std::vector<int>& a, long long l, long long r) {
  if (l >= r) {
    return;
  }
  HoarePartition(a, l, r);
}

void HoareSortSimpleMergeTBB::MergeTwo(const std::vector<int>& src, Segment left, Segment right,
                                       std::vector<int>& dst) {
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

bool HoareSortSimpleMergeTBB::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  const std::size_t n = task_data->inputs_count[0];
  input_.resize(n);
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (std::size_t i = 0; i < n; ++i) {
    input_[i] = in_ptr[i];
  }
  output_.assign(n, 0);
  return true;
}

bool HoareSortSimpleMergeTBB::ValidationImpl() {
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->inputs_count[0] == 0U) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool HoareSortSimpleMergeTBB::RunImpl() {
  const std::size_t n = input_.size();
  if (n == 0U) {
    output_.clear();
    return true;
  }

  const std::size_t mid = n / 2U;

  oneapi::tbb::task_group tg;
  tg.run([this, mid]() { QuickSortHoare(input_, 0, static_cast<long long>(mid == 0 ? 0 : (mid - 1))); });
  tg.run([this, n, mid]() {
    if (mid < n) {
      QuickSortHoare(input_, static_cast<long long>(mid), static_cast<long long>(n - 1));
    }
  });
  tg.wait();

  MergeTwo(input_, Segment{.begin = 0U, .end = mid}, Segment{.begin = mid, .end = n}, output_);
  return true;
}

bool HoareSortSimpleMergeTBB::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (std::size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace ersoz_b_hoare_sort_simple_merge_tbb
