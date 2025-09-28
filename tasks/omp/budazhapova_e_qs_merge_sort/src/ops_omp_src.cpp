#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "omp/budazhapova_e_qs_merge_sort/include/ops_omp_inc.hpp"

namespace budazhapova_e_qs_merge_sort_omp {
namespace {

bool IsValidRange(int low, int high, size_t size) {
  return low >= 0 && high >= low && static_cast<size_t>(high) < size;
}

int PartitionHoare(std::vector<int>& arr, int low, int high) {
  if (!IsValidRange(low, high, arr.size())) {
    return low;
  }

  int pivot = arr[low + ((high - low) / 2)];
  int i = low - 1;
  int j = high + 1;

  while (true) {
    do {
      i++;
    } while (arr[i] < pivot);

    do {
      j--;
    } while (arr[j] > pivot);

    if (i >= j) {
      return j;
    }
    std::swap(arr[i], arr[j]);
  }
}

void SequentialMerge(std::vector<int>& arr, int start, int mid, int end) {
  if (start > end || mid < start || mid > end) {
    return;
  }

  std::vector<int> temp(end - start + 1);
  int i = start;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= end) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= end) {
    temp[k++] = arr[j++];
  }

  for (i = start, k = 0; i <= end; i++, k++) {
    arr[i] = temp[k];
  }
}

void ParallelMerge(std::vector<int>& arr, int start, int mid, int end) {
  if (start >= end || mid < start || mid >= end) {
    return;
  }

  const int merge_threshold = 500;
  if (end - start + 1 <= merge_threshold) {
    SequentialMerge(arr, start, mid, end);
    return;
  }

  int i = start + ((mid - start) / 2);
  if (i < start || i > mid) {
    i = start;
  }

  auto start_it = arr.begin() + (mid + 1);
  auto end_it = arr.begin() + end + 1;
  if (start_it >= arr.end() || end_it > arr.end()) {
    SequentialMerge(arr, start, mid, end);
    return;
  }

  auto j_iter = std::lower_bound(start_it, end_it, arr[i]);
  int j = static_cast<int>(j_iter - arr.begin());

  if (start <= i - 1 && j - 1 >= start && i - 1 >= start) {
#pragma omp task
    { ParallelMerge(arr, start, i - 1, j - 1); }
  }

  if (i <= mid && end >= j && j <= end) {
#pragma omp task
    { ParallelMerge(arr, i, mid, end); }
  }

#pragma omp taskwait
}

void QuickSortMergeParallel(std::vector<int>& arr, int low, int high) {
  if (low >= high || low < 0 || static_cast<size_t>(high) >= arr.size()) {
    return;
  }

  const int min_size = 1000;

  if (high - low + 1 > min_size) {
    int pi = PartitionHoare(arr, low, high);

    if (pi < low || pi > high) {
      pi = low + ((high - low) / 2);
    }

    if (low < pi) {
#pragma omp task
      { QuickSortMergeParallel(arr, low, pi); }
    }

    if (pi + 1 < high) {
#pragma omp task
      { QuickSortMergeParallel(arr, pi + 1, high); }
    }

#pragma omp taskwait

    if (low <= pi && pi < high) {
      ParallelMerge(arr, low, pi, high);
    }
  } else {
    std::sort(arr.begin() + low, arr.begin() + high + 1);
  }
}

}  // namespace

bool budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP::RunImpl() {
  output_ = input_;

  if (!output_.empty() && output_.size() > 1) {
#pragma omp parallel
    {
#pragma omp single
      { QuickSortMergeParallel(output_, 0, static_cast<int>(output_.size()) - 1); }
    }
  }

  return true;
}

bool budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
}  // namespace budazhapova_e_qs_merge_sort_omp