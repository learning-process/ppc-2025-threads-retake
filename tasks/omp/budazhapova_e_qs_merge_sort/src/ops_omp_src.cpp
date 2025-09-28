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

struct MergeRange {
  int start;
  int mid;
  int end;
};

void SequentialMerge(std::vector<int>& arr, const MergeRange& range) {
  if (range.start > range.end || range.mid < range.start || range.mid > range.end) {
    return;
  }

  std::vector<int> temp(range.end - range.start + 1);
  int i = range.start;
  int j = range.mid + 1;
  int k = 0;

  while (i <= range.mid && j <= range.end) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= range.mid) {
    temp[k++] = arr[i++];
  }
  while (j <= range.end) {
    temp[k++] = arr[j++];
  }

  for (i = range.start, k = 0; i <= range.end; i++, k++) {
    arr[i] = temp[k];
  }
}

void ParallelMerge(std::vector<int>& arr, const MergeRange& range) {
  if (range.start >= range.end || range.mid < range.start || range.mid >= range.end) {
    return;
  }

  const int merge_threshold = 500;
  if (range.end - range.start + 1 <= merge_threshold) {
    SequentialMerge(arr, range);
    return;
  }

  int i = range.start + ((range.mid - range.start) / 2);
  if (i < range.start || i > range.mid) {
    i = range.start;
  }

  auto start_it = arr.begin() + (range.mid + 1);
  auto end_it = arr.begin() + range.end + 1;
  if (start_it >= arr.end() || end_it > arr.end()) {
    SequentialMerge(arr, range);
    return;
  }

  auto j_iter = std::lower_bound(start_it, end_it, arr[i]);
  int j = static_cast<int>(j_iter - arr.begin());

  if (range.start <= i - 1 && j - 1 >= range.start && i - 1 >= range.start) {
    MergeRange left_range{/*start=*/range.start, /*mid=*/i - 1, /*end=*/j - 1};
    if (left_range.start <= left_range.mid && left_range.mid <= left_range.end) {
#pragma omp task default(none) shared(arr) firstprivate(left_range)
      { ParallelMerge(arr, left_range); }
    }
  }

  if (i <= range.mid && range.end >= j && j <= range.end) {
    MergeRange right_range{/*start=*/i, /*mid=*/range.mid, /*end=*/range.end};
    if (right_range.start <= right_range.mid && right_range.mid <= right_range.end) {
#pragma omp task default(none) shared(arr) firstprivate(right_range)
      { ParallelMerge(arr, right_range); }
    }
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
#pragma omp task default(none) shared(arr) firstprivate(low, pi)
      { QuickSortMergeParallel(arr, low, pi); }
    }

    if (pi + 1 < high) {
#pragma omp task default(none) shared(arr) firstprivate(pi, high)
      { QuickSortMergeParallel(arr, pi + 1, high); }
    }

#pragma omp taskwait

    if (low <= pi && pi < high) {
      MergeRange merge_range{/*start=*/low, /*mid=*/pi, /*end=*/high};
      if (merge_range.start <= merge_range.mid && merge_range.mid < merge_range.end) {
        ParallelMerge(arr, merge_range);
      }
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
#pragma omp single nowait
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