#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "omp/budazhapova_e_qs_merge_sort/include/ops_omp_inc.hpp"

namespace budazhapova_e_qs_merge_sort_omp {
namespace {
int PartitionHoare(std::vector<int>& arr, int low, int high) {
  int pivot = arr[low + ((high - low) / 2)];
  int i = low;
  int j = high;

  while (true) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }
    if (i >= j) {
      return j;
    }
    std::swap(arr[i], arr[j]);
    i++;
    j--;
  }
}

struct MergeRange {
  int start;
  int mid;
  int end;
};

void SequentialMerge(std::vector<int>& arr, const MergeRange& range) {
  if (range.start >= range.end) {
    return;
  }

  std::vector<int> temp(range.end - range.start + 1);
  int i = range.start;
  int j = range.mid + 1;
  int k = 0;

  while (i <= range.mid && j <= range.end) {
    if (arr[i] <= arr[j]) {
      temp[k] = arr[i];
      k++;
      i++;
    } else {
      temp[k] = arr[j];
      k++;
      j++;
    }
  }

  while (i <= range.mid) {
    temp[k] = arr[i];
    k++;
    i++;
  }
  while (j <= range.end) {
    temp[k] = arr[j];
    k++;
    j++;
  }

  for (i = range.start, k = 0; i <= range.end; i++, k++) {
    arr[i] = temp[k];
  }
}

void ParallelMerge(std::vector<int>& arr, const MergeRange& range) {
  if (range.start >= range.end) {
    return;
  }

  const int merge_threshold = 500;

  if (range.end - range.start + 1 <= merge_threshold) {
    SequentialMerge(arr, range);
    return;
  }

  int i = range.start + ((range.mid - range.start) / 2);

  auto start_it = arr.begin() + range.mid + 1;
  auto end_it = arr.begin() + range.end + 1;
  auto j_iter = std::lower_bound(start_it, end_it, arr[i]);
  int j = static_cast<int>(j_iter - arr.begin());

  if (range.start <= i - 1 && j - 1 >= range.start) {
    MergeRange left_range{.start = range.start, .mid = i - 1, .end = j - 1};
#pragma omp task default(none) shared(arr) firstprivate(left_range)
    { ParallelMerge(arr, left_range); }
  }

  if (i <= range.mid && range.end >= i) {
    MergeRange right_range{.start = i, .mid = range.mid, .end = range.end};
#pragma omp task default(none) shared(arr) firstprivate(right_range)
    { ParallelMerge(arr, right_range); }
  }

#pragma omp taskwait
}

void QuickSortMergeParallel(std::vector<int>& arr, int low, int high) {
  if (low >= high) {
    return;
  }

  const int min_size = 10;

  if (low < high && high - low >= min_size) {
    int pi = PartitionHoare(arr, low, high);

#pragma omp task default(none) shared(arr) firstprivate(low, pi)
    { QuickSortMergeParallel(arr, low, pi); }

#pragma omp task default(none) shared(arr) firstprivate(pi, high)
    { QuickSortMergeParallel(arr, pi + 1, high); }

#pragma omp taskwait

    if (low <= pi && pi <= high) {
      MergeRange merge_range{.start = low, .mid = pi, .end = high};
      ParallelMerge(arr, merge_range);
    }
  } else if (low < high) {
    int pi = PartitionHoare(arr, low, high);
    if (low < pi) {
      QuickSortMergeParallel(arr, low, pi);
    }
    if (pi + 1 < high) {
      QuickSortMergeParallel(arr, pi + 1, high);
    }
    MergeRange merge_range{.start = low, .mid = pi, .end = high};
    SequentialMerge(arr, merge_range);
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