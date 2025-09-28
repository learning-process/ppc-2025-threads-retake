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

void SequentialMerge(std::vector<int>& arr, int left, int middle, int right) {
  std::vector<int> temp(right - left + 1);
  int i = left;
  int j = middle + 1;
  int k = 0;

  while (i <= middle && j <= right) {
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

  while (i <= middle) {
    temp[k] = arr[i];
    k++;
    i++;
  }
  while (j <= right) {
    temp[k] = arr[j];
    k++;
    j++;
  }

  for (i = left, k = 0; i <= right; i++, k++) {
    arr[i] = temp[k];
  }
}

void ParallelMerge(std::vector<int>& arr, int left, int middle, int right) {
  const int merge_threshold = 500;

  if (right - left + 1 <= merge_threshold) {
    SequentialMerge(arr, left, middle, right);
    return;
  }

  int i = left + ((middle - left) / 2);
  auto j_iter = std::lower_bound(arr.begin() + middle + 1, arr.begin() + right + 1, arr[i]);
  int j = static_cast<int>(j_iter - arr.begin());

#pragma omp task default(none) shared(arr) firstprivate(left, i, middle, j)
  { ParallelMerge(arr, left, i - 1, j - 1); }

#pragma omp task default(none) shared(arr) firstprivate(i, middle, j, right)
  { ParallelMerge(arr, i, middle, right); }

#pragma omp taskwait
}

void QuickSortMergeParallel(std::vector<int>& arr, int low, int high) {
  const int min_size = 10;

  if (low < high && high - low >= min_size) {
    int pi = PartitionHoare(arr, low, high);
#pragma omp task default(none) shared(arr) firstprivate(low, pi)
    { QuickSortMergeParallel(arr, low, pi); }

#pragma omp task default(none) shared(arr) firstprivate(pi, high)
    { QuickSortMergeParallel(arr, pi + 1, high); }

#pragma omp taskwait

    ParallelMerge(arr, low, pi, high);
  } else if (low < high) {
    int pi = PartitionHoare(arr, low, high);
    if (low < pi) {
      QuickSortMergeParallel(arr, low, pi);
    }
    if (pi + 1 < high) {
      QuickSortMergeParallel(arr, pi + 1, high);
    }
    SequentialMerge(arr, low, pi, high);
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