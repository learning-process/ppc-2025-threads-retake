#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

#include "tbb/budazhapova_e_qs_merge_sort/include/ops_tbb.hpp"

namespace budazhapova_e_qs_merge_sort_tbb {
namespace {
//empty comment for clangggg
const int kSequentialThreshold = 1000;
const int kParallelMergeThreshold = 5000;

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

void SequentialQuickSort(std::vector<int>& arr, int low, int high) {
  if (low >= high || low < 0 || static_cast<size_t>(high) >= arr.size()) {
    return;
  }

  struct Range {
    int low;
    int high;
  };
  std::vector<Range> stack;
  stack.push_back({low, high});

  while (!stack.empty()) {
    Range current = stack.back();
    stack.pop_back();

    int low_val = current.low;
    int high_val = current.high;

    if (low_val >= high_val) {
      continue;
    }

    int pi = PartitionHoare(arr, low_val, high_val);

    if (pi - low_val > high_val - pi) {
      stack.push_back({low_val, pi});
      stack.push_back({pi + 1, high_val});
    } else {
      stack.push_back({pi + 1, high_val});
      stack.push_back({low_val, pi});
    }
  }
}

void SequentialMerge(std::vector<int>& arr, int left, int mid, int right) {
  if (left >= right || mid < left || mid >= right) {
    return;
  }

  std::vector<int> temp(right - left + 1);
  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= right) {
    temp[k++] = arr[j++];
  }

  for (i = left, k = 0; i <= right; i++, k++) {
    arr[i] = temp[k];
  }
}

void ParallelMerge(std::vector<int>& arr, int left, int mid, int right) {
  if (left >= right || mid < left || mid >= right) {
    return;
  }

  if (right - left + 1 <= kParallelMergeThreshold) {
    SequentialMerge(arr, left, mid, right);
    return;
  }

  int left_mid = left + ((mid - left) / 2);

  auto it = std::lower_bound(arr.begin() + mid + 1, arr.begin() + right + 1, arr[left_mid]);
  auto right_pos = std::distance(arr.begin(), it);

  tbb::parallel_invoke(
      [&] {
        if (left <= left_mid - 1 && right_pos - 1 >= left) {
          ParallelMerge(arr, left, left_mid - 1, static_cast<int>(right_pos - 1));
        }
      },
      [&] {
        if (left_mid <= mid && right >= right_pos) {
          ParallelMerge(arr, left_mid, mid, right);
        }
      });
}

void HybridSort(std::vector<int>& arr, int low, int high, bool parallel = true) {
  if (low >= high || low < 0 || static_cast<size_t>(high) >= arr.size()) {
    return;
  }

  if (high - low + 1 <= kSequentialThreshold) {
    SequentialQuickSort(arr, low, high);
    return;
  }

  int pi = PartitionHoare(arr, low, high);

  if (parallel) {
    tbb::parallel_invoke([&] { HybridSort(arr, low, pi, true); }, [&] { HybridSort(arr, pi + 1, high, true); });
    ParallelMerge(arr, low, pi, high);
  } else {
    HybridSort(arr, low, pi, false);
    HybridSort(arr, pi + 1, high, false);
    SequentialMerge(arr, low, pi, high);
  }
}

}  // namespace

bool budazhapova_e_qs_merge_sort_tbb::QSMergeSortTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool budazhapova_e_qs_merge_sort_tbb::QSMergeSortTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool budazhapova_e_qs_merge_sort_tbb::QSMergeSortTBB::RunImpl() {
  output_ = input_;

  if (!output_.empty() && output_.size() > 1) {
    if (output_.size() <= kSequentialThreshold) {
      std::ranges::sort(output_);
    } else {
      oneapi::tbb::task_arena arena;
      arena.execute([&] { HybridSort(output_, 0, static_cast<int>(output_.size()) - 1, true); });
    }
  }

  return true;
}

bool budazhapova_e_qs_merge_sort_tbb::QSMergeSortTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
}  // namespace budazhapova_e_qs_merge_sort_tbb