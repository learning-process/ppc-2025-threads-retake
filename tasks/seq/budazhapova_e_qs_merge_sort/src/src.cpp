
#include <cmath>
#include <cstddef>
#include <vector>

#include "seq/budazhapova_e_qs_merge_sort/include/inc.h"

namespace budazhapova_e_qs_merge_sort_seq {
namespace {
void quickSortHoare(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pi = partitionHoare(arr, low, high);
    quickSortHoare(arr, low, pi);
    quickSortHoare(arr, pi + 1, high);
  }
}

int partitionHoare(std::vector<int>& arr, int low, int high) {
  int pivot = arr[(low + high) / 2];
  int i = low;
  int j = high;

  while (i <= j) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;

    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

  return j;
}
}  // namespace
}  // namespace budazhapova_e_qs_merge_sort_seq

bool budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential::RunImpl() {
  output_ = input_;

  if (!output_.empty()) {
    quickSortHoare(output_, 0, static_cast<int>(output_.size()) - 1);
  }

  return true;
}

bool budazhapova_e_qs_merge_sort_seq::QSMergeSortSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}