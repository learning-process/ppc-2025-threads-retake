#include "omp/strakhov_a_double_radix_merge/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
bool strakhov_a_double_radix_merge_omp::DoubleRadixMergeOmp::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  unsigned int size = task_data->inputs_count[0];
  input_ = std::vector<double>(in_ptr, in_ptr + size);

  output_ = std::vector<double>(size, 0);

  return true;
}

bool strakhov_a_double_radix_merge_omp::DoubleRadixMergeOmp::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool strakhov_a_double_radix_merge_omp::DoubleRadixMergeOmp::RunImpl() {
  unsigned int size = task_data->inputs_count[0];
  if (size == 0) {
    return true;
  }

  const int type_length = sizeof(double) * 8;

  // float to uint64_t
  std::vector<uint64_t> temp_vector(size);
  for (unsigned int i = 0; i < size; ++i) {
    uint64_t bits = 0;
    std::memcpy(&bits, &input_[i], sizeof(double));
    if ((bits >> 63) != 0ULL) {
      // negative: invert all bits
      temp_vector[i] = ~bits;
    } else {
      // non-negative: flip sign bit
      temp_vector[i] = bits ^ 0x8000000000000000ULL;
    }
  }

  std::vector<uint64_t> true_vector(size);
  std::vector<uint64_t> false_vector(size);

  for (int i = 0; i < type_length; ++i) {
    const uint64_t bit_mask = (uint64_t{1} << i);
    unsigned int cnt_true = 0;
    unsigned int cnt_false = 0;
    for (unsigned int j = 0; j < size; ++j) {
      if ((temp_vector[j] & bit_mask) == bit_mask) {
        true_vector[cnt_true++] = temp_vector[j];
      } else {
        false_vector[cnt_false++] = temp_vector[j];
      }
    }

    if (cnt_false > 0) {
      std::memcpy(temp_vector.data(), false_vector.data(), static_cast<size_t>(cnt_false) * sizeof(uint64_t));
    }

    if (cnt_true > 0) {
      std::memcpy(temp_vector.data() + cnt_false, true_vector.data(), static_cast<size_t>(cnt_true) * sizeof(uint64_t));
    }
  }

  // uint64 to float
  for (unsigned int i = 0; i < size; ++i) {
    uint64_t k = temp_vector[i];
    if ((k >> 63) != 0ULL) {
      k ^= 0x8000000000000000ULL;
    } else {
      k = ~k;
    }
    std::memcpy(&output_[i], &k, sizeof(double));
  }

  return true;
}

bool strakhov_a_double_radix_merge_omp::DoubleRadixMergeOmp::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
