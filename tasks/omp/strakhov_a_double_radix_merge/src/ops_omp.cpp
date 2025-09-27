#include "omp/strakhov_a_double_radix_merge/include/ops_omp.hpp"

#include <algorithm>
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

inline void strakhov_a_double_radix_merge_omp::MergeSorted(size_t left_start, size_t left_end, const uint64_t *input,
                                                           size_t right_start, size_t right_end, uint64_t *output,
                                                           size_t output_start) {
  size_t i = left_start;
  size_t j = right_start;
  size_t k = output_start;
  while ((i < left_end) && (j < right_end)) {
    if (input[i] <= input[j]) {
      output[k] = input[i];
      i++;
      k++;
    } else {
      output[k] = input[j];
      j++;
      k++;
    }
  }
  while (i < left_end) {
    output[k] = input[i];
    i++;
    k++;
  }
  while (j < right_end) {
    output[k] = input[j];
    j++;
    k++;
  }
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
#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < size; ++i) {
    uint64_t bits = 0;
    std::memcpy(&bits, &input_[i], sizeof(double));
    if ((bits >> 63) != 0ULL) {
      temp_vector[i] = ~bits;
    } else {
      temp_vector[i] = bits ^ 0x8000000000000000ULL;
    }
  }
  // radix sort
  const size_t n = static_cast<size_t>(size);
  const size_t chunk_size = 1u << 14;
  const size_t num_chunks = (n + chunk_size - 1) / chunk_size;

#pragma omp parallel for schedule(static)
  for (ptrdiff_t t = 0; t < (ptrdiff_t)num_chunks; ++t) {
    const size_t start = (size_t)t * chunk_size;
    const size_t end = std::min(start + chunk_size, n);
    const size_t chunk_size = end - start;
    if (chunk_size == 0) continue;
    std::vector<uint64_t> true_vector(chunk_size);
    std::vector<uint64_t> false_vector(chunk_size);
    for (int i = 0; i < type_length; ++i) {
      const uint64_t bit_mask = (uint64_t{1} << i);
      unsigned int cnt_true = 0;
      unsigned int cnt_false = 0;
      for (unsigned int j = start; j < end; ++j) {
        if ((temp_vector[j] & bit_mask) == bit_mask) {
          true_vector[cnt_true++] = temp_vector[j];
        } else {
          false_vector[cnt_false++] = temp_vector[j];
        }
      }

      if (cnt_false > 0) {
        std::memcpy(temp_vector.data() + start, false_vector.data(), static_cast<size_t>(cnt_false) * sizeof(uint64_t));
      }

      if (cnt_true > 0) {
        std::memcpy(temp_vector.data() + start + cnt_false, true_vector.data(),
                    static_cast<size_t>(cnt_true) * sizeof(uint64_t));
      }
    }
  }
  // simple merge
  std::vector<uint64_t> shadow_vector(size);
  uint64_t *temp_ptr = temp_vector.data();
  uint64_t *shadow_ptr = shadow_vector.data();

  for (size_t run = chunk_size; run < n; run <<= 1) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t base = 0; base < (ptrdiff_t)n; base += (ptrdiff_t)(run << 1)) {
      size_t left_start = (size_t)base;
      size_t left_end = std::min(left_start + run, n);
      size_t right_start = left_end;
      size_t right_end = std::min(left_end + run, n);
      if (right_start < right_end) {
        MergeSorted(left_start, left_end, temp_ptr, right_start, right_end, shadow_ptr, left_start);
      } else {
        for (size_t i = left_start; i < left_end; i++) shadow_ptr[i] = temp_ptr[i];
      }
    }
    std::swap(temp_ptr, shadow_ptr);
  }
  if (temp_ptr != temp_vector.data()) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i) temp_vector[(size_t)i] = temp_ptr[(size_t)i];
  }

  // uint64 to float
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
