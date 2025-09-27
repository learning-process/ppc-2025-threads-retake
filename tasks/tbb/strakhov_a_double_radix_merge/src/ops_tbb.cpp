#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tbb/strakhov_a_double_radix_merge/include/ops_tbb.hpp"

bool strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  unsigned int size = task_data->inputs_count[0];
  input_ = std::vector<double>(in_ptr, in_ptr + size);

  output_ = std::vector<double>(size, 0);

  return true;
}
struct strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::Range {
  size_t start, end;
};
inline void strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::MergeSorted(const uint64_t *input, Range left,
                                                                                Range right, uint64_t *output,
                                                                                size_t output_start) {
  size_t left_end = left.end;
  size_t right_end = right.end;
  size_t i = left.start;
  size_t j = right.start;
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

inline void strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::FloatToNormalized(std::vector<uint64_t> &input,
                                                                                      unsigned size) {
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, size),
                            [&](const oneapi::tbb::blocked_range<size_t> &r) {
                              for (size_t i = r.begin(); i != r.end(); ++i) {
                                uint64_t bits = 0;
                                std::memcpy(&bits, &input_[i], sizeof(double));
                                if ((bits >> 63) != 0ULL) {
                                  input[i] = ~bits;
                                } else {
                                  input[i] = bits ^ 0x8000000000000000ULL;
                                }
                              }
                            });
}
inline void strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::NormalizedToFloat(std::vector<uint64_t> &input,
                                                                                      unsigned size) {
  // uint64 to float
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, size),
                            [&](const oneapi::tbb::blocked_range<size_t> &r) {
                              for (size_t i = r.begin(); i != r.end(); ++i) {
                                uint64_t k = input[i];
                                if ((k >> 63) != 0ULL) {
                                  k ^= 0x8000000000000000ULL;
                                } else {
                                  k = ~k;
                                }
                                std::memcpy(&output_[i], &k, sizeof(double));
                              }
                            });
}

inline void strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::RadixGrandSort(size_t n,
                                                                                   std::vector<uint64_t> &input,
                                                                                   size_t chunk_size) {
  const size_t num_chunks = (n + chunk_size - 1) / chunk_size;
  const int type_length = sizeof(double) * 8;
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, num_chunks), [&](const oneapi::tbb::blocked_range<size_t> &rr) {
        for (size_t t = rr.begin(); t != rr.end(); ++t) {
          const size_t start = (size_t)t * chunk_size;
          const size_t end = std::min(start + chunk_size, n);
          const size_t chunk_length = end - start;
          if (chunk_length == 0) {
            continue;
          }
          std::vector<uint64_t> true_vector(chunk_length);
          std::vector<uint64_t> false_vector(chunk_length);
          for (int i = 0; i < type_length; ++i) {
            const uint64_t bit_mask = (uint64_t{1} << i);
            unsigned int cnt_true = 0;
            unsigned int cnt_false = 0;
            for (unsigned int j = start; j < end; ++j) {
              if ((input[j] & bit_mask) == bit_mask) {
                true_vector[cnt_true++] = input[j];
              } else {
                false_vector[cnt_false++] = input[j];
              }
            }

            if (cnt_false > 0) {
              std::memcpy(input.data() + start, false_vector.data(), static_cast<size_t>(cnt_false) * sizeof(uint64_t));
            }

            if (cnt_true > 0) {
              std::memcpy(input.data() + start + cnt_false, true_vector.data(),
                          static_cast<size_t>(cnt_true) * sizeof(uint64_t));
            }
          }
        }
      });
}

bool strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::RunImpl() {
  unsigned int size = task_data->inputs_count[0];
  if (size == 0) {
    return true;
  }

  std::vector<uint64_t> temp_vector(size);
  FloatToNormalized(temp_vector, size);
  // radix sort
  auto n = static_cast<size_t>(size);
  const size_t chunk_size = 1U << 14;
  RadixGrandSort(n, temp_vector, chunk_size);
  // simple merge
  std::vector<uint64_t> shadow_vector(size);
  uint64_t *temp_ptr = temp_vector.data();
  uint64_t *shadow_ptr = shadow_vector.data();

  for (size_t run = chunk_size; run < n; run <<= 1) {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, n, (run << 1)),
                              [&](const oneapi::tbb::blocked_range<size_t> &r) {
                                for (size_t base = r.begin(); base < r.end(); base += (run << 1)) {
                                  auto left_start = (size_t)base;
                                  size_t left_end = std::min(left_start + run, n);
                                  size_t right_start = left_end;
                                  size_t right_end = std::min(left_end + run, n);
                                  if (right_start < right_end) {
                                    MergeSorted(temp_ptr, Range{.start = left_start, .end = left_end},
                                                Range{.start = right_start, .end = right_end}, shadow_ptr, left_start);
                                  } else {
                                    for (size_t i = left_start; i < left_end; i++) {
                                      shadow_ptr[i] = temp_ptr[i];
                                    }
                                  }
                                }
                              });
    std::swap(temp_ptr, shadow_ptr);
  }
  if (temp_ptr != temp_vector.data()) {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, n), [&](const oneapi::tbb::blocked_range<size_t> &r) {
          std::memcpy(temp_vector.data() + r.begin(), temp_ptr + r.begin(), (r.end() - r.begin()) * sizeof(uint64_t));
        });
    NormalizedToFloat(temp_vector, size);
    return true;
  }

  bool strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb::PostProcessingImpl() {
    unsigned long size = output_.size();

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, size),
                              [&](const oneapi::tbb::blocked_range<size_t> &r) {
                                for (size_t i = r.begin(); i != r.end(); ++i) {
                                  reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
                                }
                              });
    return true;
  }
