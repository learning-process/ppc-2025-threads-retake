#include "omp/kavtorev_d_batcher_sort/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace kavtorev_d_batcher_sort_omp {

static inline void CompSwap(double& a, double& b) {
  if (a > b) {
    std::swap(a, b);
  }
}

inline uint64_t RadixBatcherSortOpenMP::ToOrderedUint64(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  if ((bits >> 63) != 0U) {
    bits = ~bits;
  } else {
    bits ^= (uint64_t{1} << 63);
  }
  return bits;
}

inline double RadixBatcherSortOpenMP::FromOrderedUint64(uint64_t key) {
  if ((key >> 63) != 0U) {
    key ^= (uint64_t{1} << 63);
  } else {
    key = ~key;
  }
  double value = NAN;
  std::memcpy(&value, &key, sizeof(double));
  return value;
}

void RadixBatcherSortOpenMP::LsdRadixSortUint64(std::vector<uint64_t>& data) {
  const size_t n = data.size();
  if (n <= 1) {
    return;
  }
  std::vector<uint64_t> buffer(n);
  constexpr int kBytes = 8;
  constexpr int kRadix = 256;

  for (int byte = 0; byte < kBytes; ++byte) {
    size_t count[kRadix] = {0};
    const int shift = byte * 8;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i) {
      auto r = static_cast<uint8_t>((data[i] >> shift) & 0xFFU);
#pragma omp atomic
      count[r]++;
    }

    size_t sum = 0;
    for (int r = 0; r < kRadix; ++r) {
      size_t c = count[r];
      count[r] = sum;
      sum += c;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i) {
      auto r = static_cast<uint8_t>((data[i] >> shift) & 0xFFU);
      size_t pos;
#pragma omp critical
      {
        pos = count[r]++;
      }
      buffer[pos] = data[i];
    }
    data.swap(buffer);
  }
}

void RadixBatcherSortOpenMP::OddEvenMerge(std::vector<double>& a, int left, int size, int stride) {
  int m = stride * 2;
  if (m < size) {
    OddEvenMerge(a, left, size, m);
    OddEvenMerge(a, left + stride, size, m);

    for (int i = left + stride; i + stride < left + size; i += m) {
      CompSwap(a[i], a[i + stride]);
    }
  } else {
    CompSwap(a[left], a[left + stride]);
  }
}

void RadixBatcherSortOpenMP::OddEvenMergeSort(std::vector<double>& a, int left, int size) {
  if (size > 1) {
    int mid = size / 2;

#pragma omp parallel sections
    {
#pragma omp section
      { OddEvenMergeSort(a, left, mid); }
#pragma omp section
      { OddEvenMergeSort(a, left + mid, mid); }
    }

    OddEvenMerge(a, left, size, 1);
  }
}

bool RadixBatcherSortOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);
  return true;
}

bool RadixBatcherSortOpenMP::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool RadixBatcherSortOpenMP::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<uint64_t> keys(input_.size());

#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(input_.size()); ++i) {
    keys[i] = ToOrderedUint64(input_[i]);
  }

  LsdRadixSortUint64(keys);

  std::vector<double> sorted(input_.size());

#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
    sorted[i] = FromOrderedUint64(keys[i]);
  }

  size_t n = sorted.size();
  size_t pow2 = 1;
  while (pow2 < n) {
    pow2 <<= 1;
  }
  std::vector<double> padded = sorted;
  if (pow2 != n) {
    padded.resize(pow2, std::numeric_limits<double>::infinity());
  }

  OddEvenMergeSort(padded, 0, static_cast<int>(padded.size()));

  output_.resize(n);
  std::copy(padded.begin(), padded.begin() + static_cast<ptrdiff_t>(n), output_.begin());
  return true;
}

bool RadixBatcherSortOpenMP::PostProcessingImpl() {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(output_.size()); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

}  // namespace kavtorev_d_batcher_sort_omp