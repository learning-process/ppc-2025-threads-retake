#include "seq/kavtorev_d_batcher_sort/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace kavtorev_d_batcher_sort_seq {

static inline void comp_swap(double& a, double& b) {
  if (a > b) std::swap(a, b);
}

inline uint64_t RadixBatcherSortSequential::ToOrderedUint64(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(double));
  if (bits >> 63) {
    bits = ~bits;
  } else {
    bits ^= (uint64_t{1} << 63);
  }
  return bits;
}

inline double RadixBatcherSortSequential::FromOrderedUint64(uint64_t key) {
  if ((key >> 63) != 0) {
    key ^= (uint64_t{1} << 63);
  } else {
    key = ~key;
  }
  double value;
  std::memcpy(&value, &key, sizeof(double));
  return value;
}

void RadixBatcherSortSequential::LsdRadixSortUint64(std::vector<uint64_t>& data) {
  const size_t n = data.size();
  if (n <= 1) return;
  std::vector<uint64_t> buffer(n);
  constexpr int kBytes = 8;
  constexpr int kRadix = 256;
  for (int byte = 0; byte < kBytes; ++byte) {
    size_t count[kRadix] = {0};
    const int shift = byte * 8;
    for (size_t i = 0; i < n; ++i) {
      count[(data[i] >> shift) & 0xFFu]++;
    }
    size_t sum = 0;
    for (int r = 0; r < kRadix; ++r) {
      size_t c = count[r];
      count[r] = sum;
      sum += c;
    }
    for (size_t i = 0; i < n; ++i) {
      uint8_t r = static_cast<uint8_t>((data[i] >> shift) & 0xFFu);
      buffer[count[r]++] = data[i];
    }
    data.swap(buffer);
  }
}

void RadixBatcherSortSequential::OddEvenMerge(std::vector<double>& a, int left, int size, int stride) {
  int m = stride * 2;
  if (m < size) {
    OddEvenMerge(a, left, size, m);
    OddEvenMerge(a, left + stride, size, m);
    for (int i = left + stride; i + stride < left + size; i += m) {
      comp_swap(a[i], a[i + stride]);
    }
  } else {
    comp_swap(a[left], a[left + stride]);
  }
}

void RadixBatcherSortSequential::OddEvenMergeSort(std::vector<double>& a, int left, int size) {
  if (size > 1) {
    int mid = size / 2;
    OddEvenMergeSort(a, left, mid);
    OddEvenMergeSort(a, left + mid, mid);
    OddEvenMerge(a, left, size, 1);
  }
}

bool RadixBatcherSortSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);
  return true;
}

bool RadixBatcherSortSequential::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool RadixBatcherSortSequential::RunImpl() {
  if (input_.empty()) return true;
  std::vector<uint64_t> keys(input_.size());
  for (size_t i = 0; i < input_.size(); ++i) keys[i] = ToOrderedUint64(input_[i]);
  LsdRadixSortUint64(keys);
  std::vector<double> sorted(input_.size());
  for (size_t i = 0; i < keys.size(); ++i) sorted[i] = FromOrderedUint64(keys[i]);
  size_t n = sorted.size();
  size_t pow2 = 1;
  while (pow2 < n) pow2 <<= 1;
  std::vector<double> padded = sorted;
  if (pow2 != n) padded.resize(pow2, std::numeric_limits<double>::infinity());
  OddEvenMergeSort(padded, 0, static_cast<int>(padded.size()));
  output_.resize(n);
  std::copy(padded.begin(), padded.begin() + static_cast<ptrdiff_t>(n), output_.begin());
  return true;
}

bool RadixBatcherSortSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

}  // namespace kavtorev_d_batcher_sort_seq
