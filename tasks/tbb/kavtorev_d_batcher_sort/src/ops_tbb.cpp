#include "tbb/kavtorev_d_batcher_sort/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace kavtorev_d_batcher_sort_tbb {

namespace {
inline void CompSwap(double& a, double& b) {
  if (a > b) {
    std::swap(a, b);
  }
}
}  // namespace

inline uint64_t RadixBatcherSortTBB::ToOrderedUint64(double value) {
  uint64_t bits = 0;
#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
  bits = std::bit_cast<uint64_t>(value);
#else
  std::memcpy(&bits, &value, sizeof(double));
#endif
  if ((bits >> 63) != 0U) {
    bits = ~bits;
  } else {
    bits ^= (uint64_t{1} << 63);
  }
  return bits;
}

inline double RadixBatcherSortTBB::FromOrderedUint64(uint64_t key) {
  if ((key >> 63) != 0U) {
    key ^= (uint64_t{1} << 63);
  } else {
    key = ~key;
  }
#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
  return std::bit_cast<double>(key);
#else
  double value = 0.0;
  std::memcpy(&value, &key, sizeof(double));
  return value;
#endif
}

void RadixBatcherSortTBB::LsdRadixSortUint64(std::vector<uint64_t>& data) {
  const size_t n = data.size();
  if (n <= 1) {
    return;
  }
  std::vector<uint64_t> buffer(n);
  constexpr int kBytes = 8;
  constexpr int kRadix = 256;

  for (int byte = 0; byte < kBytes; ++byte) {
    const int shift = byte * 8;
    tbb::combinable<std::vector<size_t>> local_hist([&] { return std::vector<size_t>(kRadix, 0); });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& range) {
      auto& hist = local_hist.local();
      for (size_t i = range.begin(); i != range.end(); ++i) {
        auto r = static_cast<uint8_t>((data[i] >> shift) & 0xFFU);
        hist[r]++;
      }
    });

    std::vector<size_t> count(kRadix, 0);
    local_hist.combine_each([&](const std::vector<size_t>& h) {
      for (int r = 0; r < kRadix; ++r) {
        count[r] += h[r];
      }
    });

    size_t sum = 0;
    for (int r = 0; r < kRadix; ++r) {
      size_t c = count[r];
      count[r] = sum;
      sum += c;
    }

    std::unique_ptr<std::atomic<size_t>[]> positions(new std::atomic<size_t>[kRadix]);
    for (int r = 0; r < kRadix; ++r) {
      positions[r].store(count[r], std::memory_order_relaxed);
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        auto r = static_cast<uint8_t>((data[i] >> shift) & 0xFFU);
        size_t pos = positions[r].fetch_add(1, std::memory_order_relaxed);
        buffer[pos] = data[i];
      }
    });

    data.swap(buffer);
  }
}

void RadixBatcherSortTBB::OddEvenMerge(std::vector<double>& a, int left, int size, int stride) {
  int m = stride * 2;
  if (m < size) {
    OddEvenMerge(a, left, size, m);
    OddEvenMerge(a, left + stride, size, m);
    std::vector<std::pair<int, int>> indices;
    for (int i = left + stride; i + stride < left + size; i += m) {
      indices.emplace_back(i, i + stride);
    }
    if (!indices.empty()) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, indices.size()), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t idx = range.begin(); idx != range.end(); ++idx) {
          CompSwap(a[indices[idx].first], a[indices[idx].second]);
        }
      });
    }
  } else {
    CompSwap(a[left], a[left + stride]);
  }
}

void RadixBatcherSortTBB::OddEvenMergeSort(std::vector<double>& a, int left, int size) {
  if (size > 1) {
    int mid = size / 2;
    tbb::parallel_invoke([&]() { OddEvenMergeSort(a, left, mid); }, [&]() { OddEvenMergeSort(a, left + mid, mid); });
    OddEvenMerge(a, left, size, 1);
  }
}

bool RadixBatcherSortTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);
  return true;
}

bool RadixBatcherSortTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool RadixBatcherSortTBB::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<uint64_t> keys(input_.size());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, input_.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      keys[i] = ToOrderedUint64(input_[i]);
    }
  });

  LsdRadixSortUint64(keys);

  std::vector<double> sorted(input_.size());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      sorted[i] = FromOrderedUint64(keys[i]);
    }
  });

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

bool RadixBatcherSortTBB::PostProcessingImpl() {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, output_.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
    }
  });
  return true;
}

}  // namespace kavtorev_d_batcher_sort_tbb
