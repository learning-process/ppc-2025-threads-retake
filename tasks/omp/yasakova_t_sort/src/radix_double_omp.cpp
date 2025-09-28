#include "omp/yasakova_t_sort/include/radix_double_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>
#include <vector>

namespace yasakova_t_sort_omp {

namespace {

void ComputeLocalKeys(const std::vector<double>& nonnan, std::vector<uint64_t>& keys) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nonnan.size()); ++i) {
    keys[static_cast<size_t>(i)] = ToKey(nonnan[static_cast<size_t>(i)]);
  }
}

void ComputeLocalCounts(const std::vector<uint64_t>& keys, std::vector<std::array<size_t, 256>>& local_counts,
                        size_t threads, size_t chunk_size, size_t n, int shift) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t chunk = 0; chunk < static_cast<std::ptrdiff_t>(threads); ++chunk) {
    const size_t begin = static_cast<size_t>(chunk) * chunk_size;
    const size_t end = std::min(n, begin + chunk_size);
    auto& local = local_counts[static_cast<size_t>(chunk)];
    local.fill(0);
    for (size_t i = begin; i < end; ++i) {
      ++local[static_cast<uint8_t>(keys[i] >> shift)];
    }
  }
}

std::array<size_t, 256> AggregateCounts(const std::vector<std::array<size_t, 256>>& local_counts) {
  std::array<size_t, 256> global_counts{};
  for (const auto& counts : local_counts) {
    for (int bucket = 0; bucket < 256; ++bucket) {
      global_counts[bucket] += counts[bucket];
    }
  }
  return global_counts;
}

void PrefixSums(std::array<size_t, 256>& counts) {
  size_t sum = 0;
  for (int bucket = 0; bucket < 256; ++bucket) {
    const auto current = counts[bucket];
    counts[bucket] = sum;
    sum += current;
  }
}

std::vector<std::array<size_t, 256>> PreparePositions(const std::vector<std::array<size_t, 256>>& local_counts,
                                                      const std::array<size_t, 256>& global_counts) {
  std::vector<std::array<size_t, 256>> positions(local_counts.size());
  for (int bucket = 0; bucket < 256; ++bucket) {
    size_t pos = global_counts[bucket];
    for (size_t chunk = 0; chunk < local_counts.size(); ++chunk) {
      positions[chunk][bucket] = pos;
      pos += local_counts[chunk][bucket];
    }
  }
  return positions;
}

void ScatterIntoBuffer(const std::vector<uint64_t>& keys, std::vector<uint64_t>& buffer,
                       const std::vector<std::array<size_t, 256>>& positions, size_t threads, size_t chunk_size,
                       size_t n, int shift) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t chunk = 0; chunk < static_cast<std::ptrdiff_t>(threads); ++chunk) {
    const size_t begin = static_cast<size_t>(chunk) * chunk_size;
    const size_t end = std::min(n, begin + chunk_size);
    auto local_pos = positions[static_cast<size_t>(chunk)];
    for (size_t i = begin; i < end; ++i) {
      const auto bucket = static_cast<uint8_t>(keys[i] >> shift);
      buffer[local_pos[bucket]++] = keys[i];
    }
  }
}

void ScatterValues(std::vector<double>& nonnan, const std::vector<uint64_t>& keys) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nonnan.size()); ++i) {
    nonnan[static_cast<size_t>(i)] = FromKey(keys[static_cast<size_t>(i)]);
  }
}

}  // namespace

bool SortTaskOpenMP::ValidationImpl() {
  if (!task_data) {
    return false;
  }
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SortTaskOpenMP::PreProcessingImpl() {
  const auto count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* input_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskOpenMP::RunImpl() {
  output_ = input_;
  RadixSortDoubleOmp(output_);
  return true;
}

bool SortTaskOpenMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_, output_ptr);
  return true;
}

void RadixSortDoubleOmp(std::vector<double>& data) {
  std::vector<double> nonnan;
  nonnan.reserve(data.size());
  std::vector<double> nans;
  nans.reserve(16);
  for (double value : data) {
    (IsNan(value) ? nans : nonnan).push_back(value);
  }

  const auto n = nonnan.size();
  if (n <= 1) {
    data = std::move(nonnan);
    data.insert(data.end(), nans.begin(), nans.end());
    return;
  }

  std::vector<uint64_t> keys(n);
  ComputeLocalKeys(nonnan, keys);

  auto threads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
  threads = std::min(threads, n);
  const size_t chunk_size = (n + threads - 1) / threads;

  std::vector<uint64_t> buffer(n);
  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;

    std::vector<std::array<size_t, 256>> local_counts(threads);
    ComputeLocalCounts(keys, local_counts, threads, chunk_size, n, shift);

    auto global_counts = AggregateCounts(local_counts);
    PrefixSums(global_counts);

    const auto positions = PreparePositions(local_counts, global_counts);
    ScatterIntoBuffer(keys, buffer, positions, threads, chunk_size, n, shift);

    keys.swap(buffer);
  }

  ScatterValues(nonnan, keys);

  data = std::move(nonnan);
  data.insert(data.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_omp
