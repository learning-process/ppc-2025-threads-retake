#include "tbb/yasakova_t_sort/include/radix_double_tbb.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

namespace yasakova_t_sort_tbb {

namespace {

struct ParallelPlan {
  size_t chunks;
  size_t chunk_size;
  size_t total;
};

ParallelPlan MakePlan(size_t total) {
  auto chunks = std::max<size_t>(1, tbb::this_task_arena::max_concurrency());
  chunks = std::min(chunks, total);
  const size_t chunk_size = (total + chunks - 1) / chunks;
  return ParallelPlan{chunks, chunk_size, total};
}

void ComputeLocalKeys(const std::vector<double>& nonnan, std::vector<uint64_t>& keys) {
  tbb::parallel_for(size_t(0), nonnan.size(), [&](size_t i) { keys[i] = ToKey(nonnan[i]); });
}

void ComputeLocalCounts(const std::vector<uint64_t>& keys, std::vector<std::array<size_t, 256>>& local_counts,
                        const ParallelPlan& plan, int shift) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, plan.chunks), [&](const auto& range) {
    for (size_t chunk = range.begin(); chunk != range.end(); ++chunk) {
      const size_t begin = chunk * plan.chunk_size;
      const size_t end = std::min(plan.total, begin + plan.chunk_size);
      auto& local = local_counts[chunk];
      local.fill(0);
      for (size_t i = begin; i < end; ++i) {
        ++local[static_cast<uint8_t>(keys[i] >> shift)];
      }
    }
  });
}

std::array<size_t, 256> AggregateCounts(const std::vector<std::array<size_t, 256>>& local_counts) {
  std::array<size_t, 256> global_counts{};
  for (const auto& local : local_counts) {
    for (int bucket = 0; bucket < 256; ++bucket) {
      global_counts[bucket] += local[bucket];
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
                       const std::vector<std::array<size_t, 256>>& positions, const ParallelPlan& plan, int shift) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, plan.chunks), [&](const auto& range) {
    for (size_t chunk = range.begin(); chunk != range.end(); ++chunk) {
      const size_t begin = chunk * plan.chunk_size;
      const size_t end = std::min(plan.total, begin + plan.chunk_size);
      auto local_pos = positions[chunk];
      for (size_t i = begin; i < end; ++i) {
        const auto bucket = static_cast<uint8_t>(keys[i] >> shift);
        buffer[local_pos[bucket]++] = keys[i];
      }
    }
  });
}

void ScatterValues(std::vector<double>& nonnan, const std::vector<uint64_t>& keys) {
  tbb::parallel_for(size_t(0), nonnan.size(), [&](size_t i) { nonnan[i] = FromKey(keys[i]); });
}

}  // namespace

bool SortTaskTBB::ValidationImpl() {
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

bool SortTaskTBB::PreProcessingImpl() {
  const auto count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* input_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskTBB::RunImpl() {
  output_ = input_;
  RadixSortDoubleTbb(output_);
  return true;
}

bool SortTaskTBB::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_, output_ptr);
  return true;
}

void RadixSortDoubleTbb(std::vector<double>& data) {
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

  const auto plan = MakePlan(n);

  std::vector<uint64_t> keys(n);
  ComputeLocalKeys(nonnan, keys);

  std::vector<uint64_t> buffer(n);
  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;

    std::vector<std::array<size_t, 256>> local_counts(plan.chunks);
    ComputeLocalCounts(keys, local_counts, plan, shift);

    auto global_counts = AggregateCounts(local_counts);
    PrefixSums(global_counts);

    const auto positions = PreparePositions(local_counts, global_counts);
    ScatterIntoBuffer(keys, buffer, positions, plan, shift);

    keys.swap(buffer);
  }

  ScatterValues(nonnan, keys);

  data = std::move(nonnan);
  data.insert(data.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_tbb
