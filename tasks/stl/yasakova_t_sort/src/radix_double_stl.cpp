#include "stl/yasakova_t_sort/include/radix_double_stl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace yasakova_t_sort_stl {

namespace {

size_t ClampThreads(size_t requested, size_t total) {
  const auto capped_total = std::max<size_t>(size_t{1}, total);
  const auto sanitized_requested = std::max<size_t>(size_t{1}, requested);
  return std::min(sanitized_requested, capped_total);
}

struct ParallelPlan {
  size_t threads;
  size_t chunk_size;
  size_t total;
};

ParallelPlan MakePlan(size_t total) {
  const int raw_requested = ppc::util::GetPPCNumThreads();
  const auto requested = raw_requested > 0 ? static_cast<size_t>(raw_requested) : size_t{0};
  const size_t threads = ClampThreads(requested, total);
  const size_t chunk_size = (total + threads - 1) / threads;
  return ParallelPlan{.threads = threads, .chunk_size = chunk_size, .total = total};
}

void RunParallel(const ParallelPlan& plan, const auto& fn) {
  std::vector<std::thread> workers;
  workers.reserve(plan.threads);
  for (size_t thread_id = 0; thread_id < plan.threads; ++thread_id) {
    const size_t begin = thread_id * plan.chunk_size;
    if (begin >= plan.total) {
      break;
    }
    const size_t end = std::min(plan.total, begin + plan.chunk_size);
    workers.emplace_back([&, begin, end, thread_id]() { fn(begin, end, thread_id); });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

void ComputeLocalKeys(const std::vector<double>& nonnan, std::vector<uint64_t>& keys, const ParallelPlan& plan) {
  RunParallel(plan, [&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      keys[i] = ToKey(nonnan[i]);
    }
  });
}

void ComputeLocalCounts(const std::vector<uint64_t>& keys, std::vector<std::array<size_t, 256>>& local_counts,
                        const ParallelPlan& plan, int shift) {
  RunParallel(plan, [&](size_t begin, size_t end, size_t thread_id) {
    auto& local = local_counts[thread_id];
    local.fill(0);
    for (size_t i = begin; i < end; ++i) {
      ++local[static_cast<uint8_t>(keys[i] >> shift)];
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
    for (size_t thread_id = 0; thread_id < local_counts.size(); ++thread_id) {
      positions[thread_id][bucket] = pos;
      pos += local_counts[thread_id][bucket];
    }
  }
  return positions;
}

void ScatterIntoBuffer(const std::vector<uint64_t>& keys, std::vector<uint64_t>& buffer,
                       const std::vector<std::array<size_t, 256>>& positions, const ParallelPlan& plan, int shift) {
  RunParallel(plan, [&](size_t begin, size_t end, size_t thread_id) {
    auto local_pos = positions[thread_id];
    for (size_t i = begin; i < end; ++i) {
      const auto bucket = static_cast<uint8_t>(keys[i] >> shift);
      buffer[local_pos[bucket]++] = keys[i];
    }
  });
}

void ScatterValues(std::vector<double>& nonnan, const std::vector<uint64_t>& keys, const ParallelPlan& plan) {
  RunParallel(plan, [&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      nonnan[i] = FromKey(keys[i]);
    }
  });
}

}  // namespace

bool SortTaskSTL::ValidationImpl() {
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

bool SortTaskSTL::PreProcessingImpl() {
  const auto count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* input_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskSTL::RunImpl() {
  output_ = input_;
  RadixSortDoubleStl(output_);
  return true;
}

bool SortTaskSTL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_, output_ptr);
  return true;
}

void RadixSortDoubleStl(std::vector<double>& data) {
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
  ComputeLocalKeys(nonnan, keys, plan);

  std::vector<uint64_t> buffer(n);
  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;

    std::vector<std::array<size_t, 256>> local_counts(plan.threads);
    ComputeLocalCounts(keys, local_counts, plan, shift);

    auto global_counts = AggregateCounts(local_counts);
    PrefixSums(global_counts);

    const auto positions = PreparePositions(local_counts, global_counts);
    ScatterIntoBuffer(keys, buffer, positions, plan, shift);

    keys.swap(buffer);
  }

  ScatterValues(nonnan, keys, plan);

  data = std::move(nonnan);
  data.insert(data.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_stl
