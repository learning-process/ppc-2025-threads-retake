#include "tbb/yasakova_t_sort/include/radix_double_tbb.hpp"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace yasakova_t_sort_tbb {

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

  std::vector<uint64_t> keys(n);
  tbb::parallel_for(size_t(0), n, [&](size_t i) { keys[i] = ToKey(nonnan[i]); });

  std::vector<uint64_t> buffer(n);
  auto chunks = std::max<size_t>(1, tbb::this_task_arena::max_concurrency());
  chunks = std::min(chunks, n);
  const size_t chunk_size = (n + chunks - 1) / chunks;

  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;
    std::vector<std::array<size_t, 256>> local_counts(chunks);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks), [&](const auto& range) {
      for (size_t chunk = range.begin(); chunk != range.end(); ++chunk) {
        const size_t begin = chunk * chunk_size;
        const size_t end = std::min(n, begin + chunk_size);
        auto& local = local_counts[chunk];
        local.fill(0);
        for (size_t i = begin; i < end; ++i) {
          ++local[static_cast<uint8_t>(keys[i] >> shift)];
        }
      }
    });

    std::array<size_t, 256> global_counts{};
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
      for (int bucket = 0; bucket < 256; ++bucket) {
        global_counts[bucket] += local_counts[chunk][bucket];
      }
    }

    size_t sum = 0;
    for (int bucket = 0; bucket < 256; ++bucket) {
      const auto current = global_counts[bucket];
      global_counts[bucket] = sum;
      sum += current;
    }

    std::vector<std::array<size_t, 256>> positions(chunks);
    for (int bucket = 0; bucket < 256; ++bucket) {
      size_t pos = global_counts[bucket];
      for (size_t chunk = 0; chunk < chunks; ++chunk) {
        positions[chunk][bucket] = pos;
        pos += local_counts[chunk][bucket];
      }
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks), [&](const auto& range) {
      for (size_t chunk = range.begin(); chunk != range.end(); ++chunk) {
        const size_t begin = chunk * chunk_size;
        const size_t end = std::min(n, begin + chunk_size);
        auto local_pos = positions[chunk];
        for (size_t i = begin; i < end; ++i) {
          const auto bucket = static_cast<uint8_t>(keys[i] >> shift);
          buffer[local_pos[bucket]++] = keys[i];
        }
      }
    });

    keys.swap(buffer);
  }

  tbb::parallel_for(size_t(0), n, [&](size_t i) { nonnan[i] = FromKey(keys[i]); });

  data = std::move(nonnan);
  data.insert(data.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_tbb
