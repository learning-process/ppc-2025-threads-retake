#include "tbb/yasakova_t_sort/include/radix_double_tbb.hpp"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace yasakova_t_sort_tbb {

bool SortTaskTBB::ValidationImpl() {
  if (!task_data) return false;
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) return false;
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) return false;
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) return false;
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SortTaskTBB::PreProcessingImpl() {
  const size_t count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* in_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskTBB::RunImpl() {
  output_ = input_;
  radix_sort_double_tbb(output_);
  return true;
}

bool SortTaskTBB::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

void radix_sort_double_tbb(std::vector<double>& a) {
  std::vector<double> nonnan;
  nonnan.reserve(a.size());
  std::vector<double> nans;
  nans.reserve(16);
  for (double x : a) {
    (is_nan(x) ? nans : nonnan).push_back(x);
  }

  const size_t n = nonnan.size();
  if (n <= 1) {
    a = std::move(nonnan);
    a.insert(a.end(), nans.begin(), nans.end());
    return;
  }

  std::vector<uint64_t> keys(n);
  tbb::parallel_for(size_t(0), n, [&](size_t i) { keys[i] = to_key(nonnan[i]); });

  std::vector<uint64_t> buf(n);
  size_t chunks = std::max<size_t>(1, tbb::this_task_arena::max_concurrency());
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
      for (int b = 0; b < 256; ++b) {
        global_counts[b] += local_counts[chunk][b];
      }
    }

    size_t sum = 0;
    for (int b = 0; b < 256; ++b) {
      size_t c = global_counts[b];
      global_counts[b] = sum;
      sum += c;
    }

    std::vector<std::array<size_t, 256>> positions(chunks);
    for (int b = 0; b < 256; ++b) {
      size_t pos = global_counts[b];
      for (size_t chunk = 0; chunk < chunks; ++chunk) {
        positions[chunk][b] = pos;
        pos += local_counts[chunk][b];
      }
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks), [&](const auto& range) {
      for (size_t chunk = range.begin(); chunk != range.end(); ++chunk) {
        const size_t begin = chunk * chunk_size;
        const size_t end = std::min(n, begin + chunk_size);
        auto local_pos = positions[chunk];
        for (size_t i = begin; i < end; ++i) {
          uint8_t b = static_cast<uint8_t>(keys[i] >> shift);
          buf[local_pos[b]++] = keys[i];
        }
      }
    });

    keys.swap(buf);
  }

  tbb::parallel_for(size_t(0), n, [&](size_t i) { nonnan[i] = from_key(keys[i]); });

  a = std::move(nonnan);
  a.insert(a.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_tbb
