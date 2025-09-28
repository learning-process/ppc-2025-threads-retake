#include "omp/yasakova_t_sort/include/radix_double_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace yasakova_t_sort_omp {

bool SortTaskOpenMP::ValidationImpl() {
  if (!task_data) return false;
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) return false;
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) return false;
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) return false;
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SortTaskOpenMP::PreProcessingImpl() {
  const size_t count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* in_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskOpenMP::RunImpl() {
  output_ = input_;
  radix_sort_double_omp(output_);
  return true;
}

bool SortTaskOpenMP::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

void radix_sort_double_omp(std::vector<double>& a) {
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
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    keys[i] = to_key(nonnan[static_cast<size_t>(i)]);
  }

  std::vector<uint64_t> buf(n);
  size_t chunks = static_cast<size_t>(std::max(1, omp_get_max_threads()));
  chunks = std::min(chunks, n);
  const size_t chunk_size = (n + chunks - 1) / chunks;

  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;
    std::vector<std::array<size_t, 256>> local_counts(chunks);

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t chunk = 0; chunk < static_cast<std::ptrdiff_t>(chunks); ++chunk) {
      const size_t begin = static_cast<size_t>(chunk) * chunk_size;
      const size_t end = std::min(n, begin + chunk_size);
      auto& local = local_counts[static_cast<size_t>(chunk)];
      local.fill(0);
      for (size_t i = begin; i < end; ++i) {
        ++local[static_cast<uint8_t>(keys[i] >> shift)];
      }
    }

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

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t chunk = 0; chunk < static_cast<std::ptrdiff_t>(chunks); ++chunk) {
      const size_t begin = static_cast<size_t>(chunk) * chunk_size;
      const size_t end = std::min(n, begin + chunk_size);
      auto local_pos = positions[static_cast<size_t>(chunk)];
      for (size_t i = begin; i < end; ++i) {
        uint8_t b = static_cast<uint8_t>(keys[i] >> shift);
        buf[local_pos[b]++] = keys[i];
      }
    }

    keys.swap(buf);
  }

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    nonnan[static_cast<size_t>(i)] = from_key(keys[static_cast<size_t>(i)]);
  }

  a = std::move(nonnan);
  a.insert(a.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_omp
