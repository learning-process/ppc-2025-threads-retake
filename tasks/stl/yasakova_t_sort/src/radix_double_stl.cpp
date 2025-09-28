#include "stl/yasakova_t_sort/include/radix_double_stl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <thread>
#include <utility>

namespace yasakova_t_sort_stl {

bool SortTaskSTL::ValidationImpl() {
  if (!task_data) return false;
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) return false;
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) return false;
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) return false;
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SortTaskSTL::PreProcessingImpl() {
  const size_t count = static_cast<size_t>(task_data->inputs_count[0]);
  const auto* in_ptr = reinterpret_cast<const double*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + count);
  output_.assign(count, 0.0);
  return true;
}

bool SortTaskSTL::RunImpl() {
  output_ = input_;
  radix_sort_double_stl(output_);
  return true;
}

bool SortTaskSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

namespace {

size_t clamp_threads(size_t requested, size_t n) {
  if (requested == 0) return std::min<size_t>(1, n);
  return std::max<size_t>(1, std::min(requested, n));
}

}  // namespace

void radix_sort_double_stl(std::vector<double>& a) {
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

  size_t threads = clamp_threads(std::thread::hardware_concurrency(), n);
  const size_t chunk_size = (n + threads - 1) / threads;

  auto run_parallel = [&](auto&& fn) {
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
      const size_t begin = t * chunk_size;
      if (begin >= n) break;
      const size_t end = std::min(n, begin + chunk_size);
      workers.emplace_back([&, begin, end, t]() { fn(begin, end, t); });
    }
    for (auto& th : workers) th.join();
  };

  std::vector<uint64_t> keys(n);
  run_parallel([&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      keys[i] = to_key(nonnan[i]);
    }
  });

  std::vector<uint64_t> buf(n);

  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;
    std::vector<std::array<size_t, 256>> local_counts(threads);

    run_parallel([&](size_t begin, size_t end, size_t tid) {
      auto& local = local_counts[tid];
      local.fill(0);
      for (size_t i = begin; i < end; ++i) {
        ++local[static_cast<uint8_t>(keys[i] >> shift)];
      }
    });

    std::array<size_t, 256> global_counts{};
    for (size_t tid = 0; tid < threads; ++tid) {
      for (int b = 0; b < 256; ++b) {
        global_counts[b] += local_counts[tid][b];
      }
    }

    size_t sum = 0;
    for (int b = 0; b < 256; ++b) {
      size_t c = global_counts[b];
      global_counts[b] = sum;
      sum += c;
    }

    std::vector<std::array<size_t, 256>> positions(threads);
    for (int b = 0; b < 256; ++b) {
      size_t pos = global_counts[b];
      for (size_t tid = 0; tid < threads; ++tid) {
        positions[tid][b] = pos;
        pos += local_counts[tid][b];
      }
    }

    run_parallel([&](size_t begin, size_t end, size_t tid) {
      auto local_pos = positions[tid];
      for (size_t i = begin; i < end; ++i) {
        uint8_t b = static_cast<uint8_t>(keys[i] >> shift);
        buf[local_pos[b]++] = keys[i];
      }
    });

    keys.swap(buf);
  }

  run_parallel([&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      nonnan[i] = from_key(keys[i]);
    }
  });

  a = std::move(nonnan);
  a.insert(a.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_stl
