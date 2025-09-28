#include "stl/yasakova_t_sort/include/radix_double_stl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace yasakova_t_sort_stl {

namespace {

size_t ClampThreads(size_t requested, size_t n) {
  if (requested == 0) {
    return std::min<size_t>(1, n);
  }
  return std::max<size_t>(1, std::min(requested, n));
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

  const size_t threads = ClampThreads(std::thread::hardware_concurrency(), n);
  const size_t chunk_size = (n + threads - 1) / threads;

  auto run_parallel = [&](auto&& fn) {
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
      const size_t begin = t * chunk_size;
      if (begin >= n) {
        break;
      }
      const size_t end = std::min(n, begin + chunk_size);
      workers.emplace_back([&, begin, end, t]() { fn(begin, end, t); });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  };

  std::vector<uint64_t> keys(n);
  run_parallel([&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      keys[i] = ToKey(nonnan[i]);
    }
  });

  std::vector<uint64_t> buffer(n);

  for (int pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;
    std::vector<std::array<size_t, 256>> local_counts(threads);

    run_parallel([&](size_t begin, size_t end, size_t thread_id) {
      auto& local = local_counts[thread_id];
      local.fill(0);
      for (size_t i = begin; i < end; ++i) {
        ++local[static_cast<uint8_t>(keys[i] >> shift)];
      }
    });

    std::array<size_t, 256> global_counts{};
    for (size_t thread_id = 0; thread_id < threads; ++thread_id) {
      for (int bucket = 0; bucket < 256; ++bucket) {
        global_counts[bucket] += local_counts[thread_id][bucket];
      }
    }

    size_t sum = 0;
    for (int bucket = 0; bucket < 256; ++bucket) {
      const auto current = global_counts[bucket];
      global_counts[bucket] = sum;
      sum += current;
    }

    std::vector<std::array<size_t, 256>> positions(threads);
    for (int bucket = 0; bucket < 256; ++bucket) {
      size_t pos = global_counts[bucket];
      for (size_t thread_id = 0; thread_id < threads; ++thread_id) {
        positions[thread_id][bucket] = pos;
        pos += local_counts[thread_id][bucket];
      }
    }

    run_parallel([&](size_t begin, size_t end, size_t thread_id) {
      auto local_pos = positions[thread_id];
      for (size_t i = begin; i < end; ++i) {
        const auto bucket = static_cast<uint8_t>(keys[i] >> shift);
        buffer[local_pos[bucket]++] = keys[i];
      }
    });

    keys.swap(buffer);
  }

  run_parallel([&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      nonnan[i] = FromKey(keys[i]);
    }
  });

  data = std::move(nonnan);
  data.insert(data.end(), nans.begin(), nans.end());
}

}  // namespace yasakova_t_sort_stl
