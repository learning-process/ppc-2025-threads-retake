#include "stl/khokhlov_a_shell_stl/include/ops_stl.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

bool khokhlov_a_shell_stl::ShellStl::PreProcessingImpl() {
  input_.resize(task_data->inputs_count[0]);
  std::ranges::copy(std::views::counted(reinterpret_cast<int*>(task_data->inputs[0]), task_data->inputs_count[0]),
                    input_.begin());
  return true;
}

bool khokhlov_a_shell_stl::ShellStl::ValidationImpl() {
  return task_data->inputs_count.size() == 1 && task_data->inputs_count[0] > 0 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khokhlov_a_shell_stl::ShellStl::RunImpl() {
  input_ = ShellSort(input_);
  return true;
}

bool khokhlov_a_shell_stl::ShellStl::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

void khokhlov_a_shell_stl::ShellStl::ShellSortChunk(std::vector<int>& vec, int start, int end, int interval) {
  for (int i = start + interval; i < end; ++i) {
    int tmp = vec[i];
    int j = i;
    while (j >= start + interval && vec[j - interval] > tmp) {
      vec[j] = vec[j - interval];
      j -= interval;
    }
    vec[j] = tmp;
  }
}

std::vector<int> khokhlov_a_shell_stl::ShellStl::ShellSort(const std::vector<int>& input) {
  std::vector<int> vec(input);
  int n = static_cast<int>(vec.size());
  unsigned int num_threads = std::thread::hardware_concurrency();
  unsigned int chunk_size = (n + num_threads - 1) / num_threads;

  std::vector<std::pair<int, int>> chunks;
  for (unsigned int i = 0; i < n; i += chunk_size) {
    chunks.emplace_back(i, std::min(i + chunk_size, n));
  }

  for (int interval = chunk_size / 2; interval > 0; interval /= 2) {
    std::vector<std::thread> threads;
    threads.reserve(chunks.size());
    for (const auto& chunk : chunks) {
      threads.emplace_back(ShellSortChunk, std::ref(vec), chunk.first, chunk.second, interval);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  for (int interval = n / 2; interval > 0; interval /= 2) {
    for (int i = interval; i < n; ++i) {
      int tmp = vec[i];
      int j = i;
      while (j >= interval && vec[j - interval] > tmp) {
        vec[j] = vec[j - interval];
        j -= interval;
      }
      vec[j] = tmp;
    }
  }

  return vec;
}

bool khokhlov_a_shell_stl::CheckSorted(const std::vector<int>& input) { return std::ranges::is_sorted(input); }

std::vector<int> khokhlov_a_shell_stl::GenerateRandomVector(int size) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{1, 100};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  std::ranges::generate(vec, gen);

  return vec;
}