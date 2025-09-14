#include "tbb/khokhlov_a_shell_tbb/include/ops_tbb.hpp"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <random>
#include <vector>

bool khokhlov_a_shell_tbb::ShellTbb::PreProcessingImpl() {
  // Init value for input and output
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[0]); i++) {
    input_.push_back(reinterpret_cast<int*>(task_data->inputs[0])[i]);
  }
  return true;
}

bool khokhlov_a_shell_tbb::ShellTbb::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count.size() == 1 && task_data->inputs_count[0] > 0 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khokhlov_a_shell_tbb::ShellTbb::RunImpl() {
  input_ = ShellSort(input_);
  return true;
}

bool khokhlov_a_shell_tbb::ShellTbb::PostProcessingImpl() {
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[0]); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}

std::vector<int> khokhlov_a_shell_tbb::ShellTbb::ShellSort(const std::vector<int>& input) {
  std::vector<int> vec(input);
  int n = static_cast<int>(vec.size());
  int num_threads = tbb::task_arena::automatic;
  int chunk_size = (n + num_threads - 1) / num_threads;

  tbb::parallel_for(tbb::blocked_range<int>(0, n, chunk_size), [&](const tbb::blocked_range<int>& range) {
    int start = range.begin();
    int end = range.end();

    for (int interval = (end - start) / 2; interval > 0; interval /= 2) {
      for (int i = start + interval; i < end; i++) {
        int tmp = vec[i];
        int j = i;
        for (; j >= start + interval && vec[j - interval] > tmp; j -= interval) {
          vec[j] = vec[j - interval];
        }
        vec[j] = tmp;
      }
    }
  });

  for (int interval = n / 2; interval > 0; interval /= 2) {
    for (int i = interval; i < n; i++) {
      int tmp = vec[i];
      int j = i;
      for (; j >= interval && vec[j - interval] > tmp; j -= interval) {
        vec[j] = vec[j - interval];
      }
      vec[j] = tmp;
    }
  }

  return vec;
}

bool khokhlov_a_shell_tbb::CheckSorted(const std::vector<int>& input) { return std::ranges::is_sorted(input); }

std::vector<int> khokhlov_a_shell_tbb::GenerateRandomVector(int size) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{1, 100};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  std::ranges::generate(vec, gen);

  return vec;
}