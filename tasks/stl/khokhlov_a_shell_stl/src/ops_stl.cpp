#include "stl/khokhlov_a_shell_stl/include/ops_stl.hpp"

#include <algorithm>
#include <execution>
#include <random>
#include <vector>
#include <ranges>
#include <thread>

bool khokhlov_a_shell_stl::ShellStl::PreProcessingImpl() {
  // Инициализация входных данных
  input_.resize(task_data->inputs_count[0]);
  std::ranges::copy(
      std::views::counted(reinterpret_cast<int*>(task_data->inputs[0]), task_data->inputs_count[0]),
      input_.begin());
  return true;
}

bool khokhlov_a_shell_stl::ShellStl::ValidationImpl() {
  // Проверка корректности входных и выходных данных
  return task_data->inputs_count.size() == 1 && task_data->inputs_count[0] > 0 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khokhlov_a_shell_stl::ShellStl::RunImpl() {
  input_ = ShellSort(input_);
  return true;
}

bool khokhlov_a_shell_stl::ShellStl::PostProcessingImpl() {
  // Копирование результатов в выходной буфер
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

std::vector<int> khokhlov_a_shell_stl::ShellStl::ShellSort(const std::vector<int>& input) {
  std::vector<int> vec(input);
  int n = static_cast<int>(vec.size());
  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = (n + num_threads - 1) / num_threads;

  // Первая фаза: параллельная сортировка подмассивов
  std::vector<std::pair<int, int>> chunks;
  for (int i = 0; i < n; i += chunk_size) {
    chunks.emplace_back(i, std::min(i + chunk_size, n));
  }

  for (int interval = chunk_size / 2; interval > 0; interval /= 2) {
    std::for_each(std::execution::par, chunks.begin(), chunks.end(), [&](const auto& chunk) {
      int start = chunk.first;
      int end = chunk.second;
      for (int i = start + interval; i < end; ++i) {
        int tmp = vec[i];
        int j = i;
        while (j >= start + interval && vec[j - interval] > tmp) {
          vec[j] = vec[j - interval];
          j -= interval;
        }
        vec[j] = tmp;
      }
    });
  }

  // Вторая фаза: финальная сортировка всего массива
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

bool khokhlov_a_shell_stl::CheckSorted(const std::vector<int>& input) {
  return std::ranges::is_sorted(input);
}

std::vector<int> khokhlov_a_shell_stl::GenerateRandomVector(int size) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{1, 100};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  std::ranges::generate(vec, gen);

  return vec;
}