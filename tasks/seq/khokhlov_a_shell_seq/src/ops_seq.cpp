#include "seq/khokhlov_a_shell_seq/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

bool khokhlov_a_shell_seq::ShellSeq::PreProcessingImpl() {
  // Init value for input and output
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[0]); i++) {
    input_.push_back(reinterpret_cast<int*>(task_data->inputs[0])[i]);
  }
  return true;
}

bool khokhlov_a_shell_seq::ShellSeq::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count.size() == 1 && task_data->inputs_count[0] > 0 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khokhlov_a_shell_seq::ShellSeq::RunImpl() {
  input_ = ShellSort(input_);
  return true;
}

bool khokhlov_a_shell_seq::ShellSeq::PostProcessingImpl() {
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[0]); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}

std::vector<int> khokhlov_a_shell_seq::ShellSeq::ShellSort(const std::vector<int>& input) {
  std::vector<int> vec(input);

  for (int interval = static_cast<int>(vec.size()) / 2; interval > 0; interval /= 2) {
    for (int i = interval; i < static_cast<int>(vec.size()); i++) {
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

bool khokhlov_a_shell_seq::CheckSorted(const std::vector<int>& input) { return std::ranges::is_sorted(input); }

std::vector<int> khokhlov_a_shell_seq::GenerateRandomVector(int size) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{1, 100};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  std::ranges::generate(vec, gen);

  return vec;
}