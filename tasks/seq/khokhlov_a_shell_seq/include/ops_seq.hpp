#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_shell_seq {

class ShellSeq : public ppc::core::Task {
 public:
  explicit ShellSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<int> shell_sort(const std::vector<int>& input);
  std::vector<int> input_;
};

bool checkSorted(std::vector<int> input);

std::vector<int> generate_random_vector(int size, int min, int max);

}  // namespace khokhlov_a_shell_seq