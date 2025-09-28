#pragma once

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
  static std::vector<int> ShellSort(const std::vector<int>& input);
  std::vector<int> input_;
};

bool CheckSorted(const std::vector<int>& input);

std::vector<int> GenerateRandomVector(int size);

}  // namespace khokhlov_a_shell_seq