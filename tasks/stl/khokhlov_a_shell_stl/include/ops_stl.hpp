#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_shell_stl {

class ShellStl : public ppc::core::Task {
 public:
  explicit ShellStl(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<int> ShellSort(const std::vector<int>& input);
  static void ShellSortChunk(std::vector<int>& vec, int start, int end, int interval);
  std::vector<int> input_;
};

bool CheckSorted(const std::vector<int>& input);

std::vector<int> GenerateRandomVector(int size);

}  // namespace khokhlov_a_shell_stl