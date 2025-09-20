#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chastov_v_shell_sort_with_even_odd_batcher_merge {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_data_;
};

}  // namespace chastov_v_shell_sort_with_even_odd_batcher_merge