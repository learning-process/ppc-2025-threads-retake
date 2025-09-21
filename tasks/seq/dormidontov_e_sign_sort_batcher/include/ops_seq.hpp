#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dormidontov_e_sign_sort_batcher_seq {
class SeqTask : public ppc::core::Task {
 public:
  explicit SeqTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void Sort();
  void Merge(int, int);
  std::vector<double> input_, tmp_;
  size_t input_size_{};
};
}  // namespace dormidontov_e_sign_sort_batcher_seq