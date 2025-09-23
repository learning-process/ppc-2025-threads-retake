#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_fox_algorithm_seq {

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  std::vector<double> result;
  size_t sizeA;
  size_t sizeB;
};

}  // namespace tarakanov_d_fox_algorithm_seq