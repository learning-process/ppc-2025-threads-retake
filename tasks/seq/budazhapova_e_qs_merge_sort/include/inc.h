#pragma once
#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_e_qs_merge_sort_seq {

class QSMergeSortSequential : public ppc::core::Task {
 public:
  explicit QSMergeSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
};

}  // namespace budazhapova_e_qs_merge_sort_seq