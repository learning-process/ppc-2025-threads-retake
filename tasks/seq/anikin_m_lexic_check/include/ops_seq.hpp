#pragma once

#include <string>

#include "core/task/include/task.hpp"

namespace anikin_m_lexic_check_seq {

class LexicCheckSequential : public ppc::core::Task {
 public:
  explicit LexicCheckSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), ret(0) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string input0_, input1_;
  int ret;
};

}  // namespace anikin_m_lexic_check_seq
