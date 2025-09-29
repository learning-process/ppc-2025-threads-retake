#pragma once

#include <utility>

#include "core/task/include/task.hpp"
#include "stl/veliev_e_montecarlo/include/my_funcs.hpp"

namespace veliev_e_stl_test {

class VelievEMonteCarloSeq : public ppc::core::Task {
 public:
  explicit VelievEMonteCarloSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  veliev_func_stl::Func function_{};
  double Int1_[2]{}, Int2_[2]{};

  int N_{};
  double res_{};
};

}  // namespace veliev_e_stl_test