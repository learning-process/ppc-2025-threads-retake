#pragma once

#include <omp.h>

#include <utility>

#include "core/task/include/task.hpp"
#include "omp/veliev_e_montecarlo/include/my_funcs.hpp"

namespace veliev_e_monte_carlo_omp {

class VelievEMonteCarloOmp : public ppc::core::Task {
 public:
  explicit VelievEMonteCarloOmp(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  veliev_func_omp::Func function_{};
  double Int1_[2]{}, Int2_[2]{};

  int N_{};
  double res_{};
};

}  // namespace veliev_e_monte_carlo_omp