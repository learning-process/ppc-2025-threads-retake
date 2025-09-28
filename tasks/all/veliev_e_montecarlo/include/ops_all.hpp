#pragma once

#include <omp.h>

#include <boost/mpi/communicator.hpp>
#include <utility>

#include "all/veliev_e_montecarlo/include/my_funcs.hpp"
#include "core/task/include/task.hpp"

namespace veliev_e_monte_carlo_all {

class VelievEMonteCarloAll : public ppc::core::Task {
 public:
  explicit VelievEMonteCarloAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SetFunc(const veliev_func_all::Func &f);

 private:
  veliev_func_all::Func function_{};
  double Int1_[2]{}, Int2_[2]{};

  int N_{};
  double res_{};

  boost::mpi::communicator world_;
};

}  // namespace veliev_e_monte_carlo_all