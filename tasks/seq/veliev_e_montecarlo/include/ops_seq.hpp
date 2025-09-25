#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_monte_carlo_seq {

using func = double (*)(double, double);

double Flin(double x, double y);
double FsinxPsiny(double x, double y);
double FcosxPcosy(double x, double y);
double Fxy(double x, double y);
double Fxyy(double x, double y);

class VelievEMonteCarloSeq : public ppc::core::Task {
 public:
  explicit VelievEMonteCarloSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  func function_{};
  double Int1_[2]{}, Int2_[2]{};

  int N_{};
  double res_{};
};

}  // namespace veliev_e_monte_carlo_seq