#include "seq/veliev_e_montecarlo/include/ops_seq.hpp"

#include <cmath>
#include <vector>

using namespace std::chrono_literals;

double veliev_e_monte_carlo_seq::Flin(double x, double y) { return x + y; }
double veliev_e_monte_carlo_seq::FsinxPsiny(double x, double y) { return sin(x) + sin(y); }
double veliev_e_monte_carlo_seq::FcosxPcosy(double x, double y) { return cos(x) + cos(y); }
double veliev_e_monte_carlo_seq::Fxy(double x, double y) { return x * y; }
double veliev_e_monte_carlo_seq::Fxyy(double x, double y) { return x * y * y; }

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::PreProcessingImpl() {
  Int1_[0] = reinterpret_cast<double *>(task_data->inputs[0])[0];
  Int1_[1] = reinterpret_cast<double *>(task_data->inputs[0])[1];
  Int2_[0] = reinterpret_cast<double *>(task_data->inputs[1])[0];
  Int2_[1] = reinterpret_cast<double *>(task_data->inputs[1])[1];
  function_ = reinterpret_cast<Func>(task_data->inputs[2]);

  N_ = *reinterpret_cast<int *>(task_data->inputs[3]);
  res_ = 0.0;
  return true;
}

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->inputs_count[1] == 2 && task_data->outputs_count[0] == 1;
}

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::RunImpl() {
  double h1 = (Int1_[1] - Int1_[0]) / N_;
  double h2 = (Int2_[1] - Int2_[0]) / N_;

  int i{};
  int j{};
  for (j = 0; j < N_; ++j) {
    double y = Int2_[0] + (h2 * j);
    for (i = 0; i < N_; ++i) {
      res_ += function_(Int1_[0] + (h1 * i), y);
    }
  }
  res_ *= h1 * h2;

  return true;
}

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
