#include "seq/Veliev_E_MonteCarlo/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
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
  function_ = reinterpret_cast<func>(task_data->inputs[2]);

  N_ = *reinterpret_cast<int *>(task_data->inputs[3]);
  res_ = 0.0;
  return true;
}

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->inputs_count[1] == 2 && task_data->outputs_count[0] == 1;
}

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::RunImpl() {
  double h1_ = (Int1_[1] - Int1_[0]) / N_;
  double h2_ = (Int2_[1] - Int2_[0]) / N_;

  int i_;
  int j_;
  for (j_ = 0; j_ < N_; ++j_) {
    double y_ = Int2_[0] + h2_ * j_;
    for (i_ = 0; i_ < N_; ++i_) {
      res_ += function_(Int1_[0] + h1_ * i_, y_);
    }
  }
  res_ *= h1_ * h2_;

  return true;
}

bool veliev_e_monte_carlo_seq::VelievEMonteCarloSeq::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
