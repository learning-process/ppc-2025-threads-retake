#include <cmath>
#include <iostream>
#include <exception>
#include <thread>
#include <vector>

#include "oneapi/tbb.h"
#include "oneapi/tbb/parallel_reduce.h"
#include "tbb/veliev_e_montecarlo/include/ops_tbb.hpp"

using namespace std::chrono_literals;

bool veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb::PreProcessingImpl() {
  Int1_[0] = reinterpret_cast<double *>(task_data->inputs[0])[0];
  Int1_[1] = reinterpret_cast<double *>(task_data->inputs[0])[1];
  Int2_[0] = reinterpret_cast<double *>(task_data->inputs[1])[0];
  Int2_[1] = reinterpret_cast<double *>(task_data->inputs[1])[1];
  function_ = reinterpret_cast<veliev_func_tbb::Func>(task_data->inputs[2]);

  N_ = *reinterpret_cast<int *>(task_data->inputs[3]);
  res_ = 0.0;
  return true;
}

bool veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->inputs_count[1] == 2 && task_data->outputs_count[0] == 1;
}

bool veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb::RunImpl() {  
  try {
    double h1 = (Int1_[1] - Int1_[0]) / N_;
    double h2 = (Int2_[1] - Int2_[0]) / N_;

    res_ = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(0, N_), 0.0,
        [&](const tbb::blocked_range<int> &r, double total) {
          for (int j = r.begin(); j < r.end(); ++j) {
            double y = Int2_[0] + (h2 * j);
            for (int i = 0; i < N_; ++i) {
              total += function_(Int1_[0] + h1 * i, y);
            }
          }
          return total;
        },
        std::plus<>());
    res_ *= h1 * h2;
  } catch (const std::exception &e) {
    std::cout << e.what() << '\n';
    return false;
  }
  
  return true;
}

bool veliev_e_monte_carlo_tbb::VelievEMonteCarloTbb::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}