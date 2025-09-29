#include "stl/veliev_e_montecarlo/include/ops_stl.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

using namespace std::chrono_literals;

bool veliev_e_monte_carlo_stl::VelievEMonteCarloStl::PreProcessingImpl() {
  Int1_[0] = reinterpret_cast<double *>(task_data->inputs[0])[0];
  Int1_[1] = reinterpret_cast<double *>(task_data->inputs[0])[1];
  Int2_[0] = reinterpret_cast<double *>(task_data->inputs[1])[0];
  Int2_[1] = reinterpret_cast<double *>(task_data->inputs[1])[1];
  function_ = reinterpret_cast<veliev_func_stl::Func>(task_data->inputs[2]);

  N_ = *reinterpret_cast<int *>(task_data->inputs[3]);
  res_ = 0.0;
  return true;
}

bool veliev_e_monte_carlo_stl::VelievEMonteCarloStl::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->inputs_count[1] == 2 && task_data->outputs_count[0] == 1;
}

bool veliev_e_monte_carlo_stl::VelievEMonteCarloStl::RunImpl() {
  try {
    double h1 = (Int1_[1] - Int1_[0]) / N_;
    double h2 = (Int2_[1] - Int2_[0]) / N_;

    int num_threads = ppc::util::GetPPCNumThreads();
    std::vector<std::thread> threads(num_threads);

    std::vector<double> tmp_res(num_threads, 0.0);

    for (int j = 0; j < num_threads; ++j) {
      threads[j] = std::thread([&, j]() {
        double local_res = 0.0;
        for (int i = j; i < N_; i += num_threads) {
          double lmb = Int2_[0] + (h2 * i);
          for (int t = 0; t < N_; ++t) {
            local_res += function_(Int1_[0] + (h1 * t), lmb);
          }
        }
        tmp_res[j] = local_res * h1 * h2;
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }

    for (double res : tmp_res) {
      res_ += res;
    }
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    return false;
  }

  return true;
}

bool veliev_e_monte_carlo_stl::VelievEMonteCarloStl::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}