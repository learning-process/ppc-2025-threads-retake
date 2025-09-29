#include "all/veliev_e_montecarlo/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <vector>

using namespace std::chrono_literals;

void veliev_e_monte_carlo_all::VelievEMonteCarloAll::SetFunc(const veliev_func_all::Func &f) { function_ = f; }

bool veliev_e_monte_carlo_all::VelievEMonteCarloAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *in1 = reinterpret_cast<double *>(task_data->inputs[0]);
    auto *in2 = reinterpret_cast<double *>(task_data->inputs[1]);

    Int1_[0] = in1[0];
    Int1_[1] = in1[1];
    Int2_[0] = in2[0];
    Int2_[1] = in2[1];

    std::memcpy(&N_, task_data->inputs[2], sizeof(N_));
    res_ = 0.0;
  }
  return true;
}

bool veliev_e_monte_carlo_all::VelievEMonteCarloAll::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == 2 && task_data->inputs_count[1] == 2 && task_data->outputs_count[0] == 1;
  }

  return true;
}

bool veliev_e_monte_carlo_all::VelievEMonteCarloAll::RunImpl() {
  try {
    double buf[4];
    if (world_.rank() == 0) {
      buf[0] = Int1_[0];
      buf[1] = Int1_[1];
      buf[2] = Int2_[0];
      buf[3] = Int2_[1];
    }
    boost::mpi::broadcast(world_, buf, 4, 0);
    boost::mpi::broadcast(world_, N_, 0);

    const double x0 = buf[0];
    const double x1 = buf[1];
    const double y0 = buf[2];
    const double y1 = buf[3];
    const int n = N_;

    const double h1 = (x1 - x0) / n;
    const double h2 = (y1 - y0) / n;

    int size = world_.size();
    int rank = world_.rank();
    int base = n / size;
    int rem = n % size;
    int j_start = (rank * base) + std::min(rank, rem);
    int j_count = base + (rank < rem ? 1 : 0);
    int j_end = j_start + j_count;

    auto func = function_;

    double local_res = 0.0;

    const int recompute_period = 1024;

#pragma omp parallel
    {
      double thread_res = 0.0;

#pragma omp for schedule(static)
      for (int j = j_start; j < j_end; ++j) {
        double y = y0 + (h2 * j);

        double x = x0;
        int i = 0;

        int limit = n - (n % 4);
        for (; i < limit; i += 4) {
          if ((i & (recompute_period - 1)) == 0) {
            x = x0 + (h1 * i);
          }
          thread_res += func(x, y);
          x += h1;
          thread_res += func(x, y);
          x += h1;
          thread_res += func(x, y);
          x += h1;
          thread_res += func(x, y);
          x += h1;
        }

        for (; i < n; ++i) {
          if ((i & (recompute_period - 1)) == 0) {
            x = x0 + (h1 * i);
          }
          thread_res += func(x, y);
          x += h1;
        }
      }

#pragma omp atomic
      local_res += thread_res;
    }  // omp parallel

    double local_res_sum = local_res * (h1 * h2);

    double global_res = 0.0;
    boost::mpi::reduce(world_, local_res_sum, global_res, std::plus<double>(), 0);

    if (rank == 0) {
      res_ = global_res;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}

bool veliev_e_monte_carlo_all::VelievEMonteCarloAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  }
  return true;
}