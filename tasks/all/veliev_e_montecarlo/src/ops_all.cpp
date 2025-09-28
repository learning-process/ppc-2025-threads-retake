#include "all/veliev_e_montecarlo/include/ops_all.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

using namespace std::chrono_literals;

void veliev_e_monte_carlo_all::VelievEMonteCarloAll::SetFunc(const veliev_func_all::Func &f) { function_ = f; }

bool veliev_e_monte_carlo_all::VelievEMonteCarloAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    Int1_[0] = reinterpret_cast<double *>(task_data->inputs[0])[0];
    Int1_[1] = reinterpret_cast<double *>(task_data->inputs[0])[1];
    Int2_[0] = reinterpret_cast<double *>(task_data->inputs[1])[0];
    Int2_[1] = reinterpret_cast<double *>(task_data->inputs[1])[1];

    N_ = *reinterpret_cast<int *>(task_data->inputs[3]);
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
    boost::mpi::broadcast(world_, Int1_[0], 0);
    boost::mpi::broadcast(world_, Int1_[1], 0);
    boost::mpi::broadcast(world_, Int2_[0], 0);
    boost::mpi::broadcast(world_, Int2_[1], 0);
    boost::mpi::broadcast(world_, N_, 0);

    const double h1 = (Int1_[1] - Int1_[0]) / N_;
    const double h2 = (Int2_[1] - Int2_[0]) / N_;

    int size = world_.size();
    int rank = world_.rank();

    int base = N_ / size;
    int rem = N_ % size;
    int j_start = rank * base + std::min(rank, rem);
    int j_count = base + (rank < rem ? 1 : 0);
    int j_end = j_start + j_count;

    double local_res = 0.0;

    for (int j = j_start; j < j_end; ++j) {
      double y = Int2_[0] + (h2 * static_cast<double>(j));
      double sum_i = 0.0;

#pragma omp parallel for reduction(+ : sum_i)
      for (int i = 0; i < N_; ++i) {
        double x = Int1_[0] + (h1 * static_cast<double>(i));
        sum_i += function_(x, y);
      }

      local_res += sum_i;
    }

    local_res *= (h1 * h2);

    double global_res = 0.0;
    boost::mpi::reduce(world_, local_res, global_res, std::plus<double>(), 0);

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