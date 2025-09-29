#include "stl/sdobnov_v_simpson/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

namespace sdobnov_v_simpson_stl {
double Polynomial3d(std::vector<double> point) {
  if (point.size() != 3) {
    return 0.0;
  }
  return (point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2]) + (point[0] * point[1]) +
         (point[1] * point[2]);
}

double Trigonometric4d(std::vector<double> point) {
  if (point.size() != 4) {
    return 0.0;
  }
  return std::sin(point[0]) + std::cos(point[1]) + std::sin(point[2]) + std::cos(point[3]);
}

double Mixed5d(std::vector<double> point) {
  if (point.size() != 5) {
    return 0.0;
  }
  return (point[0] * point[0]) + std::sin(point[1]) + (point[2] * std::cos(point[3])) + point[4];
}

bool SimpsonIntegralStl::PreProcessingImpl() {
  if (task_data->inputs_count.size() < 5) {
    return false;
  }

  if (task_data->inputs_count[0] != sizeof(int)) {
    return false;
  }
  dimensions_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  if (dimensions_ <= 0) {
    return false;
  }

  if (task_data->inputs_count[1] != dimensions_ * sizeof(double)) {
    return false;
  }
  lower_bounds_.resize(dimensions_);
  for (int i = 0; i < dimensions_; ++i) {
    lower_bounds_[i] = reinterpret_cast<double*>(task_data->inputs[1])[i];
  }

  if (task_data->inputs_count[2] != dimensions_ * sizeof(double)) {
    return false;
  }
  upper_bounds_.resize(dimensions_);
  for (int i = 0; i < dimensions_; ++i) {
    upper_bounds_[i] = reinterpret_cast<double*>(task_data->inputs[2])[i];
    if (upper_bounds_[i] <= lower_bounds_[i]) {
      return false;
    }
  }

  if (task_data->inputs_count[3] != dimensions_ * sizeof(int)) {
    return false;
  }
  n_points_.resize(dimensions_);
  for (int i = 0; i < dimensions_; ++i) {
    n_points_[i] = reinterpret_cast<int*>(task_data->inputs[3])[i];
    if (n_points_[i] <= 0) {
      return false;
    }
  }

  if (task_data->inputs_count[4] != sizeof(Func)) {
    return false;
  }
  integrand_function_ = reinterpret_cast<Func>(task_data->inputs[4]);

  return integrand_function_ != nullptr;
}

bool SimpsonIntegralStl::ValidationImpl() {
  if (task_data->inputs_count.size() < 5) {
    return false;
  }
  if (task_data->inputs_count[0] != sizeof(int)) {
    return false;
  }

  int dimensions = *reinterpret_cast<int*>(task_data->inputs[0]);
  if (dimensions <= 0) {
    return false;
  }

  if (task_data->inputs_count[1] != dimensions * sizeof(double)) {
    return false;
  }
  if (task_data->inputs_count[2] != dimensions * sizeof(double)) {
    return false;
  }
  if (task_data->inputs_count[3] != dimensions * sizeof(int)) {
    return false;
  }
  if (task_data->inputs_count[4] != sizeof(Func)) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }

  return true;
}

bool SimpsonIntegralStl::RunImpl() {
  if (integrand_function_ == nullptr) {
    return false;
  }

  if (dimensions_ >= 2) {
    result_ = ParallelSimpsonAsync();
  } else {
    std::vector<double> current_point;
    double raw_result = SimpsonRecursive(0, current_point);

    int n = n_points_[0];
    if (n % 2 != 0) {
      n++;
    }
    double h = (upper_bounds_[0] - lower_bounds_[0]) / n;
    double coefficient = h / 3.0;

    result_ = raw_result * coefficient;
  }

  return true;
}

bool SimpsonIntegralStl::PostProcessingImpl() {
  if (task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result_;
  return true;
}

double SimpsonIntegralStl::ParallelSimpsonAsync() {
  int outer_dim = 0;
  double a = lower_bounds_[outer_dim];
  double b = upper_bounds_[outer_dim];
  int n = n_points_[outer_dim];
  if (n % 2 != 0) {
    n++;
  }
  double h = (b - a) / n;

  unsigned int num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 4;
  }
  std::vector<std::future<double>> futures;
  int chunk_size = std::max(1, n / static_cast<int>(num_threads));

  for (int chunk_start = 0; chunk_start <= n; chunk_start += chunk_size) {
    int chunk_end = std::min(chunk_start + chunk_size, n + 1);

    futures.push_back(std::async(std::launch::async, [=, this]() {
      double local_sum = 0.0;
      for (int i = chunk_start; i < chunk_end; i++) {
        double x = a + (i * h);
        double weight = 0;
        if (i == 0 || i == n) {
          weight = 1;
        } else if (i % 2 == 0) {
          weight = 2;
        } else {
          weight = 4;
        }

        std::vector<double> point = {x};
        double partial_sum = SimpsonRecursive(1, point);
        local_sum += weight * partial_sum;
      }
      return local_sum;
    }));
  }

  double total_sum = 0.0;
  for (auto& future : futures) {
    total_sum += future.get();
  }

  double coefficient = h / 3.0;
  for (int i = 1; i < dimensions_; ++i) {
    int n_inner = n_points_[i];
    if (n_inner % 2 != 0) {
      n_inner++;
    }
    double h_inner = (upper_bounds_[i] - lower_bounds_[i]) / n_inner;
    coefficient *= h_inner / 3.0;
  }

  return total_sum * coefficient;
}

double SimpsonIntegralStl::ParallelSimpsonThreads() {
  int outer_dim = 0;
  double a = lower_bounds_[outer_dim];
  double b = upper_bounds_[outer_dim];
  int n = n_points_[outer_dim];
  if (n % 2 != 0) {
    n++;
  }
  double h = (b - a) / n;

  unsigned int num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 4;
  }

  std::vector<std::thread> threads;
  std::vector<double> partial_sums(num_threads, 0.0);
  int chunk_size = std::max(1, n / static_cast<int>(num_threads));

  for (unsigned int t = 0; t < num_threads; ++t) {
    int chunk_start = static_cast<int>(t) * chunk_size;
    int chunk_end = (t == num_threads - 1) ? n + 1 : chunk_start + chunk_size;

    threads.emplace_back([=, this, &partial_sums]() {
      double local_sum = 0.0;
      for (int i = chunk_start; i < chunk_end && i <= n; i++) {
        double x = a + (i * h);
        double weight = 0;
        if (i == 0 || i == n) {
          weight = 1;
        } else if (i % 2 == 0) {
          weight = 2;
        } else {
          weight = 4;
        }

        std::vector<double> point = {x};
        double partial_sum = SimpsonRecursive(1, point);
        local_sum += weight * partial_sum;
      }
      partial_sums[t] = local_sum;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  double total_sum = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);

  double coefficient = h / 3.0;
  for (int i = 1; i < dimensions_; ++i) {
    int n_inner = n_points_[i];
    if (n_inner % 2 != 0) {
      n_inner++;
    }
    double h_inner = (upper_bounds_[i] - lower_bounds_[i]) / n_inner;
    coefficient *= h_inner / 3.0;
  }

  return total_sum * coefficient;
}

double SimpsonIntegralStl::ProcessPoint(int i) {
  double a = lower_bounds_[0];
  double b = upper_bounds_[0];
  int n = n_points_[0];
  if (n % 2 != 0) {
    n++;
  }
  double h = (b - a) / n;

  double x = a + (i * h);
  double weight = 0;
  if (i == 0 || i == n) {
    weight = 1;
  } else if (i % 2 == 0) {
    weight = 2;
  } else {
    weight = 4;
  }

  std::vector<double> point = {x};
  return weight * SimpsonRecursive(1, point);
}

double SimpsonIntegralStl::SimpsonRecursive(int dim_index, const std::vector<double>& current_point) {
  if (dim_index == dimensions_) {
    return integrand_function_(current_point);
  }

  double a = lower_bounds_[dim_index];
  double b = upper_bounds_[dim_index];
  int n = n_points_[dim_index];
  if (n % 2 != 0) {
    n++;
  }
  double h = (b - a) / n;

  double sum = 0.0;

  for (int i = 0; i <= n; i++) {
    double x = a + (i * h);
    double weight = 0;
    if (i == 0 || i == n) {
      weight = 1;
    } else if (i % 2 == 0) {
      weight = 2;
    } else {
      weight = 4;
    }

    std::vector<double> new_point;
    new_point.reserve(current_point.size() + 1);
    new_point = current_point;
    new_point.push_back(x);

    double partial_sum = SimpsonRecursive(dim_index + 1, new_point);
    sum += weight * partial_sum;
  }

  return sum;
}

}  // namespace sdobnov_v_simpson_stl