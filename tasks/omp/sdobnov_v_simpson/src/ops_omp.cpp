#include <omp.h>

#include <vector>

#include "omp/sdobnov_v_simpson/include/ops_omp.hpp"

namespace sdobnov_v_simpson_omp {

double Polynomial3d(std::vector<double> point) {
  if (point.size() != 3) return 0.0;
  return point[0] * point[0] + point[1] * point[1] + point[2] * point[2] + point[0] * point[1] + point[1] * point[2];
}

double Trigonometric4d(std::vector<double> point) {
  if (point.size() != 4) return 0.0;
  return sin(point[0]) + cos(point[1]) + sin(point[2]) + cos(point[3]);
}

double Mixed5d(std::vector<double> point) {
  if (point.size() != 5) return 0.0;
  return point[0] * point[0] + sin(point[1]) + point[2] * cos(point[3]) + point[4];
}

bool SimpsonIntegralOmp::PreProcessingImpl() {
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

bool SimpsonIntegralOmp::ValidationImpl() {
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

bool SimpsonIntegralOmp::RunImpl() {
  if (integrand_function_ == nullptr) {
    return false;
  }

  if (dimensions_ >= 2) {
    result_ = ParallelSimpson();
  } else {
    std::vector<double> current_point;
    double raw_result = SimpsonRecursive(0, current_point);

    int n = n_points_[0];
    if (n % 2 != 0) n++;
    double h = (upper_bounds_[0] - lower_bounds_[0]) / n;
    double coefficient = h / 3.0;

    result_ = raw_result * coefficient;
  }

  return true;
}

bool SimpsonIntegralOmp::PostProcessingImpl() {
  if (task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }

  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result_;
  return true;
}

double SimpsonIntegralOmp::ParallelSimpson() {
  int outer_dim = 0;
  double a = lower_bounds_[outer_dim];
  double b = upper_bounds_[outer_dim];
  int n = n_points_[outer_dim];
  if (n % 2 != 0) n++;
  double h = (b - a) / n;

  double total_sum = 0.0;

#pragma omp parallel for reduction(+ : total_sum) schedule(dynamic)
  for (int i = 0; i <= n; i++) {
    double x = a + i * h;
    double weight;
    if (i == 0 || i == n) {
      weight = 1;
    } else if (i % 2 == 0) {
      weight = 2;
    } else {
      weight = 4;
    }

    std::vector<double> current_point;
    current_point.push_back(x);

    double partial_sum = SimpsonRecursive(1, current_point);
    total_sum += weight * partial_sum;
  }

  double coefficient = h / 3.0;
  for (int i = 1; i < dimensions_; ++i) {
    int n_inner = n_points_[i];
    if (n_inner % 2 != 0) n_inner++;
    double h_inner = (upper_bounds_[i] - lower_bounds_[i]) / n_inner;
    coefficient *= h_inner / 3.0;
  }

  return total_sum * coefficient;
}

double SimpsonIntegralOmp::SimpsonRecursive(int dim_index, const std::vector<double>& current_point) {
  if (dim_index == dimensions_) {
    return integrand_function_(current_point);
  }

  double a = lower_bounds_[dim_index];
  double b = upper_bounds_[dim_index];
  int n = n_points_[dim_index];
  if (n % 2 != 0) n++;
  double h = (b - a) / n;

  double sum = 0.0;

  for (int i = 0; i <= n; i++) {
    double x = a + i * h;
    double weight;
    if (i == 0 || i == n) {
      weight = 1;
    } else if (i % 2 == 0) {
      weight = 2;
    } else {
      weight = 4;
    }

    std::vector<double> new_point = current_point;
    new_point.push_back(x);
    double partial_sum = SimpsonRecursive(dim_index + 1, new_point);
    sum += weight * partial_sum;
  }

  return sum;
}

}  // namespace sdobnov_v_simpson_omp