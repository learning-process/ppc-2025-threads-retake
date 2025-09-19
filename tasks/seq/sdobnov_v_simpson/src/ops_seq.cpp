#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "seq/sdobnov_v_simpson/include/ops_seq.hpp"

namespace sdobnov_v_simpson {

bool SimpsonIntegralSequential::PreProcessingImpl() {
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
  integrand_function_ = *reinterpret_cast<Func>(task_data->inputs[4]);

  return integrand_function_ != nullptr;
}

bool SimpsonIntegralSequential::ValidationImpl() {
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

bool SimpsonIntegralSequential::RunImpl() {
  if (integrand_function_ == nullptr) {
    return false;
  }

  std::vector<double> current_point;
  double raw_result = SimpsonRecursive(0, current_point);

  double coefficient = 1.0;
  for (int i = 0; i < dimensions_; ++i) {
    int n = n_points_[i];
    if (n % 2 != 0) n++;
    double h = (upper_bounds_[i] - lower_bounds_[i]) / n;
    coefficient *= h / 3.0;
  }

  result_ = raw_result * coefficient;
  return true;
}

bool SimpsonIntegralSequential::PostProcessingImpl() {
  if (task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }

  double* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result_;
  return true;
}

double SimpsonIntegralSequential::SimpsonRecursive(int dim_index, std::vector<double> current_point) {
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

}  // namespace sdobnov_v_simpson