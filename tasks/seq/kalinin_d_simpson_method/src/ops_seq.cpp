#include "seq/kalinin_d_simpson_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace kalinin_d_simpson_method_seq {

static double EvaluateById(int id, const std::vector<double>& x) {
  switch (id) {
    case 0: {
      return 1.0;
    }
    case 1: {
      double s = 0.0;
      for (double v : x) s += v;
      return s;
    }
    case 2: {
      double p = 1.0;
      for (double v : x) p *= v;
      return p;
    }
    case 3: {
      double s = 0.0;
      for (double v : x) s += v * v;
      return s;
    }
    default:
      return 0.0;
  }
}

bool SimpsonNDSequential::PreProcessingImpl() {
  dimension_ = static_cast<int>(task_data->inputs_count[0]);
  lower_bounds_.assign(reinterpret_cast<double*>(task_data->inputs[0]),
                       reinterpret_cast<double*>(task_data->inputs[0]) + dimension_);
  upper_bounds_.assign(reinterpret_cast<double*>(task_data->inputs[1]),
                       reinterpret_cast<double*>(task_data->inputs[1]) + dimension_);

  const int* params = reinterpret_cast<int*>(task_data->inputs[2]);
  segments_per_dim_ = params[0];
  function_id_ = params[1];

  result_ = 0.0;
  return true;
}

bool SimpsonNDSequential::ValidationImpl() {
  if (task_data->inputs_count.size() < 3) return false;
  if (task_data->outputs_count.size() < 1) return false;
  if (task_data->outputs_count[0] != 1) return false;

  const int dim = static_cast<int>(task_data->inputs_count[0]);
  if (dim <= 0) return false;
  if (static_cast<int>(task_data->inputs_count[1]) != dim) return false;
  if (task_data->inputs_count[2] != 2) return false;

  const double* lb = reinterpret_cast<double*>(task_data->inputs[0]);
  const double* ub = reinterpret_cast<double*>(task_data->inputs[1]);
  for (int i = 0; i < dim; ++i) {
    if (!(ub[i] > lb[i])) return false;
  }

  const int* params = reinterpret_cast<int*>(task_data->inputs[2]);
  const int segments = params[0];
  if (segments <= 0 || (segments % 2) != 0) return false;
  return true;
}

bool SimpsonNDSequential::RunImpl() {
  std::vector<double> h(dimension_);
  for (int d = 0; d < dimension_; ++d) {
    h[d] = (upper_bounds_[d] - lower_bounds_[d]) / static_cast<double>(segments_per_dim_);
  }

  std::vector<int> idx(dimension_, 0);
  std::vector<double> x(dimension_, 0.0);
  const long long points_per_dim = static_cast<long long>(segments_per_dim_) + 1;
  const long long total_points = static_cast<long long>(std::pow(points_per_dim, dimension_));

  for (long long linear = 0; linear < total_points; ++linear) {
    long long tmp = linear;
    double weight = 1.0;
    for (int d = 0; d < dimension_; ++d) {
      idx[d] = static_cast<int>(tmp % points_per_dim);
      tmp /= points_per_dim;
      x[d] = lower_bounds_[d] + h[d] * static_cast<double>(idx[d]);
      if (idx[d] == 0 || idx[d] == segments_per_dim_) {
        weight *= 1.0;
      } else if ((idx[d] % 2) == 1) {
        weight *= 4.0;
      } else {
        weight *= 2.0;
      }
    }
    result_ += weight * EvaluateById(function_id_, x);
  }

  double scale = 1.0;
  for (int d = 0; d < dimension_; ++d) {
    scale *= h[d] / 3.0;
  }
  result_ *= scale;
  return true;
}

bool SimpsonNDSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double SimpsonNDSequential::EvaluateFunction(const std::vector<double>& point) const {
  return EvaluateById(function_id_, point);
}

}  // namespace kalinin_d_simpson_method_seq

