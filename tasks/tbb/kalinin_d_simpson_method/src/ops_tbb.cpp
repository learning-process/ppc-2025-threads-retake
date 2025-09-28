#include "tbb/kalinin_d_simpson_method/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <vector>

namespace kalinin_d_simpson_method_tbb {
namespace {
double EvaluateById(int id, const std::vector<double>& x) {
  switch (id) {
    case 0:
      return 1.0;
    case 1: {
      double s = 0.0;
      for (double v : x) {
        s += v;
      }
      return s;
    }
    case 2: {
      double p = 1.0;
      for (double v : x) {
        p *= v;
      }
      return p;
    }
    case 3: {
      double s = 0.0;
      for (double v : x) {
        s += v * v;
      }
      return s;
    }
    default:
      return 0.0;
  }
}

inline int SimpsonWeight(int idx, int segments) {
  if (idx == 0 || idx == segments) {
    return 1;
  }
  return (idx % 2 == 1) ? 4 : 2;
}
}  // namespace

bool SimpsonNDTBB::PreProcessingImpl() {
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

bool SimpsonNDTBB::ValidationImpl() {
  if (task_data->inputs_count.size() < 3) {
    return false;
  }
  if (task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->outputs_count[0] != 1) {
    return false;
  }

  const int dim = static_cast<int>(task_data->inputs_count[0]);
  if (dim <= 0) {
    return false;
  }
  if (static_cast<int>(task_data->inputs_count[1]) != dim) {
    return false;
  }
  if (task_data->inputs_count[2] != 2) {
    return false;
  }

  const double* lb = reinterpret_cast<double*>(task_data->inputs[0]);
  const double* ub = reinterpret_cast<double*>(task_data->inputs[1]);
  for (int i = 0; i < dim; ++i) {
    if (!(ub[i] > lb[i])) {
      return false;
    }
  }

  const int* params = reinterpret_cast<int*>(task_data->inputs[2]);
  const int segments = params[0];
  return segments > 0 && (segments % 2) == 0;
}

bool SimpsonNDTBB::RunImpl() {
  std::vector<double> h(dimension_);
  for (int d = 0; d < dimension_; ++d) {
    h[d] = (upper_bounds_[d] - lower_bounds_[d]) / static_cast<double>(segments_per_dim_);
  }

  const long long points_per_dim = static_cast<long long>(segments_per_dim_) + 1;
  const auto total_points = static_cast<long long>(std::pow(points_per_dim, dimension_));

  auto sum = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<long long>(0, total_points, 200000), 0.0,
      [&](const oneapi::tbb::blocked_range<long long>& r, double local_sum) -> double {
        std::vector<int> idx(dimension_, 0);
        std::vector<double> x(dimension_, 0.0);

        long long tmp = r.begin();
        for (int d = 0; d < dimension_; ++d) {
          idx[d] = static_cast<int>(tmp % points_per_dim);
          tmp /= points_per_dim;
          x[d] = lower_bounds_[d] + h[d] * static_cast<double>(idx[d]);
        }

        double weight = 1.0;
        for (int d = 0; d < dimension_; ++d) {
          weight *= SimpsonWeight(idx[d], segments_per_dim_);
        }

        for (long long linear = r.begin(); linear < r.end(); ++linear) {
          local_sum += weight * EvaluateById(function_id_, x);

          int d = 0;
          while (d < dimension_) {
            const int old = idx[d];
            int old_wd = SimpsonWeight(old, segments_per_dim_);

            int new_idx = old + 1;
            if (new_idx <= segments_per_dim_) {
              idx[d] = new_idx;
              x[d] += h[d];
              int new_wd = SimpsonWeight(idx[d], segments_per_dim_);
              weight = weight / static_cast<double>(old_wd) * static_cast<double>(new_wd);
              break;
            }

            idx[d] = 0;
            x[d] = lower_bounds_[d];
            weight = weight / static_cast<double>(old_wd) * 1.0;
            ++d;
          }
        }
        return local_sum;
      },
      [](double a, double b) -> double { return a + b; });

  double scale = 1.0;
  for (int d = 0; d < dimension_; ++d) {
    scale *= h[d] / 3.0;
  }
  result_ = sum * scale;
  return true;
}

bool SimpsonNDTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double SimpsonNDTBB::EvaluateFunction(const std::vector<double>& point) const {
  return EvaluateById(function_id_, point);
}

}  // namespace kalinin_d_simpson_method_tbb