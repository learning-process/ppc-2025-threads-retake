// ops_tbb.cpp (исправленный под clang-tidy)

#include "ops_tbb.h"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <thread>
#include <vector>

namespace {

double Sum(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) {
    s += v;
  }
  return s;
}

double Product(const std::vector<double>& x) {
  double p = 1.0;
  for (double v : x) {
    p *= v;
  }
  return p;
}

double SquareSum(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) {
    s += v * v;
  }
  return s;
}

int SimpsonWeight(int idx, int segments) {
  if (idx == 0 || idx == segments) {
    return 1;
  }
  if (idx % 2 == 0) {
    return 2;
  }
  return 4;
}

}  // namespace

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
  int dim = static_cast<int>(task_data->inputs_count[0]);
  if (dim <= 0) {
    return false;
  }
  if (static_cast<int>(task_data->inputs_count[1]) != dim) {
    return false;
  }
  if (task_data->inputs_count[2] != 2) {
    return false;
  }
  auto lb = task_data->inputs[1];
  auto ub = task_data->inputs[2];
  for (int i = 0; i < dim; i++) {
    if (!(ub[i] > lb[i])) {
      return false;
    }
  }
  return true;
}

bool SimpsonNDTBB::RunImpl() {
  dimension_ = static_cast<int>(task_data->inputs_count[0]);
  lower_bounds_ = task_data->inputs[1];
  upper_bounds_ = task_data->inputs[2];

  h.resize(dimension_);
  for (int d = 0; d < dimension_; ++d) {
    h[d] = (upper_bounds_[d] - lower_bounds_[d]) / static_cast<double>(segments_per_dim_);
  }

  oneapi::tbb::global_control c(oneapi::tbb::global_control::max_allowed_parallelism,
                                std::thread::hardware_concurrency());

  auto integral = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<long long>(0, static_cast<long long>(std::pow(segments_per_dim_ + 1, dimension_))),
      0.0,
      [&](const oneapi::tbb::blocked_range<long long>& r, double local_sum) -> double {
        std::vector<int> idx(dimension_);
        std::vector<double> x(dimension_);
        for (long long i = r.begin(); i != r.end(); ++i) {
          long long t = i;
          double weight = 1.0;
          for (int d = 0; d < dimension_; ++d) {
            idx[d] = t % (segments_per_dim_ + 1);
            t /= (segments_per_dim_ + 1);
            x[d] = lower_bounds_[d] + idx[d] * h[d];
            weight *= SimpsonWeight(idx[d], segments_per_dim_);
          }
          local_sum += weight * func_(x);
        }
        return local_sum;
      },
      std::plus<>());

  double scale = 1.0;
  for (int d = 0; d < dimension_; ++d) {
    scale *= h[d] / 3.0;
  }
  integral *= scale;

  task_data->outputs[0] = {integral};
  return true;
}