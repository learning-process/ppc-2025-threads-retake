#include "tbb/leontev_n_graham/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"

bool leontev_n_graham_tbb::GrahamTbb::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr_x = reinterpret_cast<float *>(task_data->inputs[0]);
  auto *in_ptr_y = reinterpret_cast<float *>(task_data->inputs[1]);
  input_X_ = std::vector<float>(in_ptr_x, in_ptr_x + input_size);
  input_Y_ = std::vector<float>(in_ptr_y, in_ptr_y + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_X_ = std::vector<float>(output_size, 0);
  output_Y_ = std::vector<float>(output_size, 0);
  return true;
}

bool leontev_n_graham_tbb::GrahamTbb::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0] &&
         task_data->inputs_count[0] > 0;
}

std::pair<float, float> leontev_n_graham_tbb::GrahamTbb::Minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_tbb::GrahamTbb::Mul(std::pair<float, float> a, std::pair<float, float> b) {
  return (a.first * b.second) - (b.first * a.second);
}

void leontev_n_graham_tbb::GrahamTbb::InitData(std::vector<std::vector<std::pair<float, float>>> &data, int threads,
                                               size_t temp_size, const std::vector<std::pair<float, float>> &points) {
  data[threads - 1].resize(temp_size + 1 + ((points.size() - 1) % threads));
  data[0][0] = points[0];
  std::copy(points.begin() + 1, points.begin() + 1 + static_cast<long>(temp_size), data[0].begin() + 1);
  for (int i = 1; i < threads - 1; i++) {
    data[i][0] = points[0];
    std::copy(points.begin() + 1 + static_cast<long>(i * temp_size),
              points.begin() + 1 + static_cast<long>((i + 1) * temp_size), data[i].begin() + 1);
  }
  data[threads - 1][0] = points[0];
  std::copy(points.begin() + 1 + static_cast<long>((threads - 1) * temp_size), points.end(),
            data[threads - 1].begin() + 1);
}

std::pair<float, float> leontev_n_graham_tbb::GrahamTbb::GetMinPoint(
    const std::vector<std::pair<float, float>> &points) {
  using Point = std::pair<float, float>;
  Point p0 = points[0];
  for (Point p : points) {
    if (p.first < p0.first || (p.first == p0.first && p.second < p0.second)) {
      p0 = p;
    }
  }
  return p0;
}

void leontev_n_graham_tbb::GrahamTbb::WhileLoop(const std::pair<float, float> &p,
                                                std::vector<std::pair<float, float>> &hull) {
  using Point = std::pair<float, float>;
  while (hull.size() >= 2) {
    Point new_vector = Minus(p, hull.back());
    Point last_vector = Minus(hull.back(), hull[hull.size() - 2]);
    if (Mul(new_vector, last_vector) >= 0.0F) {
      hull.pop_back();
    } else {
      break;
    }
  }
  hull.push_back(p);
}

bool leontev_n_graham_tbb::GrahamTbb::RunImpl() {
  size_t amount_of_points = input_X_.size();
  if (amount_of_points == 0) {
    return false;
  }
  using Point = std::pair<float, float>;
  std::vector<Point> points(amount_of_points);
  for (size_t i = 0; i < amount_of_points; i++) {
    points[i] = Point(input_X_[i], input_Y_[i]);
  }
  Point p0 = GetMinPoint(points);

  // sort by polar angle
  std::ranges::sort(points, [&](Point a, Point b) {
    float res = Mul(Minus(a, p0), Minus(b, p0));
    if (res == 0) {
      return (Minus(a, p0).first * Minus(a, p0).first) + (Minus(a, p0).second * Minus(a, p0).second) <
             (Minus(b, p0).first * Minus(b, p0).first) + (Minus(b, p0).second * Minus(b, p0).second);
    }
    return res > 0.0F;
  });
  int threads = ppc::util::GetPPCNumThreads();
  if (threads < 1) {
    return false;
  }
  size_t temp_size = (points.size() - 1) / threads;
  std::vector<std::vector<Point>> data(threads, std::vector<Point>(temp_size + 1));
  std::vector<std::vector<Point>> outputs(threads);
  std::vector<Point> hull1;
  InitData(data, threads, temp_size, points);
  tbb::parallel_for(tbb::blocked_range<int>(0, threads), [&](tbb::blocked_range<int> r) {
    for (int thread = r.begin(); thread < r.end(); thread++) {
      for (Point p : data[thread]) {
        WhileLoop(p, outputs[thread]);
      }
    }
  });
  for (int thread = 0; thread < threads; thread++) {
    for (size_t i = (thread == 0) ? 0 : 1; i < outputs[thread].size(); i++) {
      Point p = outputs[thread][i];
      WhileLoop(p, hull1);
    }
  }
  output_X_.resize(hull1.size());
  output_Y_.resize(hull1.size());
  for (size_t i = 0; i < hull1.size(); i++) {
    output_X_[i] = hull1[i].first;
    output_Y_[i] = hull1[i].second;
  }
  return true;
}

bool leontev_n_graham_tbb::GrahamTbb::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[2])[0] = static_cast<int>(output_X_.size());
  for (size_t i = 0; i < output_X_.size(); i++) {
    reinterpret_cast<float *>(task_data->outputs[0])[i] = output_X_[i];
    reinterpret_cast<float *>(task_data->outputs[1])[i] = output_Y_[i];
  }
  return true;
}

bool leontev_n_graham_tbb::GrahamSeq::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr_x = reinterpret_cast<float *>(task_data->inputs[0]);
  auto *in_ptr_y = reinterpret_cast<float *>(task_data->inputs[1]);
  input_X_ = std::vector<float>(in_ptr_x, in_ptr_x + input_size);
  input_Y_ = std::vector<float>(in_ptr_y, in_ptr_y + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_X_ = std::vector<float>(output_size, 0);
  output_Y_ = std::vector<float>(output_size, 0);
  return true;
}

bool leontev_n_graham_tbb::GrahamSeq::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0] &&
         task_data->inputs_count[0] > 0;
}

std::pair<float, float> leontev_n_graham_tbb::GrahamSeq::Minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_tbb::GrahamSeq::Mul(std::pair<float, float> a, std::pair<float, float> b) {
  return (a.first * b.second) - (b.first * a.second);
}

bool leontev_n_graham_tbb::GrahamSeq::RunImpl() {
  size_t amount_of_points = input_X_.size();
  if (amount_of_points == 0) {
    return false;
  }
  using Point = std::pair<float, float>;
  std::vector<Point> points(amount_of_points);
  for (size_t i = 0; i < amount_of_points; i++) {
    points[i] = Point(input_X_[i], input_Y_[i]);
  }
  Point p0 = points[0];
  for (Point p : points) {
    if (p.first < p0.first || (p.first == p0.first && p.second < p0.second)) {
      p0 = p;
    }
  }

  // sort by polar angle
  std::ranges::sort(points, [&](Point a, Point b) {
    float res = Mul(Minus(a, p0), Minus(b, p0));
    if (res == 0) {
      return (Minus(a, p0).first * Minus(a, p0).first) + (Minus(a, p0).second * Minus(a, p0).second) <
             (Minus(b, p0).first * Minus(b, p0).first) + (Minus(b, p0).second * Minus(b, p0).second);
    }
    return res > 0.0F;
  });
  std::vector<Point> hull;
  for (Point p : points) {
    while (hull.size() >= 2) {
      Point new_vector = Minus(p, hull.back());
      Point last_vector = Minus(hull.back(), hull[hull.size() - 2]);
      if (Mul(new_vector, last_vector) >= 0.0F) {
        hull.pop_back();
      } else {
        break;
      }
    }
    hull.push_back(p);
  }
  output_X_.resize(hull.size());
  output_Y_.resize(hull.size());
  for (size_t i = 0; i < hull.size(); i++) {
    output_X_[i] = hull[i].first;
    output_Y_[i] = hull[i].second;
  }
  return true;
}

bool leontev_n_graham_tbb::GrahamSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[2])[0] = static_cast<int>(output_X_.size());
  for (size_t i = 0; i < output_X_.size(); i++) {
    reinterpret_cast<float *>(task_data->outputs[0])[i] = output_X_[i];
    reinterpret_cast<float *>(task_data->outputs[1])[i] = output_Y_[i];
  }
  return true;
}
