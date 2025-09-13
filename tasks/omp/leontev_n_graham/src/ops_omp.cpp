#include "omp/leontev_n_graham/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool leontev_n_graham_omp::GrahamOmp::PreProcessingImpl() {
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

bool leontev_n_graham_omp::GrahamOmp::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0] &&
         task_data->inputs_count[0] > 0;
}

std::pair<float, float> leontev_n_graham_omp::GrahamOmp::Minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_omp::GrahamOmp::Mul(std::pair<float, float> a, std::pair<float, float> b) {
  return (a.first * b.second) - (b.first * a.second);
}

void leontev_n_graham_omp::GrahamOmp::InitData(std::vector<std::vector<std::pair<float, float>>> &data, int threads,
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

std::pair<float, float> leontev_n_graham_omp::GrahamOmp::GetMinPoint(
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

bool leontev_n_graham_omp::GrahamOmp::RunImpl() {
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
    if (a == p0) {
      return Mul((Point(1.0F, 0.0F)), Minus(b, p0)) > 0.0F;
    }
    if (b == p0) {
      return Mul(Minus(a, p0), Point(1.0F, 0.0F)) > 0.0F;
    }
    return Mul(Minus(a, p0), Minus(b, p0)) > 0.0F;
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
#pragma omp parallel for num_threads(threads)
  for (int thread = 0; thread < threads; thread++) {
    for (Point p : data[thread]) {
      while (outputs[thread].size() >= 2) {
        Point new_vector = Minus(p, outputs[thread].back());
        Point last_vector = Minus(outputs[thread].back(), outputs[thread][outputs[thread].size() - 2]);
        if (Mul(new_vector, last_vector) >= 0.0F) {
          outputs[thread].pop_back();
        } else {
          break;
        }
      }
      outputs[thread].push_back(p);
    }
  }
  for (int thread = 0; thread < threads; thread++) {
    for (size_t i = (thread == 0) ? 0 : 1; i < outputs[thread].size(); i++) {
      Point p = outputs[thread][i];
      while (hull1.size() >= 2) {
        Point new_vector = Minus(p, hull1.back());
        Point last_vector = Minus(hull1.back(), hull1[hull1.size() - 2]);
        if (Mul(new_vector, last_vector) >= 0.0F) {
          hull1.pop_back();
        } else {
          break;
        }
      }
      hull1.push_back(p);
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

bool leontev_n_graham_omp::GrahamOmp::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[2])[0] = static_cast<int>(output_X_.size());
  for (size_t i = 0; i < output_X_.size(); i++) {
    reinterpret_cast<float *>(task_data->outputs[0])[i] = output_X_[i];
    reinterpret_cast<float *>(task_data->outputs[1])[i] = output_Y_[i];
  }
  return true;
}

bool leontev_n_graham_omp::GrahamSeq::PreProcessingImpl() {
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

bool leontev_n_graham_omp::GrahamSeq::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0] &&
         task_data->inputs_count[0] > 0;
}

std::pair<float, float> leontev_n_graham_omp::GrahamSeq::Minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_omp::GrahamSeq::Mul(std::pair<float, float> a, std::pair<float, float> b) {
  return (a.first * b.second) - (b.first * a.second);
}

bool leontev_n_graham_omp::GrahamSeq::RunImpl() {
  size_t amount_of_points = input_X_.size();
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
    if (a == p0) {
      return Mul((Point(1.0F, 0.0F)), Minus(b, p0)) > 0.0F;
    }
    if (b == p0) {
      return Mul(Minus(a, p0), Point(1.0F, 0.0F)) > 0.0F;
    }
    return Mul(Minus(a, p0), Minus(b, p0)) > 0.0F;
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

bool leontev_n_graham_omp::GrahamSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[2])[0] = static_cast<int>(output_X_.size());
  for (size_t i = 0; i < output_X_.size(); i++) {
    reinterpret_cast<float *>(task_data->outputs[0])[i] = output_X_[i];
    reinterpret_cast<float *>(task_data->outputs[1])[i] = output_Y_[i];
  }
  return true;
}
