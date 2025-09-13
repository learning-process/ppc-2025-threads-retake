#include "omp/leontev_n_graham/include/ops_omp.hpp"

#include <omp.h>
#include <cmath>
#include <cstddef>
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
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0];
}

std::pair<float, float> leontev_n_graham_omp::GrahamOmp::minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_omp::GrahamOmp::mul(std::pair<float, float> a, std::pair<float, float> b) {
  return a.first * b.second - b.first * a.second;
}

bool leontev_n_graham_omp::GrahamOmp::RunImpl() {
  int amount_of_points = input_X_.size();
  typedef std::pair<float, float> point;
  std::vector<point> points(amount_of_points);
  for (int i = 0; i < amount_of_points; i++) {
    points[i] = point(input_X_[i], input_Y_[i]);
  }
  point p0 = points[0];
  for (point p : points)
    if (p.first < p0.first || (p.first == p0.first && p.second < p0.second)) p0 = p;

  // sort by polar angle
  std::sort(points.begin(), points.end(), [&](point a, point b) {
    if (a == p0) {
      return mul((point(1.0f, 0.0f)), minus(b, p0)) > 0.0f;
    }
    if (b == p0) {
      return mul(minus(a, p0), point(1.0f, 0.0f)) > 0.0f;
    }
    return mul(minus(a, p0), minus(b, p0)) > 0.0f;
  });
  int threads = ppc::util::GetPPCNumThreads();
  if (threads < 1) {
    return false;
  }
  size_t temp_size = (points.size() - 1) / threads; 
  std::vector<std::vector<point>> data(threads, std::vector<point>(temp_size + 1));
  std::vector<std::vector<point>> outputs(threads);
  std::vector<point> hull;
  std::vector<point> hull1;
  hull.push_back(points[0]);
  data[threads - 1].resize(temp_size + 1 + (points.size() - 1) % threads);
  data[0][0] = points[0];
  std::copy(points.begin() + 1, points.begin() + 1 + temp_size, data[0].begin() + 1);
  for (int i = 1; i < threads - 1; i++) {
    data[i][0] = points[0];
    std::copy(points.begin() + 1 + i * temp_size, points.begin() + 1 + (i + 1) * temp_size, data[i].begin() + 1);
  }
  data[threads - 1][0] = points[0];
  std::copy(points.begin() + 1 + (threads - 1) * temp_size, points.end(), data[threads - 1].begin() + 1);
#pragma omp parallel for num_threads(threads)
  for (int thread = 0; thread < threads; thread++) {
    for (point p : data[thread]) {
      while (outputs[thread].size() >= 2) {
        point new_vector = minus(p, outputs[thread].back());
        point last_vector = minus(outputs[thread].back(), outputs[thread][outputs[thread].size() - 2]);
        if (mul(new_vector, last_vector) >= 0.0f)
          outputs[thread].pop_back();
        else
          break;
      }
      outputs[thread].push_back(p);
    }
  }
  for (int thread = 0; thread < threads; thread++) {
    for (int i = 1; i < outputs[thread].size(); i++) {
      hull.push_back(outputs[thread][i]);
    }
  }
  for (point p : hull) {
    while (hull1.size() >= 2) {
      point new_vector = minus(p, hull1.back());
      point last_vector = minus(hull1.back(), hull1[hull1.size() - 2]);
      if (mul(new_vector, last_vector) >= 0.0f)
        hull1.pop_back();
      else
        break;
    }
    hull1.push_back(p);
  }
  output_X_.resize(hull1.size());
  output_Y_.resize(hull1.size());
  for (int i = 0; i < hull1.size(); i++) {
    output_X_[i] = hull1[i].first;
    output_Y_[i] = hull1[i].second;
  }
  return true;
}

bool leontev_n_graham_omp::GrahamOmp::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[2])[0] = static_cast<int>(output_X_.size());
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
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0];
}

std::pair<float, float> leontev_n_graham_omp::GrahamSeq::minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_omp::GrahamSeq::mul(std::pair<float, float> a, std::pair<float, float> b) {
  return a.first * b.second - b.first * a.second;
}

bool leontev_n_graham_omp::GrahamSeq::RunImpl() {
  int amount_of_points = input_X_.size();
  typedef std::pair<float, float> point;
  std::vector<point> points(amount_of_points);
  for (int i = 0; i < amount_of_points; i++) {
    points[i] = point(input_X_[i], input_Y_[i]);
  }
  point p0 = points[0];
  for (point p : points)
    if (p.first < p0.first || (p.first == p0.first && p.second < p0.second)) p0 = p;

  // sort by polar angle
  std::sort(points.begin(), points.end(), [&](point a, point b) {
    if (a == p0) {
      return mul((point(1.0f, 0.0f)), minus(b, p0)) > 0.0f;
    }
    if (b == p0) {
      return mul(minus(a, p0), point(1.0f, 0.0f)) > 0.0f;
    }
    return mul(minus(a, p0), minus(b, p0)) > 0.0f;
  });
  std::vector<point> hull;
  for (point p : points) {
    while (hull.size() >= 2) {
      point new_vector = minus(p, hull.back());
      point last_vector = minus(hull.back(), hull[hull.size() - 2]);
      if (mul(new_vector, last_vector) >= 0.0f)
        hull.pop_back();
      else
        break;
    }
    hull.push_back(p);
  }
  output_X_.resize(hull.size());
  output_Y_.resize(hull.size());
  for (int i = 0; i < hull.size(); i++) {
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