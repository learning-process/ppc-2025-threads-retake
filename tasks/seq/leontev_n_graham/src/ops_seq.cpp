#include "seq/leontev_n_graham/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool leontev_n_graham_seq::GrahamSeq::PreProcessingImpl() {
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

bool leontev_n_graham_seq::GrahamSeq::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs.size() == 2 && task_data->outputs_count <= task_data->inputs_count;
}

std::pair<float, float> leontev_n_graham_seq::GrahamSeq::minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_seq::GrahamSeq::mul(std::pair<float, float> a, std::pair<float, float> b) {
  return a.first * b.second - b.first * a.second;
}

bool leontev_n_graham_seq::GrahamSeq::RunImpl() {
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
  std::sort(points.begin(), points.end(), [&](point a, point b) { return mul(minus(a, p0), minus(b, p0)) > 0.0f; });

  std::vector<point> hull;
  for (point p : points) {
    while (hull.size() >= 2) {
      point new_vector = minus(p, hull.back());
      point last_vector = minus(hull.back(), hull[hull.size() - 2]);
      if (mul(new_vector, last_vector) > 0.0f)
        hull.pop_back();
      else
        break;
    }
    hull.push_back(p);
  }
  for (int i = 0; i < hull.size(); i++) {
    output_X_[i] = hull[i].first;
    output_Y_[i] = hull[i].second;
  }
  return true;
}

bool leontev_n_graham_seq::GrahamSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_X_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_X_[i];
    reinterpret_cast<int *>(task_data->outputs[1])[i] = output_Y_[i];
  }
  return true;
}
