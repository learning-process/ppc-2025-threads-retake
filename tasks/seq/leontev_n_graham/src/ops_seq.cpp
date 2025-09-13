#include "seq/leontev_n_graham/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <ranges>
#include <vector>
#include <utility>

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
  return task_data->inputs.size() == 2 && task_data->outputs_count[0] <= task_data->inputs_count[0];
}

std::pair<float, float> leontev_n_graham_seq::GrahamSeq::Minus(std::pair<float, float> a, std::pair<float, float> b) {
  return {a.first - b.first, a.second - b.second};
}

float leontev_n_graham_seq::GrahamSeq::Mul(std::pair<float, float> a, std::pair<float, float> b) {
  return (a.first * b.second) - (b.first * a.second);
}

bool leontev_n_graham_seq::GrahamSeq::RunImpl() {
  size_t amount_of_points = input_X_.size();
  using point = std::pair<float, float>;
  std::vector<point> points(amount_of_points);
  for (int i = 0; i < amount_of_points; i++) {
    points[i] = point(input_X_[i], input_Y_[i]);
  }
  point p0 = points[0];
  for (point p : points) {
    if (p.first < p0.first || (p.first == p0.first && p.second < p0.second)) {
      p0 = p;
    }
  }

  // sort by polar angle
  std::ranges::sort(points, [&](point a, point b) {
    if (a == p0) {
      return Mul((point(1.0F, 0.0F)), Minus(b, p0)) > 0.0F;
    }
    if (b == p0) {
      return Mul(Minus(a, p0), point(1.0F, 0.0F)) > 0.0F;
    }
    return Mul(Minus(a, p0), Minus(b, p0)) > 0.0F;
  });

  std::vector<point> hull;
  for (point p : points) {
    while (hull.size() >= 2) {
      point new_vector = Minus(p, hull.back());
      point last_vector = Minus(hull.back(), hull[hull.size() - 2]);
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

bool leontev_n_graham_seq::GrahamSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[2])[0] = static_cast<int>(output_X_.size());
  for (size_t i = 0; i < output_X_.size(); i++) {
    reinterpret_cast<float *>(task_data->outputs[0])[i] = output_X_[i];
    reinterpret_cast<float *>(task_data->outputs[1])[i] = output_Y_[i];
  }
  return true;
}
