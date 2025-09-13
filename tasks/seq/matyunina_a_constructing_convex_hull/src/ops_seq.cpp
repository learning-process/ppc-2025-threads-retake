#include "seq/matyunina_a_constructing_convex_hull/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint> 
#include <set>
#include <stack>
#include <utility>
#include <vector>

bool matyunina_a_constructing_convex_hull_seq::Point::operator<(const Point& other) const {
  return (x < other.x) || (x == other.x && y < other.y);
}
bool matyunina_a_constructing_convex_hull_seq::Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

int matyunina_a_constructing_convex_hull_seq::Point::Orientation(Point& a, Point& b, Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double matyunina_a_constructing_convex_hull_seq::Point::DistanceToLine(Point& a, Point& b, Point& c) {
  return std::abs(Orientation(a, b, c));
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::PreProcessingImpl() {
  width_ = static_cast<int>(task_data->inputs_count[0]);
  height_ = static_cast<int>(task_data->inputs_count[1]);

  int size = width_ * height_;

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  return true;
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

void matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::FindPoints() {
  points_.clear();
  int size = width_ * height_;
  int estimated_points = 0;
  for (int i = 0; i < std::min(1000, size); i++) {
    if (input_[i] == 1) { estimated_points++; };
  }
  double density = static_cast<double>(estimated_points) / std::min(1000, size);
  points_.reserve(static_cast<int>(size * density * 1.2));

  for (int i = 0; i < width_; i++) {
    for (int j = 0; j < height_; j++) {
      if (input_[(j * width_) + i] == 1) {
        points_.emplace_back(i, j);
      }
    }
  }
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::RunImpl() {
  FindPoints();

  if (points_.size() < 3) {
    output_ = points_;
    return true;
  }

  Point leftmost = points_[0];
  Point rightmost = points_[0];

  for (Point& p : points_) {
    if (p.x < leftmost.x) { leftmost = p; };
    if (p.x > rightmost.x) { rightmost = p; };
  }

  std::stack<std::pair<Point, Point>> segment_stack;
  std::set<Point> hull_set;

  hull_set.insert(leftmost);
  hull_set.insert(rightmost);
  segment_stack.emplace(leftmost, rightmost);
  segment_stack.emplace(rightmost, leftmost);

  while (!segment_stack.empty()) {
    Point a = segment_stack.top().first;
    Point b = segment_stack.top().second;
    segment_stack.pop();

    double max_distance = -1;
    Point farthest_point;
    bool found = false;

    for (Point& p : points_) {
      if (Point::Orientation(a, b, p) > 0) {
        double dist = Point::DistanceToLine(a, b, p);
        if (dist > max_distance) {
          max_distance = dist;
          farthest_point = p;
          found = true;
        }
      }
    }

    if (found) {
      hull_set.insert(farthest_point);

      segment_stack.emplace(a, farthest_point);
      segment_stack.emplace(farthest_point, b);
    }
  }

  output_.assign(hull_set.begin(), hull_set.end());

  return true;
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::PostProcessingImpl() {
  std::ranges::sort(output_, [](const Point& a, const Point& b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
  });

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
