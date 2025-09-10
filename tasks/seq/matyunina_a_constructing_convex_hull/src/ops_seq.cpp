#include "seq/matyunina_a_constructing_convex_hull/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <stack>
#include <set>
#include <algorithm>

bool matyunina_a_constructing_convex_hull_seq::Point::operator<(const Point& other) const {
  return (x < other.x) || (x == other.x && y < other.y);
}
bool matyunina_a_constructing_convex_hull_seq::Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

int matyunina_a_constructing_convex_hull_seq::Point::orientation(Point& a, Point& b, Point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double matyunina_a_constructing_convex_hull_seq::Point::distanceToLine(Point& a, Point& b,Point& c) {
  return std::abs(orientation(a, b, c));
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

  int size = width_ * height_;

  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  for (int i = 0; i < width_; i++) {
    for (int j = 0; j < height_; j++) {
      if (input_[j * width_ + i] == 1) {
        points_.push_back({i, j});
      }
    }
  }

  return true;
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::RunImpl() {
  if (points_.size() < 3) {
    output_ = points_;
    return true;
  }


  Point leftmost = points_[0];
  Point rightmost = points_[0];

  for (Point& p : points_) {
    if (p.x < leftmost.x) leftmost = p;
    if (p.x > rightmost.x) rightmost = p;
  }

  std::stack<std::pair<Point, Point>> segmentStack;
  std::set<Point> hullSet;


  hullSet.insert(leftmost);
  hullSet.insert(rightmost);
  segmentStack.push({ leftmost, rightmost });
  segmentStack.push({ rightmost, leftmost });

  while (!segmentStack.empty()) {
      Point a = segmentStack.top().first;
      Point b = segmentStack.top().second;
      segmentStack.pop();

      double maxDistance = -1;
      Point farthestPoint;
      bool found = false;

      for (Point& p : points_) {
        if (Point::orientation(a, b, p) > 0) {
          double dist = Point::distanceToLine(a, b, p);
          if (dist > maxDistance) {
            maxDistance = dist;
            farthestPoint = p;
            found = true;
          }
        }
      }

      if (found) {
        hullSet.insert(farthestPoint);

        segmentStack.push({ a, farthestPoint });
        segmentStack.push({ farthestPoint, b });
      }
  }

  output_.assign(hullSet.begin(), hullSet.end());

  return true;
}

bool matyunina_a_constructing_convex_hull_seq::ConstructingConvexHull::PostProcessingImpl() {

  std::sort(output_.begin(), output_.end());

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
