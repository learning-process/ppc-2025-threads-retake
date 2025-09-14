#include "omp/matyunina_a_constructing_convex_hull/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stack>
#include <utility>
#include <vector>

bool matyunina_a_constructing_convex_hull_omp::Point::operator<(const Point& other) const {
  return (x < other.x) || (x == other.x && y < other.y);
}
bool matyunina_a_constructing_convex_hull_omp::Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

int matyunina_a_constructing_convex_hull_omp::Point::Orientation(Point& a, Point& b, Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double matyunina_a_constructing_convex_hull_omp::Point::DistanceToLine(Point& a, Point& b, Point& c) {
  return std::abs(Orientation(a, b, c));
}

double matyunina_a_constructing_convex_hull_omp::Point::Distance(const Point& p1, const Point& p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt((dx * dx) + (dy * dy));
}

bool matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::PreProcessingImpl() {
  width_ = static_cast<int>(task_data->inputs_count[0]);
  height_ = static_cast<int>(task_data->inputs_count[1]);

  int size = width_ * height_;

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  return true;
}

bool matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

void matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::FindPoints() {
  points_.clear();

  int size = width_ * height_;
  int estimated_points = 0;

  for (int i = 0; i < std::min(1000, size); i++) {
    if (input_[i] == 1) {
      estimated_points++;
    }
  }
  double density = static_cast<double>(estimated_points) / std::min(1000, size);
  points_.reserve(static_cast<int>(size * density * 1.2));

#pragma omp parallel
  {
    std::vector<Point> local_points;
#pragma omp for nowait
    for (int i = 0; i < width_; i++) {
      for (int j = 0; j < height_; j++) {
        if (input_[(j * width_) + i] == 1) {
          local_points.emplace_back(i, j);
        }
      }
    }

#pragma omp critical
    points_.insert(points_.end(), local_points.begin(), local_points.end());
  }
}

bool matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::RunImpl() {
  FindPoints();

  if (points_.size() < 3) {
    output_ = points_;
    return true;
  }

  Point leftmost = points_[0];
  Point rightmost = points_[0];

  for (Point& p : points_) {
    if (p.x < leftmost.x) {
      leftmost = p;
    }
    if (p.x > rightmost.x) {
      rightmost = p;
    }
  }

  std::stack<std::pair<Point, Point>> segment_stack;
  std::set<Point> hull_set;

  hull_set.insert(leftmost);
  hull_set.insert(rightmost);
  segment_stack.emplace(leftmost, rightmost);
  segment_stack.emplace(rightmost, leftmost);

  while (!segment_stack.empty()) {
    std::vector<std::pair<Point, Point>> segmentsToProcess;

    size_t batch_size = std::min(segment_stack.size(), (size_t)omp_get_max_threads() * 4);
    for (size_t i = 0; i < batch_size && !segment_stack.empty(); ++i) {
      segmentsToProcess.push_back(segment_stack.top());
      segment_stack.pop();
    }

    std::vector<std::pair<Point, Point>> new_segments;

#pragma omp parallel
    {
      std::vector<std::pair<Point, Point>> local_new_segments;

#pragma omp for nowait
      for (int i = 0; i < (int)segmentsToProcess.size(); ++i) {
        Point a = segmentsToProcess[i].first;
        Point b = segmentsToProcess[i].second;
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
#pragma omp critical
          { hull_set.insert(farthest_point); }

          local_new_segments.emplace_back(a, farthest_point);
          local_new_segments.emplace_back(farthest_point, b);
        }
      }

#pragma omp critical
      { new_segments.insert(new_segments.end(), local_new_segments.begin(), local_new_segments.end()); }
    }

    for (const auto& seg : new_segments) {
      segment_stack.push(seg);
    }
  }

  DeleteDublecate(hull_set);

  return true;
}

void matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::DeleteDublecate(std::set<Point>& hull_set) {
  std::vector<Point> tempHull(hull_set.begin(), hull_set.end());
  std::vector<Point> final_hull;

  Point center;
  for (const auto& p : tempHull) {
    center.x += p.x;
    center.y += p.y;
  }
  center.x /= static_cast<int>(tempHull.size());
  center.y /= static_cast<int>(tempHull.size());

  std::ranges::sort(tempHull, [&center](const Point& a, const Point& b) {
    return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
  });

  for (int i = 0; i < (int)tempHull.size(); i++) {
    Point prev = tempHull[(i - 1 + tempHull.size()) % tempHull.size()];
    Point curr = tempHull[i];
    Point next = tempHull[(i + 1) % tempHull.size()];

    if (Point::Orientation(prev, curr, next) == 0) {
      double dist1 = Point::Distance(prev, curr);
      double dist2 = Point::Distance(curr, next);
      double dist3 = Point::Distance(prev, next);

      if (std::abs(dist1 + dist2 - dist3) < 1e-9) {
        continue;
      }
    }
    final_hull.push_back(curr);
  }

  output_ = final_hull;
}

bool matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::PostProcessingImpl() {
  std::sort(output_.begin(), output_.end());

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
