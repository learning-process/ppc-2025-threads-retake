#include "omp/matyunina_a_constructing_convex_hull/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stack>
#include <utility>
#include <vector>

using namespace matyunina_a_constructing_convex_hull_omp;

bool Point::operator<(const Point& other) const { return (x < other.x) || (x == other.x && y < other.y); }

bool Point::operator==(const Point& other) const { return x == other.x && y == other.y; }

int Point::Orientation(Point& a, Point& b, Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double Point::DistanceToLine(Point& a, Point& b, Point& c) { return std::abs(Orientation(a, b, c)); }

double Point::Distance(const Point& p1, const Point& p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt((dx * dx) + (dy * dy));
}

bool ConstructingConvexHull::PreProcessingImpl() {
  width_ = static_cast<int>(task_data->inputs_count[0]);
  height_ = static_cast<int>(task_data->inputs_count[1]);

  int size = width_ * height_;

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  return true;
}

bool ConstructingConvexHull::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

void ConstructingConvexHull::FindPoints() {
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

bool ConstructingConvexHull::RunImpl() {
  FindPoints();

  if (points_.size() < 3) {
    output_ = points_;
    return true;
  }

  auto [leftmost, rightmost] = FindExtremePoints();
  std::stack<std::pair<Point, Point>> segment_stack;
  std::set<Point> hull_set;

  InitializeHull(leftmost, rightmost, hull_set, segment_stack);
  ProcessAllSegments(segment_stack, hull_set);
  DeleteDublecate(hull_set);

  return true;
}

std::pair<Point, Point> ConstructingConvexHull::FindExtremePoints() {
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

  return {leftmost, rightmost};
}

void ConstructingConvexHull::InitializeHull(const Point& leftmost, const Point& rightmost, std::set<Point>& hull_set,
                                            std::stack<std::pair<Point, Point>>& segment_stack) {
  hull_set.insert(leftmost);
  hull_set.insert(rightmost);
  segment_stack.emplace(leftmost, rightmost);
  segment_stack.emplace(rightmost, leftmost);
}

void ConstructingConvexHull::ProcessAllSegments(std::stack<std::pair<Point, Point>>& segment_stack,
                                                std::set<Point>& hull_set) {
  while (!segment_stack.empty()) {
    std::vector<std::pair<Point, Point>> segments_to_process;
    size_t batch_size = std::min(segment_stack.size(), (size_t)omp_get_max_threads() * 4);

    for (size_t i = 0; i < batch_size && !segment_stack.empty(); ++i) {
      segments_to_process.push_back(segment_stack.top());
      segment_stack.pop();
    }

    std::vector<std::pair<Point, Point>> new_segments;

#pragma omp parallel
    {
      std::vector<std::pair<Point, Point>> local_new_segments;

#pragma omp for nowait
      for (int i = 0; i < (int)segments_to_process.size(); ++i) {
        Point a = segments_to_process[i].first;
        Point b = segments_to_process[i].second;
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
          hull_set.insert(farthest_point);

          local_new_segments.emplace_back(a, farthest_point);
          local_new_segments.emplace_back(farthest_point, b);
        }
      }

#pragma omp critical
      new_segments.insert(new_segments.end(), local_new_segments.begin(), local_new_segments.end());
    }

    for (const auto& seg : new_segments) {
      segment_stack.push(seg);
    }
  }
}

void ConstructingConvexHull::DeleteDublecate(std::set<Point>& hull_set) {
  std::vector<Point> temp_hull(hull_set.begin(), hull_set.end());
  std::vector<Point> final_hull;

  Point center;
  for (const auto& p : temp_hull) {
    center.x += p.x;
    center.y += p.y;
  }
  center.x /= static_cast<int>(temp_hull.size());
  center.y /= static_cast<int>(temp_hull.size());

  std::ranges::sort(temp_hull, [&center](const Point& a, const Point& b) {
    return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
  });

  for (int i = 0; i < (int)temp_hull.size(); i++) {
    Point prev = temp_hull[(i - 1 + temp_hull.size()) % temp_hull.size()];
    Point curr = temp_hull[i];
    Point next = temp_hull[(i + 1) % temp_hull.size()];

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

bool ConstructingConvexHull::PostProcessingImpl() {
  std::ranges::sort(output_, [](const Point& a, const Point& b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
