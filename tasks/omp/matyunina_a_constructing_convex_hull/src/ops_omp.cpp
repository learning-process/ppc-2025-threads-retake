#include "omp/matyunina_a_constructing_convex_hull/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <stack>
#include <vector>

bool matyunina_a_constructing_convex_hull_omp::Point::operator<(const Point& other) const {
  return (x < other.x) || (x == other.x && y < other.y);
}
bool matyunina_a_constructing_convex_hull_omp::Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

int matyunina_a_constructing_convex_hull_omp::Point::orientation(Point& a, Point& b, Point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double matyunina_a_constructing_convex_hull_omp::Point::distanceToLine(Point& a, Point& b, Point& c) {
  return std::abs(orientation(a, b, c));
}

double matyunina_a_constructing_convex_hull_omp::Point::distance(const Point& p1, const Point& p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt(dx * dx + dy * dy);
}

bool matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

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
    if (input_[i] == 1) estimated_points++;
  }
  double density = static_cast<double>(estimated_points) / std::min(1000, size);
  points_.reserve(static_cast<int>(size * density * 1.2));

#pragma omp parallel
  {
    std::vector<Point> local_points;
#pragma omp for nowait
    for (int i = 0; i < width_; i++) {
      for (int j = 0; j < height_; j++) {
        if (input_[j * width_ + i] == 1) {
          local_points.emplace_back(Point(i, j));
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
    if (p.x < leftmost.x) leftmost = p;
    if (p.x > rightmost.x) rightmost = p;
  }

  std::stack<std::pair<Point, Point>> segmentStack;
  std::set<Point> hullSet;

  hullSet.insert(leftmost);
  hullSet.insert(rightmost);
  segmentStack.push({leftmost, rightmost});
  segmentStack.push({rightmost, leftmost});

  while (!segmentStack.empty()) {
    std::vector<std::pair<Point, Point>> segmentsToProcess;

    size_t batch_size = std::min(segmentStack.size(), (size_t)omp_get_max_threads() * 4);
    for (size_t i = 0; i < batch_size && !segmentStack.empty(); ++i) {
      segmentsToProcess.push_back(segmentStack.top());
      segmentStack.pop();
    }

    std::vector<std::pair<Point, Point>> newSegments;

#pragma omp parallel
    {
      std::vector<std::pair<Point, Point>> local_newSegments;

#pragma omp for nowait
      for (int i = 0; i < (int)segmentsToProcess.size(); ++i) {
        Point a = segmentsToProcess[i].first;
        Point b = segmentsToProcess[i].second;
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
#pragma omp critical
          { hullSet.insert(farthestPoint); }

          local_newSegments.push_back({a, farthestPoint});
          local_newSegments.push_back({farthestPoint, b});
        }
      }

#pragma omp critical
      { newSegments.insert(newSegments.end(), local_newSegments.begin(), local_newSegments.end()); }
    }

    for (const auto& seg : newSegments) {
      segmentStack.push(seg);
    }
  }

  DeleteDublecate(hullSet);

  return true;
}

void matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::DeleteDublecate(std::set<Point>& hullSet) {
  std::vector<Point> tempHull(hullSet.begin(), hullSet.end());
  std::vector<Point> finalHull;

  Point center;
  for (const auto& p : tempHull) {
    center.x += p.x;
    center.y += p.y;
  }
  center.x /= tempHull.size();
  center.y /= tempHull.size();

  std::sort(tempHull.begin(), tempHull.end(), [&center](const Point& a, const Point& b) {
    return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
  });

  for (int i = 0; i < (int)tempHull.size(); i++) {
    Point prev = tempHull[(i - 1 + tempHull.size()) % tempHull.size()];
    Point curr = tempHull[i];
    Point next = tempHull[(i + 1) % tempHull.size()];

    if (Point::orientation(prev, curr, next) == 0) {
      double dist1 = Point::distance(prev, curr);
      double dist2 = Point::distance(curr, next);
      double dist3 = Point::distance(prev, next);

      if (std::abs(dist1 + dist2 - dist3) < 1e-9) {
        continue;
      }
    }
    finalHull.push_back(curr);
  }

  output_ = finalHull;
}

bool matyunina_a_constructing_convex_hull_omp::ConstructingConvexHull::PostProcessingImpl() {
  std::sort(output_.begin(), output_.end());

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
