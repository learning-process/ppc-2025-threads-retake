#include "tbb/matyunina_a_constructing_convex_hull/include/ops_tbb.hpp"

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool matyunina_a_constructing_convex_hull_tbb::Point::operator<(const Point& other) const {
  return (x < other.x) || (x == other.x && y < other.y);
}
bool matyunina_a_constructing_convex_hull_tbb::Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

int matyunina_a_constructing_convex_hull_tbb::Point::Orientation(Point& a, Point& b, Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double matyunina_a_constructing_convex_hull_tbb::Point::DistanceToLine(Point& a, Point& b, Point& c) {
  return std::abs(Orientation(a, b, c));
}

double matyunina_a_constructing_convex_hull_tbb::Point::Distance(const Point& p1, const Point& p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt((dx * dx) + (dy * dy));
}

bool matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::PreProcessingImpl() {
  width_ = static_cast<int>(task_data->inputs_count[0]);
  height_ = static_cast<int>(task_data->inputs_count[1]);

  int size = width_ * height_;

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size);

  return true;
}

bool matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

void matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::FindPoints() {
  points_.clear();

  const int size = width_ * height_;
  const int sample_n = std::min(1000, size);

  int estimated_points = 0;
  for (int i = 0; i < sample_n; ++i) {
    if (input_[i] == 1) {
      ++estimated_points;
    }
  }
  const double density = (sample_n != 0) ? static_cast<double>(estimated_points) / sample_n : 0.0;
  points_.reserve(static_cast<int>(size * density * 1.2));

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);
  oneapi::tbb::enumerable_thread_specific<std::vector<Point>> tls_points;

  arena.execute([&] {
    oneapi::tbb::parallel_for(0, height_, [&](int y) {
      auto& local = tls_points.local();
      if (local.capacity() == 0) {
        local.reserve(256);
      }

      const int row_off = y * width_;
      for (int x = 0; x < width_; ++x) {
        if (input_[row_off + x] == 1) {
          local.emplace_back(x, y);
        }
      }
    });
  });

  size_t total = 0;
  for (auto& v : tls_points) {
    total += v.size();
  }
  if (points_.capacity() < total) points_.reserve(static_cast<int>(total));

  for (auto& v : tls_points) {
    points_.insert(points_.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
  }
}

bool matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::RunImpl() {
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

  int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  while (!segment_stack.empty()) {
    Point a = segment_stack.top().first;
    Point b = segment_stack.top().second;
    segment_stack.pop();

    std::pair<double, Point> best{-1.0, Point{}};

    arena.execute([&] {
      best = oneapi::tbb::parallel_reduce(
          oneapi::tbb::blocked_range<std::size_t>(0, points_.size()), std::pair<double, Point>{-1.0, Point{}},
          [&](const oneapi::tbb::blocked_range<std::size_t>& r, std::pair<double, Point> local) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
              Point& p = points_[i];
              if (Point::Orientation(a, b, p) > 0) {
                double d = Point::DistanceToLine(a, b, p);
                if (d > local.first) {
                  local.first = d;
                  local.second = p;
                }
              }
            }
            return local;
          },
          [&](const std::pair<double, Point>& x, const std::pair<double, Point>& y) {
            if (x.first > y.first) {
              return x;
            }
            if (y.first > x.first) {
              return y;
            }
            return (x.second.x < y.second.x || (x.second.x == y.second.x && x.second.y < y.second.y)) ? x : y;
          });
    });

    if (best.first >= 0.0) {
      hull_set.insert(best.second);
      segment_stack.emplace(a, best.second);
      segment_stack.emplace(best.second, b);
    }
  }

  DeleteDublecate(hull_set);

  return true;
}

void matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::DeleteDublecate(std::set<Point>& hull_set) {
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

bool matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::PostProcessingImpl() {
  std::ranges::sort(output_, [](const Point& a, const Point& b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
