#include "tbb/matyunina_a_constructing_convex_hull/include/ops_tbb.hpp"

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <stack>
#include <vector>

#include "core/util/include/util.hpp"

bool matyunina_a_constructing_convex_hull_tbb::Point::operator<(const Point& other) const {
  return (x < other.x) || (x == other.x && y < other.y);
}
bool matyunina_a_constructing_convex_hull_tbb::Point::operator==(const Point& other) const {
  return x == other.x && y == other.y;
}

int matyunina_a_constructing_convex_hull_tbb::Point::orientation(Point& a, Point& b, Point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double matyunina_a_constructing_convex_hull_tbb::Point::distanceToLine(Point& a, Point& b, Point& c) {
  return std::abs(orientation(a, b, c));
}

bool matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

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
    if (input_[i] == 1) ++estimated_points;
  }
  const double density = sample_n ? static_cast<double>(estimated_points) / sample_n : 0.0;
  points_.reserve(static_cast<int>(size * density * 1.2));

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);
  oneapi::tbb::enumerable_thread_specific<std::vector<Point>> tls_points;

  arena.execute([&] {
    oneapi::tbb::parallel_for(0, height_, [&](int y) {
      auto& local = tls_points.local();
      if (local.capacity() == 0) local.reserve(256);

      const int row_off = y * width_;
      for (int x = 0; x < width_; ++x) {
        if (input_[row_off + x] == 1) {
          local.emplace_back(Point(x, y));
        }
      }
    });
  });

  size_t total = 0;
  for (auto& v : tls_points) total += v.size();
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
    if (p.x < leftmost.x) leftmost = p;
    if (p.x > rightmost.x) rightmost = p;
  }

  std::stack<std::pair<Point, Point>> segmentStack;
  std::set<Point> hullSet;

  hullSet.insert(leftmost);
  hullSet.insert(rightmost);
  segmentStack.push({leftmost, rightmost});
  segmentStack.push({rightmost, leftmost});

  int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  while (!segmentStack.empty()) {
    Point a = segmentStack.top().first;
    Point b = segmentStack.top().second;
    segmentStack.pop();

    std::pair<double, Point> best{-1.0, Point{}};

    arena.execute([&] {
      best = oneapi::tbb::parallel_reduce(
          oneapi::tbb::blocked_range<std::size_t>(0, points_.size()), std::pair<double, Point>{-1.0, Point{}},
          [&](const oneapi::tbb::blocked_range<std::size_t>& r, std::pair<double, Point> local) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
              Point& p = points_[i];
              if (Point::orientation(a, b, p) > 0) {
                double d = Point::distanceToLine(a, b, p);
                if (d > local.first) {
                  local.first = d;
                  local.second = p;
              }
            }
          }
            return local;
          },
          [&](const std::pair<double, Point>& x, const std::pair<double, Point>& y) {
            if (x.first > y.first) return x;
            if (y.first > x.first) return y;
            return (x.second.x < y.second.x || (x.second.x == y.second.x && x.second.y < y.second.y)) ? x : y;
          });
    });

    if (best.first >= 0.0) {
      hullSet.insert(best.second);
      segmentStack.push({a, best.second});
      segmentStack.push({best.second, b});
    }
  }

  output_.assign(hullSet.begin(), hullSet.end());

  return true;
}

bool matyunina_a_constructing_convex_hull_tbb::ConstructingConvexHull::PostProcessingImpl() {
  std::sort(output_.begin(), output_.end());

  task_data->outputs_count.push_back(output_.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_.data()));

  return true;
}
