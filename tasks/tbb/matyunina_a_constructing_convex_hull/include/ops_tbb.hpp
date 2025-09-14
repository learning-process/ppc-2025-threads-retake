#pragma once

#include <oneapi/tbb/blocked_range.h>
#include <tbb/tbb.h>

#include <cstddef>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace matyunina_a_constructing_convex_hull_tbb {

struct XCoord {
  int value;
  XCoord(int v = 0) : value(v) {}
};
struct YCoord {
  int value;
  YCoord(int v = 0) : value(v) {}
};

struct Point {
  int x, y;
  Point(XCoord x = {0}, YCoord y = {0}) : x(x.value), y(y.value) {}
  bool operator<(const Point& other) const;
  bool operator==(const Point& other) const;
  static int Orientation(Point& a, Point& b, Point& c);
  static double DistanceToLine(Point& a, Point& b, Point& c);
  static double Distance(const Point& p1, const Point& p2);
};

class ConstructingConvexHull : public ppc::core::Task {
 public:
  explicit ConstructingConvexHull(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void FindPoints();
  void DeleteDublecate(std::set<Point>& hull_set);
  std::pair<Point, Point> FindExtremePoints();
  static void InitializeHull(const Point& leftmost, const Point& rightmost, std::set<Point>& hull_set,
                             std::stack<std::pair<Point, Point>>& segment_stack);
  void ProcessAllSegments(std::stack<std::pair<Point, Point>>& segment_stack, std::set<Point>& hull_set);
  std::pair<double, Point> ProcessSegmentRange(Point& a, Point& b, const oneapi::tbb::blocked_range<std::size_t>& r,
                                               std::pair<double, Point> local);
  static std::pair<double, Point> CombineResults(const std::pair<double, Point>& x, const std::pair<double, Point>& y);
  std::vector<int> input_;
  std::vector<Point> points_;
  std::vector<Point> output_;
  int width_{}, height_{};
};

}  // namespace matyunina_a_constructing_convex_hull_tbb