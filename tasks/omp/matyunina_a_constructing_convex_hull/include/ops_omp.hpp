#pragma once

#include <utility>
#include <vector>
#include <set>

#include "core/task/include/task.hpp"

namespace matyunina_a_constructing_convex_hull_omp {

struct Point {
  int x, y;
  Point(int x = 0, int y = 0) : x(x), y(y) {}
  bool operator<(const Point& other) const;
  bool operator==(const Point& other) const;
  static int orientation(Point& a, Point& b, Point& c);
  static double distanceToLine(Point& a, Point& b, Point& c);
  static double distance(const Point& p1, const Point& p2);
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
  void DeleteDublecate(std::set<Point>& hullSet);
  std::vector<int> input_;
  std::vector<Point> points_{};
  std::vector<Point> output_{};
  int width_{}, height_{};
};

}  // namespace matyunina_a_constructing_convex_hull_omp