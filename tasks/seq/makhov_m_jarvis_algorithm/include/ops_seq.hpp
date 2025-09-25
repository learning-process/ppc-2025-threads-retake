#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_jarvis_algorithm_seq {

struct XCoord {
  double value;
  explicit XCoord(double v) : value(v) {}
};

struct YCoord {
  double value;
  explicit YCoord(double v) : value(v) {}
};

struct Point {
 public:
  double x, y;

  Point() : x(0), y(0) {}
  Point(XCoord x, YCoord y) : x(x.value), y(y.value) {}

  [[nodiscard]] double GetX() const { return x; }
  [[nodiscard]] double GetY() const { return y; }
  void SetX(double x_coord) { x = x_coord; }
  void SetY(double y_coord) { y = y_coord; }
  void Set(XCoord x_coord, YCoord y_coord) {
    x = x_coord.value;
    y = y_coord.value;
  }

  [[nodiscard]] double DistanceTo(const Point& other) const {
    double dx = x - other.x;
    double dy = y - other.y;
    return std::sqrt((dx * dx) + (dy * dy));
  }

  friend std::ostream& operator<<(std::ostream& os, const Point& point) {
    std::ios old_state(nullptr);
    old_state.copyfmt(os);
    os << "(" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";
    os.copyfmt(old_state);
    return os;
  }

  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
  bool operator!=(const Point& other) const { return !(*this == other); }
};

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Публичные статические методы для использования в тестах
  static double Cross(const Point& a, const Point& b, const Point& c);
  static double Dist(const Point& a, const Point& b);
  static uint8_t* ConvertPointsToByteArray(const std::vector<Point>& points, uint32_t& out_size);
  static std::vector<Point> ConvertByteArrayToPoints(const uint8_t* byte_array, uint32_t byte_array_size);
  static Point GetRandomPoint(XCoord min_x, XCoord max_x, YCoord min_y, YCoord max_y);

 private:
  [[nodiscard]] static size_t FindLeftmostPoint(const std::vector<Point>& points);
  [[nodiscard]] static size_t FindNextPoint(size_t current, const std::vector<Point>& points);

  std::vector<Point> input_;
  std::vector<Point> result_;
};

}  // namespace makhov_m_jarvis_algorithm_seq