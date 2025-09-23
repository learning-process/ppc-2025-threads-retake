#pragma once

#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_jarvis_algorithm_seq {

struct Point {
 public:
  double x_, y_;
  // Конструкторы
  Point() : x_(0), y_(0) {}
  Point(double x, double y) : x_(x), y_(y) {}

  // Методы доступа
  double getX() const { return x_; }
  double getY() const { return y_; }
  void setX(double x) { x_ = x; }
  void setY(double y) { y_ = y; }
  void set(double x, double y) {
    x_ = x;
    y_ = y;
  }

  double distanceTo(const Point& other) const {
    double dx = x_ - other.x_;
    double dy = y_ - other.y_;
    return std::sqrt(dx * dx + dy * dy);
  }

  // Перегрузка оператора вывода с форматированием
  friend std::ostream& operator<<(std::ostream& os, const Point& point) {
    // Сохраняем оригинальные настройки потока
    std::ios oldState(nullptr);
    oldState.copyfmt(os);

    // Устанавливаем фиксированный формат и точность (2 знака после запятой)
    os << "(" << std::fixed << std::setprecision(2) << point.x_ << ", " << point.y_ << ")";

    // Восстанавливаем оригинальные настройки потока
    os.copyfmt(oldState);

    return os;
  }

  // Перегрузка операторов сравнения
  bool operator==(const Point& other) const { return x_ == other.x_ && y_ == other.y_; }
  bool operator!=(const Point& other) const { return !(*this == other); }
};

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  double cross(const Point& a, const Point& b, const Point& c);
  double dist(const Point& a, const Point& b);
  uint8_t* convertPointsToByteArray(const std::vector<Point>& points, uint32_t& outSize);
  static std::vector<Point> convertByteArrayToPoints(const uint8_t* byteArray, uint32_t byteArraySize);
  static Point GetRandomPoint(double minX, double maxX, double minY, double maxY);

 private:
  std::vector<Point> input_, output_, result_;
};

}  // namespace makhov_m_jarvis_algorithm_seq