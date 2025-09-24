#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_jarvis_algorithm_seq {

struct Point {
 public:
  using XCoord = double;
  using YCoord = double;
  double x, y;
  // Конструкторы
  Point() : x(0), y(0) {}
  Point(XCoord x, YCoord y) : x(x), y(y) {}

  // Методы доступа
  [[nodiscard]] double GetX() const { return x; }
  [[nodiscard]] double GetY() const { return y; }
  void SetX(double x_coord) { x = x_coord; }
  void SetY(double y_coord) { y = y_coord; }
  void Set(XCoord x_coord, YCoord y_coord) {
    x = x_coord;
    y = y_coord;
  }

  [[nodiscard]] double DistanceTo(const Point& other) const {
    double dx = x - other.x;
    double dy = y - other.y;
    return std::sqrt((dx * dx) + (dy * dy));
  }

  // Перегрузка оператора вывода с форматированием
  friend std::ostream& operator<<(std::ostream& os, const Point& point) {
    // Сохраняем оригинальные настройки потока
    std::ios old_state(nullptr);
    old_state.copyfmt(os);

    // Устанавливаем фиксированный формат и точность (2 знака после запятой)
    os << "(" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";

    // Восстанавливаем оригинальные настройки потока
    os.copyfmt(old_state);

    return os;
  }

  // Перегрузка операторов сравнения
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
  static double Cross(const Point& a, const Point& b, const Point& c) {
    return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
  }
  static double Dist(const Point& a, const Point& b) {
    return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
  }
  static uint8_t* ConvertPointsToByteArray(const std::vector<Point>& points, uint32_t& out_size) {
    out_size = static_cast<uint32_t>(points.size() * 2 * sizeof(double));
    auto* buffer = new uint8_t[out_size];
    auto* double_buffer = reinterpret_cast<double*>(buffer);

    for (uint32_t i = 0; i < points.size(); ++i) {
      double_buffer[2 * i] = points[i].GetX();
      double_buffer[(2 * i) + 1] = points[i].GetY();
    }

    return buffer;
  }
  static std::vector<Point> ConvertByteArrayToPoints(const uint8_t* byte_array, uint32_t byte_array_size) {
    std::vector<makhov_m_jarvis_algorithm_seq::Point> points;

    if (byte_array == nullptr || byte_array_size == 0) {
      return points;
    }

    uint32_t point_count = byte_array_size / (2 * sizeof(double));

    if (byte_array_size % (2 * sizeof(double)) != 0) {
      point_count = byte_array_size / (2 * sizeof(double));
    }

    const auto* data = reinterpret_cast<const double*>(byte_array);

    for (uint32_t i = 0; i < point_count; ++i) {
      makhov_m_jarvis_algorithm_seq::Point point;
      point.SetX(data[2 * i]);
      point.SetY(data[(2 * i) + 1]);
      points.push_back(point);
    }

    return points;
  }
  static Point GetRandomPoint(Point::XCoord min_x, Point::XCoord max_x, Point::YCoord min_y, Point::YCoord max_y) {
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    static std::mt19937 generator(seed);

    // Создаем распределения для координат X и Y
    std::uniform_real_distribution<double> dist_x(min_x, max_x);
    std::uniform_real_distribution<double> dist_y(min_y, max_y);

    // Генерируем случайные координаты
    Point point;
    point.x = dist_x(generator);
    point.y = dist_y(generator);

    return point;
  }

 private:
  std::vector<Point> input_, output_, result_;
};

}  // namespace makhov_m_jarvis_algorithm_seq