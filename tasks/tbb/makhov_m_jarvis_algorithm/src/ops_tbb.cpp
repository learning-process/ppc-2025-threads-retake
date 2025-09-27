#include "tbb/makhov_m_jarvis_algorithm/include/ops_tbb.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "tbb/tbb.h"

bool makhov_m_jarvis_algorithm_tbb::TaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] / (2 * sizeof(double)) >= 3;
}

bool makhov_m_jarvis_algorithm_tbb::TaskTBB::PreProcessingImpl() {
  const uint8_t* input_buffer = task_data->inputs[0];
  uint32_t byte_array_size = task_data->inputs_count[0];
  input_ = ConvertByteArrayToPoints(input_buffer, byte_array_size);
  return true;
}

bool makhov_m_jarvis_algorithm_tbb::TaskTBB::RunImpl() {
  size_t n = input_.size();

  if (n < 3) {
    result_ = input_;
    return true;
  }

  if (n == 3) {
    result_ = input_;
    return true;
  }

  size_t leftmost = FindLeftmostPoint(input_);
  size_t current = leftmost;

  do {
    result_.push_back(input_[current]);
    current = FindNextPoint(current, input_);
  } while (current != leftmost);

  return true;
}

bool makhov_m_jarvis_algorithm_tbb::TaskTBB::PostProcessingImpl() {
  uint32_t output_size = 0;
  uint8_t* output_buffer = ConvertPointsToByteArray(result_, output_size);

  if (task_data->outputs.empty()) {
    task_data->outputs.push_back(output_buffer);
    task_data->outputs_count.push_back(output_size);
  } else {
    // Освобождаем старую память, если нужно
    if (task_data->outputs[0] != nullptr) {
      delete[] task_data->outputs[0];
    }
    task_data->outputs[0] = output_buffer;
    task_data->outputs_count[0] = output_size;
  }

  return true;
}

double makhov_m_jarvis_algorithm_tbb::TaskTBB::Cross(const Point& a, const Point& b, const Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double makhov_m_jarvis_algorithm_tbb::TaskTBB::Dist(const Point& a, const Point& b) {
  return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

uint8_t* makhov_m_jarvis_algorithm_tbb::TaskTBB::ConvertPointsToByteArray(const std::vector<Point>& points,
                                                                          uint32_t& out_size) {
  out_size = static_cast<uint32_t>(points.size() * 2 * sizeof(double));
  uint8_t* buffer = new uint8_t[out_size];
  double* double_buffer = reinterpret_cast<double*>(buffer);

  // Распараллеливаем с помощью TBB parallel_for
  tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      double_buffer[2 * i] = points[i].GetX();
      double_buffer[(2 * i) + 1] = points[i].GetY();
    }
  });

  return buffer;
}

std::vector<makhov_m_jarvis_algorithm_tbb::Point> makhov_m_jarvis_algorithm_tbb::TaskTBB::ConvertByteArrayToPoints(
    const uint8_t* byte_array, uint32_t byte_array_size) {
  std::vector<Point> points;

  if (byte_array == nullptr || byte_array_size == 0) {
    return points;
  }

  size_t point_count = byte_array_size / (2 * sizeof(double));
  const auto* data = reinterpret_cast<const double*>(byte_array);

  points.resize(point_count);

  // Распараллеливаем с помощью TBB parallel_for
  tbb::parallel_for(tbb::blocked_range<size_t>(0, point_count), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      points[i].SetX(data[2 * i]);
      points[i].SetY(data[(2 * i) + 1]);
    }
  });

  return points;
}

size_t makhov_m_jarvis_algorithm_tbb::TaskTBB::FindLeftmostPoint(const std::vector<Point>& points) {
  if (points.empty()) return 0;

  // Используем TBB parallel_reduce для поиска самой левой точки
  size_t leftmost = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(1, points.size()), 0,
      [&](const tbb::blocked_range<size_t>& range, size_t local_leftmost) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          if (points[i].x < points[local_leftmost].x ||
              (points[i].x == points[local_leftmost].x && points[i].y < points[local_leftmost].y)) {
            local_leftmost = i;
          }
        }
        return local_leftmost;
      },
      [&](size_t left1, size_t left2) {
        if (points[left1].x < points[left2].x ||
            (points[left1].x == points[left2].x && points[left1].y < points[left2].y)) {
          return left1;
        }
        return left2;
      });

  return leftmost;
}

size_t makhov_m_jarvis_algorithm_tbb::TaskTBB::FindNextPoint(size_t current, const std::vector<Point>& points) {
  if (points.size() < 2) return current;

  // Используем TBB parallel_reduce для поиска следующей точки
  size_t next = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, points.size()), current,
      [&](const tbb::blocked_range<size_t>& range, size_t local_next) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          if (i == current) continue;

          double cross_product = Cross(points[current], points[local_next], points[i]);

          if (local_next == current || cross_product > 0) {
            local_next = i;
          } else if (cross_product == 0) {
            if (Dist(points[current], points[i]) > Dist(points[current], points[local_next])) {
              local_next = i;
            }
          }
        }
        return local_next;
      },
      [&](size_t next1, size_t next2) {
        if (next1 == current) return next2;
        if (next2 == current) return next1;

        double cross_product = Cross(points[current], points[next1], points[next2]);

        if (cross_product > 0) {
          return next2;
        } else if (cross_product == 0) {
          if (Dist(points[current], points[next2]) > Dist(points[current], points[next1])) {
            return next2;
          }
        }
        return next1;
      });

  return next;
}

makhov_m_jarvis_algorithm_tbb::Point makhov_m_jarvis_algorithm_tbb::TaskTBB::GetRandomPoint(XCoord min_x, XCoord max_x,
                                                                                            YCoord min_y,
                                                                                            YCoord max_y) {
  unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
  static std::mt19937 generator(seed);

  std::uniform_real_distribution<double> dist_x(min_x.value, max_x.value);
  std::uniform_real_distribution<double> dist_y(min_y.value, max_y.value);

  Point point;
  point.x = dist_x(generator);
  point.y = dist_y(generator);

  return point;
}