#include "omp/makhov_m_jarvis_algorithm/include/ops_omp.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

bool makhov_m_jarvis_algorithm_omp::TaskOmp::ValidationImpl() {
  return task_data->inputs_count[0] / (2 * sizeof(double)) >= 3;
}

bool makhov_m_jarvis_algorithm_omp::TaskOmp::PreProcessingImpl() {
  const uint8_t* input_buffer = task_data->inputs[0];
  uint32_t byte_array_size = task_data->inputs_count[0];
  input_ = ConvertByteArrayToPoints(input_buffer, byte_array_size);
  return true;
}

bool makhov_m_jarvis_algorithm_omp::TaskOmp::RunImpl() {
  size_t n = input_.size();

  if (n < 3) {
    result_ = input_;
    return true;
  }

  if (n == 3) {
    result_ = input_;
    return true;
  }

  // Распараллеливаем поиск самой левой точки
  size_t leftmost = FindLeftmostPoint(input_);
  size_t current = leftmost;

  do {
    result_.push_back(input_[current]);
    // Распараллеливаем поиск следующей точки
    current = FindNextPoint(current, input_);
  } while (current != leftmost);

  return true;
}

bool makhov_m_jarvis_algorithm_omp::TaskOmp::PostProcessingImpl() {
  uint32_t output_size = 0;
  uint8_t* output_buffer = ConvertPointsToByteArray(result_, output_size);

  if (task_data->outputs.empty()) {
    task_data->outputs.push_back(output_buffer);
    task_data->outputs_count.push_back(output_size);
  } else {
    if (task_data->outputs[0] != nullptr) {
      delete[] task_data->outputs[0];
    }
    task_data->outputs[0] = output_buffer;
    task_data->outputs_count[0] = output_size;
  }

  return true;
}

double makhov_m_jarvis_algorithm_omp::TaskOmp::Cross(const Point& a, const Point& b, const Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double makhov_m_jarvis_algorithm_omp::TaskOmp::Dist(const Point& a, const Point& b) {
  return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

uint8_t* makhov_m_jarvis_algorithm_omp::TaskOmp::ConvertPointsToByteArray(const std::vector<Point>& points,
                                                                          uint32_t& out_size) {
  out_size = static_cast<uint32_t>(points.size() * 2 * sizeof(double));
  auto* buffer = new uint8_t[out_size];
  auto* double_buffer = reinterpret_cast<double*>(buffer);

// Распараллеливаем заполнение буфера
#pragma omp parallel for
  for (int i = 0; i < (int)points.size(); ++i) {
    double_buffer[2 * i] = points[i].GetX();
    double_buffer[(2 * i) + 1] = points[i].GetY();
  }

  return buffer;
}

std::vector<makhov_m_jarvis_algorithm_omp::Point> makhov_m_jarvis_algorithm_omp::TaskOmp::ConvertByteArrayToPoints(
    const uint8_t* byte_array, uint32_t byte_array_size) {
  std::vector<Point> points;

  if (byte_array == nullptr || byte_array_size == 0) {
    return points;
  }

  size_t point_count = byte_array_size / (2 * sizeof(double));
  const auto* data = reinterpret_cast<const double*>(byte_array);

  points.resize(point_count);  // Заранее выделяем память

// Распараллеливаем преобразование байтов в точки
#pragma omp parallel for
  for (int i = 0; i < (int)point_count; ++i) {
    points[i].SetX(data[2 * i]);
    points[i].SetY(data[(2 * i) + 1]);
  }

  return points;
}

size_t makhov_m_jarvis_algorithm_omp::TaskOmp::FindLeftmostPoint(const std::vector<Point>& points) {
  size_t leftmost = 0;
  size_t n = points.size();

  if (n < 2) {
    return leftmost;
  }

// Распараллеливаем поиск самой левой точки
#pragma omp parallel
  {
    size_t local_leftmost = 0;

// Каждый поток находит свою самую левую точку в своей части массива
#pragma omp for nowait
    for (int i = 1; i < (int)n; ++i) {
      if (points[i].x < points[local_leftmost].x ||
          (points[i].x == points[local_leftmost].x && points[i].y < points[local_leftmost].y)) {
        local_leftmost = i;
      }
    }

// Критическая секция для сравнения локальных результатов
#pragma omp critical
    {
      if (points[local_leftmost].x < points[leftmost].x ||
          (points[local_leftmost].x == points[leftmost].x && points[local_leftmost].y < points[leftmost].y)) {
        leftmost = local_leftmost;
      }
    }
  }

  return leftmost;
}

size_t makhov_m_jarvis_algorithm_omp::TaskOmp::FindNextPoint(size_t current, const std::vector<Point>& points) {
  size_t next = current;
  size_t n = points.size();

  if (n < 2) {
    return next;
  }

// Распараллеливаем поиск следующей точки
#pragma omp parallel
  {
    size_t local_next = current;

// Каждый поток находит своего кандидата на следующую точку
#pragma omp for nowait
    for (int i = 0; i < (int)n; ++i) {
      if (i == (int)current) {
        continue;
      }

      double cross_product = Cross(points[current], points[local_next], points[i]);

      if (local_next == current || cross_product > 0) {
        local_next = i;
      } else if (cross_product == 0) {
        if (Dist(points[current], points[i]) > Dist(points[current], points[local_next])) {
          local_next = i;
        }
      }
    }

// Критическая секция для выбора лучшего кандидата среди потоков
#pragma omp critical
    {
      double cross_product = Cross(points[current], points[next], points[local_next]);

      if (next == current || cross_product > 0) {
        next = local_next;
      } else if (cross_product == 0) {
        if (Dist(points[current], points[local_next]) > Dist(points[current], points[next])) {
          next = local_next;
        }
      }
    }
  }

  return next;
}

makhov_m_jarvis_algorithm_omp::Point makhov_m_jarvis_algorithm_omp::TaskOmp::GetRandomPoint(XCoord min_x, XCoord max_x,
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