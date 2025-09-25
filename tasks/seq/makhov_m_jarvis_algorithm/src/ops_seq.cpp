#include "seq/makhov_m_jarvis_algorithm/include/ops_seq.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

bool makhov_m_jarvis_algorithm_seq::TaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] / (2 * sizeof(double)) >= 3;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::PreProcessingImpl() {
  const uint8_t* input_buffer = task_data->inputs[0];
  uint32_t byte_array_size = task_data->inputs_count[0];
  input_ = ConvertByteArrayToPoints(input_buffer, byte_array_size);
  return true;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::RunImpl() {
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

bool makhov_m_jarvis_algorithm_seq::TaskSequential::PostProcessingImpl() {
  uint32_t output_size = 0;
  uint8_t* output_buffer = ConvertPointsToByteArray(result_, output_size);

  if (task_data->outputs.empty()) {
    task_data->outputs.push_back(output_buffer);
    task_data->outputs_count.push_back(output_size);
  } else {
    // if (task_data->outputs[0] != nullptr) {
    //   delete[] task_data->outputs[0];
    // }
    task_data->outputs[0] = output_buffer;
    task_data->outputs_count[0] = output_size;
  }

  if (output_buffer != nullptr) {
    delete[] output_buffer;
  }
  return true;
}

double makhov_m_jarvis_algorithm_seq::TaskSequential::Cross(const Point& a, const Point& b, const Point& c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

double makhov_m_jarvis_algorithm_seq::TaskSequential::Dist(const Point& a, const Point& b) {
  return ((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y));
}

uint8_t* makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(const std::vector<Point>& points,
                                                                                 uint32_t& out_size) {
  out_size = static_cast<uint32_t>(points.size() * 2 * sizeof(double));
  auto* buffer = new uint8_t[out_size];
  auto* double_buffer = reinterpret_cast<double*>(buffer);

  for (size_t i = 0; i < points.size(); ++i) {
    double_buffer[2 * i] = points[i].GetX();
    double_buffer[(2 * i) + 1] = points[i].GetY();
  }

  return buffer;
}

std::vector<makhov_m_jarvis_algorithm_seq::Point>
makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(const uint8_t* byte_array,
                                                                        uint32_t byte_array_size) {
  std::vector<Point> points;

  if (byte_array == nullptr || byte_array_size == 0) {
    return points;
  }

  size_t point_count = byte_array_size / (2 * sizeof(double));
  const auto* data = reinterpret_cast<const double*>(byte_array);

  for (size_t i = 0; i < point_count; ++i) {
    Point point;
    point.SetX(data[2 * i]);
    point.SetY(data[(2 * i) + 1]);
    points.push_back(point);
  }

  return points;
}

static size_t makhov_m_jarvis_algorithm_seq::TaskSequential::FindLeftmostPoint(const std::vector<Point>& points) const {
  size_t leftmost = 0;

  for (size_t i = 1; i < points.size(); ++i) {
    if (points[i].x < points[leftmost].x || (points[i].x == points[leftmost].x && points[i].y < points[leftmost].y)) {
      leftmost = i;
    }
  }

  return leftmost;
}

static size_t makhov_m_jarvis_algorithm_seq::TaskSequential::FindNextPoint(size_t current,
                                                                           const std::vector<Point>& points) const {
  size_t next = current;

  for (size_t i = 0; i < points.size(); ++i) {
    if (i == current) {
      continue
    };

    double cross_product = Cross(points[current], points[next], points[i]);

    if (next == current || cross_product > 0) {
      next = i;
    } else if (cross_product == 0) {
      if (Dist(points[current], points[i]) > Dist(points[current], points[next])) {
        next = i;
      }
    }
  }

  return next;
}

makhov_m_jarvis_algorithm_seq::Point makhov_m_jarvis_algorithm_seq::TaskSequential::GetRandomPoint(XCoord min_x,
                                                                                                   XCoord max_x,
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