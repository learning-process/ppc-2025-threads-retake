#include "seq/makhov_m_jarvis_algorithm/include/ops_seq.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "iostream"

double makhov_m_jarvis_algorithm_seq::TaskSequential::cross(const makhov_m_jarvis_algorithm_seq::Point& a,
                                                            const makhov_m_jarvis_algorithm_seq::Point& b,
                                                            const makhov_m_jarvis_algorithm_seq::Point& c) {
  return (b.x_ - a.x_) * (c.y_ - a.y_) - (b.y_ - a.y_) * (c.x_ - a.x_);
}

double makhov_m_jarvis_algorithm_seq::TaskSequential::dist(const makhov_m_jarvis_algorithm_seq::Point& a,
                                                           const makhov_m_jarvis_algorithm_seq::Point& b) {
  return (a.x_ - b.x_) * (a.x_ - b.x_) + (a.y_ - b.y_) * (a.y_ - b.y_);
}

uint8_t* makhov_m_jarvis_algorithm_seq::TaskSequential::convertPointsToByteArray(
    const std::vector<makhov_m_jarvis_algorithm_seq::Point>& points, uint32_t& outSize) {
  // Вычисляем размер необходимого буфера в байтах
  outSize = static_cast<uint32_t>(points.size() * 2 * sizeof(double));

  // Выделяем память под буфер
  uint8_t* buffer = new uint8_t[outSize];

  double* doubleBuffer = reinterpret_cast<double*>(buffer);

  // Копируем данные точек в буфер
  for (size_t i = 0; i < points.size(); ++i) {
    doubleBuffer[2 * i] = points[i].getX();
    doubleBuffer[2 * i + 1] = points[i].getY();
  }

  return buffer;
}

std::vector<makhov_m_jarvis_algorithm_seq::Point>
makhov_m_jarvis_algorithm_seq::TaskSequential::convertByteArrayToPoints(const uint8_t* byteArray,
                                                                        uint32_t byteArraySize) {
  std::vector<makhov_m_jarvis_algorithm_seq::Point> points;

  // Проверяем валидность входных данных
  if (byteArray == nullptr || byteArraySize == 0) {
    return points;  // Возвращаем пустой вектор
  }

  // Вычисляем количество точек в массиве
  // Каждая точка представлена двумя double (x и y)
  size_t pointCount = byteArraySize / (2 * sizeof(double));

  // Проверяем, что размер данных корректен
  if (byteArraySize % (2 * sizeof(double)) != 0) {
    // Размер не кратен размеру двух double - возможно, ошибка в данных
    pointCount = byteArraySize / (2 * sizeof(double));  // Округляем в меньшую сторону
  }

  const double* data = reinterpret_cast<const double*>(byteArray);

  // Восстанавливаем точки из байтового массива
  for (size_t i = 0; i < pointCount; ++i) {
    makhov_m_jarvis_algorithm_seq::Point point;
    point.setX(data[2 * i]);
    point.setY(data[2 * i + 1]);
    points.push_back(point);
  }

  return points;
}

makhov_m_jarvis_algorithm_seq::Point makhov_m_jarvis_algorithm_seq::TaskSequential::GetRandomPoint(double minX,
                                                                                                   double maxX,
                                                                                                   double minY,
                                                                                                   double maxY) {
  // Используем текущее время как seed для генератора случайных чисел
  unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
  static std::mt19937 generator(seed);

  // Создаем распределения для координат X и Y
  std::uniform_real_distribution<double> distX(minX, maxX);
  std::uniform_real_distribution<double> distY(minY, maxY);

  // Генерируем случайные координаты
  Point point;
  point.x_ = distX(generator);
  point.y_ = distY(generator);

  return point;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] / (2 * sizeof(double)) < 3) {
    return false;
  }
  return true;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::PreProcessingImpl() {
  // Init value for input and output
  const uint8_t* inputBuffer = task_data->inputs[0];
  uint32_t byteArraySize = task_data->inputs_count[0];
  // Преобразуем входные байты в вектор точек

  input_ = makhov_m_jarvis_algorithm_seq::TaskSequential::convertByteArrayToPoints(inputBuffer, byteArraySize);
  return true;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::RunImpl() {
  size_t n = task_data->inputs_count[0] / (2 * sizeof(double));
  if (n == 3) {
    result_.resize(3);
    for (size_t i = 0; i < n; i++) {
      result_[i] = input_[i];
    }
    return true;  // Возвращаем все точки, если их 3
  }

  // Шаг 1: Находим самую левую точку
  size_t leftmost = 0;
  for (size_t i = 1; i < n; i++) {
    if (input_[i].x_ < input_[leftmost].x_ ||
        (input_[i].x_ == input_[leftmost].x_ && input_[i].y_ < input_[leftmost].y_)) {
      leftmost = i;
    }
  }

  size_t current = leftmost;

  do {
    // Добавляем текущую точку в оболочку
    result_.push_back(input_[current]);

    // Шаг 2: Ищем следующую точку для добавления
    size_t next = current;
    for (size_t i = 0; i < n; i++) {
      if (i == current) continue;

      // Если next ещё не инициализирован, выбираем первую подходящую точку
      if (next == current) {
        next = i;
      } else {
        double cr = cross(input_[current], input_[next], input_[i]);
        if (cr > 0) {
          // Точка i лежит левее, обновляем next
          next = i;
        } else if (cr == 0) {
          // Если точки коллинеарны, выбираем более дальнюю
          if (dist(input_[current], input_[i]) > dist(input_[current], input_[next])) {
            next = i;
          }
        }
      }
    }

    current = next;  // Переходим к следующей точке
  } while (current != leftmost);  // Повторяем, пока не вернёмся в начальную точку

  return true;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::PostProcessingImpl() {
  uint32_t outputSize;
  uint8_t* outputBuffer = convertPointsToByteArray(result_, outputSize);

  if (task_data->outputs.empty()) {
    task_data->outputs.push_back(outputBuffer);
    task_data->outputs_count.push_back(static_cast<uint32_t>(outputSize));
  } else {
    // Освобождаем старую память, если нужно

    if (task_data->outputs[0] != nullptr) {
      delete[] task_data->outputs[0];
    }
    task_data->outputs[0] = outputBuffer;
    task_data->outputs_count[0] = static_cast<uint32_t>(outputSize);
  }

  return true;
}
