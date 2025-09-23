#include "seq/makhov_m_jarvis_algorithm/include/ops_seq.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool makhov_m_jarvis_algorithm_seq::TaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] / (2 * sizeof(double)) >= 3;
}

bool makhov_m_jarvis_algorithm_seq::TaskSequential::PreProcessingImpl() {
  // Init value for input and output
  const uint8_t* input_buffer = task_data->inputs[0];
  uint32_t byte_array_size = task_data->inputs_count[0];
  // Преобразуем входные байты в вектор точек

  input_ = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(input_buffer, byte_array_size);
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
    if (input_[i].x < input_[leftmost].x || (input_[i].x == input_[leftmost].x && input_[i].y < input_[leftmost].y)) {
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
      if (i == current) {
        continue;
      }

      // Если next ещё не инициализирован, выбираем первую подходящую точку
      if (next == current) {
        next = i;
      } else {
        double cr = Cross(input_[current], input_[next], input_[i]);
        if (cr > 0) {
          // Точка i лежит левее, обновляем next
          next = i;
        } else if (cr == 0) {
          // Если точки коллинеарны, выбираем более дальнюю
          if (Dist(input_[current], input_[i]) > Dist(input_[current], input_[next])) {
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
  uint32_t output_size = 0;
  uint8_t* output_buffer = ConvertPointsToByteArray(result_, output_size);

  if (task_data->outputs.empty()) {
    task_data->outputs.push_back(output_buffer);
    task_data->outputs_count.push_back(output_size);
  } else {
    // Освобождаем старую память, если нужно

    if (task_data->outputs[0] != nullptr) {
      std::cout << "task_data->outputs[0] is not nullptr" << '\n';
      delete[] task_data->outputs[0];
    }
    task_data->outputs[0] = output_buffer;
    task_data->outputs_count[0] = output_size;
  }

  return true;
}
