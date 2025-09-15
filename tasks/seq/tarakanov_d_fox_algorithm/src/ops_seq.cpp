#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"
#include <vector>

namespace tarakanov_d_fox_algorithm_seq {

bool TaskSequential::PreProcessingImpl() {
  // Создаем матрицы A и B, заполненные 1.0 для тестирования
  size_t size = 2; // Размер матриц (2x2 для примера)
  
  // Создаем матрицу A
  std::vector<double> matrixA(size * size, 1.0);
  
  // Создаем матрицу B
  std::vector<double> matrixB(size * size, 1.0);
  
  // Инициализируем данные для умножения
  task_data_->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data_->inputs_count.emplace_back(matrixA.size() * sizeof(double));
  
  task_data_->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data_->outputs_count.emplace_back(matrixB.size() * sizeof(double));
  
  return true;
}

bool TaskSequential::ValidationImpl() {
  // Проверяем, что размеры матриц соответствуют для умножения
  // Для упрощения берем первые две матрицы
  if (task_data_->inputs_count.size() < 2) {
    return false;
  }
  
  size_t sizeA = task_data_->inputs_count[0] / sizeof(double);
  size_t sizeB = task_data_->inputs_count[1] / sizeof(double);
  
  // Для матриц A (sizeA x sizeA) и B (sizeA x sizeA) результат будет (sizeA x sizeA)
  // Проверяем, что матрицы могут быть умножены
  return (sizeA == sizeB);
}

bool TaskSequential::RunImpl() {
  // Извлекаем матрицы A и B из task_data_
  double* matrixA = reinterpret_cast<double*>(task_data_->inputs[0]);
  double* matrixB = reinterpret_cast<double*>(task_data_->inputs[1]);
  
  size_t sizeA = task_data_->inputs_count[0] / sizeof(double);
  size_t sizeB = task_data_->inputs_count[1] / sizeof(double);
  
  // Создаем результирующую матрицу C
  std::vector<double> matrixC(sizeA * sizeA, 0.0);
  double* result = matrixC.data();
  
  // Базовый алгоритм умножения матриц
  for (size_t i = 0; i < sizeA; ++i) {
    for (size_t j = 0; j < sizeA; ++j) {
      for (size_t k = 0; k < sizeA; ++k) {
        result[i * sizeA + j] += matrixA[i * sizeA + k] * matrixB[k * sizeA + j];
      }
    }
  }
  
  // Сохраняем результат в outputs
  task_data_->outputs[0] = reinterpret_cast<uint8_t*>(result);
  task_data_->outputs_count[0] = sizeA * sizeA * sizeof(double);
  
  return true;
}

bool TaskSequential::PostProcessingImpl() {
  // Очистка ресурсов (если требуется)
  return true;
}

}  // namespace tarakanov_d_fox_algorithm_seq