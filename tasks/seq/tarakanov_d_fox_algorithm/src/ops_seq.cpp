#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace tarakanov_d_fox_algorithm_seq {

bool TaskSequential::PreProcessingImpl() {
  // Не создаем новые данные, используем переданные
  return true;
}

bool TaskSequential::ValidationImpl() {
  if (task_data->inputs_count.size() < 2 || task_data->outputs_count.size() < 1) {
    return false;
  }

  size_t sizeA = task_data->inputs_count[0] / sizeof(double);
  size_t sizeB = task_data->inputs_count[1] / sizeof(double);

  // Проверяем, что матрицы квадратные и одного размера
  size_t dimA = std::sqrt(sizeA);
  size_t dimB = std::sqrt(sizeB);

  return (dimA * dimA == sizeA) && (dimB * dimB == sizeB) && (dimA == dimB);
}

bool TaskSequential::RunImpl() {
  // Получаем данные из task_data
  double* matrixA = reinterpret_cast<double*>(task_data->inputs[0]);
  double* matrixB = reinterpret_cast<double*>(task_data->inputs[1]);
  double* result = reinterpret_cast<double*>(task_data->outputs[0]);

  size_t totalSize = task_data->inputs_count[0] / sizeof(double);
  size_t n = std::sqrt(totalSize);  // размер матрицы (n x n)

  // Инициализируем результат нулями
  for (size_t i = 0; i < n * n; ++i) {
    result[i] = 0.0;
  }

  // Алгоритм Фокса (блочное умножение)
  constexpr size_t blockSize = 2;

  for (size_t i = 0; i < n; i += blockSize) {
    for (size_t j = 0; j < n; j += blockSize) {
      for (size_t k = 0; k < n; k += blockSize) {
        // Границы блоков
        size_t iEnd = std::min(i + blockSize, n);
        size_t jEnd = std::min(j + blockSize, n);
        size_t kEnd = std::min(k + blockSize, n);

        // Умножение блоков
        for (size_t ii = i; ii < iEnd; ++ii) {
          for (size_t jj = j; jj < jEnd; ++jj) {
            double sum = 0.0;
            for (size_t kk = k; kk < kEnd; ++kk) {
              sum += matrixA[ii * n + kk] * matrixB[kk * n + jj];
            }
            result[ii * n + jj] += sum;
          }
        }
      }
    }
  }

  return true;
}

bool TaskSequential::PostProcessingImpl() {
  // Очистка не требуется, так как память управляется извне
  return true;
}

}  // namespace tarakanov_d_fox_algorithm_seq