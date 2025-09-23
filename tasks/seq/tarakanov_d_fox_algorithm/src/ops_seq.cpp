#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace tarakanov_d_fox_algorithm_seq {

bool TaskSequential::ValidationImpl() {
  if (task_data->inputs_count.size() < 2 || task_data->outputs_count.size() < 1) {
    return false;
  }

  sizeA = task_data->inputs_count[0] / sizeof(double);
  sizeB = task_data->inputs_count[1] / sizeof(double);

  size_t dimA = std::sqrt(sizeA);
  size_t dimB = std::sqrt(sizeB);

  return (dimA * dimA == sizeA) && (dimB * dimB == sizeB) && (dimA == dimB);
}

bool TaskSequential::PreProcessingImpl() {
  double *doublePtrMatrixA = reinterpret_cast<double *>(task_data->inputs[0]);
  double *doublePtrMatrixB = reinterpret_cast<double *>(task_data->inputs[1]);

  matrixA = std::vector<double>(doublePtrMatrixA, doublePtrMatrixA + sizeA);
  matrixB = std::vector<double>(doublePtrMatrixA, doublePtrMatrixB + sizeB);

  result.reserve(sizeA);

  return true;
}

bool TaskSequential::RunImpl() {
  size_t totalSize = sizeA;
  size_t n = std::sqrt(totalSize);

  for (size_t i = 0; i < n * n; ++i) {
    result[i] = 0.0;
  }

  constexpr size_t blockSize = 2;

  for (size_t i = 0; i < n; i += blockSize) {
    for (size_t j = 0; j < n; j += blockSize) {
      for (size_t k = 0; k < n; k += blockSize) {
        size_t iEnd = std::min(i + blockSize, n);
        size_t jEnd = std::min(j + blockSize, n);
        size_t kEnd = std::min(k + blockSize, n);

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
  std::copy(result.begin(), result.begin() + sizeA, task_data->outputs[0]);
  return true;
}

}  // namespace tarakanov_d_fox_algorithm_seq