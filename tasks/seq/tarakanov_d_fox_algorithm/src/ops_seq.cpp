#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tarakanov_d_fox_algorithm_seq {

bool TaskSequential::ValidationImpl() {
  if (task_data->inputs_count.size() < 2 || task_data->outputs_count.empty()) {
    return false;
  }

  sizeA_ = task_data->inputs_count[0] / sizeof(double);
  sizeB_ = task_data->inputs_count[1] / sizeof(double);

  size_t dim_a = std::sqrt(sizeA_);
  size_t dim_b = std::sqrt(sizeB_);

  return (dim_a * dim_a == sizeA_) && (dim_b * dim_b == sizeB_) && (dim_a == dim_b);
}

bool TaskSequential::PreProcessingImpl() {
  double *doublePtrmatrixA_ = reinterpret_cast<double *>(task_data->inputs[0]);
  double *doublePtrmatrixB_ = reinterpret_cast<double *>(task_data->inputs[1]);

  matrixA_ = std::vector<double>(doublePtrmatrixA_, doublePtrmatrixA_ + sizeA_);
  matrixB_ = std::vector<double>(doublePtrmatrixA_, doublePtrmatrixB_ + sizeB_);

  result_.reserve(sizeA_);

  return true;
}

bool TaskSequential::RunImpl() {
  size_t totalSize = sizeA_;
  size_t n = std::sqrt(totalSize);

  for (size_t i = 0; i < n * n; ++i) {
    result_[i] = 0.0;
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
              sum += matrixA_[ii * n + kk] * matrixB_[kk * n + jj];
            }
            result_[ii * n + jj] += sum;
          }
        }
      }
    }
  }

  return true;
}

bool TaskSequential::PostProcessingImpl() {
  std::copy(result_.begin(), result_.begin() + sizeA_, task_data->outputs[0]);
  return true;
}

}  // namespace tarakanov_d_fox_algorithm_seq