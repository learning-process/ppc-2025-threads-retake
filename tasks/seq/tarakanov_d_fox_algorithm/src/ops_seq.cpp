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
#include "seq/tarakanov_d_fox_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool tarakanov_d_fox_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }

  unsigned int matrix_size = input_size / 2;
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);

  A_ = std::vector<double>(in_ptr, in_ptr + matrix_size);
  B_ = std::vector<double>(in_ptr + matrix_size, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);

  n_ = static_cast<int>(std::sqrt(matrix_size));
  if (n_ * n_ != static_cast<int>(matrix_size)) {
    return false;
  }

  block_size_ = n_ / 2;
  for (int i = 1; i <= n_; ++i) {
    if (n_ % i == 0) {
      block_size_ = i;
      break;
    }
  }
  return block_size_ > 0;
}

bool tarakanov_d_fox_algorithm_seq::TestTaskSequential::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));
  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

bool tarakanov_d_fox_algorithm_seq::TestTaskSequential::RunImpl() {
  int num_blocks = (n_ + block_size_ - 1) / block_size_;

  for (int stage = 0; stage < num_blocks; ++stage) {
    for (int i = 0; i < n_; i += block_size_) {
      for (int j = 0; j < n_; j += block_size_) {
        for (int bi = i; bi < i + block_size_ && bi < n_; ++bi) {
          for (int bj = j; bj < j + block_size_ && bj < n_; ++bj) {
            int start_k = stage * block_size_;
            for (int bk = start_k; bk < std::min((stage + 1) * block_size_, n_); ++bk) {
              output_[(bi * n_) + bj] += A_[(bi * n_) + bk] * B_[(bk * n_) + bj];
            }
          }
        }
      }
    }
  }
  return true;
}
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