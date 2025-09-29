#include "tbb/chernova_n_cannon_matrix_mul/include/ops_tbb.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

std::vector<double> chernova_n_cannon_matrix_mul_tbb::CannonMatrixMultiplicationTBB(const std::vector<double> &mat_a,
                                                                                    const std::vector<double> &mat_b,
                                                                                    int n) {
  if (n <= 0) {
    return {};
  }

  int block_size = 32;
  if (n < 32) {
    block_size = 8;
  }
  if (n < 8) {
    block_size = 2;
  }

  std::vector<double> matrix_c(n * n, 0.0);

  int num_blocks = (n + block_size - 1) / block_size;

  tbb::parallel_for(0, num_blocks, [&](int block_i) {
    for (int block_j = 0; block_j < num_blocks; block_j++) {
      for (int block_k = 0; block_k < num_blocks; block_k++) {
        int i_start = block_i * block_size;
        int j_start = block_j * block_size;
        int k_start = block_k * block_size;
        int i_end = std::min(i_start + block_size, n);
        int j_end = std::min(j_start + block_size, n);
        int k_end = std::min(k_start + block_size, n);

        for (int i = i_start; i < i_end; i++) {
          for (int k = k_start; k < k_end; k++) {
            double a_val = mat_a[(i * n) + k];
            for (int j = j_start; j < j_end; j++) {
              matrix_c[(i * n) + j] += a_val * mat_b[(k * n) + j];
            }
          }
        }
      }
    }
  });

  return matrix_c;
}

std::vector<double> chernova_n_cannon_matrix_mul_tbb::MultiplyMatrixTBB(const std::vector<double> &mat_a,
                                                                        const std::vector<double> &mat_b, int n) {
  if (n == 0) {
    return {};
  }

  std::vector<double> matrix_c(n * n, 0.0);

  tbb::parallel_for(0, n, [&](int i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += mat_a[(i * n) + k] * mat_b[(k * n) + j];
      }
      matrix_c[(i * n) + j] = sum;
    }
  });

  return matrix_c;
}

bool chernova_n_cannon_matrix_mul_tbb::TestTaskTBB::PreProcessingImpl() {
  n_ = *reinterpret_cast<int *>(task_data->inputs[2]);

  size_t matrix_size = n_ * n_;

  if (matrix_size == 0) {
    matrixA_.clear();
    matrixB_.clear();
    return true;
  }

  matrixA_.resize(matrix_size);
  matrixB_.resize(matrix_size);

  auto *tmp_ptr_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *tmp_ptr_b = reinterpret_cast<double *>(task_data->inputs[1]);

  std::copy(tmp_ptr_a, tmp_ptr_a + matrix_size, matrixA_.begin());
  std::copy(tmp_ptr_b, tmp_ptr_b + matrix_size, matrixB_.begin());

  return true;
}

bool chernova_n_cannon_matrix_mul_tbb::TestTaskTBB::ValidationImpl() {
  if (task_data->inputs.size() < 3 || task_data->outputs.empty()) {
    return false;
  }

  int n = *reinterpret_cast<int *>(task_data->inputs[2]);

  if (n <= 0) {
    return false;
  }

  size_t expected_size = n * n;

  return task_data->inputs_count[0] == expected_size && task_data->inputs_count[1] == expected_size &&
         task_data->outputs_count[0] == expected_size;
}

bool chernova_n_cannon_matrix_mul_tbb::TestTaskTBB::RunImpl() {
  res_ = CannonMatrixMultiplicationTBB(matrixA_, matrixB_, n_);
  return true;
}

bool chernova_n_cannon_matrix_mul_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(res_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}