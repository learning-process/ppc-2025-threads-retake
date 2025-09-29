#include "omp/chernova_n_cannon_matrix_mul/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

std::vector<double> chernova_n_cannon_matrix_mul_omp::CannonMatrixMultiplicationOMP(const std::vector<double>& a,
                                                                                    const std::vector<double>& b,
                                                                                    int n) {
  if (n <= 0) return {};

  int block_size = 32;
  if (n < 32) block_size = 8;
  if (n < 8) block_size = 2;

  std::vector<double> matrixC(n * n, 0.0);

  int num_blocks = (n + block_size - 1) / block_size;

#pragma omp parallel for
  for (int block_i = 0; block_i < num_blocks; block_i++) {
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
            double a_val = a[i * n + k];
            for (int j = j_start; j < j_end; j++) {
              matrixC[i * n + j] += a_val * b[k * n + j];
            }
          }
        }
      }
    }
  }

  return matrixC;
}
std::vector<double> chernova_n_cannon_matrix_mul_omp::MultiplyMatrixOMP(const std::vector<double>& a,
                                                                        const std::vector<double>& b, int n) {
  std::vector<double> matrixC(n * n, 0.0);

  if (n == 0) {
    return {};
  }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      matrixC[(i * n) + j] = sum;
    }
  }
  return matrixC;
}
bool chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP::PreProcessingImpl() {
  n_ = *reinterpret_cast<int*>(task_data->inputs[2]);

  size_t matrix_size = n_ * n_;

  if (matrix_size == 0) {
    matrixA.clear();
    matrixB.clear();
    return true;
  }

  matrixA.resize(matrix_size);
  matrixB.resize(matrix_size);

  auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  std::copy(tmp_ptr_a, tmp_ptr_a + matrix_size, matrixA.begin());
  std::copy(tmp_ptr_b, tmp_ptr_b + matrix_size, matrixB.begin());

  return true;
}

bool chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP::ValidationImpl() {
  if (task_data->inputs.size() < 3 || task_data->outputs.size() < 1) {
    return false;
  }

  int n = *reinterpret_cast<int*>(task_data->inputs[2]);

  if (n <= 0) {
    return false;
  }

  size_t expected_size = n * n;

  return task_data->inputs_count[0] == expected_size && task_data->inputs_count[1] == expected_size &&
         task_data->outputs_count[0] == expected_size;
}

bool chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP::RunImpl() {
  res = CannonMatrixMultiplicationOMP(matrixA, matrixB, n_);
  return true;
}

bool chernova_n_cannon_matrix_mul_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::copy(res.begin(), res.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}