#include "seq/chernova_n_cannon_matrix_mul/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

std::vector<double> chernova_n_cannon_matrix_mul_seq::CannonMatrixMultiplication(const std::vector<double>& mat_a,
                                                                                 const std::vector<double>& mat_b,
                                                                                 int n) {
  if (n <= 0) {
    return {};
  }

  if (mat_a.empty() || mat_b.empty()) {
    return std::vector<double>(n * n, 0.0);
  }

  int p = static_cast<int>(std::sqrt(n));
  if (p * p != n) {
    p = 2;
  }

  int block_size = n / p;

  std::vector<double> matrix_c(n * n, 0.0);

  std::vector<double> a_temp = mat_a;
  std::vector<double> b_temp = mat_b;
  if (a_temp.empty() || b_temp.empty()) {
    return std::vector<double>(n * n, 0.0);
  }

  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      int new_j_a = (j - i + p) % p;
      int new_i_b = (i - j + p) % p;

      for (int ii = 0; ii < block_size; ++ii) {
        for (int jj = 0; jj < block_size; ++jj) {
          int orig_row = i * block_size + ii;
          int orig_col = j * block_size + jj;
          int new_row_a = i * block_size + ii;
          int new_col_a = new_j_a * block_size + jj;
          int new_row_b = new_i_b * block_size + ii;
          int new_col_b = j * block_size + jj;

          a_temp[(new_row_a * n) + new_col_a] = mat_a[(orig_row * n) + orig_col];
          b_temp[(new_row_b * n) + new_col_b] = mat_b[(orig_row * n) + orig_col];
        }
      }
    }
  }

  for (int step = 0; step < p; ++step) {
    for (int i = 0; i < p; ++i) {
      for (int j = 0; j < p; ++j) {
        for (int ii = 0; ii < block_size; ++ii) {
          for (int jj = 0; jj < block_size; ++jj) {
            for (int kk = 0; kk < block_size; ++kk) {
              int row_a = i * block_size + ii;
              int col_a = j * block_size + kk;
              int row_b = i * block_size + kk;
              int col_b = j * block_size + jj;
              int row_c = i * block_size + ii;
              int col_c = j * block_size + jj;

              matrix_c[(row_c * n) + col_c] += a_temp[(row_a * n) + col_a] * b_temp[(row_b * n) + col_b];
            }
          }
        }
      }
    }

    if (step < p - 1) {
      std::vector<double> a_shifted = a_temp;
      std::vector<double> b_shifted = b_temp;

      for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
          int new_j = (j - 1 + p) % p;
          for (int ii = 0; ii < block_size; ++ii) {
            for (int jj = 0; jj < block_size; ++jj) {
              int orig_row = i * block_size + ii;
              int orig_col = j * block_size + jj;
              int new_row = i * block_size + ii;
              int new_col = new_j * block_size + jj;

              a_shifted[(new_row * n) + new_col] = a_temp[(orig_row * n) + orig_col];
            }
          }
        }
      }

      for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
          int new_i = (i - 1 + p) % p;
          for (int ii = 0; ii < block_size; ++ii) {
            for (int jj = 0; jj < block_size; ++jj) {
              int orig_row = i * block_size + ii;
              int orig_col = j * block_size + jj;
              int new_row = new_i * block_size + ii;
              int new_col = j * block_size + jj;

              b_shifted[(new_row * n) + new_col] = b_temp[(orig_row * n) + orig_col];
            }
          }
        }
      }
      if (a_temp.empty() || b_temp.empty() || a_shifted.empty() || b_shifted.empty()) {
        return std::vector<double>(n * n, 0.0);
      }
      a_temp = a_shifted;
      b_temp = b_shifted;
    }
  }

  return matrix_c;
}

bool chernova_n_cannon_matrix_mul_seq::TestTaskSequential::PreProcessingImpl() {
  if (task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr) {
    return false;
  }
  matrixA_ = std::vector<double>(task_data->inputs_count[0]);
  matrixB_ = std::vector<double>(task_data->inputs_count[1]);
  n_ = *reinterpret_cast<int*>(task_data->inputs[2]);

  auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    matrixA_[i] = tmp_ptr_a[i];
  }

  auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
  for (size_t i = 0; i < task_data->inputs_count[1]; i++) {
    matrixB_[i] = tmp_ptr_b[i];
  }
  return true;
}

bool chernova_n_cannon_matrix_mul_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr || task_data->inputs[2] == nullptr ||
      task_data->outputs[0] == nullptr) {
    return false;
  }
  int n = *reinterpret_cast<int*>(task_data->inputs[2]);
  if (n <= 0) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         task_data->inputs_count[1] == task_data->outputs_count[0] && task_data->inputs_count[1] > 0 &&
         task_data->inputs_count[0] > 0;
}

bool chernova_n_cannon_matrix_mul_seq::TestTaskSequential::RunImpl() {
  res_ = CannonMatrixMultiplication(matrixA_, matrixB_, n_);
  return true;
}

bool chernova_n_cannon_matrix_mul_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(res_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}

std::vector<double> chernova_n_cannon_matrix_mul_seq::MultiplyMatrix(const std::vector<double>& mat_a,
                                                                     const std::vector<double>& mat_b, int n) {
  std::vector<double> matrix_c(n * n, 0.0);

  if (n == 0) {
    return {};
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        matrix_c[(i * n) + j] += mat_a[(i * n) + k] * mat_b[(k * n) + j];
      }
    }
  }
  return matrix_c;
}