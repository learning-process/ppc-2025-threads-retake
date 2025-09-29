#include "seq/chernova_n_cannon_matrix_mul/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {
struct MatrixParams {
  int matrix_size;
  int param;
  int block_size;
};
void InitialAlignment(const std::vector<double>& mat_a, const std::vector<double>& mat_b, std::vector<double>& a_temp,
                      std::vector<double>& b_temp, const MatrixParams& params) {
  for (int i = 0; i < params.param; ++i) {
    for (int j = 0; j < params.param; ++j) {
      const int new_j_a = (j - i + params.param) % params.param;
      const int new_i_b = (i - j + params.param) % params.param;

      for (int ii = 0; ii < params.block_size; ++ii) {
        for (int jj = 0; jj < params.block_size; ++jj) {
          const int orig_idx = (((i * params.block_size) + ii) * params.matrix_size) + ((j * params.block_size) + jj);
          const int new_idx_a =
              (((i * params.block_size) + ii) * params.matrix_size) + ((new_j_a * params.block_size) + jj);
          const int new_idx_b =
              (((new_i_b * params.block_size) + ii) * params.matrix_size) + ((j * params.block_size) + jj);

          a_temp[new_idx_a] = mat_a[orig_idx];
          b_temp[new_idx_b] = mat_b[orig_idx];
        }
      }
    }
  }
}

void MultiplyBlocks(const std::vector<double>& a_temp, const std::vector<double>& b_temp, std::vector<double>& matrix_c,
                    const MatrixParams& params) {
  for (int i = 0; i < params.param; ++i) {
    for (int j = 0; j < params.param; ++j) {
      for (int ii = 0; ii < params.block_size; ++ii) {
        for (int jj = 0; jj < params.block_size; ++jj) {
          for (int kk = 0; kk < params.block_size; ++kk) {
            const int row_a = (i * params.block_size) + ii;
            const int col_a = (j * params.block_size) + kk;
            const int row_b = (i * params.block_size) + kk;
            const int col_b = (j * params.block_size) + jj;
            const int idx_c = (((i * params.block_size) + ii) * params.matrix_size) + ((j * params.block_size) + jj);

            matrix_c[idx_c] +=
                a_temp[(row_a * params.matrix_size) + col_a] * b_temp[(row_b * params.matrix_size) + col_b];
          }
        }
      }
    }
  }
}

void ShiftBlocks(std::vector<double>& a_temp, std::vector<double>& b_temp, const MatrixParams& params) {
  std::vector<double> a_shifted(params.matrix_size * params.matrix_size);
  std::vector<double> b_shifted(params.matrix_size * params.matrix_size);

  for (int i = 0; i < params.param; ++i) {
    for (int j = 0; j < params.param; ++j) {
      const int new_j = (j - 1 + params.param) % params.param;
      for (int ii = 0; ii < params.block_size; ++ii) {
        for (int jj = 0; jj < params.block_size; ++jj) {
          const int orig_idx = (((i * params.block_size) + ii) * params.matrix_size) + ((j * params.block_size) + jj);
          const int new_idx =
              (((i * params.block_size) + ii) * params.matrix_size) + ((new_j * params.block_size) + jj);
          a_shifted[new_idx] = a_temp[orig_idx];
        }
      }
    }
  }

  for (int i = 0; i < params.param; ++i) {
    for (int j = 0; j < params.param; ++j) {
      const int new_i = (i - 1 + params.param) % params.param;
      for (int ii = 0; ii < params.block_size; ++ii) {
        for (int jj = 0; jj < params.block_size; ++jj) {
          const int orig_idx = (((i * params.block_size) + ii) * params.matrix_size) + ((j * params.block_size) + jj);
          const int new_idx =
              (((new_i * params.block_size) + ii) * params.matrix_size) + ((j * params.block_size) + jj);
          b_shifted[new_idx] = b_temp[orig_idx];
        }
      }
    }
  }

  a_temp = std::move(a_shifted);
  b_temp = std::move(b_shifted);
}

}  // namespace

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
  const int block_size = n / p;
  MatrixParams params{.matrix_size = n, .param = p, .block_size = block_size};
  std::vector<double> matrix_c(n * n, 0.0);
  std::vector<double> a_temp(n * n);
  std::vector<double> b_temp(n * n);

  if (a_temp.empty() || b_temp.empty()) {
    return std::vector<double>(n * n, 0.0);
  }

  InitialAlignment(mat_a, mat_b, a_temp, b_temp, params);

  for (int step = 0; step < params.param; ++step) {
    MultiplyBlocks(a_temp, b_temp, matrix_c, params);

    if (step < params.param - 1) {
      ShiftBlocks(a_temp, b_temp, params);
      if (a_temp.empty() || b_temp.empty()) {
        return std::vector<double>(n * n, 0.0);
      }
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