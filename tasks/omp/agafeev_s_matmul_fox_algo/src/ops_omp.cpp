#include "omp/agafeev_s_matmul_fox_algo/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void agafeev_s_matmul_fox_algo_omp::BlockMultiply(const std::vector<double> &matr_a, unsigned long row,
                                                  const std::vector<double> &matr_b, unsigned long col,
                                                  std::vector<double> &matr_res, size_t block_index, size_t block_size,
                                                  size_t n) {
  size_t block_h = std::min(block_size, n - (row * block_size));
  size_t block_w = std::min(block_size, n - (col * block_size));
  size_t block_k = std::min(block_size, n - (block_index * block_size));

  double *matr_res_ptr = &matr_res[((row * block_size) * n) + (col * block_size)];
  const double *matr_a_ptr = &matr_a[((row * block_size) * n) + (block_index * block_size)];
  const double *matr_b_ptr = &matr_b[((block_index * block_size) * n) + (col * block_size)];

  for (size_t ii = 0; ii < block_h; ii++) {
    for (size_t jj = 0; jj < block_w; jj++) {
      double sum = 0.0;
      const double *matr_a_row = matr_a_ptr + (ii * n);
      const double *matr_b_col = matr_b_ptr + jj;
      for (size_t kk = 0; kk < block_k - 1; kk += 2) {
        sum += matr_a_row[kk] * matr_b_col[kk * n] + matr_a_row[kk + 1] * matr_b_col[(kk + 1) * n];
      }
      if (block_k % 2 != 0) {
        sum += matr_a_row[block_k - 1] * matr_b_col[(block_k - 1) * n];
      }

      matr_res_ptr[(ii * n) + jj] += sum;
    }
  }
}

bool agafeev_s_matmul_fox_algo_omp::MultiplMatrixOpenMP::PreProcessingImpl() {
  first_input_.clear();
  second_input_.clear();
  result_.clear();
  size_ = static_cast<int>(task_data->inputs_count[0]);
  auto *temp_ptr1 = reinterpret_cast<double *>(task_data->inputs[0]);
  first_input_.insert(first_input_.begin(), temp_ptr1, temp_ptr1 + (size_ * size_));
  auto *temp_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  second_input_.insert(second_input_.begin(), temp_ptr2, temp_ptr2 + (size_ * size_));
  block_size_ = task_data->inputs_count[1];
  result_.resize(size_ * size_, 0.0);

  return true;
}

bool agafeev_s_matmul_fox_algo_omp::MultiplMatrixOpenMP::ValidationImpl() {
  return (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[0] &&
          task_data->inputs_count[0] * task_data->inputs_count[0] % task_data->inputs_count[1] == 0);
}

bool agafeev_s_matmul_fox_algo_omp::MultiplMatrixOpenMP::RunImpl() {
  size_t num_blocks = (size_ + block_size_ - 1) / block_size_;
#pragma omp parallel
  for (size_t step = 0; step < num_blocks; step++) {
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < num_blocks; i++) {
      size_t block_index = (i + step) % num_blocks;
      for (size_t j = 0; j < num_blocks; ++j) {
        BlockMultiply(first_input_, i, second_input_, j, result_, block_index, block_size_, size_);
      }
    }
  }
  return true;
}

bool agafeev_s_matmul_fox_algo_omp::MultiplMatrixOpenMP::PostProcessingImpl() {
  for (unsigned int i = 0; i < task_data->outputs_count[0]; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = result_[i];
  }
  return true;
}
