#include "seq/agafeev_s_matmul_fox_algo/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void agafeev_s_matmul_fox_algo_seq::BlockMultiply(const std::vector<double> &matr_a, const std::vector<double> &matr_b,
                                                  std::vector<double> &matr_res, size_t i, size_t j, size_t block_index,
                                                  size_t block_size, size_t n) {
  size_t block_h = std::min(block_size, n - (i * block_size));
  size_t block_w = std::min(block_size, n - (j * block_size));
  for (size_t ii = 0; ii < block_h; ii++) {
    for (size_t jj = 0; jj < block_w; jj++) {
      double sum = 0.0;
      for (size_t kk = 0; kk < std::min(block_size, n - (block_index * block_size)); kk++) {
        size_t row_a = (i * block_size) + ii;
        size_t col_a = (block_index * block_size) + kk;
        size_t row_b = (block_index * block_size) + kk;
        size_t col_b = (j * block_size) + jj;
        if (row_a < n && col_a < n && row_b < n && col_b < n)
          sum += matr_a[(row_a * n) + col_a] * matr_b[(row_b * n) + col_b];
      }
      size_t row_c = (i * block_size) + ii;
      size_t col_c = (j * block_size) + jj;
      if (row_c < n && col_c < n) matr_res[(row_c * n) + col_c] += sum;
    }
  }
}

bool agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental::PreProcessingImpl() {
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

bool agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental::ValidationImpl() {
  return (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[0] &&
          task_data->inputs_count[0] * task_data->inputs_count[0] % task_data->inputs_count[1] == 0);
}

bool agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental::RunImpl() {
  size_t num_blocks = (size_ + block_size_ - 1) / block_size_;
  for (size_t step = 0; step < num_blocks; step++) {
    for (size_t i = 0; i < num_blocks; i++) {
      size_t block_index = (i + step) % num_blocks;
      for (size_t j = 0; j < num_blocks; ++j)
        BlockMultiply(first_input_, second_input_, result_, i, j, block_index, block_size_, size_);
    }
  }
  return true;
}

bool agafeev_s_matmul_fox_algo_seq::MultiplMatrixSequental::PostProcessingImpl() {
  for (unsigned int i = 0; i < task_data->outputs_count[0]; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = result_[i];
  }
  return true;
}
