#include "seq/guseynov_e_sparse_matrix_multiply_crs/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace guseynov_e_sparse_matrix_multiply_crs {

CRSMatrix T(const CRSMatrix& M) {
  CRSMatrix temp_matrix;
  temp_matrix.n_rows = M.n_cols;
  temp_matrix.n_cols = M.n_rows;
  temp_matrix.pointer.assign(temp_matrix.n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(temp_matrix.n_rows);
  for (int i = 0; i < M.n_rows; i++) {
    for (int k = M.pointer[i]; k < M.pointer[i + 1]; k++) {
      int j = M.col_indexes[k];
      temp[j].emplace_back(i, M.non_zero_values[k]);
    }
  }

  for (int i = 0; i < temp_matrix.n_rows; i++) {
    temp_matrix.pointer[i + 1] = temp_matrix.pointer[i];
    for (auto& j : temp[i]) {
      temp_matrix.col_indexes.push_back(j.first);
      temp_matrix.non_zero_values.push_back(j.second);
      temp_matrix.pointer[i + 1]++;
    }
  }

  return temp_matrix;
}

bool is_crs(const CRSMatrix& M) {
  if (M.pointer.size() != size_t(M.n_rows + 1)) return false;
  int non_zero_elems_count = M.non_zero_values.size();
  if (M.col_indexes.size() != size_t(non_zero_elems_count) || M.pointer[M.n_rows] != non_zero_elems_count) return false;
  if (M.pointer[0] != 0) return false;
  for (int i = 1; i <= M.n_rows; i++) {
    if (M.pointer[i] < M.pointer[i - 1]) return false;
  }
  for (int i = 0; i < non_zero_elems_count; i++) {
    if (M.col_indexes[i] < 0 || M.col_indexes[i] >= M.n_cols) return false;
  }
  return true;
}

}  // namespace guseynov_e_sparse_matrix_multiply_crs

bool guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial::PreProcessingImpl() {
  A_mat_ = reinterpret_cast<CRSMatrix*>(task_data->inputs[0]);
  B_mat_ = reinterpret_cast<CRSMatrix*>(task_data->inputs[1]);
  Result_ = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  return true;
}

bool guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial::ValidationImpl() {
  if (task_data->inputs.size() != 2 || task_data->outputs.size() != 1 || !task_data->inputs_count.empty() ||
      !task_data->outputs_count.empty())
    return false;

  auto* A = reinterpret_cast<CRSMatrix*>(task_data->inputs[0]);
  auto* B = reinterpret_cast<CRSMatrix*>(task_data->inputs[1]);
  auto* R = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  if (A == nullptr || B == nullptr || R == nullptr) return false;
  if (!is_crs(*A) || !is_crs(*B)) return false;
  if (A->n_cols != B->n_rows) return false;

  return true;
}
bool guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial::RunImpl() {
  *B_mat_ = T(*B_mat_);

  Result_->n_rows = A_mat_->n_rows;
  Result_->n_cols = B_mat_->n_rows;
  Result_->pointer.assign(Result_->n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(Result_->n_rows);

  for (int i = 0; i < Result_->n_rows; i++) {
    for (int j = 0; j < B_mat_->n_rows; j++) {
      double sum = 0.0;
      for (int k_A = A_mat_->pointer[i]; k_A < A_mat_->pointer[i + 1]; k_A++) {
        for (int k_B = B_mat_->pointer[j]; k_B < B_mat_->pointer[j + 1]; k_B++) {
          if (A_mat_->col_indexes[k_A] == B_mat_->col_indexes[k_B]) {
            sum += A_mat_->non_zero_values[k_A] * B_mat_->non_zero_values[k_B];
          }
        }
      }
      if (std::abs(sum) > 1e-12) {  // отсекаем нули
        temp[i].emplace_back(j, sum);
      }
    }
  }

  for (int i = 0; i < Result_->n_rows; i++) {
    Result_->pointer[i + 1] = Result_->pointer[i];
    for (auto& j : temp[i]) {
      Result_->col_indexes.push_back(j.first);
      Result_->non_zero_values.push_back(j.second);
      Result_->pointer[i + 1]++;
    }
  }
  return true;
}

bool guseynov_e_sparse_matrix_multiply_crs::SparseMatMultSequantial::PostProcessingImpl() {
  CRSMatrix* output = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  output->n_rows = Result_->n_rows;
  output->n_cols = Result_->n_cols;
  output->pointer = Result_->pointer;
  output->col_indexes = Result_->col_indexes;
  output->non_zero_values = Result_->non_zero_values;

  return true;
}
