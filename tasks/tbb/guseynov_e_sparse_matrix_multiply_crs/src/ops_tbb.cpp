#include "tbb/guseynov_e_sparse_matrix_multiply_crs/include/ops_tbb.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <tbb/mutex.h>

#include "oneapi/tbb/parallel_for.h"

namespace guseynov_e_sparse_matrix_multiply_crs_tbb {

namespace {
CRSMatrix T(const CRSMatrix& m) {
  CRSMatrix temp_matrix;
  temp_matrix.n_rows = m.n_cols;
  temp_matrix.n_cols = m.n_rows;
  temp_matrix.pointer.assign(temp_matrix.n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(temp_matrix.n_rows);
  for (int i = 0; i < m.n_rows; i++) {
    for (int k = m.pointer[i]; k < m.pointer[i + 1]; k++) {
      int j = m.col_indexes[k];
      temp[j].emplace_back(i, m.non_zero_values[k]);
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

bool IsCrs(const CRSMatrix& m) {
  if (m.pointer.size() != static_cast<size_t>(m.n_rows) + 1) {
    return false;
  }

  size_t non_zero_elems_count = m.non_zero_values.size();
  if (m.col_indexes.size() != non_zero_elems_count ||
      static_cast<size_t>(m.pointer[m.n_rows]) != non_zero_elems_count) {
    return false;
  }

  if (m.pointer[0] != 0) {
    return false;
  }

  for (int i = 1; i <= m.n_rows; i++) {
    if (m.pointer[i] < m.pointer[i - 1]) {
      return false;
    }
  }
  for (size_t i = 0; i < non_zero_elems_count; i++) {
    if (m.col_indexes[i] < 0 || m.col_indexes[i] >= m.n_cols) {
      return false;
    }
  }
  return true;
}
}  // namespace

bool guseynov_e_sparse_matrix_multiply_crs_tbb::SparseMatMultTBB::PreProcessingImpl() {
  A_mat_ = reinterpret_cast<CRSMatrix*>(task_data->inputs[0]);
  B_mat_ = reinterpret_cast<CRSMatrix*>(task_data->inputs[1]);
  Result_ = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  return true;
}

bool guseynov_e_sparse_matrix_multiply_crs_tbb::SparseMatMultTBB::ValidationImpl() {
  if (task_data->inputs.size() != 2 || task_data->outputs.size() != 1 || !task_data->inputs_count.empty() ||
      !task_data->outputs_count.empty()) {
    return false;
  }

  auto* a = reinterpret_cast<CRSMatrix*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<CRSMatrix*>(task_data->inputs[1]);
  auto* r = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  if (a == nullptr || b == nullptr || r == nullptr) {
    return false;
  }

  if (!IsCrs(*a) || !IsCrs(*b)) {
    return false;
  }

  if (a->n_cols != b->n_rows) {
    return false;
  }

  return true;
}

bool guseynov_e_sparse_matrix_multiply_crs_tbb::SparseMatMultTBB::RunImpl() {
  *B_mat_ = T(*B_mat_);

  Result_->n_rows = A_mat_->n_rows;
  Result_->n_cols = B_mat_->n_rows;
  Result_->pointer.assign(Result_->n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(Result_->n_rows);
  
  std::vector<tbb::mutex> mutexes(Result_->n_rows);

  tbb::parallel_for(tbb::blocked_range<int>(0, Result_->n_rows),
    [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        for (int j = 0; j < B_mat_->n_rows; j++) {
          double sum = 0.0;
          for (int k_a = A_mat_->pointer[i]; k_a < A_mat_->pointer[i + 1]; k_a++) {
            for (int k_b = B_mat_->pointer[j]; k_b < B_mat_->pointer[j + 1]; k_b++) {
              if (A_mat_->col_indexes[k_a] == B_mat_->col_indexes[k_b]) {
                sum += A_mat_->non_zero_values[k_a] * B_mat_->non_zero_values[k_b];
              }
            }
          }
          if (std::abs(sum) > 1e-12) {  // отсекаем нули
            tbb::mutex::scoped_lock lock(mutexes[i]);
            temp[i].emplace_back(j, sum);
          }
        }
      }
    });

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



bool guseynov_e_sparse_matrix_multiply_crs_tbb::SparseMatMultTBB::PostProcessingImpl() {
  auto* output = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  output->n_rows = Result_->n_rows;
  output->n_cols = Result_->n_cols;
  output->pointer = Result_->pointer;
  output->col_indexes = Result_->col_indexes;
  output->non_zero_values = Result_->non_zero_values;

  return true;
}
}  // namespace guseynov_e_sparse_matrix_multiply_crs_tbb