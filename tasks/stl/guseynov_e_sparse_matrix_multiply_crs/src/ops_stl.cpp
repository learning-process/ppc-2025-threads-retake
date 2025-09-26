#include "stl/guseynov_e_sparse_matrix_multiply_crs/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace guseynov_e_sparse_matrix_multiply_crs_stl {

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
void MultiplyRowSTL(std::size_t i, const guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix* a,
                    const guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix* b,
                    std::vector<std::vector<std::pair<int, double>>>& temp) {
  for (int j = 0; j < b->n_rows; ++j) {
    double sum = 0.0;
    for (int k_a = a->pointer[i]; k_a < a->pointer[i + 1]; ++k_a) {
      for (int k_b = b->pointer[j]; k_b < b->pointer[j + 1]; ++k_b) {
        if (a->col_indexes[k_a] == b->col_indexes[k_b]) {
          sum += a->non_zero_values[k_a] * b->non_zero_values[k_b];
        }
      }
    }
    if (std::abs(sum) > 1e-12) {
      temp[i].emplace_back(j, sum);
    }
  }
}
}  // namespace

bool guseynov_e_sparse_matrix_multiply_crs_stl::SparseMatMultSTL::PreProcessingImpl() {
  A_mat_ = reinterpret_cast<CRSMatrix*>(task_data->inputs[0]);
  B_mat_ = reinterpret_cast<CRSMatrix*>(task_data->inputs[1]);
  Result_ = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  return true;
}

bool guseynov_e_sparse_matrix_multiply_crs_stl::SparseMatMultSTL::ValidationImpl() {
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

bool guseynov_e_sparse_matrix_multiply_crs_stl::SparseMatMultSTL::RunImpl() {
  *B_mat_ = T(*B_mat_);

  Result_->n_rows = A_mat_->n_rows;
  Result_->n_cols = B_mat_->n_rows;
  Result_->pointer.assign(Result_->n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(Result_->n_rows);

  auto n_threads = ppc::util::GetPPCNumThreads();
  if (n_threads == 0){
    n_threads = 4;
  }

  const std::size_t rows = static_cast<std::size_t>(Result_->n_rows);
  const std::size_t threads_sz = static_cast<std::size_t>(n_threads);
  const std::size_t rows_per_thread = (rows + threads_sz - 1) / threads_sz;

  auto worker = [&](std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      MultiplyRowSTL(i, A_mat_, B_mat_, temp);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (size_t t = 0; t < n_threads; ++t) {
    const std::size_t start = static_cast<std::size_t>(t) * rows_per_thread;
    const std::size_t end = std::min(rows, (static_cast<std::size_t>(t) + 1u) * rows_per_thread);
    if (start < end) {
      threads.emplace_back(worker, start, end);
    }
  }

  for (auto& th : threads) {
    th.join();
  }

  for (std::size_t i = 0; i < rows; ++i) {
    Result_->pointer[i + 1] = Result_->pointer[i];
    for (auto& p : temp[i]) {
      Result_->col_indexes.push_back(p.first);
      Result_->non_zero_values.push_back(p.second);
      Result_->pointer[i + 1]++;
    }
  }

  return true;
}

bool guseynov_e_sparse_matrix_multiply_crs_stl::SparseMatMultSTL::PostProcessingImpl() {
  auto* output = reinterpret_cast<CRSMatrix*>(task_data->outputs[0]);

  output->n_rows = Result_->n_rows;
  output->n_cols = Result_->n_cols;
  output->pointer = Result_->pointer;
  output->col_indexes = Result_->col_indexes;
  output->non_zero_values = Result_->non_zero_values;

  return true;
}
}  // namespace guseynov_e_sparse_matrix_multiply_crs_stl