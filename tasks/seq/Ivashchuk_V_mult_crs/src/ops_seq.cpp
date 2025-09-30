#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::PreProcessingImpl() {
  const size_t input_size_a = task_data->inputs_count[0];
  const size_t input_size_b = task_data->inputs_count[1];

  auto* in_ptr_a = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<std::complex<double>*>(task_data->inputs[1]);

  std::vector<std::complex<double>> dense_a(in_ptr_a, in_ptr_a + input_size_a);
  std::vector<std::complex<double>> dense_b(in_ptr_b, in_ptr_b + input_size_b);

  const int size = static_cast<int>(std::sqrt(static_cast<double>(input_size_a)));

  ConvertToCRS(dense_a, size, size, &matrix_a_);
  ConvertToCRS(dense_b, size, size, &matrix_b_);

  return true;
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ValidationImpl() {
  return task_data->inputs_count.size() >= 2 && task_data->outputs_count.size() >= 1 &&
         task_data->inputs_count[0] == task_data->inputs_count[1];
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::RunImpl() {
  MultiplySparseMatrices(matrix_a_, matrix_b_, &result_matrix_);
  return true;
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::PostProcessingImpl() {
  std::vector<std::complex<double>> dense_result;
  ConvertFromCRS(result_matrix_, &dense_result);

  for (size_t i = 0; i < dense_result.size() && i < task_data->outputs_count[0]; ++i) {
    reinterpret_cast<std::complex<double>*>(task_data->outputs[0])[i] = dense_result[i];
  }

  return true;
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ConvertToCRS(const std::vector<std::complex<double>>& dense,
                                                                         const int rows, const int cols,
                                                                         SparseMatrixCRS* sparse) {
  sparse->rows_ = rows;
  sparse->cols_ = cols;
  sparse->row_pointers_.clear();
  sparse->row_pointers_.reserve(static_cast<size_t>(rows) + 1);
  sparse->values_.clear();
  sparse->col_indices_.clear();

  sparse->row_pointers_.push_back(0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const std::complex<double> value =
          dense[static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j)];
      if (std::abs(value.real()) > 1e-10 || std::abs(value.imag()) > 1e-10) {
        sparse->values_.push_back(value);
        sparse->col_indices_.push_back(j);
      }
    }
    sparse->row_pointers_.push_back(static_cast<int>(sparse->values_.size()));
  }
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::MultiplySparseMatrices(const SparseMatrixCRS& a,
                                                                                   const SparseMatrixCRS& b,
                                                                                   SparseMatrixCRS* c) {
  c->rows_ = a.rows_;
  c->cols_ = b.cols_;
  c->row_pointers_.clear();
  c->row_pointers_.reserve(static_cast<size_t>(c->rows_) + 1);
  c->values_.clear();
  c->col_indices_.clear();

  std::vector<std::complex<double>> temp(static_cast<size_t>(c->cols_), 0);
  std::vector<bool> temp_used(static_cast<size_t>(c->cols_), false);

  c->row_pointers_.push_back(0);

  for (int i = 0; i < a.rows_; ++i) {
    std::fill(temp.begin(), temp.end(), 0);
    std::fill(temp_used.begin(), temp_used.end(), false);

    for (int a_idx = a.row_pointers_[i]; a_idx < a.row_pointers_[i + 1]; ++a_idx) {
      const int k = a.col_indices_[a_idx];
      const std::complex<double> a_val = a.values_[a_idx];

      for (int b_idx = b.row_pointers_[k]; b_idx < b.row_pointers_[k + 1]; ++b_idx) {
        const int j = b.col_indices_[b_idx];
        const std::complex<double> b_val = b.values_[b_idx];
        temp[static_cast<size_t>(j)] += a_val * b_val;
        temp_used[static_cast<size_t>(j)] = true;
      }
    }

    for (int j = 0; j < c->cols_; ++j) {
      if (temp_used[static_cast<size_t>(j)] && (std::abs(temp[static_cast<size_t>(j)].real()) > 1e-10 ||
                                                std::abs(temp[static_cast<size_t>(j)].imag()) > 1e-10)) {
        c->values_.push_back(temp[static_cast<size_t>(j)]);
        c->col_indices_.push_back(j);
      }
    }

    c->row_pointers_.push_back(static_cast<int>(c->values_.size()));
  }
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ConvertFromCRS(const SparseMatrixCRS& sparse,
                                                                           std::vector<std::complex<double>>* dense) {
  dense->assign(static_cast<size_t>(sparse.rows_) * static_cast<size_t>(sparse.cols_), 0);

  for (int i = 0; i < sparse.rows_; ++i) {
    for (int idx = sparse.row_pointers_[i]; idx < sparse.row_pointers_[i + 1]; ++idx) {
      const int j = sparse.col_indices_[idx];
      (*dense)[static_cast<size_t>(i) * static_cast<size_t>(sparse.cols_) + static_cast<size_t>(j)] =
          sparse.values_[idx];
    }
  }
}