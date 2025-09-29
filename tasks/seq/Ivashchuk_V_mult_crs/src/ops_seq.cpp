#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::PreProcessingImpl() {
  const size_t input_size_a = taskData->inputs_count[0];
  const size_t input_size_b = taskData->inputs_count[1];

  auto* in_ptr_a = reinterpret_cast<std::complex<double>*>(taskData->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<std::complex<double>*>(taskData->inputs[1]);

  std::vector<std::complex<double>> dense_a(in_ptr_a, in_ptr_a + input_size_a);
  std::vector<std::complex<double>> dense_b(in_ptr_b, in_ptr_b + input_size_b);

  const int size = static_cast<int>(std::sqrt(input_size_a));

  ConvertToCRS(dense_a, size, size, &matrixA_);
  ConvertToCRS(dense_b, size, size, &matrixB_);

  return true;
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ValidationImpl() {
  return taskData->inputs_count.size() >= 2 && taskData->outputs_count.size() >= 1 &&
         taskData->inputs_count[0] == taskData->inputs_count[1];
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::RunImpl() {
  MultiplySparseMatrices(matrixA_, matrixB_, &resultMatrix_);
  return true;
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::PostProcessingImpl() {
  std::vector<std::complex<double>> dense_result;
  ConvertFromCRS(resultMatrix_, &dense_result);

  for (size_t i = 0; i < dense_result.size() && i < taskData->outputs_count[0]; ++i) {
    reinterpret_cast<std::complex<double>*>(taskData->outputs[0])[i] = dense_result[i];
  }

  return true;
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ConvertToCRS(const std::vector<std::complex<double>>& dense,
                                                                         const int rows, const int cols,
                                                                         SparseMatrixCRS* sparse) {
  sparse->rows = rows;
  sparse->cols = cols;
  sparse->row_pointers.clear();
  sparse->row_pointers.reserve(rows + 1);
  sparse->values.clear();
  sparse->col_indices.clear();

  sparse->row_pointers.push_back(0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const std::complex<double> value = dense[i * cols + j];
      if (std::abs(value.real()) > 1e-10 || std::abs(value.imag()) > 1e-10) {
        sparse->values.push_back(value);
        sparse->col_indices.push_back(j);
      }
    }
    sparse->row_pointers.push_back(static_cast<int>(sparse->values.size()));
  }
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::MultiplySparseMatrices(const SparseMatrixCRS& A,
                                                                                   const SparseMatrixCRS& B,
                                                                                   SparseMatrixCRS* C) {
  C->rows = A.rows;
  C->cols = B.cols;
  C->row_pointers.clear();
  C->row_pointers.reserve(C->rows + 1);
  C->values.clear();
  C->col_indices.clear();

  std::vector<std::complex<double>> temp(C->cols, 0);
  std::vector<bool> temp_used(C->cols, false);

  C->row_pointers.push_back(0);

  for (int i = 0; i < A.rows; ++i) {
    std::fill(temp.begin(), temp.end(), 0);
    std::fill(temp_used.begin(), temp_used.end(), false);

    for (int a_idx = A.row_pointers[i]; a_idx < A.row_pointers[i + 1]; ++a_idx) {
      const int k = A.col_indices[a_idx];
      const std::complex<double> a_val = A.values[a_idx];

      for (int b_idx = B.row_pointers[k]; b_idx < B.row_pointers[k + 1]; ++b_idx) {
        const int j = B.col_indices[b_idx];
        const std::complex<double> b_val = B.values[b_idx];
        temp[j] += a_val * b_val;
        temp_used[j] = true;
      }
    }

    for (int j = 0; j < C->cols; ++j) {
      if (temp_used[j] && (std::abs(temp[j].real()) > 1e-10 || std::abs(temp[j].imag()) > 1e-10)) {
        C->values.push_back(temp[j]);
        C->col_indices.push_back(j);
      }
    }

    C->row_pointers.push_back(static_cast<int>(C->values.size()));
  }
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ConvertFromCRS(const SparseMatrixCRS& sparse,
                                                                           std::vector<std::complex<double>>* dense) {
  dense->assign(sparse.rows * sparse.cols, 0);

  for (int i = 0; i < sparse.rows; ++i) {
    for (int idx = sparse.row_pointers[i]; idx < sparse.row_pointers[i + 1]; ++idx) {
      const int j = sparse.col_indices[idx];
      (*dense)[i * sparse.cols + j] = sparse.values[idx];
    }
  }
}