#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::PreProcessingImpl() {
  unsigned int input_size_A = task_data->inputs_count[0];
  unsigned int input_size_B = task_data->inputs_count[1];

  auto* in_ptr_A = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);
  auto* in_ptr_B = reinterpret_cast<std::complex<double>*>(task_data->inputs[1]);

  std::vector<std::complex<double>> denseA(in_ptr_A, in_ptr_A + input_size_A);
  std::vector<std::complex<double>> denseB(in_ptr_B, in_ptr_B + input_size_B);

  int size = static_cast<int>(std::sqrt(input_size_A));

  ConvertToCRS(denseA, size, size, matrixA);
  ConvertToCRS(denseB, size, size, matrixB);

  return true;
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ValidationImpl() {
  return task_data->inputs_count.size() >= 2 && task_data->outputs_count.size() >= 1 &&
         task_data->inputs_count[0] == task_data->inputs_count[1];
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::RunImpl() {
  MultiplySparseMatrices(matrixA, matrixB, resultMatrix);
  return true;
}

bool Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::PostProcessingImpl() {
  std::vector<std::complex<double>> denseResult;
  ConvertFromCRS(resultMatrix, denseResult);

  for (size_t i = 0; i < denseResult.size() && i < task_data->outputs_count[0]; i++) {
    reinterpret_cast<std::complex<double>*>(task_data->outputs[0])[i] = denseResult[i];
  }

  return true;
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ConvertToCRS(const std::vector<std::complex<double>>& dense,
                                                                         int rows, int cols, SparseMatrixCRS& sparse) {
  sparse.rows = rows;
  sparse.cols = cols;
  sparse.row_pointers.clear();
  sparse.row_pointers.reserve(rows + 1);
  sparse.values.clear();
  sparse.col_indices.clear();

  sparse.row_pointers.push_back(0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::complex<double> value = dense[i * cols + j];
      if (std::abs(value.real()) > 1e-10 || std::abs(value.imag()) > 1e-10) {
        sparse.values.push_back(value);
        sparse.col_indices.push_back(j);
      }
    }
    sparse.row_pointers.push_back(static_cast<int>(sparse.values.size()));
  }
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::MultiplySparseMatrices(const SparseMatrixCRS& A,
                                                                                   const SparseMatrixCRS& B,
                                                                                   SparseMatrixCRS& C) {
  C.rows = A.rows;
  C.cols = B.cols;
  C.row_pointers.clear();
  C.row_pointers.reserve(C.rows + 1);
  C.values.clear();
  C.col_indices.clear();

  // Временный вектор для накопления результатов строки
  std::vector<std::complex<double>> temp(C.cols, 0);
  // Вектор для отслеживания использованных столбцов в текущей строке
  std::vector<bool> temp_used(C.cols, false);

  C.row_pointers.push_back(0);

  for (int i = 0; i < A.rows; ++i) {
    // Очищаем временный вектор
    std::fill(temp.begin(), temp.end(), 0);
    std::fill(temp_used.begin(), temp_used.end(), false);

    // Умножаем строку i матрицы A на матрицу B
    for (int a_idx = A.row_pointers[i]; a_idx < A.row_pointers[i + 1]; ++a_idx) {
      int k = A.col_indices[a_idx];
      std::complex<double> a_val = A.values[a_idx];

      // Умножаем на соответствующие элементы строки k матрицы B
      for (int b_idx = B.row_pointers[k]; b_idx < B.row_pointers[k + 1]; ++b_idx) {
        int j = B.col_indices[b_idx];
        std::complex<double> b_val = B.values[b_idx];
        temp[j] += a_val * b_val;
        temp_used[j] = true;
      }
    }

    // Добавляем ненулевые элементы из временного вектора в результат
    for (int j = 0; j < C.cols; ++j) {
      if (temp_used[j] && (std::abs(temp[j].real()) > 1e-10 || std::abs(temp[j].imag()) > 1e-10)) {
        C.values.push_back(temp[j]);
        C.col_indices.push_back(j);
      }
    }

    C.row_pointers.push_back(static_cast<int>(C.values.size()));
  }
}

void Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier::ConvertFromCRS(const SparseMatrixCRS& sparse,
                                                                           std::vector<std::complex<double>>& dense) {
  dense.assign(sparse.rows * sparse.cols, 0);

  for (int i = 0; i < sparse.rows; ++i) {
    for (int idx = sparse.row_pointers[i]; idx < sparse.row_pointers[i + 1]; ++idx) {
      int j = sparse.col_indices[idx];
      dense[i * sparse.cols + j] = sparse.values[idx];
    }
  }
}