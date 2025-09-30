#include "omp/polyakov_a_mult_complex_matrix_CRS/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

polyakov_a_mult_complex_matrix_CRS_omp::MatrixCRS polyakov_a_mult_complex_matrix_CRS_omp::GetRandomMatrixCRS(
    size_t n, size_t m, size_t sparsity_coeff) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> real_dist(-1000.0, 1000.0);
  std::uniform_real_distribution<double> imag_dist(-1000.0, 1000.0);
  std::uniform_int_distribution<int> try_dist(1, 100);

  std::vector<std::complex<double>> values;
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.push_back(0);

  for (size_t i = 0; i < n; i++) {
    size_t nz_row = 0;
    for (size_t j = 0; j < m; j++) {
      if (static_cast<size_t>(try_dist(gen)) <= sparsity_coeff) {
        values.emplace_back(real_dist(gen), imag_dist(gen));
        col_ind.push_back(j);
        nz_row++;
      }
    }
    row_ptr.push_back(row_ptr.back() + nz_row);
  }

  MatrixCRS Matrix(n, m, values, col_ind, row_ptr);

  return Matrix;
}

bool polyakov_a_mult_complex_matrix_CRS_omp::TestTaskOMP::PreProcessingImpl() {
  a_rows = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->rows_;
  a_cols = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->cols_;
  b_rows = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->rows_;
  b_cols = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->cols_;

  A = reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  B = reinterpret_cast<MatrixCRS *>(task_data->inputs[1]);
  C = reinterpret_cast<MatrixCRS *>(task_data->outputs[0]);
  c_rows = a_rows;
  c_cols = b_cols;

  return true;
}

bool polyakov_a_mult_complex_matrix_CRS_omp::TestTaskOMP::ValidationImpl() {
  a_rows = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->rows_;
  a_cols = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->cols_;
  b_rows = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->rows_;
  b_cols = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->cols_;
  return a_cols == b_rows && a_rows != 0 && a_cols != 0 && b_cols != 0;
}

bool polyakov_a_mult_complex_matrix_CRS_omp::TestTaskOMP::RunImpl() {
  const double eps = 1e-9;

  // подсчёт количества ненулевых в каждой строке
  int a_rows_int = static_cast<int>(a_rows);
  std::vector<int> row_nnz(a_rows, 0);

#pragma omp parallel
  {
    std::vector<bool> local_marked(c_cols, 0);
#pragma omp for
    for (int r = 0; r < a_rows_int; r++) {
      std::fill(local_marked.begin(), local_marked.end(), 0);
      for (size_t i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++) {
        size_t k = A->col_ind[i];
        for (size_t j = B->row_ptr[k]; j < B->row_ptr[k + 1]; j++) {
          size_t t = B->col_ind[j];
          local_marked[t] = 1;
        }
      }
      row_nnz[r] = std::accumulate(local_marked.begin(), local_marked.end(), 0);
    }
  }

  C->row_ptr.resize(a_rows + 1);
  for (size_t r = 0; r < a_rows; r++) {
    C->row_ptr[r + 1] = C->row_ptr[r] + row_nnz[r];
  }

  size_t total_nnz = C->row_ptr[a_rows];
  C->values.assign(total_nnz, 0.0);
  C->col_ind.assign(total_nnz, 0);

  // Вычисление результата
#pragma omp parallel
  {
    std::vector<std::complex<double>> local_temp(c_cols);
#pragma omp for
    for (int r = 0; r < a_rows_int; r++) {
      std::fill(local_temp.begin(), local_temp.end(), std::complex<double>(0.0));

      // Умножение строки r матрицы A на B
      for (size_t i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++) {
        std::complex<double> a_val = A->values[i];
        size_t k = A->col_ind[i];
        for (size_t j = B->row_ptr[k]; j < B->row_ptr[k + 1]; j++) {
          std::complex<double> b_val = B->values[j];
          size_t t = B->col_ind[j];
          local_temp[t] += a_val * b_val;
        }
      }

      size_t write_pos = C->row_ptr[r];
      for (size_t j = 0; j < c_cols; j++) {
        if (std::abs(local_temp[j]) > eps) {
          C->values[write_pos] = local_temp[j];
          C->col_ind[write_pos] = static_cast<size_t>(j);
          write_pos++;
        }
      }
    }
  }

  return true;
}

bool polyakov_a_mult_complex_matrix_CRS_omp::TestTaskOMP::PostProcessingImpl() { return true; }
