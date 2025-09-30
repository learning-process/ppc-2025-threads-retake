#include "seq/polyakov_a_mult_complex_matrix_CRS/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

polyakov_a_mult_complex_matrix_CRS_seq::MatrixCRS polyakov_a_mult_complex_matrix_CRS_seq::GetRandomMatrixCRS(
    size_t n, size_t m, size_t sparsity_coeff) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> real_dist(-1000.0, 1000.0);
  std::uniform_real_distribution<double> imag_dist(-1000.0, 1000.0);
  std::uniform_int_distribution<int> try_dist(0, 100);

  std::vector<std::complex<double>> values;
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.push_back(0);

  for (size_t i = 0; i < n; i++) {
    size_t nz_row = 0;
    for (size_t j = 0; j < m; j++) {
      if (try_dist(gen) <= sparsity_coeff) {
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

bool polyakov_a_mult_complex_matrix_CRS_seq::TestTaskSequential::PreProcessingImpl() {
  A = reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  B = reinterpret_cast<MatrixCRS *>(task_data->inputs[1]);
  c_rows = a_rows;
  c_cols = b_cols;

  C = new MatrixCRS(c_rows, c_cols);
  return true;
}

bool polyakov_a_mult_complex_matrix_CRS_seq::TestTaskSequential::ValidationImpl() {
  a_rows = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->rows_;
  a_cols = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->cols_;
  b_rows = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->rows_;
  b_cols = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->cols_;
  return a_cols == b_rows && a_rows != 0 && a_cols != 0 && b_cols != 0;
}

bool polyakov_a_mult_complex_matrix_CRS_seq::TestTaskSequential::RunImpl() {
  double eps = 1e-9;

  for (int r = 0; r < a_rows; r++) {                        // для каждой строки
    std::vector<std::complex<double>> temp_row(c_cols, 0);  // создаём вектор для временного хранения строки матрицы C

    for (int i = A->row_ptr[r]; i < A->row_ptr[r + 1]; i++) {  // проходимся по i-ой строке A, если она не нулевая
      std::complex<double> a_value = A->values[i];
      size_t k = A->col_ind[i];

      for (int j = B->row_ptr[k]; j < B->row_ptr[k + 1]; j++) {  // проходимся по соответсвующей строке B
        std::complex<double> b_value = B->values[j];
        int t = B->col_ind[j];
        temp_row[t] += a_value * b_value;
      }
    }

    for (int i = 0; i < c_cols; i++) {
      if (std::abs(temp_row[i]) > eps) {  // isComplexNoneZero
        C->values.push_back(temp_row[i]);
        C->col_ind.push_back(i);
      }
    }
    C->row_ptr.push_back(C->values.size());
  }
  return true;
}

bool polyakov_a_mult_complex_matrix_CRS_seq::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<MatrixCRS *>(task_data->outputs[0]) = *C;
  return true;
}
