#include "seq/polyakov_a_mult_complex_matrix_CRS/include/ops_seq.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace pcrs = polyakov_a_mult_complex_matrix_crs_seq;

pcrs::MatrixCRS pcrs::GetRandomMatrixCRS(pcrs::Rows num_rows, pcrs::Cols num_cols, size_t sparsity_coeff) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> real_dist(-1000.0, 1000.0);
  std::uniform_real_distribution<double> imag_dist(-1000.0, 1000.0);
  std::uniform_int_distribution<int> try_dist(1, 100);

  std::vector<std::complex<double>> values;
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.push_back(0);

  for (size_t i = 0; i < num_rows.value; ++i) {
    size_t nz_row = 0;
    for (size_t j = 0; j < num_cols.value; ++j) {
      if (static_cast<size_t>(try_dist(gen)) <= sparsity_coeff) {
        values.emplace_back(real_dist(gen), imag_dist(gen));
        col_ind.push_back(j);
        nz_row++;
      }
    }
    row_ptr.push_back(row_ptr.back() + nz_row);
  }

  return pcrs::MatrixCRS(num_rows, num_cols, std::move(values), std::move(col_ind), std::move(row_ptr));
}

bool polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential::PreProcessingImpl() {
  a_rows_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->rows;
  a_cols_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->cols;
  b_rows_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->rows;
  b_cols_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->cols;

  a_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  b_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1]);
  c_ = reinterpret_cast<MatrixCRS *>(task_data->outputs[0]);
  c_rows_ = a_rows_;
  c_cols_ = b_cols_;

  return true;
}

bool polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential::ValidationImpl() {
  a_rows_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->rows;
  a_cols_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->cols;
  b_rows_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->rows;
  b_cols_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->cols;
  return a_cols_ == b_rows_ && a_rows_ != 0 && a_cols_ != 0 && b_cols_ != 0;
}

bool polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential::RunImpl() {
  double eps = 1e-9;

  for (size_t r = 0; r < a_rows_; r++) {
    std::vector<std::complex<double>> temp_row(c_cols_, 0);  // создаём вектор для временного хранения строки матрицы c_

    for (size_t i = a_->row_ptr[r]; i < a_->row_ptr[r + 1]; i++) {  // проходимся по i-ой строке a_, если она не нулевая
      std::complex<double> a_value = a_->values[i];
      size_t k = a_->col_ind[i];

      for (size_t j = b_->row_ptr[k]; j < b_->row_ptr[k + 1]; j++) {  // проходимся по соответсвующей строке b_
        std::complex<double> b_value = b_->values[j];
        size_t t = b_->col_ind[j];
        temp_row[t] += a_value * b_value;
      }
    }

    for (size_t i = 0; i < c_cols_; i++) {
      if (std::abs(temp_row[i]) > eps) {  // isComplexNoneZero
        c_->values.push_back(temp_row[i]);
        c_->col_ind.push_back(i);
      }
    }
    c_->row_ptr.push_back(c_->values.size());
  }
  return true;
}

bool polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential::PostProcessingImpl() { return true; }
