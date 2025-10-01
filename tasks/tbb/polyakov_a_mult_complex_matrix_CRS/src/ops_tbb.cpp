#include "tbb/polyakov_a_mult_complex_matrix_CRS/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace pcrs = polyakov_a_mult_complex_matrix_crs_tbb;

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

  return {num_rows, num_cols, std::move(values), std::move(col_ind), std::move(row_ptr)};
}

bool polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB::PreProcessingImpl() {
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

bool polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB::ValidationImpl() {
  a_rows_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->rows;
  a_cols_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[0])->cols;
  b_rows_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->rows;
  b_cols_ = reinterpret_cast<MatrixCRS *>(task_data->inputs[1])->cols;
  return a_cols_ == b_rows_ && a_rows_ != 0 && a_cols_ != 0 && b_cols_ != 0;
}

// Функция для подсчёта ненулевых элементов в строке матрицы A
int polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB::CountNonZeroInRow(size_t r) {
  std::vector<char> local_marked(c_cols_, 0);
  for (auto &x : local_marked) {
    x = 0;
  }

  for (size_t i = a_->row_ptr[r]; i < a_->row_ptr[r + 1]; ++i) {
    size_t k = a_->col_ind[i];
    for (size_t j = b_->row_ptr[k]; j < b_->row_ptr[k + 1]; ++j) {
      size_t t = b_->col_ind[j];
      local_marked[t] = 1;
    }
  }

  return std::accumulate(local_marked.begin(), local_marked.end(), 0);
}

// Функция для вычисления произведения строк A и B и обновления local_temp
void polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB::ComputeRowProduct(
    size_t r, std::vector<std::complex<double>> &local_temp) {
  for (size_t i = a_->row_ptr[r]; i < a_->row_ptr[r + 1]; ++i) {
    std::complex<double> a_val = a_->values[i];
    size_t k = a_->col_ind[i];
    for (size_t j = b_->row_ptr[k]; j < b_->row_ptr[k + 1]; ++j) {
      std::complex<double> b_val = b_->values[j];
      size_t t = b_->col_ind[j];
      local_temp[t] += a_val * b_val;
    }
  }
}

bool polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB::RunImpl() {
  const double eps = 1e-9;

  // Подсчёт количества ненулевых в каждой строке
  std::vector<int> row_nnz(a_rows_, 0);

  // Подсчёт ненулевых элементов в строках матрицы A
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a_rows_), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t r = range.begin(); r < range.end(); ++r) {
      row_nnz[r] = CountNonZeroInRow(r);
    }
  });

  c_->row_ptr.resize(a_rows_ + 1);
  for (size_t r = 0; r < a_rows_; ++r) {
    c_->row_ptr[r + 1] = c_->row_ptr[r] + row_nnz[r];
  }

  size_t total_nnz = c_->row_ptr[a_rows_];
  c_->values.assign(total_nnz, std::complex<double>(0.0));
  c_->col_ind.assign(total_nnz, 0);

  // Вычисление результата умножения A * B и запись в матрицу C
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a_rows_), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t r = range.begin(); r < range.end(); ++r) {
      std::vector<std::complex<double>> local_temp(c_cols_, std::complex<double>{0.0});
      ComputeRowProduct(r, local_temp);

      size_t write_pos = c_->row_ptr[r];
      for (size_t j = 0; j < c_cols_; ++j) {
        if (std::abs(local_temp[j]) > eps) {
          c_->values[write_pos] = local_temp[j];
          c_->col_ind[write_pos] = j;
          write_pos++;
        }
      }
    }
  });

  return true;
}

bool polyakov_a_mult_complex_matrix_crs_tbb::TestTaskTBB::PostProcessingImpl() { return true; }
