#include "../include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace {

void ConvertDenseToCRS(const std::vector<std::complex<double>>& dense, int num_rows, int num_cols,
                       std::vector<std::complex<double>>& values, std::vector<int>& col_indices,
                       std::vector<int>& row_ptr) {
  row_ptr.clear();
  values.clear();
  col_indices.clear();

  row_ptr.push_back(0);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      const auto& elem = dense[(i * num_cols) + j];
      if (std::abs(elem) > 1e-10) {
        values.push_back(elem);
        col_indices.push_back(j);
      }
    }
    row_ptr.push_back(static_cast<int>(values.size()));
  }
}

}  // namespace

bool ivashchuk_v_tbb::SparseMatrixComplexCRS::PreProcessingImpl() {
  unsigned int total_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);

  int matrix_size = static_cast<int>(std::sqrt(total_size / 2 / sizeof(std::complex<double>)));

  std::vector<std::complex<double>> dense_a(matrix_size * matrix_size);
  std::vector<std::complex<double>> dense_b(matrix_size * matrix_size);

  std::copy(in_ptr, in_ptr + (matrix_size * matrix_size), dense_a.begin());
  std::copy(in_ptr + (matrix_size * matrix_size), in_ptr + (2 * matrix_size * matrix_size), dense_b.begin());

  ConvertToCRS(dense_a, matrix_size, matrix_size, matrix_a_);
  ConvertToCRS(dense_b, matrix_size, matrix_size, matrix_b_);

  result_.rows = matrix_size;
  result_.cols = matrix_size;

  return true;
}

bool ivashchuk_v_tbb::SparseMatrixComplexCRS::ValidationImpl() {
  unsigned int total_input_size = task_data->inputs_count[0];
  unsigned int total_output_size = task_data->outputs_count[0];

  int matrix_size = static_cast<int>(std::sqrt(total_input_size / 2 / sizeof(std::complex<double>)));
  int expected_output_size = matrix_size * matrix_size;

  return total_input_size >= 2 * matrix_size * matrix_size * sizeof(std::complex<double>) &&
         total_output_size >= expected_output_size * sizeof(std::complex<double>);
}

bool ivashchuk_v_tbb::SparseMatrixComplexCRS::RunImpl() {
  SparseMatMul(matrix_a_, matrix_b_, result_);
  return true;
}

bool ivashchuk_v_tbb::SparseMatrixComplexCRS::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<std::complex<double>*>(task_data->outputs[0]);
  int size = result_.rows * result_.cols;
  std::fill(out_ptr, out_ptr + size, std::complex<double>(0.0));

  for (int i = 0; i < result_.rows; ++i) {
    int row_start = result_.row_ptr[i];
    int row_end = result_.row_ptr[i + 1];

    for (int j = row_start; j < row_end; ++j) {
      int col = result_.col_indices[j];
      out_ptr[(i * result_.cols) + col] = result_.values[j];
    }
  }

  return true;
}

void ivashchuk_v_tbb::SparseMatrixComplexCRS::ConvertToCRS(const std::vector<std::complex<double>>& dense, int num_rows,
                                                           int num_cols, CRSMatrix& crs) {
  crs.rows = num_rows;
  crs.cols = num_cols;
  ConvertDenseToCRS(dense, num_rows, num_cols, crs.values, crs.col_indices, crs.row_ptr);
}

void ivashchuk_v_tbb::SparseMatrixComplexCRS::SparseMatMul(const CRSMatrix& a, const CRSMatrix& b, CRSMatrix& c) {
  c.rows = a.rows;
  c.cols = b.cols;
  c.row_ptr.clear();
  c.values.clear();
  c.col_indices.clear();

  c.row_ptr.push_back(0);

  std::vector<std::vector<std::complex<double>>> temp_results(a.rows);
  std::vector<std::vector<int>> temp_cols(a.rows);

  tbb::parallel_for(tbb::blocked_range<int>(0, a.rows), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      std::vector<std::complex<double>> row_result(b.cols, 0.0);

      int a_row_start = a.row_ptr[i];
      int a_row_end = a.row_ptr[i + 1];

      for (int k = a_row_start; k < a_row_end; ++k) {
        int col_a = a.col_indices[k];
        const std::complex<double>& val_a = a.values[k];

        int b_row_start = b.row_ptr[col_a];
        int b_row_end = b.row_ptr[col_a + 1];

        for (int l = b_row_start; l < b_row_end; ++l) {
          int col_b = b.col_indices[l];
          row_result[col_b] += val_a * b.values[l];
        }
      }

      for (int j = 0; j < b.cols; ++j) {
        if (std::abs(row_result[j]) > 1e-10) {
          temp_results[i].push_back(row_result[j]);
          temp_cols[i].push_back(j);
        }
      }
    }
  });

  for (int i = 0; i < a.rows; ++i) {
    c.values.insert(c.values.end(), temp_results[i].begin(), temp_results[i].end());
    c.col_indices.insert(c.col_indices.end(), temp_cols[i].begin(), temp_cols[i].end());
    c.row_ptr.push_back(static_cast<int>(c.values.size()));
  }
}