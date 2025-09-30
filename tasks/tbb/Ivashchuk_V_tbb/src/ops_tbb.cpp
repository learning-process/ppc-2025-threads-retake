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

void ComputeRowProduct(const std::vector<std::complex<double>>& a_values, const std::vector<int>& a_col_indices,
                       int a_row_start, int a_row_end, const std::vector<std::complex<double>>& b_values,
                       const std::vector<int>& b_col_indices, const std::vector<int>& b_row_ptr,
                       std::vector<std::complex<double>>& temp_row) {
  for (int idx = a_row_start; idx < a_row_end; ++idx) {
    int col_a = a_col_indices[idx];
    const std::complex<double>& val_a = a_values[idx];

    int b_row_start_local = b_row_ptr[col_a];
    int b_row_end_local = b_row_ptr[col_a + 1];

    for (int inner_idx = b_row_start_local; inner_idx < b_row_end_local; ++inner_idx) {
      int col_b = b_col_indices[inner_idx];
      temp_row[col_b] += val_a * b_values[inner_idx];
    }
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

  tbb::parallel_for(tbb::blocked_range<int>(0, a.rows), [&](const tbb::blocked_range<int>& range) {
    std::vector<std::complex<double>> local_values;
    std::vector<int> local_col_indices;
    std::vector<int> local_row_ptr;

    for (int i = range.begin(); i != range.end(); ++i) {
      std::vector<std::complex<double>> temp_row(b.cols, 0.0);

      int a_row_start = a.row_ptr[i];
      int a_row_end = a.row_ptr[i + 1];

      ComputeRowProduct(a.values, a.col_indices, a_row_start, a_row_end, b.values, b.col_indices, b.row_ptr, temp_row);

      for (int j = 0; j < b.cols; ++j) {
        if (std::abs(temp_row[j]) > 1e-10) {
          local_values.push_back(temp_row[j]);
          local_col_indices.push_back(j);
        }
      }
      local_row_ptr.push_back(static_cast<int>(local_values.size()));
    }

#pragma omp critical
    {
      c.values.insert(c.values.end(), local_values.begin(), local_values.end());
      c.col_indices.insert(c.col_indices.end(), local_col_indices.begin(), local_col_indices.end());
      if (c.row_ptr.size() == 1) {
        for (int ptr : local_row_ptr) {
          c.row_ptr.push_back(ptr);
        }
      } else {
        int last_ptr = c.row_ptr.back();
        for (int ptr : local_row_ptr) {
          c.row_ptr.push_back(last_ptr + ptr);
        }
      }
    }
  });
}