#include "ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <cmath>

namespace {

void ConvertDenseToCRS(const std::vector<std::complex<double>>& dense, int rows, int cols,
                       std::vector<std::complex<double>>& values, std::vector<int>& col_indices,
                       std::vector<int>& row_ptr) {
  row_ptr.clear();
  values.clear();
  col_indices.clear();

  row_ptr.push_back(0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const auto& elem = dense[(i * cols) + j];
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
  for (int k = a_row_start; k < a_row_end; ++k) {
    int col_a = a_col_indices[k];
    const std::complex<double>& val_a = a_values[k];

    int b_row_start = b_row_ptr[col_a];
    int b_row_end = b_row_ptr[col_a + 1];

    for (int l = b_row_start; l < b_row_end; ++l) {
      int col_b = b_col_indices[l];
      temp_row[col_b] += val_a * b_values[l];
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

  std::copy(in_ptr, in_ptr + matrix_size * matrix_size, dense_a.begin());
  std::copy(in_ptr + matrix_size * matrix_size, in_ptr + 2 * matrix_size * matrix_size, dense_b.begin());

  ConvertToCRS(dense_a, matrix_size, matrix_size, matrix_a_);
  ConvertToCRS(dense_b, matrix_size, matrix_size, matrix_b_);

  result_.rows_ = matrix_size;
  result_.cols_ = matrix_size;

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
  int size = result_.rows_ * result_.cols_;
  std::fill(out_ptr, out_ptr + size, std::complex<double>(0.0));

  for (int i = 0; i < result_.rows_; ++i) {
    int row_start = result_.row_ptr_[i];
    int row_end = result_.row_ptr_[i + 1];

    for (int j = row_start; j < row_end; ++j) {
      int col = result_.col_indices_[j];
      out_ptr[(i * result_.cols_) + col] = result_.values_[j];
    }
  }

  return true;
}

void ivashchuk_v_tbb::SparseMatrixComplexCRS::ConvertToCRS(const std::vector<std::complex<double>>& dense, int rows,
                                                           int cols, CRSMatrix& crs) {
  crs.rows_ = rows;
  crs.cols_ = cols;
  ConvertDenseToCRS(dense, rows, cols, crs.values_, crs.col_indices_, crs.row_ptr_);
}

void ivashchuk_v_tbb::SparseMatrixComplexCRS::SparseMatMul(const CRSMatrix& a, const CRSMatrix& b, CRSMatrix& c) {
  c.rows_ = a.rows_;
  c.cols_ = b.cols_;
  c.row_ptr_.clear();
  c.values_.clear();
  c.col_indices_.clear();

  c.row_ptr_.push_back(0);

  tbb::parallel_for(tbb::blocked_range<int>(0, a.rows_), [&](const tbb::blocked_range<int>& range) {
    std::vector<std::complex<double>> local_values;
    std::vector<int> local_col_indices;
    std::vector<int> local_row_ptr;

    for (int i = range.begin(); i != range.end(); ++i) {
      std::vector<std::complex<double>> temp_row(b.cols_, 0.0);

      int a_row_start = a.row_ptr_[i];
      int a_row_end = a.row_ptr_[i + 1];

      ComputeRowProduct(a.values_, a.col_indices_, a_row_start, a_row_end, b.values_, b.col_indices_, b.row_ptr_,
                        temp_row);

      int non_zero_count = 0;
      for (int j = 0; j < b.cols_; ++j) {
        if (std::abs(temp_row[j]) > 1e-10) {
          local_values.push_back(temp_row[j]);
          local_col_indices.push_back(j);
          non_zero_count++;
        }
      }
      local_row_ptr.push_back(static_cast<int>(local_values.size()));
    }

#pragma omp critical
    {
      c.values_.insert(c.values_.end(), local_values.begin(), local_values.end());
      c.col_indices_.insert(c.col_indices_.end(), local_col_indices.begin(), local_col_indices.end());
      if (c.row_ptr_.size() == 1) {
        for (size_t ptr : local_row_ptr) {
          c.row_ptr_.push_back(ptr);
        }
      } else {
        int last_ptr = c.row_ptr_.back();
        for (size_t ptr : local_row_ptr) {
          c.row_ptr_.push_back(last_ptr + ptr);
        }
      }
    }
  });
}