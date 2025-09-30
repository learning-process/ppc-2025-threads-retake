#include "ivashchuk_v_tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

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
      if (std::abs(elem) > 1e-10) {  // Non-zero element
        values.push_back(elem);
        col_indices.push_back(j);
      }
    }
    row_ptr.push_back(static_cast<int>(values.size()));
  }
}

}  // namespace

bool ivashchuk_v_tbb::SparseMatrixComplexCRS::PreProcessingImpl() {
  // Assuming input data contains two matrices in dense format
  unsigned int total_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);

  // Split input into two matrices (assuming they are square and same size)
  int matrix_size = static_cast<int>(std::sqrt(total_size / 2 / sizeof(std::complex<double>)));

  std::vector<std::complex<double>> denseA(matrix_size * matrix_size);
  std::vector<std::complex<double>> denseB(matrix_size * matrix_size);

  // Copy first matrix
  std::copy(in_ptr, in_ptr + matrix_size * matrix_size, denseA.begin());
  // Copy second matrix
  std::copy(in_ptr + matrix_size * matrix_size, in_ptr + 2 * matrix_size * matrix_size, denseB.begin());

  // Convert to CRS format
  ConvertToCRS(denseA, matrix_size, matrix_size, matrix_a_);
  ConvertToCRS(denseB, matrix_size, matrix_size, matrix_b_);

  // Initialize result matrix
  result_.rows = matrix_size;
  result_.cols = matrix_size;

  return true;
}

bool ivashchuk_v_tbb::SparseMatrixComplexCRS::ValidationImpl() {
  // Check if we have enough data for two matrices
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
  // Convert result back to dense format for output
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

void ivashchuk_v_tbb::SparseMatrixComplexCRS::ConvertToCRS(const std::vector<std::complex<double>>& dense, int rows,
                                                           int cols, CRSMatrix& crs) {
  crs.rows = rows;
  crs.cols = cols;
  ConvertDenseToCRS(dense, rows, cols, crs.values, crs.col_indices, crs.row_ptr);
}

void ivashchuk_v_tbb::SparseMatrixComplexCRS::SparseMatMul(const CRSMatrix& a, const CRSMatrix& b, CRSMatrix& c) {
  c.rows = a.rows;
  c.cols = b.cols;
  c.row_ptr.clear();
  c.values.clear();
  c.col_indices.clear();

  c.row_ptr.push_back(0);

  // Parallel computation of result rows using TBB
  tbb::parallel_for(tbb::blocked_range<int>(0, a.rows), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      std::vector<std::complex<double>> temp_row(b.cols, 0.0);

      int a_row_start = a.row_ptr[i];
      int a_row_end = a.row_ptr[i + 1];

      // Multiply row i of a with matrix b
      for (int k = a_row_start; k < a_row_end; ++k) {
        int col_a = a.col_indices[k];
        const std::complex<double>& val_a = a.values[k];

        int b_row_start = b.row_ptr[col_a];
        int b_row_end = b.row_ptr[col_a + 1];

        for (int l = b_row_start; l < b_row_end; ++l) {
          int col_b = b.col_indices[l];
          temp_row[col_b] += val_a * b.values[l];
        }
      }

      // Store non-zero elements of result row
      for (int j = 0; j < b.cols; ++j) {
        if (std::abs(temp_row[j]) > 1e-10) {
// Need synchronization for parallel insertion
#pragma omp critical
          {
            c.values.push_back(temp_row[j]);
            c.col_indices.push_back(j);
          }
        }
      }

// Update row pointer (needs synchronization)
#pragma omp critical
      {
        c.row_ptr.push_back(static_cast<int>(c.values.size()));
      }
    }
  });
}