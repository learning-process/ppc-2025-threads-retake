#include "ops_omp.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

bool Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP::PreProcessingImpl() {
  // Parse input data
  auto* input_data1 = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);
  auto* input_data2 = reinterpret_cast<std::complex<double>*>(task_data->inputs[1]);

  int rows1 = reinterpret_cast<int*>(task_data->inputs[2])[0];
  int cols1 = reinterpret_cast<int*>(task_data->inputs[2])[1];
  int rows2 = reinterpret_cast<int*>(task_data->inputs[3])[0];
  int cols2 = reinterpret_cast<int*>(task_data->inputs[3])[1];

  // Initialize matrix1
  matrix1.rows = rows1;
  matrix1.cols = cols1;
  matrix1.row_pointers.push_back(0);

  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols1; ++j) {
      std::complex<double> val = input_data1[i * cols1 + j];
      if (std::abs(val.real()) > 1e-10 || std::abs(val.imag()) > 1e-10) {
        matrix1.values.push_back(val);
        matrix1.col_indices.push_back(j);
      }
    }
    matrix1.row_pointers.push_back(matrix1.values.size());
  }

  // Initialize matrix2
  matrix2.rows = rows2;
  matrix2.cols = cols2;
  matrix2.row_pointers.push_back(0);

  for (int i = 0; i < rows2; ++i) {
    for (int j = 0; j < cols2; ++j) {
      std::complex<double> val = input_data2[i * cols2 + j];
      if (std::abs(val.real()) > 1e-10 || std::abs(val.imag()) > 1e-10) {
        matrix2.values.push_back(val);
        matrix2.col_indices.push_back(j);
      }
    }
    matrix2.row_pointers.push_back(matrix2.values.size());
  }

  return true;
}

bool Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP::ValidationImpl() {
  // Check if matrices can be multiplied
  if (matrix1.cols != matrix2.rows) {
    return false;
  }

  // Check if output data is not nullptr
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  return true;
}

bool Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP::RunImpl() {
  multiplySparseMatrices();
  return true;
}

void Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP::multiplySparseMatrices() {
  int rows = matrix1.rows;
  int cols = matrix2.cols;

  result.rows = rows;
  result.cols = cols;
  result.row_pointers.resize(rows + 1, 0);

  std::vector<std::vector<std::complex<double>>> temp_values(rows);
  std::vector<std::vector<int>> temp_col_indices(rows);

#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::complex<double> sum(0.0, 0.0);

      int start1 = matrix1.row_pointers[i];
      int end1 = matrix1.row_pointers[i + 1];

      for (int k1 = start1; k1 < end1; ++k1) {
        int col1 = matrix1.col_indices[k1];
        std::complex<double> val1 = matrix1.values[k1];

        int start2 = matrix2.row_pointers[col1];
        int end2 = matrix2.row_pointers[col1 + 1];

        for (int k2 = start2; k2 < end2; ++k2) {
          int col2 = matrix2.col_indices[k2];
          if (col2 == j) {
            sum += val1 * matrix2.values[k2];
            break;
          }
        }
      }

      if (std::abs(sum.real()) > 1e-10 || std::abs(sum.imag()) > 1e-10) {
        temp_values[i].push_back(sum);
        temp_col_indices[i].push_back(j);
      }
    }
  }

  // Convert temporary structure to CRS format
  result.row_pointers[0] = 0;
  for (int i = 0; i < rows; ++i) {
    result.values.insert(result.values.end(), temp_values[i].begin(), temp_values[i].end());
    result.col_indices.insert(result.col_indices.end(), temp_col_indices[i].begin(), temp_col_indices[i].end());
    result.row_pointers[i + 1] = result.values.size();
  }
}

bool Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP::PostProcessingImpl() {
  // Convert result matrix to dense format for output
  auto* output_data = reinterpret_cast<std::complex<double>*>(task_data->outputs[0]);
  int rows = result.rows;
  int cols = result.cols;

  // Initialize output with zeros
  for (int i = 0; i < rows * cols; ++i) {
    output_data[i] = std::complex<double>(0.0, 0.0);
  }

  // Fill non-zero elements
  for (int i = 0; i < rows; ++i) {
    int start = result.row_pointers[i];
    int end = result.row_pointers[i + 1];

    for (int j = start; j < end; ++j) {
      int col = result.col_indices[j];
      output_data[i * cols + col] = result.values[j];
    }
  }

  return true;
}
