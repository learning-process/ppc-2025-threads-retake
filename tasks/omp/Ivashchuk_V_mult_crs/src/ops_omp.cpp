#include "omp/Ivashchuk_V_mult_crs/include/ops_omp.hpp"

#include <cmath>
#include <complex>
#include <vector>

bool ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP::PreProcessingImpl() {
  // Parse input data
  auto* input_data1 = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);
  auto* input_data2 = reinterpret_cast<std::complex<double>*>(task_data->inputs[1]);

  int rows1 = reinterpret_cast<int*>(task_data->inputs[2])[0];
  int cols1 = reinterpret_cast<int*>(task_data->inputs[2])[1];
  int rows2 = reinterpret_cast<int*>(task_data->inputs[3])[0];
  int cols2 = reinterpret_cast<int*>(task_data->inputs[3])[1];

  // Initialize matrix1_
  matrix1_.rows = rows1;
  matrix1_.cols = cols1;
  matrix1_.row_pointers.push_back(0);

  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols1; ++j) {
      std::complex<double> val = input_data1[(i * cols1) + j];
      if (std::abs(val.real()) > 1e-10 || std::abs(val.imag()) > 1e-10) {
        matrix1_.values.push_back(val);
        matrix1_.col_indices.push_back(j);
      }
    }
    matrix1_.row_pointers.push_back(static_cast<int>(matrix1_.values.size()));
  }

  // Initialize matrix2_
  matrix2_.rows = rows2;
  matrix2_.cols = cols2;
  matrix2_.row_pointers.push_back(0);

  for (int i = 0; i < rows2; ++i) {
    for (int j = 0; j < cols2; ++j) {
      std::complex<double> val = input_data2[(i * cols2) + j];
      if (std::abs(val.real()) > 1e-10 || std::abs(val.imag()) > 1e-10) {
        matrix2_.values.push_back(val);
        matrix2_.col_indices.push_back(j);
      }
    }
    matrix2_.row_pointers.push_back(static_cast<int>(matrix2_.values.size()));
  }

  return true;
}

bool ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP::ValidationImpl() {
  if (task_data->inputs.size() < 4) {
    return false;
  }
  if (task_data->outputs.empty()) {
    return false;
  }

  int* dims1 = reinterpret_cast<int*>(task_data->inputs[2]);
  int* dims2 = reinterpret_cast<int*>(task_data->inputs[3]);

  if (dims1 == nullptr || dims2 == nullptr) {
    return false;
  }

  int cols1 = dims1[1];
  int rows2 = dims2[0];

  if (cols1 != rows2) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  return true;
}

bool ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP::RunImpl() {
  MultiplySparseMatrices();
  return true;
}

void ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP::MultiplySparseMatrices() {
  int rows = matrix1_.rows;
  int cols = matrix2_.cols;

  result_.rows = rows;
  result_.cols = cols;
  result_.row_pointers.resize(rows + 1, 0);

  std::vector<std::vector<std::complex<double>>> temp_values(rows);
  std::vector<std::vector<int>> temp_col_indices(rows);

#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::complex<double> sum(0.0, 0.0);

      int start1 = matrix1_.row_pointers[i];
      int end1 = matrix1_.row_pointers[i + 1];

      for (int k1 = start1; k1 < end1; ++k1) {
        int col1 = matrix1_.col_indices[k1];
        std::complex<double> val1 = matrix1_.values[k1];

        int start2 = matrix2_.row_pointers[col1];
        int end2 = matrix2_.row_pointers[col1 + 1];

        for (int k2 = start2; k2 < end2; ++k2) {
          int col2 = matrix2_.col_indices[k2];
          if (col2 == j) {
            sum += val1 * matrix2_.values[k2];
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
  result_.row_pointers[0] = 0;
  for (int i = 0; i < rows; ++i) {
    result_.values.insert(result_.values.end(), temp_values[i].begin(), temp_values[i].end());
    result_.col_indices.insert(result_.col_indices.end(), temp_col_indices[i].begin(), temp_col_indices[i].end());
    result_.row_pointers[i + 1] = static_cast<int>(result_.values.size());
  }
}

bool ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP::PostProcessingImpl() {
  auto* output_data = reinterpret_cast<std::complex<double>*>(task_data->outputs[0]);
  int rows = result_.rows;
  int cols = result_.cols;

  // Initialize output with zeros
  for (int i = 0; i < rows * cols; ++i) {
    output_data[i] = std::complex<double>(0.0, 0.0);
  }

  // Fill non-zero elements
  for (int i = 0; i < rows; ++i) {
    int start = result_.row_pointers[i];
    int end = result_.row_pointers[i + 1];

    for (int j = start; j < end; ++j) {
      int col = result_.col_indices[j];
      output_data[(i * cols) + col] = result_.values[j];
    }
  }

  return true;
}