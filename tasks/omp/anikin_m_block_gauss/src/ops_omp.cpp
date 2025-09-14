#include "omp/anikin_m_block_gauss/include/ops_omp.hpp"

#include <cmath>
#include <vector>

bool anikin_m_block_gauss_omp::BlockGaussOMP::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (task_data->inputs_count[1] != task_data->outputs_count[1]) {
    return false;
  }
  if (task_data->inputs_count[0] <= 0 || task_data->inputs_count[1] <= 0) {
    return false;
  }
  return true;
}

bool anikin_m_block_gauss_omp::BlockGaussOMP::PreProcessingImpl() {
  xres_ = static_cast<int>(task_data->inputs_count[0]);
  yres_ = static_cast<int>(task_data->inputs_count[1]);

  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (xres_ * yres_));
  output_ = std::vector<double>(xres_ * yres_, 0);

  return true;
}

bool anikin_m_block_gauss_omp::BlockGaussOMP::RunImpl() {
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};

#pragma omp parallel for
  for (int i = 0; i < xres_; ++i) {
    for (int j = 0; j < yres_; ++j) {
      if (i == 0 || j == 0 || i == xres_ - 1 || j == yres_ - 1) {
        output_[(i * yres_) + j] = input_[(i * yres_) + j];
      } else {
        double sum = 0.0;
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += input_[((i + ki) * yres_) + (j + kj)] * kernel[ki + 1][kj + 1];
          }
        }
        output_[(i * yres_) + j] = sum;
      }
    }
  }

  return true;
}

bool anikin_m_block_gauss_omp::BlockGaussOMP::PostProcessingImpl() {
  for (int i = 0; i < xres_; i++) {
    for (int j = 0; j < yres_; j++) {
      reinterpret_cast<double *>(task_data->outputs[0])[(i * yres_) + j] = output_[(i * yres_) + j];
    }
  }
  return true;
}
