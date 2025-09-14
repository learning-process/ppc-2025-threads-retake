#include "seq/anikin_m_block_gauss/include/ops_seq.hpp"

bool anikin_m_block_gauss_seq::BlockGaussSequential::ValidationImpl() {
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

bool anikin_m_block_gauss_seq::BlockGaussSequential::PreProcessingImpl() {
  xres = static_cast<int>(task_data->inputs_count[0]);
  yres = static_cast<int>(task_data->inputs_count[1]);

  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (xres * yres));
  output_ = std::vector<double>(xres * yres, 0);

  return true;
}

bool anikin_m_block_gauss_seq::BlockGaussSequential::RunImpl() {
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};

  for (int i = 0; i < xres; ++i) {
    for (int j = 0; j < yres; ++j) {
      if (i == 0 || j == 0 || i == xres - 1 || j == yres - 1) {
        output_[(i * yres) + j] = input_[(i * yres) + j];
      } else {
        double sum = 0.0;
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += input_[((i + ki) * yres) + (j + kj)] * kernel[ki + 1][kj + 1];
          }
        }
        output_[(i * yres) + j] = sum;
      }
    }
  }

  return true;
}

bool anikin_m_block_gauss_seq::BlockGaussSequential::PostProcessingImpl() {
  for (int i = 0; i < xres; i++) {
    for (int j = 0; j < yres; j++) {
      reinterpret_cast<double *>(task_data->outputs[0])[(i * yres) + j] = output_[(i * yres) + j];
    }
  }
  return true;
}
