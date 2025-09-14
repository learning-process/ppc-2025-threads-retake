#include "tbb/anikin_m_block_gauss/include/ops_tbb.hpp"

#include <cmath>
#include <vector>

#include "oneapi/tbb/blocked_range2d.h"
#include "oneapi/tbb/parallel_for.h"

bool anikin_m_block_gauss_tbb::BlockGaussTBB::ValidationImpl() {
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

bool anikin_m_block_gauss_tbb::BlockGaussTBB::PreProcessingImpl() {
  xres_ = static_cast<int>(task_data->inputs_count[0]);
  yres_ = static_cast<int>(task_data->inputs_count[1]);

  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (xres_ * yres_));
  output_ = std::vector<double>(xres_ * yres_, 0);

  return true;
}

bool anikin_m_block_gauss_tbb::BlockGaussTBB::RunImpl() {
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};

  tbb::parallel_for(
      tbb::blocked_range2d<int>(0, xres_, 0, yres_),
      [&](const tbb::blocked_range2d<int> &r) {
        const int yres = yres_;
        const int xres = xres_;

        for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
          for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
            if (i == 0 || j == 0 || i == xres - 1 || j == yres - 1) {
              output_[i * yres + j] = input_[i * yres + j];
            } else {
              double sum = 0.0;
              sum += input_[(i - 1) * yres + (j - 1)] * kernel[0][0];
              sum += input_[(i - 1) * yres + j] * kernel[0][1];
              sum += input_[(i - 1) * yres + (j + 1)] * kernel[0][2];

              sum += input_[i * yres + (j - 1)] * kernel[1][0];
              sum += input_[i * yres + j] * kernel[1][1];
              sum += input_[i * yres + (j + 1)] * kernel[1][2];

              sum += input_[(i + 1) * yres + (j - 1)] * kernel[2][0];
              sum += input_[(i + 1) * yres + j] * kernel[2][1];
              sum += input_[(i + 1) * yres + (j + 1)] * kernel[2][2];

              output_[i * yres + j] = sum;
            }
          }
        }
      },
      tbb::simple_partitioner());

  return true;
}

bool anikin_m_block_gauss_tbb::BlockGaussTBB::PostProcessingImpl() {
  for (int i = 0; i < xres_; i++) {
    for (int j = 0; j < yres_; j++) {
      reinterpret_cast<double *>(task_data->outputs[0])[(i * yres_) + j] = output_[(i * yres_) + j];
    }
  }
  return true;
}
