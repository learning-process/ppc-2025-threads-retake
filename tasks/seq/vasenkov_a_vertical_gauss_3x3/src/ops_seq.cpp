#include "seq/vasenkov_a_vertical_gauss_3x3/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

bool vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss::PreProcessingImpl() {
  img_width_ = static_cast<int32_t>(task_data->inputs_count[0]);
  img_height_ = static_cast<int32_t>(task_data->inputs_count[1]);

  const size_t total_pixels = img_width_ * img_height_ * kCHANNELS;
  source_img_.resize(total_pixels);
  std::memcpy(source_img_.data(), task_data->inputs[0], total_pixels);

  filter_kernel_.resize(9);
  std::memcpy(filter_kernel_.data(), task_data->inputs[1], 9 * sizeof(float));

  filtered_img_ = source_img_;

  return true;
}

bool vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss::ValidationImpl() {
  return task_data->inputs_count[2] == 9 &&
         (task_data->inputs_count[0] * task_data->inputs_count[1] * kCHANNELS) == task_data->outputs_count[0];
}

bool vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss::RunImpl() {
  if (img_height_ <= 2 || img_width_ <= 2) {
    return true;
  }

  const int row_stride = img_width_ * kCHANNELS;
  const float* kernel = filter_kernel_.data();

  for (int y = 1; y < img_height_ - 1; ++y) {
    for (int x = 1; x < img_width_ - 1; ++x) {
      const int base_pixel_idx = (y * img_width_ + x) * kCHANNELS;
      const int top_pixel_idx = base_pixel_idx - row_stride;
      const int bottom_pixel_idx = base_pixel_idx + row_stride;

      for (int c = 0; c < kCHANNELS; ++c) {
        const int left_idx = base_pixel_idx - kCHANNELS + c;
        const int center_idx = base_pixel_idx + c;
        const int right_idx = base_pixel_idx + kCHANNELS + c;

        const int top_left_idx = top_pixel_idx - kCHANNELS + c;
        const int top_center_idx = top_pixel_idx + c;
        const int top_right_idx = top_pixel_idx + kCHANNELS + c;

        const int bottom_left_idx = bottom_pixel_idx - kCHANNELS + c;
        const int bottom_center_idx = bottom_pixel_idx + c;
        const int bottom_right_idx = bottom_pixel_idx + kCHANNELS + c;

        float sum = ((static_cast<float>(source_img_[top_left_idx]) * kernel[0]) +
                     (static_cast<float>(source_img_[top_center_idx]) * kernel[1]) +
                     (static_cast<float>(source_img_[top_right_idx]) * kernel[2]) +
                     (static_cast<float>(source_img_[left_idx]) * kernel[3]) +
                     (static_cast<float>(source_img_[center_idx]) * kernel[4]) +
                     (static_cast<float>(source_img_[right_idx]) * kernel[5]) +
                     (static_cast<float>(source_img_[bottom_left_idx]) * kernel[6]) +
                     (static_cast<float>(source_img_[bottom_center_idx]) * kernel[7]) +
                     (static_cast<float>(source_img_[bottom_right_idx]) * kernel[8]));

        filtered_img_[center_idx] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0F, 255.0F));
      }
    }
  }

  return true;
}

bool vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], filtered_img_.data(), filtered_img_.size() * sizeof(uint8_t));
  return true;
}