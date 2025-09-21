#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vasenkov_a_vertical_gauss_3x3/include/ops_seq.hpp"

namespace {

std::vector<uint8_t> GenerateRandomImage(int width, int height) {
  std::vector<uint8_t> image(width * height * 3);
  for (auto& pixel : image) {
    pixel = rand() % 256;
  }
  return image;
}

std::vector<float> GenerateGaussianKernel() {
  return {1.0F / 16, 2.0F / 16, 1.0F / 16, 2.0F / 16, 4.0F / 16, 2.0F / 16, 1.0F / 16, 2.0F / 16, 1.0F / 16};
}

struct RGB {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

std::vector<uint8_t> CreateSolidColorImage(int width, int height, RGB color) {
  std::vector<uint8_t> image(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    image[(i * 3)] = color.r;
    image[(i * 3) + 1] = color.g;
    image[(i * 3) + 2] = color.b;
  }
  return image;
}

}  // namespace

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_random_image) {
  constexpr int kWidth = 15;
  constexpr int kHeight = 15;

  auto input_image = GenerateRandomImage(kWidth, kHeight);
  auto output_image = input_image;
  auto kernel = GenerateGaussianKernel();

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(9);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data->outputs_count.emplace_back(output_image.size());

  vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (const auto& pixel : output_image) {
    EXPECT_GE(pixel, 0);
    EXPECT_LE(pixel, 255);
  }
}

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_validation_failure) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;

  auto input_image = CreateSolidColorImage(kWidth, kHeight, {.r = 255, .g = 255, .b = 255});
  auto output_image = input_image;
  std::vector<float> wrong_kernel = {1.0F, 2.0F, 3.0F, 4.0F};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(wrong_kernel.data()));
  task_data->inputs_count.emplace_back(4);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data->outputs_count.emplace_back(output_image.size());

  vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_edge_detection_kernel) {
  constexpr int kWidth = 7;
  constexpr int kHeight = 7;

  auto input_image = CreateSolidColorImage(kWidth, kHeight, {.r = 100, .g = 100, .b = 100});
  for (int y = 0; y < kHeight; ++y) {
    int idx = (y * kWidth + 3) * 3;
    input_image[idx] = 255;
    input_image[idx + 1] = 255;
    input_image[idx + 2] = 255;
  }

  auto output_image = input_image;
  std::vector<float> edge_kernel = {-1.0F, -1.0F, -1.0F, -1.0F, 8.0F, -1.0F, -1.0F, -1.0F, -1.0F};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(edge_kernel.data()));
  task_data->inputs_count.emplace_back(9);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data->outputs_count.emplace_back(output_image.size());

  vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  bool edges_detected = false;
  for (const auto& pixel : output_image) {
    if (pixel != 100 && pixel != 255) {
      edges_detected = true;
      break;
    }
  }
  EXPECT_TRUE(edges_detected);
}

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_empty_image) {
  constexpr int kWidth = 0;
  constexpr int kHeight = 0;

  std::vector<uint8_t> input_image;
  std::vector<uint8_t> output_image;
  auto kernel = GenerateGaussianKernel();

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(9);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data->outputs_count.emplace_back(output_image.size());

  vasenkov_a_vertical_gauss_3x3_seq::VerticalGauss task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_TRUE(output_image.empty());
}