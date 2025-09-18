#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vasenkov_a_vertical_gauss_3x3/include/ops_seq.hpp"

namespace {

std::vector<uint8_t> GenerateRandomImage(int width, int height) {
  std::vector<uint8_t> image(width * height * 3);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = static_cast<uint8_t>(dis(gen));
  }
  return image;
}

std::vector<float> GenerateGaussianKernel() {
  return {1.0F / 16, 2.0F / 16, 1.0F / 16,
          2.0F / 16, 4.0F / 16, 2.0F / 16,
          1.0F / 16, 2.0F / 16, 1.0F / 16};
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

std::vector<uint8_t> CreateGradientImage(int width, int height) {
  std::vector<uint8_t> image(width * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      image[idx] = static_cast<uint8_t>((x * 255) / width);
      image[idx + 1] = static_cast<uint8_t>((y * 255) / height);
      image[idx + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height));
    }
  }
  return image;
}

}  // namespace

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_small_image) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;

  auto input_image = CreateSolidColorImage(kWidth, kHeight, {0, 0, 0});
  input_image[((2 * kWidth + 2) * 3)] = 255;
  input_image[((2 * kWidth + 2) * 3) + 1] = 255;
  input_image[((2 * kWidth + 2) * 3) + 2] = 255;

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

  EXPECT_NE(input_image, output_image);
}

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_gradient_image) {
  constexpr int kWidth = 8;
  constexpr int kHeight = 8;

  auto input_image = CreateGradientImage(kWidth, kHeight);
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

  bool changed = false;
  for (size_t i = 0; i < output_image.size(); ++i) {
    if (input_image[i] != output_image[i]) {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed);
}

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

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_minimum_size_image) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 3;

  auto input_image = CreateSolidColorImage(kWidth, kHeight, {0, 0, 0});
  input_image[((1 * kWidth + 1) * 3)] = 255;
  input_image[((1 * kWidth + 1) * 3) + 1] = 255;
  input_image[((1 * kWidth + 1) * 3) + 2] = 255;

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

  const int center_idx = (1 * kWidth + 1) * 3;

  EXPECT_NE(output_image[center_idx], 255);
  EXPECT_NE(output_image[center_idx + 1], 255);
  EXPECT_NE(output_image[center_idx + 2], 255);

  EXPECT_GE(output_image[center_idx], 0);
  EXPECT_LE(output_image[center_idx], 255);
  EXPECT_GE(output_image[center_idx + 1], 0);
  EXPECT_LE(output_image[center_idx + 1], 255);
  EXPECT_GE(output_image[center_idx + 2], 0);
  EXPECT_LE(output_image[center_idx + 2], 255);

  for (int y = 0; y < kHeight; y++) {
    for (int x = 0; x < kWidth; x++) {
      if (y == 1 && x == 1) {
        continue;
      }

      const int idx = (y * kWidth + x) * 3;
      EXPECT_EQ(input_image[idx], output_image[idx]);
      EXPECT_EQ(input_image[idx + 1], output_image[idx + 1]);
      EXPECT_EQ(input_image[idx + 2], output_image[idx + 2]);
    }
  }
}

TEST(vasenkov_a_vertical_gauss_3x3_seq, test_validation_failure) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;

  auto input_image = CreateSolidColorImage(kWidth, kHeight, {255, 255, 255});
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

  auto input_image = CreateSolidColorImage(kWidth, kHeight, {100, 100, 100});
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