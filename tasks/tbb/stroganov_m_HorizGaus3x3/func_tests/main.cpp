#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/stroganov_m_HorizGaus3x3/include/ops_tbb.hpp"

TEST(stroganov_m_horiz_gaus3x3_tbb, AllOnes_BordersAdjusted) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 1.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 1.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    expected_output[(i * kWidth) + 0] = 0.75;
    expected_output[(i * kWidth) + (kWidth - 1)] = 0.75;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 1e-5);
    }
  }
}

TEST(stroganov_m_horiz_gaus3x3_tbb, VerticalLines_Smoothed) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    input_image[((i * kWidth)) + 3] = 2.0;
    input_image[((i * kWidth)) + 6] = 0.5;
    expected_output[((i * kWidth)) + 2] = 0.5;
    expected_output[((i * kWidth)) + 3] = 1.0;
    expected_output[((i * kWidth)) + 4] = 0.5;
    expected_output[((i * kWidth)) + 5] = 0.125;
    expected_output[((i * kWidth)) + 6] = 0.25;
    expected_output[((i * kWidth)) + 7] = 0.125;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[((i * kWidth)) + j], expected_output[((i * kWidth)) + j], 1e-5);
    }
  }
}

TEST(stroganov_m_horiz_gaus3x3_tbb, HorizontalLines_Preserved) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t j = 0; j < kWidth; ++j) {
    input_image[(2 * kWidth) + j] = 1.0;
    input_image[(7 * kWidth) + j] = 1.0;
  }

  for (size_t row : {2, 7}) {
    expected_output[(row * kWidth) + 0] = 0.75;
    expected_output[(row * kWidth) + kWidth - 1] = 0.75;
    for (size_t j = 1; j < kWidth - 1; ++j) {
      expected_output[(row * kWidth) + j] = 1.0;
    }
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_TRUE(image_filter_tbb.Validation());

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 1e-5);
    }
  }
}

TEST(stroganov_m_horiz_gaus3x3_tbb, EmptyImage_NoChange) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 1e-5);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_tbb, SharpTransitions_SmoothedEdges) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth / 2; ++j) {
      input_image[(i * kWidth) + j] = 0.0;
    }
    for (size_t j = kWidth / 2; j < kWidth; ++j) {
      input_image[(i * kWidth) + j] = 1.0;
    }
  }

  for (size_t i = 0; i < kHeight; ++i) {
    expected_output[(i * kWidth) + 4] = 0.25;
    expected_output[(i * kWidth) + 5] = 0.75;
    expected_output[(i * kWidth) + 6] = 1.0;
    expected_output[(i * kWidth) + 7] = 1.0;
    expected_output[(i * kWidth) + 8] = 1.0;
    expected_output[(i * kWidth) + 9] = 0.75;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 1e-5);
    }
  }
}

TEST(stroganov_m_horiz_gaus3x3_tbb, SmoothGradient_Preserved) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      input_image[(i * kWidth) + j] = expected_output[(i * kWidth) + j] = static_cast<double>(j) / (kWidth - 1);
    }
  }

  for (size_t i = 0; i < kHeight; ++i) {
    expected_output[(i * kWidth)] = 0.03;
    expected_output[((i + 1) * kWidth) - 1] = 0.72;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 0.5);
    }
  }
}

TEST(stroganov_m_horiz_gaus3x3_tbb, AllMax_BordersAdjusted) {
  constexpr size_t kWidth = 10;
  constexpr size_t kHeight = 10;
  std::vector<double> input_image(kWidth * kHeight, 255.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 255.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    expected_output[(i * kWidth)] = 191.25;
    expected_output[((i + 1) * kWidth) - 1] = 191.25;
  }

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 1e-5);
    }
  }
}

TEST(stroganov_m_horiz_gaus3x3_tbb, RandomImage_MeanInvariant) {
  constexpr size_t kWidth = 100;
  constexpr size_t kHeight = 100;

  std::vector<double> input_image(kWidth * kHeight);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 255.0);

  for (size_t i = 0; i < kWidth * kHeight; ++i) {
    input_image[i] = dis(gen);
  }

  std::vector<int> kernel = {1, 2, 1};

  std::vector<double> output_image(kWidth * kHeight, 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  double avg_input =
      std::accumulate(input_image.begin(), input_image.end(), 0.0) / static_cast<double>(input_image.size());
  double avg_output =
      std::accumulate(output_image.begin(), output_image.end(), 0.0) / static_cast<double>(output_image.size());

  ASSERT_NEAR(avg_input, avg_output, 1);
}

TEST(stroganov_m_horiz_gaus3x3_tbb, PointSource_Spread) {
  constexpr size_t kWidth = 5;
  constexpr size_t kHeight = 5;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected_output(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  input_image[(2 * kWidth) + 2] = 10.0;
  expected_output[(2 * kWidth) + 1] = 2.5;
  expected_output[(2 * kWidth) + 2] = 5.0;
  expected_output[(2 * kWidth) + 3] = 2.5;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  stroganov_m_horiz_gaus3x3_tbb::ImageFilterTbb image_filter_tbb(task_data_tbb);

  ASSERT_EQ(image_filter_tbb.Validation(), true);

  image_filter_tbb.PreProcessing();
  image_filter_tbb.Run();
  image_filter_tbb.PostProcessing();

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected_output[(i * kWidth) + j], 1e-5);
    }
  }
}
