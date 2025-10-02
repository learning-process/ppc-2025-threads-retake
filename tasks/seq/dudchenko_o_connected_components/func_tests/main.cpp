#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_connected_components/include/ops_seq.hpp"

namespace {

struct OutputData {
  const std::vector<int>& data;
};

struct ForegroundIndices {
  const std::vector<size_t>& indices;
};

struct BackgroundIndices {
  const std::vector<size_t>& indices;
};

struct Indices {
  const std::vector<size_t>& indices;
};

void CheckForegroundBackground(const OutputData& output, const ForegroundIndices& foreground,
                               const BackgroundIndices& background) {
  for (size_t idx : foreground.indices) {
    EXPECT_NE(output.data[idx], 0);
  }
  for (size_t idx : background.indices) {
    EXPECT_EQ(output.data[idx], 0);
  }
}

void CheckAllLabelsUnique(const OutputData& output, const Indices& indices) {
  std::vector<int> labels;
  for (size_t idx : indices.indices) {
    if (output.data[idx] != 0 && std::ranges::find(labels, output.data[idx]) == labels.end()) {
      labels.push_back(output.data[idx]);
    }
  }

  EXPECT_EQ(labels.size(), indices.indices.size());
}

void CheckComponentPoints(const OutputData& output, int component_label, const Indices& indices) {
  for (size_t idx : indices.indices) {
    EXPECT_EQ(output.data[idx], component_label);
  }
}

// Helper functions for the random test
struct ImageDimensions {
  int width;
  int height;
};

struct Data {
  std::vector<int> expected_image_data;
  std::vector<int> actual_output_data;
};

std::vector<int> GenerateRandomImageData(const ImageDimensions& dims, int foreground_percentage = 20) {
  const size_t total_pixels = dims.width * dims.height;
  std::vector<int> image_data(total_pixels);

  for (size_t i = 0; i < total_pixels; ++i) {
    image_data[i] = (std::rand() % 100 < foreground_percentage) ? 0 : 255;
  }

  return image_data;
}

void VerifyForegroundBackgroundLabels(const Data& data) {
  const size_t total_pixels = data.expected_image_data.size();

  for (size_t i = 0; i < total_pixels; ++i) {
    if (data.expected_image_data[i] == 0) {  // foreground
      EXPECT_NE(data.actual_output_data[i], 0);
    } else {  // background
      EXPECT_EQ(data.actual_output_data[i], 0);
    }
  }
}

std::vector<int> CollectUniqueLabels(const std::vector<int>& output_data) {
  std::vector<int> unique_labels;
  for (int label : output_data) {
    if (label != 0 && std::ranges::find(unique_labels, label) == unique_labels.end()) {
      unique_labels.push_back(label);
    }
  }
  return unique_labels;
}

void VerifyComponentConsistency(const std::vector<int>& output_data, int component_label) {
  std::vector<size_t> component_indices;

  for (size_t i = 0; i < output_data.size(); ++i) {
    if (output_data[i] == component_label) {
      component_indices.push_back(i);
    }
  }

  // Verify all pixels in the component have the same label
  for (size_t idx : component_indices) {
    EXPECT_EQ(output_data[idx], component_label);
  }
}

void PrintTestStatistics(size_t foreground_count, size_t background_count, size_t component_count) {
  std::cout << "Random test: " << foreground_count << " foreground, " << background_count << " background, "
            << component_count << " components\n";
}

size_t CountForegroundPixels(const std::vector<int>& image_data) {
  return std::count(image_data.begin(), image_data.end(), 0);
}

}  // namespace

TEST(dudchenko_o_connected_components_seq, test_small_image) {
  int width = 3;
  int height = 3;
  std::vector<int> image_data = {0, 255, 0, 255, 0, 255, 0, 255, 0};

  std::vector<int> input_data;
  input_data.push_back(width);
  input_data.push_back(height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  std::vector<int> output_data(width * height);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  dudchenko_o_connected_components::TestTaskSequential test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.Validation(), true);
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();

  OutputData output{output_data};
  ForegroundIndices foreground{{0, 2, 4, 6, 8}};
  BackgroundIndices background{{1, 3, 5, 7}};

  CheckForegroundBackground(output, foreground, background);
  CheckAllLabelsUnique(output, Indices{{0, 2, 4, 6, 8}});
}

TEST(dudchenko_o_connected_components_seq, test_single_component) {
  int width = 4;
  int height = 4;
  std::vector<int> image_data(width * height, 0);

  std::vector<int> input_data;
  input_data.push_back(width);
  input_data.push_back(height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  std::vector<int> output_data(width * height);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  dudchenko_o_connected_components::TestTaskSequential test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.Validation(), true);
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();

  int first_label = output_data[0];
  EXPECT_NE(first_label, 0);
  for (size_t i = 1; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], first_label);
  }
}

TEST(dudchenko_o_connected_components_seq, test_no_components) {
  int width = 4;
  int height = 4;
  std::vector<int> image_data(width * height, 255);

  std::vector<int> input_data;
  input_data.push_back(width);
  input_data.push_back(height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  std::vector<int> output_data(width * height);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  dudchenko_o_connected_components::TestTaskSequential test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.Validation(), true);
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], 0);
  }
}

TEST(dudchenko_o_connected_components_seq, test_two_separate_components) {
  int width = 5;
  int height = 5;
  std::vector<int> image_data = {0,   0,   255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 255,
                                 255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0,   0};

  std::vector<int> input_data;
  input_data.push_back(width);
  input_data.push_back(height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  std::vector<int> output_data(width * height);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  dudchenko_o_connected_components::TestTaskSequential test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.Validation(), true);
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();

  int comp1 = output_data[0];
  int comp2 = output_data[18];

  EXPECT_NE(comp1, 0);
  EXPECT_NE(comp2, 0);
  EXPECT_NE(comp1, comp2);

  OutputData output{output_data};
  CheckComponentPoints(output, comp1, Indices{{0, 1, 5, 6}});
  CheckComponentPoints(output, comp2, Indices{{18, 19, 23, 24}});
}

TEST(dudchenko_o_connected_components_seq, test_random_data_simple) {
  // Initialize random generator with fixed seed for reproducibility
  std::srand(42);

  ImageDimensions dims;
  dims.width = 50;
  dims.height = 50;
  const size_t total_pixels = dims.width * dims.height;

  // Generate random image data
  std::vector<int> image_data = GenerateRandomImageData(dims, 20);
  std::vector<int> output_data(total_pixels);

  // Подготовка входных данных
  std::vector<int> input_data;
  input_data.push_back(dims.width);
  input_data.push_back(dims.height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  // Создание и выполнение задачи
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  dudchenko_o_connected_components::TestTaskSequential test_task_seq(task_data_seq);

  // Проверка валидации
  ASSERT_EQ(test_task_seq.Validation(), true);

  // Выполнение задачи
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();

  // Verify foreground/background labels
  Data data;
  data.expected_image_data = image_data;
  data.actual_output_data = output_data;
  VerifyForegroundBackgroundLabels(data);

  // Collect unique labels
  std::vector<int> unique_labels = CollectUniqueLabels(output_data);

  // Verify component consistency for the first found component
  if (!unique_labels.empty()) {
    VerifyComponentConsistency(output_data, unique_labels[0]);
  }

  // Calculate statistics
  size_t foreground_count = CountForegroundPixels(image_data);
  size_t background_count = total_pixels - foreground_count;

  // Print statistics
  PrintTestStatistics(foreground_count, background_count, unique_labels.size());

  // Verify basic expectations
  EXPECT_GT(foreground_count, 0UL);
  EXPECT_GT(background_count, 0UL);
  EXPECT_LE(unique_labels.size(), foreground_count);
}