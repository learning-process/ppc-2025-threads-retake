#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_connected_components/include/ops_seq.hpp"

namespace {

// Структуры для устранения предупреждения о легко переставляемых параметрах
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

  // Все метки уже уникальны по построению, проверяем что их количество равно количеству индексов
  EXPECT_EQ(labels.size(), indices.indices.size());
}

void CheckComponentPoints(const OutputData& output, int component_label, const Indices& indices) {
  for (size_t idx : indices.indices) {
    EXPECT_EQ(output.data[idx], component_label);
  }
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

  dudchenko_o_connected_components::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Проверяем базовые свойства
  OutputData output{output_data};
  ForegroundIndices foreground{{0, 2, 4, 6, 8}};
  BackgroundIndices background{{1, 3, 5, 7}};

  CheckForegroundBackground(output, foreground, background);
  CheckAllLabelsUnique(output, Indices{{0, 2, 4, 6, 8}});
}

TEST(dudchenko_o_connected_components_seq, test_single_component) {
  int width = 4;
  int height = 4;
  // Все точки foreground (0)
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

  dudchenko_o_connected_components::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  int first_label = output_data[0];
  EXPECT_NE(first_label, 0);
  for (size_t i = 1; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], first_label);
  }
}

TEST(dudchenko_o_connected_components_seq, test_no_components) {
  int width = 4;
  int height = 4;
  // Все точки background (не 0)
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

  dudchenko_o_connected_components::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

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

  dudchenko_o_connected_components::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Проверяем два компонента
  int comp1 = output_data[0];
  int comp2 = output_data[18];

  EXPECT_NE(comp1, 0);
  EXPECT_NE(comp2, 0);
  EXPECT_NE(comp1, comp2);

  // Проверяем точки компонентов
  OutputData output{output_data};
  CheckComponentPoints(output, comp1, Indices{{0, 1, 5, 6}});
  CheckComponentPoints(output, comp2, Indices{{18, 19, 23, 24}});
}