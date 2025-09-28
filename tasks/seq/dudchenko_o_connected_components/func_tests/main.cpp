#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_connected_components/include/ops_seq.hpp"

TEST(dudchenko_o_connected_components_seq, test_small_image) {
  int width = 3;
  int height = 3;
  std::vector<int> image_data = {0, 255, 0, 
                                 255, 0, 255, 
                                 0, 255, 0};

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

  // Проверяем foreground точки имеют ненулевые метки
  const std::vector<size_t> foreground_indices = {0, 2, 4, 6, 8};
  for (size_t idx : foreground_indices) {
    EXPECT_NE(output_data[idx], 0) << "Foreground point at index " << idx << " should have non-zero label";
  }

  // Проверяем background точки имеют нулевые метки
  const std::vector<size_t> background_indices = {1, 3, 5, 7};
  for (size_t idx : background_indices) {
    EXPECT_EQ(output_data[idx], 0) << "Background point at index " << idx << " should have zero label";
  }

  // Проверяем что все foreground точки имеют разные метки (они не связаны)
  for (size_t i = 0; i < foreground_indices.size(); ++i) {
    for (size_t j = i + 1; j < foreground_indices.size(); ++j) {
      EXPECT_NE(output_data[foreground_indices[i]], output_data[foreground_indices[j]])
          << "Points at indices " << foreground_indices[i] << " and " << foreground_indices[j] 
          << " should have different labels";
    }
  }
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
  // Два отдельных компонента из foreground точек (0)
  std::vector<int> image_data = {0, 0, 255, 255, 255,
                                 0, 0, 255, 255, 255, 
                                 255, 255, 255, 255, 255, 
                                 255, 255, 255, 0, 0, 
                                 255, 255, 255, 0, 0};

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

  int comp1 = output_data[0];  // левый верхний компонент
  int comp2 = output_data[18]; // правый нижний компонент

  EXPECT_NE(comp1, 0);
  EXPECT_NE(comp2, 0);
  EXPECT_NE(comp1, comp2);

  // Проверяем точки первого компонента
  EXPECT_EQ(output_data[1], comp1);
  EXPECT_EQ(output_data[5], comp1);
  EXPECT_EQ(output_data[6], comp1);

  // Проверяем точки второго компонента
  EXPECT_EQ(output_data[19], comp2);
  EXPECT_EQ(output_data[23], comp2);
  EXPECT_EQ(output_data[24], comp2);
}