#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/dudchenko_o_connected_components/include/ops_seq.hpp"

TEST(dudchenko_o_connected_components_seq, test_small_image) {
  int width = 3;
  int height = 3;
  std::vector<int> image_data = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  
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

  EXPECT_NE(output_data[0], 0);
  EXPECT_EQ(output_data[1], 0);
  EXPECT_NE(output_data[2], 0);
  EXPECT_EQ(output_data[3], 0);
  EXPECT_NE(output_data[4], 0);
  EXPECT_EQ(output_data[5], 0);
  EXPECT_NE(output_data[6], 0);
  EXPECT_EQ(output_data[7], 0);
  EXPECT_NE(output_data[8], 0);

  EXPECT_NE(output_data[0], output_data[2]);
  EXPECT_NE(output_data[0], output_data[4]);
  EXPECT_NE(output_data[0], output_data[6]);
  EXPECT_NE(output_data[0], output_data[8]);
  EXPECT_NE(output_data[2], output_data[4]);
  EXPECT_NE(output_data[2], output_data[6]);
  EXPECT_NE(output_data[2], output_data[8]);
  EXPECT_NE(output_data[4], output_data[6]);
  EXPECT_NE(output_data[4], output_data[8]);
  EXPECT_NE(output_data[6], output_data[8]);
}

TEST(dudchenko_o_connected_components_seq, test_single_component) {
  int width = 4;
  int height = 4;
  std::vector<int> image_data(width * height, 1);

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

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], 0);
  }
}

TEST(dudchenko_o_connected_components_seq, test_two_separate_components) {
  int width = 5;
  int height = 5;
  std::vector<int> image_data = {1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1};

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

  int comp1 = output_data[0];
  int comp2 = output_data[18];

  EXPECT_NE(comp1, 0);
  EXPECT_NE(comp2, 0);
  EXPECT_NE(comp1, comp2);

  EXPECT_EQ(output_data[1], comp1);
  EXPECT_EQ(output_data[5], comp1);
  EXPECT_EQ(output_data[6], comp1);
  EXPECT_EQ(output_data[19], comp2);
  EXPECT_EQ(output_data[23], comp2);
  EXPECT_EQ(output_data[24], comp2);
}