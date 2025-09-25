#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/makhov_m_jarvis_algorithm/include/ops_seq.hpp"

TEST(makhov_m_jarvis_algorithm_seq, test_two_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(2);
  in[0].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  in[1].Set(makhov_m_jarvis_algorithm_seq::XCoord(2.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), false);

  // Cleanup
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_seq, test_three_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(3);
  in[0].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  in[1].Set(makhov_m_jarvis_algorithm_seq::XCoord(2.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  in[2].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(2.0));

  // Create reference
  std::vector<makhov_m_jarvis_algorithm_seq::Point> reference = in;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  ASSERT_EQ(task_sequential.PreProcessing(), true);
  ASSERT_EQ(task_sequential.Run(), true);
  ASSERT_EQ(task_sequential.PostProcessing(), true);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);
  ASSERT_EQ(reference.size(), restored_points.size());

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_seq, test_five_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(5);
  in[0].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  in[1].Set(makhov_m_jarvis_algorithm_seq::XCoord(4.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  in[2].Set(makhov_m_jarvis_algorithm_seq::XCoord(4.0), makhov_m_jarvis_algorithm_seq::YCoord(4.0));
  in[3].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(4.0));
  in[4].Set(makhov_m_jarvis_algorithm_seq::XCoord(2.0), makhov_m_jarvis_algorithm_seq::YCoord(2.0));

  // Create reference (convex hull should be the 4 corner points)
  std::vector<makhov_m_jarvis_algorithm_seq::Point> reference(4);
  reference[0].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  reference[1].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(4.0));
  reference[2].Set(makhov_m_jarvis_algorithm_seq::XCoord(4.0), makhov_m_jarvis_algorithm_seq::YCoord(4.0));
  reference[3].Set(makhov_m_jarvis_algorithm_seq::XCoord(4.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  ASSERT_EQ(task_sequential.PreProcessing(), true);
  ASSERT_EQ(task_sequential.Run(), true);
  ASSERT_EQ(task_sequential.PostProcessing(), true);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);

  ASSERT_EQ(reference.size(), restored_points.size());

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_seq, test_ten_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(10);
  in[0].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(0.0));
  in[1].Set(makhov_m_jarvis_algorithm_seq::XCoord(3.0), makhov_m_jarvis_algorithm_seq::YCoord(1.0));
  in[2].Set(makhov_m_jarvis_algorithm_seq::XCoord(7.0), makhov_m_jarvis_algorithm_seq::YCoord(1.0));
  in[3].Set(makhov_m_jarvis_algorithm_seq::XCoord(8.0), makhov_m_jarvis_algorithm_seq::YCoord(4.0));
  in[4].Set(makhov_m_jarvis_algorithm_seq::XCoord(6.0), makhov_m_jarvis_algorithm_seq::YCoord(6.0));
  in[5].Set(makhov_m_jarvis_algorithm_seq::XCoord(3.0), makhov_m_jarvis_algorithm_seq::YCoord(8.0));
  in[6].Set(makhov_m_jarvis_algorithm_seq::XCoord(5.0), makhov_m_jarvis_algorithm_seq::YCoord(3.0));
  in[7].Set(makhov_m_jarvis_algorithm_seq::XCoord(3.0), makhov_m_jarvis_algorithm_seq::YCoord(5.0));
  in[8].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(5.0));
  in[9].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(2.0));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  ASSERT_EQ(task_sequential.PreProcessing(), true);
  ASSERT_EQ(task_sequential.Run(), true);
  ASSERT_EQ(task_sequential.PostProcessing(), true);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);

  // Should have a convex hull with at least 3 points
  size_t expected_size = 3;
  ASSERT_GE(restored_points.size(), expected_size);

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_seq, test_five_points_negative_coords) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(5);
  in[0].Set(makhov_m_jarvis_algorithm_seq::XCoord(-2.0), makhov_m_jarvis_algorithm_seq::YCoord(-2.0));
  in[1].Set(makhov_m_jarvis_algorithm_seq::XCoord(2.0), makhov_m_jarvis_algorithm_seq::YCoord(1.0));
  in[2].Set(makhov_m_jarvis_algorithm_seq::XCoord(-3.0), makhov_m_jarvis_algorithm_seq::YCoord(4.0));
  in[3].Set(makhov_m_jarvis_algorithm_seq::XCoord(-1.0), makhov_m_jarvis_algorithm_seq::YCoord(2.0));
  in[4].Set(makhov_m_jarvis_algorithm_seq::XCoord(0.0), makhov_m_jarvis_algorithm_seq::YCoord(1.0));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  ASSERT_EQ(task_sequential.PreProcessing(), true);
  ASSERT_EQ(task_sequential.Run(), true);
  ASSERT_EQ(task_sequential.PostProcessing(), true);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);

  // Should have a convex hull with at least 3 points
  size_t expected_size = 3;
  ASSERT_GE(restored_points.size(), expected_size);

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_seq, test_hundred_rand_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(100);

  for (size_t i = 0; i < in.size(); i++) {
    makhov_m_jarvis_algorithm_seq::Point point = makhov_m_jarvis_algorithm_seq::TaskSequential::GetRandomPoint(
        makhov_m_jarvis_algorithm_seq::XCoord(-10.0), makhov_m_jarvis_algorithm_seq::XCoord(10.0),
        makhov_m_jarvis_algorithm_seq::YCoord(-10.0), makhov_m_jarvis_algorithm_seq::YCoord(10.0));
    in[i] = point;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  ASSERT_EQ(task_sequential.PreProcessing(), true);
  ASSERT_EQ(task_sequential.Run(), true);
  ASSERT_EQ(task_sequential.PostProcessing(), true);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);

  // Should have a convex hull with at least 3 points
  size_t expected_size = 3;
  ASSERT_GE(restored_points.size(), expected_size);

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_seq, test_thousand_rand_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(1000);

  for (size_t i = 0; i < in.size(); i++) {
    makhov_m_jarvis_algorithm_seq::Point point = makhov_m_jarvis_algorithm_seq::TaskSequential::GetRandomPoint(
        makhov_m_jarvis_algorithm_seq::XCoord(-10.0), makhov_m_jarvis_algorithm_seq::XCoord(10.0),
        makhov_m_jarvis_algorithm_seq::YCoord(-10.0), makhov_m_jarvis_algorithm_seq::YCoord(10.0));
    in[i] = point;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertPointsToByteArray(in, buffer_size);

  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  ASSERT_EQ(task_sequential.PreProcessing(), true);
  ASSERT_EQ(task_sequential.Run(), true);
  ASSERT_EQ(task_sequential.PostProcessing(), true);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);

  // Should have a convex hull with at least 3 points
  size_t expected_size = 3;
  ASSERT_GE(restored_points.size(), expected_size);

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}