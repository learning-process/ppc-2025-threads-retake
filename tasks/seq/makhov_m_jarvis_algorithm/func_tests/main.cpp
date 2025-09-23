#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/makhov_m_jarvis_algorithm/include/ops_seq.hpp"

TEST(makhov_m_jarvis_algorithm_seq, test_two_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(2);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  in[0].set(0, 0);
  in[1].set(2, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), false);
}

TEST(makhov_m_jarvis_algorithm_seq, test_three_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(3);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  in[0].set(0, 0);
  in[1].set(2, 0);
  in[2].set(0, 2);

  // Create reference
  std::vector<makhov_m_jarvis_algorithm_seq::Point> reference = in;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();

  // Restore result from task_data
  size_t resored_size = task_data_seq->outputs_count[0];
  size_t pointCount = resored_size / (2 * sizeof(double));
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::convertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);
  ASSERT_EQ(reference, restored_points);

  // DEBUG
  std::cout << std::endl;
  std::cout << "Algorythm result (" << pointCount << " points):" << std::endl;
  std::cout << "RESTORED FROM TASKDATA" << std::endl;
  for (size_t i = 0; i < pointCount; i++) std::cout << "Point " << i + 1 << ": " << restored_points[i] << std::endl;

  for (uint8_t *buffer : task_data_seq->outputs) {
    delete[] buffer;
  }
}

TEST(makhov_m_jarvis_algorithm_seq, test_five_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(5);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  in[0].set(0, 0);
  in[1].set(4, 0);
  in[2].set(4, 4);
  in[3].set(0, 4);
  in[4].set(2, 2);

  // Create reference
  std::vector<makhov_m_jarvis_algorithm_seq::Point> reference(4);
  reference[0].set(0, 0);
  reference[1].set(0, 4);
  reference[2].set(4, 4);
  reference[3].set(4, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();

  // Restore result from task_data
  size_t resored_size = task_data_seq->outputs_count[0];
  size_t pointCount = resored_size / (2 * sizeof(double));
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::convertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);
  ASSERT_EQ(reference, restored_points);

  // DEBUG
  std::cout << std::endl;
  std::cout << "Algorythm result (" << pointCount << " points):" << std::endl;
  std::cout << "RESTORED FROM TASKDATA" << std::endl;
  for (size_t i = 0; i < pointCount; i++) std::cout << "Point " << i + 1 << ": " << restored_points[i] << std::endl;

  for (uint8_t *buffer : task_data_seq->outputs) {
    delete[] buffer;
  }
}

TEST(makhov_m_jarvis_algorithm_seq, test_ten_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(10);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  in[0].set(0, 0);
  in[1].set(3, 1);
  in[2].set(7, 1);
  in[3].set(8, 4);
  in[4].set(6, 6);
  in[5].set(3, 8);
  in[6].set(5, 3);
  in[7].set(3, 5);
  in[8].set(0, 5);
  in[9].set(0, 2);

  // Create reference
  std::vector<makhov_m_jarvis_algorithm_seq::Point> reference(6);
  reference[0].set(0, 0);
  reference[1].set(0, 5);
  reference[2].set(3, 8);
  reference[3].set(6, 6);
  reference[4].set(8, 4);
  reference[5].set(7, 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();

  // Restore result from task_data
  size_t resored_size = task_data_seq->outputs_count[0];
  size_t pointCount = resored_size / (2 * sizeof(double));
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::convertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);
  ASSERT_EQ(reference, restored_points);

  // DEBUG
  std::cout << std::endl;
  std::cout << "Algorythm result (" << pointCount << " points):" << std::endl;
  std::cout << "RESTORED FROM TASKDATA" << std::endl;
  for (size_t i = 0; i < pointCount; i++) std::cout << "Point " << i + 1 << ": " << restored_points[i] << std::endl;

  for (uint8_t *buffer : task_data_seq->outputs) {
    delete[] buffer;
  }
}

TEST(makhov_m_jarvis_algorithm_seq, test_five_points_negative_coords) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(5);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  in[0].set(-2, -2);
  in[1].set(2, 1);
  in[2].set(-3, 4);
  in[3].set(-1, 2);
  in[4].set(0, 1);

  // Create reference
  std::vector<makhov_m_jarvis_algorithm_seq::Point> reference(3);
  reference[0].set(-3, 4);
  reference[1].set(2, 1);
  reference[2].set(-2, -2);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();

  // Restore result from task_data
  size_t resored_size = task_data_seq->outputs_count[0];
  size_t pointCount = resored_size / (2 * sizeof(double));
  std::vector<makhov_m_jarvis_algorithm_seq::Point> restored_points =
      makhov_m_jarvis_algorithm_seq::TaskSequential::convertByteArrayToPoints(task_data_seq->outputs[0],
                                                                              task_data_seq->outputs_count[0]);
  ASSERT_EQ(reference, restored_points);

  // DEBUG
  std::cout << std::endl;
  std::cout << "Algorythm result (" << pointCount << " points):" << std::endl;
  std::cout << "RESTORED FROM TASKDATA" << std::endl;
  for (size_t i = 0; i < pointCount; i++) std::cout << "Point " << i + 1 << ": " << restored_points[i] << std::endl;

  for (uint8_t *buffer : task_data_seq->outputs) {
    delete[] buffer;
  }
}

TEST(makhov_m_jarvis_algorithm_seq, test_hundred_rand_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(100);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  for (size_t i = 0; i < in.size(); i++) {
    makhov_m_jarvis_algorithm_seq::Point point =
        makhov_m_jarvis_algorithm_seq::TaskSequential::GetRandomPoint(-10.0, 10.0, -10.0, 10.0);
    in[i] = point;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();
}

TEST(makhov_m_jarvis_algorithm_seq, test_thousand_rand_points) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_seq::Point> in(1000);
  uint32_t in_size = in.size() * 2 * sizeof(double);
  for (size_t i = 0; i < in.size(); i++) {
    makhov_m_jarvis_algorithm_seq::Point point =
        makhov_m_jarvis_algorithm_seq::TaskSequential::GetRandomPoint(-10.0, 10.0, -10.0, 10.0);
    in[i] = point;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in_size);

  // Create Task
  makhov_m_jarvis_algorithm_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();
}