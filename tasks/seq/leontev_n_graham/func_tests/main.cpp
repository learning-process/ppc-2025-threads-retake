#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/leontev_n_graham/include/ops_seq.hpp"

namespace {
std::vector<float> GenVec(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> dist(-5, 5);
  std::vector<float> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace

TEST(leontev_n_graham_seq, test_10_points_basic) {
  constexpr size_t kCount = 10;

  // Create data
  std::vector<float> in_x = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
  std::vector<float> in_y = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 8.0F};
  std::vector<float> out_x(kCount, 0.0F);
  std::vector<float> out_y(kCount, 0.0F);
  int out_size = 0;
  std::vector<float> true_result_x = {0.0F, 9.0F, 8.0F};
  std::vector<float> true_result_y = {0.0F, 8.0F, 8.0F};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_seq->inputs_count.emplace_back(in_x.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_seq->outputs_count.emplace_back(out_x.size());

  // Create Task
  leontev_n_graham_seq::GrahamSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  out_x.resize(out_size);
  out_y.resize(out_size);
  EXPECT_EQ(out_x, true_result_x);
  EXPECT_EQ(out_y, true_result_y);
}

TEST(leontev_n_graham_seq, test_20_points_square) {
  constexpr size_t kCount = 20;

  // Create data
  std::vector<float> in_x = GenVec(kCount);
  std::vector<float> in_y = GenVec(kCount);
  in_x[0] = -10.0F;
  in_y[0] = -10.0F;
  in_x[1] = 10.0F;
  in_y[1] = -10.0F;
  in_x[2] = 10.0F;
  in_y[2] = 10.0F;
  in_x[3] = -10.0F;
  in_y[3] = 10.0F;
  std::vector<float> out_x(kCount, 0.0F);
  std::vector<float> out_y(kCount, 0.0F);
  int out_size = 0;
  std::vector<float> true_result_x = {-10.0F, 10.0F, 10.0F, -10.0F};
  std::vector<float> true_result_y = {-10.0F, -10.0F, 10.0F, 10.0F};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_seq->inputs_count.emplace_back(in_x.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_seq->outputs_count.emplace_back(out_x.size());

  // Create Task
  leontev_n_graham_seq::GrahamSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  out_x.resize(out_size);
  out_y.resize(out_size);
  EXPECT_EQ(out_x, true_result_x);
  EXPECT_EQ(out_y, true_result_y);
}
