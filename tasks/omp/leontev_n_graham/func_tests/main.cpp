#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <iostream>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "omp/leontev_n_graham/include/ops_omp.hpp"

namespace {
std::vector<float> GenVec(int size, float max = 5.0f) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> dist(-max, max);
  std::vector<float> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace

TEST(leontev_n_graham_omp, test_10_points_basic) {
  constexpr size_t kCount = 10;

  // Create data
  std::vector<float> in_X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> in_Y = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f};
  std::vector<float> out_X(kCount, 0.0f);
  std::vector<float> out_Y(kCount, 0.0f);
  int out_size = 0;
  std::vector<float> true_result_X = {0.0f, 9.0f, 8.0f};
  std::vector<float> true_result_Y = {0.0f, 8.0f, 8.0f};

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_X.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_Y.data()));
  task_data_omp->inputs_count.emplace_back(in_X.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_X.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_Y.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_omp->outputs_count.emplace_back(out_X.size());

  // Create Task
  leontev_n_graham_omp::GrahamOmp test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  out_X.resize(out_size);
  out_Y.resize(out_size);
  EXPECT_EQ(out_X, true_result_X);
  EXPECT_EQ(out_Y, true_result_Y);
}

TEST(leontev_n_graham_omp, test_20_points_square) {
  constexpr size_t kCount = 20;

  // Create data
  std::vector<float> in_X = GenVec(kCount);
  std::vector<float> in_Y = GenVec(kCount);
  in_X[0] = -10.0f;
  in_Y[0] = -10.0f;
  in_X[1] = 10.0f;
  in_Y[1] = -10.0f;
  in_X[2] = 10.0f;
  in_Y[2] = 10.0f;
  in_X[3] = -10.0f;
  in_Y[3] = 10.0f;
  std::vector<float> out_X(kCount, 0.0f);
  std::vector<float> out_Y(kCount, 0.0f);
  int out_size = 0;
  std::vector<float> true_result_X = {-10.0f, 10.0f, 10.0f, -10.0f};
  std::vector<float> true_result_Y = {-10.0f, -10.0f, 10.0f, 10.0f};

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_X.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_Y.data()));
  task_data_omp->inputs_count.emplace_back(in_X.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_X.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_Y.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_omp->outputs_count.emplace_back(out_X.size());

  // Create Task
  leontev_n_graham_omp::GrahamOmp test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  out_X.resize(out_size);
  out_Y.resize(out_size);
  EXPECT_EQ(out_X, true_result_X);
  EXPECT_EQ(out_Y, true_result_Y);
}

TEST(leontev_n_graham_omp, test_lot_of_points) {
  constexpr size_t kCount = 500;

  // Create data
  std::vector<float> in_X = GenVec(kCount, 5.0f);
  std::vector<float> in_Y = GenVec(kCount, 5.0f);
  std::vector<float> out_X(kCount, 0.0f);
  std::vector<float> out_Y(kCount, 0.0f);
  int out_size = 0;
  std::vector<float> true_result_X(kCount, 0.0f);
  std::vector<float> true_result_Y(kCount, 0.0f);
  int out_size_seq = 0;

    // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_X.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_Y.data()));
  task_data_seq->inputs_count.emplace_back(in_X.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(true_result_X.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(true_result_Y.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size_seq));
  task_data_seq->outputs_count.emplace_back(true_result_X.size());

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_X.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_Y.data()));
  task_data_omp->inputs_count.emplace_back(in_X.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_X.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_Y.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_omp->outputs_count.emplace_back(out_X.size());

  // Create Task
  leontev_n_graham_omp::GrahamSeq test_task_seq(task_data_seq);
  leontev_n_graham_omp::GrahamOmp test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_seq.Validation(), true);
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();
  true_result_X.resize(out_size_seq);
  true_result_Y.resize(out_size_seq);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  out_X.resize(out_size);
  out_Y.resize(out_size);
  EXPECT_EQ(out_X, true_result_X);
  EXPECT_EQ(out_Y, true_result_Y);
}
