#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/leontev_n_graham/include/ops_omp.hpp"

namespace {
std::vector<float> GenVec(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> dist(-5.0F, 5.0F);
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
  std::vector<float> in_x = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
  std::vector<float> in_y = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 8.0F};
  std::vector<float> out_x(kCount, 0.0F);
  std::vector<float> out_y(kCount, 0.0F);
  int out_size = 0;
  std::vector<float> true_result_x = {0.0F, 9.0F, 8.0F};
  std::vector<float> true_result_y = {0.0F, 8.0F, 8.0F};

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_omp->inputs_count.emplace_back(in_x.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_omp->outputs_count.emplace_back(out_x.size());

  // Create Task
  leontev_n_graham_omp::GrahamOmp test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  out_x.resize(out_size);
  out_y.resize(out_size);
  EXPECT_EQ(out_x, true_result_x);
  EXPECT_EQ(out_y, true_result_y);
}

TEST(leontev_n_graham_omp, test_20_points_square) {
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
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_omp->inputs_count.emplace_back(in_x.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_omp->outputs_count.emplace_back(out_x.size());

  // Create Task
  leontev_n_graham_omp::GrahamOmp test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  out_x.resize(out_size);
  out_y.resize(out_size);
  EXPECT_EQ(out_x, true_result_x);
  EXPECT_EQ(out_y, true_result_y);
}

TEST(leontev_n_graham_omp, test_lot_of_points) {
  constexpr size_t kCount = 500;

  // Create data
  std::vector<float> in_x = GenVec(kCount);
  std::vector<float> in_y = GenVec(kCount);
  std::vector<float> out_x(kCount, 0.0F);
  std::vector<float> out_y(kCount, 0.0F);
  int out_size = 0;
  std::vector<float> true_result_x(kCount, 0.0F);
  std::vector<float> true_result_y(kCount, 0.0F);
  int out_size_seq = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_seq->inputs_count.emplace_back(in_x.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(true_result_x.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(true_result_y.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size_seq));
  task_data_seq->outputs_count.emplace_back(true_result_x.size());

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_omp->inputs_count.emplace_back(in_x.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_omp->outputs_count.emplace_back(out_x.size());

  // Create Task
  leontev_n_graham_omp::GrahamSeq test_task_seq(task_data_seq);
  leontev_n_graham_omp::GrahamOmp test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_seq.Validation(), true);
  test_task_seq.PreProcessing();
  test_task_seq.Run();
  test_task_seq.PostProcessing();
  true_result_x.resize(out_size_seq);
  true_result_y.resize(out_size_seq);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  out_x.resize(out_size);
  out_y.resize(out_size);
  EXPECT_EQ(out_x, true_result_x);
  EXPECT_EQ(out_y, true_result_y);
}
