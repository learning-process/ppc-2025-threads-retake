#include <gtest/gtest.h>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/anikin_m_block_gauss/include/ops_seq.hpp"

TEST(anikin_m_block_gauss_seq, zero_test) {
  int x = 5;
  int y = 5;

  std::vector<double> image = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector image_res(x * y, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(x);
  task_data_seq->inputs_count.emplace_back(y);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(x);
  task_data_seq->outputs_count.emplace_back(y);

  // Create Task
  anikin_m_block_gauss_seq::BlockGaussSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(image, image_res);
}

TEST(anikin_m_block_gauss_seq, calc_test) {
  int x = 5;
  int y = 5;

  std::vector<double> image = {2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2};

  std::vector image_res(x * y, 0.0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(x);
  task_data_seq->inputs_count.emplace_back(y);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(x);
  task_data_seq->outputs_count.emplace_back(y);

  // Create Task
  anikin_m_block_gauss_seq::BlockGaussSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<double> real_res = {2,    2, 3, 2,    2,   2,    2.25, 2.5, 2.25, 2, 2, 2.25, 2.5,
                                  2.25, 2, 2, 2.25, 2.5, 2.25, 2,    2,   2,    3, 2, 2};

  EXPECT_EQ(real_res, image_res);
}

TEST(anikin_m_block_gauss_seq, size_validation_test0) {
  int n = 0;
  int m = 0;
  std::vector<int> image(n * m, 1);
  std::vector<int> image_res(n * m, 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  anikin_m_block_gauss_seq::BlockGaussSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(anikin_m_block_gauss_seq, size_validation_test1) {
  int n = 0;
  int m = 0;
  std::vector<int> image(n * m, 1);
  std::vector<int> image_res((n + 1) * m, 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res.data()));
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  anikin_m_block_gauss_seq::BlockGaussSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
