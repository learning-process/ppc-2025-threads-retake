#include <gtest/gtest.h>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/anikin_m_lexic_check/include/ops_seq.hpp"

TEST(anikin_m_lexic_check_seq, test_str5_str5_ret0) {
  // Create data
  std::string in0 = "aaaaa";
  std::string in1 = "aaaaa";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  anikin_m_lexic_check_seq::LexicCheckSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ret, 0);
}

TEST(anikin_m_lexic_check_seq, test_str5_str5_retn1) {
  // Create data
  std::string in0 = "aaaaa";
  std::string in1 = "bbbbb";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  anikin_m_lexic_check_seq::LexicCheckSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ret, -1);
}

TEST(anikin_m_lexic_check_seq, test_str5_str5_ret1) {
  // Create data
  std::string in0 = "bbbbb";
  std::string in1 = "aaaaa";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  anikin_m_lexic_check_seq::LexicCheckSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ret, 1);
}

TEST(anikin_m_lexic_check_seq, test_empty) {
  // Create data
  std::string in0 = "";
  std::string in1 = "";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  anikin_m_lexic_check_seq::LexicCheckSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ret, 0);
}

TEST(anikin_m_lexic_check_seq, test_str1_str0_ret1) {
  // Create data
  std::string in0 = "a";
  std::string in1 = "";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  anikin_m_lexic_check_seq::LexicCheckSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ret, 1);
}

TEST(anikin_m_lexic_check_seq, test_str0_str1_retn1) {
  // Create data
  std::string in0 = "";
  std::string in1 = "a";
  int ret = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in0.data()));
  task_data_seq->inputs_count.emplace_back(in0.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&ret));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  anikin_m_lexic_check_seq::LexicCheckSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ret, -1);
}
