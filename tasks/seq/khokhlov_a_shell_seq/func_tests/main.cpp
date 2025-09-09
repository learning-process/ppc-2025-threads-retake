#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "seq/khokhlov_a_shell_seq/include/ops_seq.hpp"

TEST(khokhlov_a_shell_seq, Shell_Validation_Fail) {
  // Create data
  std::vector<int> in1 = khokhlov_a_shell_seq::GenerateRandomVector(10, 1, 100);
  std::vector<int> in2 = std::vector<int>(5);
  std::vector<int> out(10, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_sequential = std::make_shared<ppc::core::TaskData>();
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  test_task_sequential->inputs_count.emplace_back(in1.size());
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  test_task_sequential->inputs_count.emplace_back(in2.size());
  test_task_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_sequential->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq test_task_sequential_task(test_task_sequential);
  ASSERT_EQ(test_task_sequential_task.ValidationImpl(), false);
}

TEST(khokhlov_a_shell_seq, Shell_Random_10) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_seq::GenerateRandomVector(10, 1, 100);
  std::vector<int> out(10, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_sequential = std::make_shared<ppc::core::TaskData>();
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_sequential->inputs_count.emplace_back(in.size());
  test_task_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_sequential->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq test_task_sequential_task(test_task_sequential);
  ASSERT_EQ(test_task_sequential_task.ValidationImpl(), true);
  test_task_sequential_task.PreProcessingImpl();
  test_task_sequential_task.RunImpl();
  test_task_sequential_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_seq::CheckSorted(out));
}

TEST(khokhlov_a_shell_seq, Shell_Random_20) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_seq::GenerateRandomVector(20, 1, 100);
  std::vector<int> out(20, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_sequential = std::make_shared<ppc::core::TaskData>();
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_sequential->inputs_count.emplace_back(in.size());
  test_task_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_sequential->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq test_task_sequential_task(test_task_sequential);
  ASSERT_EQ(test_task_sequential_task.ValidationImpl(), true);
  test_task_sequential_task.PreProcessingImpl();
  test_task_sequential_task.RunImpl();
  test_task_sequential_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_seq::CheckSorted(out));
}

TEST(khokhlov_a_shell_seq, Shell_Random_50) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_seq::GenerateRandomVector(50, 1, 100);
  std::vector<int> out(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_sequential = std::make_shared<ppc::core::TaskData>();
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_sequential->inputs_count.emplace_back(in.size());
  test_task_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_sequential->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq test_task_sequential_task(test_task_sequential);
  ASSERT_EQ(test_task_sequential_task.ValidationImpl(), true);
  test_task_sequential_task.PreProcessingImpl();
  test_task_sequential_task.RunImpl();
  test_task_sequential_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_seq::CheckSorted(out));
}

TEST(khokhlov_a_shell_seq, Shell_Random_70) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_seq::GenerateRandomVector(70, 1, 100);
  std::vector<int> out(70, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_sequential = std::make_shared<ppc::core::TaskData>();
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_sequential->inputs_count.emplace_back(in.size());
  test_task_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_sequential->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq test_task_sequential_task(test_task_sequential);
  ASSERT_EQ(test_task_sequential_task.ValidationImpl(), true);
  test_task_sequential_task.PreProcessingImpl();
  test_task_sequential_task.RunImpl();
  test_task_sequential_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_seq::CheckSorted(out));
}

TEST(khokhlov_a_shell_seq, Shell_Random_100) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_seq::GenerateRandomVector(100, 1, 100);
  std::vector<int> out(100, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_sequential = std::make_shared<ppc::core::TaskData>();
  test_task_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_sequential->inputs_count.emplace_back(in.size());
  test_task_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_sequential->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_seq::ShellSeq test_task_sequential_task(test_task_sequential);
  ASSERT_EQ(test_task_sequential_task.ValidationImpl(), true);
  test_task_sequential_task.PreProcessingImpl();
  test_task_sequential_task.RunImpl();
  test_task_sequential_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_seq::CheckSorted(out));
}