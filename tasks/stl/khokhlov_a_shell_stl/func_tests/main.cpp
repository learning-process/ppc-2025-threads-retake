#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/khokhlov_a_shell_stl/include/ops_stl.hpp"

TEST(khokhlov_a_shell_stl, Shell_Validation_Fail) {
  // Create data
  std::vector<int> in1 = khokhlov_a_shell_stl::GenerateRandomVector(10);
  std::vector<int> in2 = std::vector<int>(5);
  std::vector<int> out(10, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_stl = std::make_shared<ppc::core::TaskData>();
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  test_task_stl->inputs_count.emplace_back(in1.size());
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  test_task_stl->inputs_count.emplace_back(in2.size());
  test_task_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_stl->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_stl::ShellStl test_task_stl_task(test_task_stl);
  ASSERT_EQ(test_task_stl_task.ValidationImpl(), false);
}

TEST(khokhlov_a_shell_stl, Shell_Random_10) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_stl::GenerateRandomVector(10);
  std::vector<int> out(10, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_stl = std::make_shared<ppc::core::TaskData>();
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_stl->inputs_count.emplace_back(in.size());
  test_task_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_stl->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_stl::ShellStl test_task_stl_task(test_task_stl);
  ASSERT_EQ(test_task_stl_task.ValidationImpl(), true);
  test_task_stl_task.PreProcessingImpl();
  test_task_stl_task.RunImpl();
  test_task_stl_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_stl::CheckSorted(out));
}

TEST(khokhlov_a_shell_stl, Shell_Random_20) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_stl::GenerateRandomVector(20);
  std::vector<int> out(20, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_stl = std::make_shared<ppc::core::TaskData>();
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_stl->inputs_count.emplace_back(in.size());
  test_task_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_stl->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_stl::ShellStl test_task_stl_task(test_task_stl);
  ASSERT_EQ(test_task_stl_task.ValidationImpl(), true);
  test_task_stl_task.PreProcessingImpl();
  test_task_stl_task.RunImpl();
  test_task_stl_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_stl::CheckSorted(out));
}

TEST(khokhlov_a_shell_stl, Shell_Random_50) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_stl::GenerateRandomVector(50);
  std::vector<int> out(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_stl = std::make_shared<ppc::core::TaskData>();
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_stl->inputs_count.emplace_back(in.size());
  test_task_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_stl->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_stl::ShellStl test_task_stl_task(test_task_stl);
  ASSERT_EQ(test_task_stl_task.ValidationImpl(), true);
  test_task_stl_task.PreProcessingImpl();
  test_task_stl_task.RunImpl();
  test_task_stl_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_stl::CheckSorted(out));
}

TEST(khokhlov_a_shell_stl, Shell_Random_70) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_stl::GenerateRandomVector(70);
  std::vector<int> out(70, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_stl = std::make_shared<ppc::core::TaskData>();
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_stl->inputs_count.emplace_back(in.size());
  test_task_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_stl->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_stl::ShellStl test_task_stl_task(test_task_stl);
  ASSERT_EQ(test_task_stl_task.ValidationImpl(), true);
  test_task_stl_task.PreProcessingImpl();
  test_task_stl_task.RunImpl();
  test_task_stl_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_stl::CheckSorted(out));
}

TEST(khokhlov_a_shell_stl, Shell_Random_100) {
  // Create data
  std::vector<int> in = khokhlov_a_shell_stl::GenerateRandomVector(100);
  std::vector<int> out(100, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test_task_stl = std::make_shared<ppc::core::TaskData>();
  test_task_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  test_task_stl->inputs_count.emplace_back(in.size());
  test_task_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  test_task_stl->outputs_count.emplace_back(out.size());

  // Create Task
  khokhlov_a_shell_stl::ShellStl test_task_stl_task(test_task_stl);
  ASSERT_EQ(test_task_stl_task.ValidationImpl(), true);
  test_task_stl_task.PreProcessingImpl();
  test_task_stl_task.RunImpl();
  test_task_stl_task.PostProcessingImpl();
  ASSERT_TRUE(khokhlov_a_shell_stl::CheckSorted(out));
}
