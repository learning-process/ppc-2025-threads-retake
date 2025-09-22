#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/dormidontov_e_sign_sort_batcher/include/ops_seq.hpp"

TEST(dormidontov_e_sign_sort_batcher_tbb, test_50) {
  constexpr size_t kCount = 50;

  // Create data
  std::vector<double> in(kCount, 0);
  std::vector<double> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<double>(i);
  }

  out = in;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  dormidontov_e_sign_sort_batcher_tbb::TbbTask test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(dormidontov_e_sign_sort_batcher_tbb, test_0) {
  // Create data
  std::vector<double> in;
  std::vector<double> out;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  dormidontov_e_sign_sort_batcher_tbb::TbbTask test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(dormidontov_e_sign_sort_batcher_tbb, test_1) {
  constexpr size_t kCount = 1;

  // Create data
  std::vector<double> in(kCount, 0);
  std::vector<double> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<double>(i);
  }

  out = in;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  dormidontov_e_sign_sort_batcher_tbb::TbbTask test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(dormidontov_e_sign_sort_batcher_tbb, test_1000_shuffe) {
  constexpr size_t kCount = 1000;

  srand(1337);
  // Create data
  std::vector<double> in(kCount, 0);
  std::vector<double> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = rand() % 1000;
  }

  out = in;
  std::ranges::sort(begin(in), end(in));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  dormidontov_e_sign_sort_batcher_tbb::TbbTask test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(dormidontov_e_sign_sort_batcher_tbb, test_wrong_size) {
  constexpr size_t kCount = 50;

  // Create data
  std::vector<double> in(kCount, 0);
  std::vector<double> out(kCount - 1, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<double>(i);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  dormidontov_e_sign_sort_batcher_tbb::TbbTask test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
