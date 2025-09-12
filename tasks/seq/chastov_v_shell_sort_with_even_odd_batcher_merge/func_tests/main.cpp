#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/chastov_v_shell_sort_with_even_odd_batcher_merge/include/ops_seq.hpp"

namespace {
std::vector<int> GenerateRandomArray(int array_size, std::pair<int, int> value_range) {
  if (array_size <= 0) {
    throw std::invalid_argument("Invalid array size");
  }

  std::random_device random_seed;
  std::mt19937 random_engine(random_seed());
  std::uniform_int_distribution<int> value_distribution(value_range.first, value_range.second);

  std::vector<int> random_array;
  random_array.reserve(array_size);
  for (int i = 0; i < array_size; i++) {
    random_array.push_back(value_distribution(random_engine));
  }
  return random_array;
}
}  // namespace

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_positive_values) {
  // Create data
  std::vector<int> in = {300, 246, 1253, 67, 8, 900, 3421, 1, 10, 1223445};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_negative_values) {
  // Create data
  std::vector<int> in = {-300, -246, -1253, -67, -8, -900, -3421, -1, -10, -1223445};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_repeating_value) {
  // Create data
  std::vector<int> in = {10, 10, 8, 9399, 10, 10, 546, 2387, 3728};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_sorted_values) {
  // Create data
  std::vector<int> in = {1, 2, 3, 10, 30, 60, 1500, 3000, 15000};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_reverse_sorted_array) {
  // Create data
  std::vector<int> in = {15000, 3000, 5678, 1500, 60, 30, 10, 3, 2, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_single_element) {
  // Create data
  std::vector<int> in = {42};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_two_elements) {
  // Create data
  std::vector<int> in = {2, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_mixed_positive_negative) {
  // Create data
  std::vector<int> in = {-5, 3, -2, 0, 7, -1, 4};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_all_identical) {
  // Create data
  std::vector<int> in = {5, 5, 5, 5, 5, 5, 5};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_large_random) {
  // Create data
  const int array_size = 1000;
  const int max_value = 1000;
  const int min_value = -1000;

  std::vector<int> in = GenerateRandomArray(array_size, {min_value, max_value});
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_zero_values) {
  // Create data
  std::vector<int> in = {0, 0, 0, 0, 0, 0};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_duplicates_sorted_forward) {
  // Create data
  std::vector<int> in = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_duplicates_sorted_reverse) {
  // Create data
  std::vector<int> in = {5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_duplicates_mixed_order) {
  // Create data
  std::vector<int> in = {3, 1, 2, 3, 1, 2, 3, 1, 2};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_duplicates_with_extremes) {
  // Create data
  std::vector<int> in = {INT_MAX, INT_MIN, 0, INT_MAX, INT_MIN, 0, INT_MAX, INT_MIN, 0};
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge, test_multiple_duplicates_random) {
  // Create data
  std::vector<int> in;
  in.reserve(60);

  for (int i = 0; i < 10; ++i) {
    in.push_back(i);
  }

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 10; ++j) {
      in.push_back(j);
    }
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(in.begin(), in.end(), g);

  std::vector<int> out(in.size(), 0);
  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(ref, out);
}
