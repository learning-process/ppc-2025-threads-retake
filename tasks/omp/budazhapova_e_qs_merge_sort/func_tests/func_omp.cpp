#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "omp/budazhapova_e_qs_merge_sort/include/ops_omp_inc.hpp"

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_small_array) {
  constexpr size_t kCount = 10;

  // Create data - small unsorted array
  std::vector<int> in = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
  std::vector<int> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> out(kCount, 0);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_medium_array) {
  constexpr size_t kCount = 100;

  std::vector<int> in(kCount);
  std::iota(in.begin(), in.end(), 0);
  std::vector<int> expected = in;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(in.begin(), in.end(), g);

  std::vector<int> out(kCount, 0);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_large_array) {
  constexpr size_t kCount = 1000;

  // Create data - large random array
  std::vector<int> in(kCount);
  std::iota(in.begin(), in.end(), 0);
  std::vector<int> expected = in;

  // Reverse the array to make it unsorted
  std::reverse(in.begin(), in.end());

  std::vector<int> out(kCount, 0);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_already_sorted) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount);
  std::iota(in.begin(), in.end(), 0);  // 0, 1, 2, ..., 49
  std::vector<int> expected = in;
  std::vector<int> out(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_reverse_sorted) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount);
  std::iota(in.rbegin(), in.rend(), 0);
  std::vector<int> expected(kCount);
  std::iota(expected.begin(), expected.end(), 0);
  std::vector<int> out(kCount, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_with_duplicates) {
  constexpr size_t kCount = 20;

  std::vector<int> in = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4};
  std::vector<int> expected = {1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7, 8, 8, 9, 9, 9};
  std::vector<int> out(kCount, 0);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_single_element) {
  constexpr size_t kCount = 1;

  std::vector<int> in = {42};
  std::vector<int> expected = {42};
  std::vector<int> out(kCount, 0);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_sort_empty_array) {
  constexpr size_t kCount = 0;

  std::vector<int> in = {};
  std::vector<int> expected = {};
  std::vector<int> out = {};

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(budazhapova_e_qs_merge_sort_omp, test_validation_fail) {
  constexpr size_t kCount = 10;

  std::vector<int> in(kCount, 1);
  std::vector<int> out(kCount + 5, 0);  // Different size!

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  // Create Task - validation should fail
  budazhapova_e_qs_merge_sort_omp::QSMergeSortOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}