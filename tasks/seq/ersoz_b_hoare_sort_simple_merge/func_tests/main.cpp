#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/ersoz_b_hoare_sort_simple_merge/include/ops_seq.hpp"

using ersoz_b_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential;

static void RunOnce(std::vector<int> in) {
  std::vector<int> out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  HoareSortSimpleMergeSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  std::sort(in.begin(), in.end());
  EXPECT_EQ(in, out);
}

TEST(ersoz_b_hoare_sort_simple_merge_seq, empty) { RunOnce({}); }

TEST(ersoz_b_hoare_sort_simple_merge_seq, single) { RunOnce({42}); }

TEST(ersoz_b_hoare_sort_simple_merge_seq, sorted) {
  std::vector<int> v(100);
  for (int i = 0; i < 100; ++i) v[i] = i;
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_seq, reversed) {
  std::vector<int> v(100);
  for (int i = 0; i < 100; ++i) v[i] = 99 - i;
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_seq, duplicates) {
  std::vector<int> v;
  for (int i = 0; i < 100; ++i) v.push_back(i % 5);
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_seq, negatives) {
  std::vector<int> v = {5, -1, 3, -7, 2, 0, -3, 8, -2, 4};
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_seq, random_large) {
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dist(-100000, 100000);
  std::vector<int> v(10000);
  for (auto& x : v) x = dist(gen);
  RunOnce(v);
}
