#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/ersoz_b_hoare_sort_simple_merge/include/ops_omp.hpp"

namespace {

void RunOnce(const std::vector<int>& input) {
  std::vector<int> in = input;
  std::vector<int> out(in.size(), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<ersoz_b_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP>(task_data);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

}  // namespace

TEST(ersoz_b_hoare_sort_simple_merge_omp, already_sorted) {
  std::vector<int> v(100);
  for (int i = 0; i < 100; ++i) {
    v[static_cast<std::size_t>(i)] = i;
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_omp, reverse_sorted) {
  std::vector<int> v(100);
  for (int i = 0; i < 100; ++i) {
    v[static_cast<std::size_t>(i)] = 99 - i;
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_omp, duplicates) {
  std::vector<int> v;
  v.reserve(200);
  for (int i = 0; i < 200; ++i) {
    v.push_back(i % 7);
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_omp, single_element) { RunOnce({42}); }

TEST(ersoz_b_hoare_sort_simple_merge_omp, random_large) {
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dist(-100000, 100000);
  std::vector<int> v(5000);
  for (auto& x : v) {
    x = dist(gen);
  }
  RunOnce(v);
}