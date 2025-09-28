#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "stl/ersoz_b_hoare_sort_simple_merge/include/ops_stl.hpp"

namespace {

bool IsNonDecreasing(const std::vector<int>& v) {
  if (v.empty()) {
    return true;
  }
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i - 1] > v[i]) {
      return false;
    }
  }
  return true;
}

void RunOnce(const std::vector<int>& input) {
  std::vector<int> in = input;
  std::vector<int> out(in.size(), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<ersoz_b_hoare_sort_simple_merge_stl::HoareSortSimpleMergeSTL>(task_data);
  ASSERT_TRUE(task->ValidationImpl());
  ASSERT_TRUE(task->PreProcessingImpl());
  ASSERT_TRUE(task->RunImpl());
  ASSERT_TRUE(task->PostProcessingImpl());
  ASSERT_TRUE(IsNonDecreasing(out));
}

}  // namespace

TEST(ersoz_b_hoare_sort_simple_merge_stl, random_small) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> v(1000);
  for (auto& x : v) {
    x = dist(gen);
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_stl, already_sorted) {
  std::vector<int> v(5000);
  for (int i = 0; i < 5000; ++i) {
    v[i] = i;
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_stl, reverse_sorted) {
  std::vector<int> v(5000);
  for (int i = 0; i < 5000; ++i) {
    v[i] = 5000 - i;
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_stl, duplicates) {
  std::vector<int> v;
  v.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    v.push_back(i % 7);
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_stl, single_element) { RunOnce({42}); }
