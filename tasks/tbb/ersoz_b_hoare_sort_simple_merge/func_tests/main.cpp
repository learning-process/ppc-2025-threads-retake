#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/ersoz_b_hoare_sort_simple_merge/include/ops_tbb.hpp"

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

  ersoz_b_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_TRUE(IsNonDecreasing(out));
}

}  // namespace

TEST(ersoz_b_hoare_sort_simple_merge_tbb, small_random) {
  std::mt19937 gen(1337);
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> v(100);
  for (auto& x : v) {
    x = dist(gen);
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_tbb, already_sorted) {
  std::vector<int> v(100);
  for (int i = 0; i < 100; ++i) {
    v[static_cast<std::size_t>(i)] = i;
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_tbb, reverse_sorted) {
  std::vector<int> v(100);
  for (int i = 0; i < 100; ++i) {
    v[static_cast<std::size_t>(i)] = 99 - i;
  }
  RunOnce(v);
}

TEST(ersoz_b_hoare_sort_simple_merge_tbb, many_duplicates) {
  std::vector<int> v;
  v.reserve(200);
  for (int i = 0; i < 200; ++i) {
    v.push_back(i % 5);
  }
  RunOnce(v);
}
