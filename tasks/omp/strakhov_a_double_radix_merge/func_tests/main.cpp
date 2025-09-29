#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
// #include "core/util/include/util.hpp"
#include "omp/strakhov_a_double_radix_merge/include/ops_omp.hpp"

namespace {
std::vector<double> RunMyTask(const std::vector<double> &input) {
  std::vector<double> out(input.size());
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input.data())));
  task_data_omp->inputs_count.emplace_back(input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  strakhov_a_double_radix_merge_omp::DoubleRadixMergeOmp task(task_data_omp);
  EXPECT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  return out;
}
}  // namespace
TEST(strakhov_a_double_radix_merge, test_simple1) {
  std::vector<double> in{1.0, 2.2, 3.3, -4.4};
  std::vector<double> expected{-4.4, 1.0, 2.2, 3.3};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}

TEST(strakhov_a_double_radix_merge, test_simple2) {
  std::vector<double> in{1.0, 3.3, 2.2, 3.3, 3.4, 3.3, -4.4, 3.3};
  std::vector<double> expected{-4.4, 1.0, 2.2, 3.3, 3.3, 3.3, 3.3, 3.4};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_one_element) {
  std::vector<double> in{1.0};
  std::vector<double> expected{1.0};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_zero_elements) {
  std::vector<double> in{};
  std::vector<double> expected{};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_equal_elements) {
  std::vector<double> in{1.1, 1.1, 1.1};
  std::vector<double> expected{1.1, 1.1, 1.1};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_zeroes) {
  std::vector<double> in{+0.0, +0.0, -0.0};
  std::vector<double> expected{-0.0, +0.0, +0.0};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_infinities) {
  std::vector<double> in{1.0, 0.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  std::vector<double> expected{-std::numeric_limits<double>::infinity(), 0.0, 1.0,
                               std::numeric_limits<double>::infinity()};
  std::vector<double> out = RunMyTask(in);
  EXPECT_EQ(out, expected);
}

TEST(strakhov_a_double_radix_merge, test_random1) {
  std::random_device randomizer;
  std::mt19937 gen(randomizer());
  std::vector<double> in(100, 0);
  std::uniform_real_distribution<double> dist(-12.0, 12.0);
  for (auto &v : in) {
    v = dist(gen);
  }
  auto out = RunMyTask(in);
  auto expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_random2) {
  std::random_device randomizer;
  std::mt19937 gen(randomizer());
  std::vector<double> in(1000, 0);
  std::uniform_real_distribution<double> dist(-0.001, 0.001);
  for (auto &v : in) {
    v = dist(gen);
  }
  auto out = RunMyTask(in);
  auto expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(out, expected);
}
TEST(strakhov_a_double_radix_merge, test_random3) {
  std::random_device randomizer;
  std::mt19937 gen(randomizer());
  std::vector<double> in(1000, 0);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
  for (auto &v : in) {
    v = dist(gen);
  }
  auto out = RunMyTask(in);
  auto expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(out, expected);
}
