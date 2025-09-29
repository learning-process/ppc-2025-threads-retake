#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "stl/yasakova_t_sort/include/radix_double_stl.hpp"

using namespace yasakova_t_sort_stl;

TEST(yasakova_t_sort_stl, small_basic) {
  std::vector<double> values{3.0, -1.0, 0.0, -0.0, 2.5, -10.0, 2.5};
  auto reference = values;
  std::ranges::sort(reference);
  RadixSortDoubleStl(values);
  EXPECT_EQ(values, reference);
}

TEST(yasakova_t_sort_stl, random_100k) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> distribution(-1e9, 1e9);
  std::vector<double> values(100000);
  for (auto& value : values) {
    value = distribution(rng);
  }
  auto reference = values;
  std::ranges::sort(reference);
  RadixSortDoubleStl(values);
  EXPECT_EQ(values, reference);
}

TEST(yasakova_t_sort_stl, nan_tail) {
  std::vector<double> values{1.0, std::numeric_limits<double>::quiet_NaN(), -2.0, +0.0};
  std::vector<double> expected_head{-2.0, 0.0, 1.0};
  RadixSortDoubleStl(values);
  ASSERT_TRUE(std::equal(values.begin(), values.begin() + 3, expected_head.begin(), expected_head.end()));
  ASSERT_TRUE(std::isnan(values.back()));
}
