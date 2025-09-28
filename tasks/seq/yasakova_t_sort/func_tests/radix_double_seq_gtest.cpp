#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>

#include "radix_double_seq.hpp"

using namespace yasakova_t_sort_seq;

TEST(yasakova_t_sort_seq, small_basic) {
  std::vector<double> v{3.0, -1.0, 0.0, -0.0, 2.5, -10.0, 2.5};
  auto ref = v;
  std::sort(ref.begin(), ref.end());
  radix_sort_double_seq(v);
  EXPECT_EQ(v, ref);
}

TEST(yasakova_t_sort_seq, random_100k) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> d(-1e9, 1e9);
  std::vector<double> v(100000);
  for (auto& x : v) x = d(rng);
  auto ref = v;
  std::sort(ref.begin(), ref.end());
  radix_sort_double_seq(v);
  EXPECT_EQ(v, ref);
}

TEST(yasakova_t_sort_seq, nan_tail) {
  std::vector<double> v{1.0, std::numeric_limits<double>::quiet_NaN(), -2.0, +0.0};
  std::vector<double> head{-2.0, 0.0, 1.0};
  radix_sort_double_seq(v);
  ASSERT_TRUE(std::equal(v.begin(), v.begin() + 3, head.begin(), head.end()));
  ASSERT_TRUE(std::isnan(v.back()));
}
