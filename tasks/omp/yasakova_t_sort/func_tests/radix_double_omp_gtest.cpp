#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "omp/yasakova_t_sort/include/radix_double_omp.hpp"

using namespace yasakova_t_sort_omp;

TEST(yasakova_t_sort_omp, small_basic) {
  std::vector<double> v{3.0, -1.0, 0.0, -0.0, 2.5, -10.0, 2.5};
  auto ref = v;
  std::sort(ref.begin(), ref.end());
  radix_sort_double_omp(v);
  EXPECT_EQ(v, ref);
}

TEST(yasakova_t_sort_omp, random_100k) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> d(-1e9, 1e9);
  std::vector<double> v(100000);
  for (auto& x : v) x = d(rng);
  auto ref = v;
  std::sort(ref.begin(), ref.end());
  radix_sort_double_omp(v);
  EXPECT_EQ(v, ref);
}

TEST(yasakova_t_sort_omp, nan_tail) {
  std::vector<double> v{1.0, std::numeric_limits<double>::quiet_NaN(), -2.0, +0.0};
  std::vector<double> head{-2.0, 0.0, 1.0};
  radix_sort_double_omp(v);
  ASSERT_TRUE(std::equal(v.begin(), v.begin() + 3, head.begin(), head.end()));
  ASSERT_TRUE(std::isnan(v.back()));
}
