
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <limits>
#include "radix_sort_double.hpp"

using namespace bobylev_m_radix_double_omp;

TEST(bobylev_m_radix_double_omp, edge_cases) {
  std::vector<double> v = {+0.0,
                           -0.0,
                           1.0,
                           -1.0,
                           1e-9,
                           -1e-9,
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           123.456,
                           -123.456,
                           0.5,
                           -0.5};
  parallel_radix_sort_double_with_simple_merge(v);
  ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(bobylev_m_radix_double_omp, random_large) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> dist(-1e12, 1e12);
  std::vector<double> v(200000);
  for (auto& x : v) x = dist(rng);
  v.push_back(std::numeric_limits<double>::infinity());
  v.push_back(-std::numeric_limits<double>::infinity());
  v.push_back(0.0);
  v.push_back(-0.0);

  auto ref = v;
  std::sort(ref.begin(), ref.end());

  parallel_radix_sort_double_with_simple_merge(v);
  ASSERT_EQ(v.size(), ref.size());
  ASSERT_TRUE(std::equal(v.begin(), v.end(), ref.begin(), [](double a, double b) {
    if (a == b) return true;
    if (a == 0.0 && b == 0.0) return true;  // +0.0/-0.0
    return false;
  }));
}
