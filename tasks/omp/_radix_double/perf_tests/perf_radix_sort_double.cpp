#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "radix_sort_double.hpp"

using namespace bobylev_m_radix_double_omp;

// Имена тестов в perf_tests должны быть ровно эти два:
TEST(bobylev_m_radix_double_omp, test_pipeline_run) {
  std::mt19937_64 rng(42);
  std::normal_distribution<double> dist(0.0, 1e6);
  std::vector<double> v(1'000'000);
  for (auto& x : v) x = dist(rng);
  parallel_radix_sort_double_with_simple_merge(v);
  ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(bobylev_m_radix_double_omp, test_task_run) {
  std::mt19937_64 rng(43);
  std::uniform_real_distribution<double> dist(-1e9, 1e9);
  std::vector<double> v(1'000'000);
  for (auto& x : v) x = dist(rng);
  parallel_radix_sort_double_with_simple_merge(v);
  ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
}
