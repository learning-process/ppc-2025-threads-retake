#include "radix_double_seq.hpp"
#include <gtest/gtest.h>
#include <random>
#include <chrono>

using namespace yasakova_t_sort_seq;

static std::vector<double> make_data(size_t n) {
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> d(-1e9, 1e9);
    std::vector<double> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

TEST(yasakova_t_sort_seq, test_pipeline_run) {
    auto v = make_data(300000);
    radix_sort_double_seq(v);
    for (size_t i = 1; i < v.size(); ++i) ASSERT_LE(v[i-1], v[i]);
}

TEST(yasakova_t_sort_seq, test_task_run) {
    auto v = make_data(300000);
    radix_sort_double_seq(v);
}
