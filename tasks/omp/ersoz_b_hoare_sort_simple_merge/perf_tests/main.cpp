#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/ersoz_b_hoare_sort_simple_merge/include/ops_omp.hpp"

TEST(ersoz_b_hoare_sort_simple_merge_omp, test_pipeline_run) {
  const std::size_t n = 8'000'000;
  std::vector<int> in(n);
  std::vector<int> out(n, 0);
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);
  for (auto& x : in) {
    x = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ersoz_b_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(ersoz_b_hoare_sort_simple_merge_omp, test_task_run) {
  const std::size_t n = 8'000'000;
  std::vector<int> in(n);
  std::vector<int> out(n, 0);
  std::mt19937 gen(456);
  std::uniform_int_distribution<int> dist(-2'000'000, 2'000'000);
  for (auto& x : in) {
    x = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ersoz_b_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(std::ranges::is_sorted(out));
}
