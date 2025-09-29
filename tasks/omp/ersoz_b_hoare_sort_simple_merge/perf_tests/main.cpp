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

using ersoz_b_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP;

TEST(ersoz_b_hoare_sort_simple_merge_omp, test_pipeline_run) {
  const size_t n = 5000000;
  std::vector<int> in(n);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(-1000000, 1000000);
  for (auto& x : in) {
    x = dist(gen);
  }
  std::vector<int> out(n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<HoareSortSimpleMergeOpenMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  const auto wall_t0 = std::chrono::high_resolution_clock::now();
  perf->PipelineRun(perf_attr, perf_results);
  auto wall_elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - wall_t0)
          .count();
  while (wall_elapsed < 1.0) {
    perf->PipelineRun(perf_attr, perf_results);
    wall_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - wall_t0)
            .count();
  }
  perf_results->time_sec = wall_elapsed;
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(out));
  ASSERT_GE(wall_elapsed, 1.0);
}

TEST(ersoz_b_hoare_sort_simple_merge_omp, test_task_run) {
  const size_t n = 5000000;
  std::vector<int> in(n);
  std::mt19937 gen(4242);
  std::uniform_int_distribution<int> dist(-1000000, 1000000);
  for (auto& x : in) {
    x = dist(gen);
  }
  std::vector<int> out(n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<HoareSortSimpleMergeOpenMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  const auto wall_t0 = std::chrono::high_resolution_clock::now();
  perf->TaskRun(perf_attr, perf_results);
  auto wall_elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - wall_t0)
          .count();
  while (wall_elapsed < 1.0) {
    perf->TaskRun(perf_attr, perf_results);
    wall_elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - wall_t0)
            .count();
  }
  perf_results->time_sec = wall_elapsed;
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(out));
  ASSERT_GE(wall_elapsed, 1.0);
}