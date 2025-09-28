#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/muradov_k_histogram_stretch/include/ops_tbb.hpp"

namespace {
void ExtraCheck(const std::vector<int>& out) {
  volatile int sum = 0;
  for (int r = 0; r < 32; ++r) {
    for (int v : out) {
      sum += v & 1;
    }
  }
  (void)sum;
}
}  // namespace

TEST(muradov_k_histogram_stretch_tbb, test_pipeline_run) {
  const int k_size = 600000;
  std::vector<int> in(k_size);
  std::vector<int> out(k_size, 0);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(30, 200);
  for (int& p : in) {
    p = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<muradov_k_histogram_stretch::HistogramStretchTBBTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto now = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto mm = std::ranges::minmax_element(out);
  ASSERT_EQ(*mm.min, 0);
  ASSERT_EQ(*mm.max, 255);
  ExtraCheck(out);
}

TEST(muradov_k_histogram_stretch_tbb, test_task_run) {
  const int k_size = 600000;
  std::vector<int> in(k_size);
  std::vector<int> out(k_size, 0);
  std::mt19937 gen(777);
  std::uniform_int_distribution<int> dist(0, 255);
  for (int& p : in) {
    p = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<muradov_k_histogram_stretch::HistogramStretchTBBTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto now = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto mm = std::ranges::minmax_element(out);
  ASSERT_EQ(*mm.min, 0);
  ASSERT_EQ(*mm.max, 255);
  ExtraCheck(out);
}
