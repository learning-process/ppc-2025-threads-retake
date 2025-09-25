#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/muradov_k_histogram_stretch/include/ops_omp.hpp"

namespace {
void ExtraWork(const std::vector<int>& v) {
  volatile int acc = 0;
  for (int r = 0; r < 16; ++r) {
    for (int x : v) {
      acc += x & 1;
    }
  }
  (void)acc;
}
}  // namespace

TEST(muradov_k_histogram_stretch_omp, test_pipeline_run) {
  const int k_size = 120000;
  std::vector<int> in(k_size);
  std::vector<int> out(k_size, 0);
  std::mt19937 gen(123);
  std::uniform_int_distribution<int> dist(0, 255);
  for (int& x : in) {
    x = dist(gen);
  }

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  td->inputs_count.emplace_back(in.size());
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<muradov_k_histogram_stretch_omp::HistogramStretchOpenMP>(td);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };

  auto perf_res = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->PipelineRun(perf_attr, perf_res);
  ppc::core::Perf::PrintPerfStatistic(perf_res);

  auto mm = std::ranges::minmax_element(out);
  ASSERT_EQ(*mm.min, 0);
  ASSERT_EQ(*mm.max, 255);
  ExtraWork(out);
}

TEST(muradov_k_histogram_stretch_omp, test_task_run) {
  const int k_size = 120000;
  std::vector<int> in(k_size);
  std::vector<int> out(k_size, 0);
  std::mt19937 gen(321);
  std::uniform_int_distribution<int> dist(0, 255);
  for (int& x : in) {
    x = dist(gen);
  }

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  td->inputs_count.emplace_back(in.size());
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<muradov_k_histogram_stretch_omp::HistogramStretchOpenMP>(td);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };

  auto perf_res = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->TaskRun(perf_attr, perf_res);
  ppc::core::Perf::PrintPerfStatistic(perf_res);

  auto mm = std::ranges::minmax_element(out);
  ASSERT_EQ(*mm.min, 0);
  ASSERT_EQ(*mm.max, 255);
  ExtraWork(out);
}
