#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/muradov_k_histogram_stretch_tbb/include/ops_tbb.hpp"

static void FillRandom(std::vector<uint8_t>& data) {
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> dist(30, 180);
  for (size_t i = 0; i < data.size(); i++) data[i] = static_cast<uint8_t>(dist(gen));
}

TEST(muradov_k_histogram_stretch_tbb, test_pipeline_run) {
  size_t n = 40000000;
  std::vector<uint8_t> in(n);
  std::vector<uint8_t> out(n, 0);
  FillRandom(in);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto cur = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(cur - t0).count();
    return static_cast<double>(d) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in.size(), out.size());
}

TEST(muradov_k_histogram_stretch_tbb, test_task_run) {
  size_t n = 40000000;
  std::vector<uint8_t> in(n);
  std::vector<uint8_t> out(n, 0);
  FillRandom(in);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto cur = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(cur - t0).count();
    return static_cast<double>(d) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in.size(), out.size());
}
