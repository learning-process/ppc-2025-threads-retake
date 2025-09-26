#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/muradov_k_histogram_stretch_tbb/include/ops_tbb.hpp"

namespace {
void FillRandom(std::vector<uint8_t>& data) {
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> dist(30, 180);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<uint8_t>(dist(gen));
  }
}

double NowSec(const std::chrono::high_resolution_clock::time_point& t0) {
  auto cur = std::chrono::high_resolution_clock::now();
  auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(cur - t0).count();
  return static_cast<double>(d) * 1e-9;
}

uint64_t CalcIters(double single_time) {
  if (single_time <= 0.0) {
    return 1ULL;
  }
  constexpr double kTarget = 1.2;
  constexpr double kMaxTotal = 8.0;
  auto need = static_cast<uint64_t>(std::ceil(kTarget / single_time));
  if (need == 0) {
    need = 1;
  }
  auto max_allowed = static_cast<uint64_t>(std::floor(kMaxTotal / single_time));
  if (max_allowed == 0) {
    max_allowed = 1;
  }
  need = std::min(need, max_allowed);
  return need;
}
}  // namespace

TEST(muradov_k_histogram_stretch_tbb, test_pipeline_run) {
  size_t n = 100000000;
  std::vector<uint8_t> in(n);
  std::vector<uint8_t> out(n, 0);
  FillRandom(in);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto warm_task = std::make_shared<muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask>(task_data);
  warm_task->Validation();
  auto warm_t0 = std::chrono::high_resolution_clock::now();
  warm_task->PreProcessing();
  warm_task->Run();
  warm_task->PostProcessing();
  double single_time = NowSec(warm_t0);
  auto task = std::make_shared<muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = CalcIters(single_time);
  auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] { return NowSec(start); };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in.size(), out.size());
}

TEST(muradov_k_histogram_stretch_tbb, test_task_run) {
  size_t n = 100000000;
  std::vector<uint8_t> in(n);
  std::vector<uint8_t> out(n, 0);
  FillRandom(in);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  auto warm_task = std::make_shared<muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask>(task_data);
  warm_task->Validation();
  warm_task->PreProcessing();
  auto warm_t0 = std::chrono::high_resolution_clock::now();
  warm_task->Run();
  double single_time = NowSec(warm_t0);
  warm_task->PostProcessing();
  auto task = std::make_shared<muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = CalcIters(single_time);
  auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] { return NowSec(start); };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto analyzer = std::make_shared<ppc::core::Perf>(task);
  analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in.size(), out.size());
}
