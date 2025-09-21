#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kavtorev_d_batcher_sort/include/ops_seq.hpp"

using kavtorev_d_batcher_sort_seq::RadixBatcherSortSequential;

TEST(kavtorev_d_batcher_sort_seq, perf_pipeline_run) {
  constexpr size_t kCount = 2000000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> d(-1e6, 1e6);
  for (auto& v : in) {
    v = d(gen);
  }
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<RadixBatcherSortSequential>(task_data_seq);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(kavtorev_d_batcher_sort_seq, perf_task_run) {
  constexpr size_t kCount = 2000000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::mt19937 gen(321);
  std::uniform_real_distribution<double> d(-1e6, 1e6);
  for (auto& v : in) {
    v = d(gen);
  }
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<RadixBatcherSortSequential>(task_data_seq);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
