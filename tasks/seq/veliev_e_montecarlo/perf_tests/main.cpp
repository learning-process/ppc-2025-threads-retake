#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/veliev_e_montecarlo/include/ops_seq.hpp"

constexpr double ESTIMATE = 1e-1;

TEST(veliev_e_monte_carlo_seq, test_pipeline_run) {
  double res = 8;
    
  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::Flin;

  // Create data
  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

  int N = 8000;

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto task_seq = std::make_shared<veliev_e_monte_carlo_seq::VelievEMonteCarloSeq>(task_data_seq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}

TEST(veliev_e_monte_carlo_seq, test_task_run) {
  double res = 8;

  veliev_e_monte_carlo_seq::func f = veliev_e_monte_carlo_seq::Flin;

  // Create data
  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

  int N = 8000;

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto task_seq = std::make_shared<veliev_e_monte_carlo_seq::VelievEMonteCarloSeq>(task_data_seq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}
