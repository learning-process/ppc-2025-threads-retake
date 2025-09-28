#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/veliev_e_montecarlo/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

constexpr double kESTIMATE = 1e-1;

TEST(veliev_e_monte_carlo_all, test_pipeline_run) {
  boost::mpi::communicator world;

  double res = 8;

  veliev_func_all::Func f = veliev_func_all::Flin;

  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

  int n = 15000;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
    task_data_all->inputs_count.emplace_back(in1.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
    task_data_all->inputs_count.emplace_back(in2.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_all->inputs_count.emplace_back(1);

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto task_seq = std::make_shared<veliev_e_monte_carlo_all::VelievEMonteCarloAll>(task_data_all);
  task_seq->SetFunc(f);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
  }
}

TEST(veliev_e_monte_carlo_all, test_task_run) {
  boost::mpi::communicator world;

  double res = 8;

  veliev_func_all::Func f = veliev_func_all::Flin;

  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

  int n = 15000;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
    task_data_all->inputs_count.emplace_back(in1.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
    task_data_all->inputs_count.emplace_back(in2.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_all->inputs_count.emplace_back(1);

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto task_seq = std::make_shared<veliev_e_monte_carlo_all::VelievEMonteCarloAll>(task_data_all);
  task_seq->SetFunc(f);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_LT(std::abs(res - out[0]), kESTIMATE);
  }
}