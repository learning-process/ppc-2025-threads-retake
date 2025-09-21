#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sdobnov_v_simpson/include/ops_seq.hpp"

TEST(sdobnov_v_simpson_seq, test_pipeline_run) {
  const int dimensions = 2;
  const double lower_bounds[2] = {0.0, 0.0};
  const double upper_bounds[2] = {10.0, 10.0};
  const int n_points[2] = {2000, 2000};

  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(2 * sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson::TestFunction2d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  auto test_task = std::make_shared<sdobnov_v_simpson::SimpsonIntegralSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const double expected_result = 20000.0 / 3.0;
  const double tolerance = 1e-4;
  EXPECT_NEAR(result, expected_result, tolerance);
}

TEST(sdobnov_v_simpson_seq, test_task_run) {
  const int dimensions = 2;
  const double lower_bounds[2] = {0.0, 0.0};
  const double upper_bounds[2] = {10.0, 10.0};
  const int n_points[2] = {2000, 2000};
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&dimensions)));
  task_data->inputs_count.emplace_back(sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(lower_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(upper_bounds)));
  task_data->inputs_count.emplace_back(2 * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(n_points)));
  task_data->inputs_count.emplace_back(2 * sizeof(int));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(sdobnov_v_simpson::TestFunction2d));
  task_data->inputs_count.emplace_back(sizeof(double (*)(std::vector<double>)));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(sizeof(double));

  auto test_task = std::make_shared<sdobnov_v_simpson::SimpsonIntegralSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const double expected_result = 20000.0 / 3.0;
  const double tolerance = 1e-4;
  EXPECT_NEAR(result, expected_result, tolerance);
}