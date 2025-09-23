#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/kalinin_d_simpson_method/include/ops_omp.hpp"

namespace {

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& lower, const std::vector<double>& upper,
                                                  int segments_per_dim, int function_id, double* result_ptr) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* lower_ptr = const_cast<double*>(lower.data());
  auto* upper_ptr = const_cast<double*>(upper.data());
  static thread_local std::deque<std::array<int, 2>> k_params_storage;
  k_params_storage.emplace_back(std::array<int, 2>{segments_per_dim, function_id});
  int* params_ptr = k_params_storage.back().data();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_ptr));
  task_data->inputs_count.emplace_back(lower.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_ptr));
  task_data->inputs_count.emplace_back(upper.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(params_ptr));
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_ptr));
  task_data->outputs_count.emplace_back(1);
  return task_data;
}

}  // namespace

TEST(kalinin_d_simpson_method_omp, perf_pipeline_run_unit_cube_linear_sum) {
  std::vector<double> a{0.0, 0.0, 0.0};
  std::vector<double> b{1.0, 1.0, 1.0};

  int n = 300;
  double result = 0.0;
  auto task_data = MakeTaskData(a, b, n, 1, &result);
  auto task = std::make_shared<kalinin_d_simpson_method_omp::SimpsonNDOpenMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_NEAR(result, 1.5, 1e-6);
}

TEST(kalinin_d_simpson_method_omp, perf_task_run_hyperrectangle_constant) {
  std::vector<double> a{0.0, 0.0, -1.0, 2.0};
  std::vector<double> b{1.0, 2.0, 1.0, 4.0};
  int n = 80;
  double result = 0.0;
  auto task_data = MakeTaskData(a, b, n, 0, &result);
  auto task = std::make_shared<kalinin_d_simpson_method_omp::SimpsonNDOpenMP>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_NEAR(result, 8.0, 1e-6);
}
