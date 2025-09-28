#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/yasakova_t_sort/include/radix_double_omp.hpp"

namespace {

std::vector<double> make_data(size_t n) {
  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> dist(-1e9, 1e9);
  std::vector<double> values(n);
  for (auto& x : values) x = dist(rng);
  return values;
}

std::shared_ptr<ppc::core::PerfAttr> make_perf_attr(uint64_t runs) {
  auto attr = std::make_shared<ppc::core::PerfAttr>();
  attr->num_running = runs;
  const auto t0 = std::chrono::high_resolution_clock::now();
  attr->current_timer = [t0]() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - t0).count();
  };
  return attr;
}

std::shared_ptr<ppc::core::TaskData> make_task_data(std::vector<double>& input, std::vector<double>& output) {
  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  data->outputs_count.emplace_back(static_cast<uint32_t>(output.size()));
  return data;
}

}  // namespace

TEST(yasakova_t_sort_omp, test_pipeline_run) {
  auto input = make_data(300000);
  std::vector<double> output(input.size(), 0.0);

  auto task_data = make_task_data(input, output);
  auto task = std::make_shared<yasakova_t_sort_omp::SortTaskOpenMP>(task_data);
  auto perf_attr = make_perf_attr(200);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 1; i < output.size(); ++i) ASSERT_LE(output[i - 1], output[i]);
}

TEST(yasakova_t_sort_omp, test_task_run) {
  auto input = make_data(300000);
  std::vector<double> output(input.size(), 0.0);

  auto task_data = make_task_data(input, output);
  auto task = std::make_shared<yasakova_t_sort_omp::SortTaskOpenMP>(task_data);
  auto perf_attr = make_perf_attr(200);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 1; i < output.size(); ++i) ASSERT_LE(output[i - 1], output[i]);
}
