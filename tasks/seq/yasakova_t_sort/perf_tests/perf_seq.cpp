#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sort/include/radix_double_seq.hpp"

namespace {

std::vector<double> MakeData(size_t count) {
  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> distribution(-1e9, 1e9);
  std::vector<double> values(count);
  for (auto& value : values) {
    value = distribution(rng);
  }
  return values;
}

std::shared_ptr<ppc::core::PerfAttr> MakePerfAttr(uint64_t runs) {
  auto attributes = std::make_shared<ppc::core::PerfAttr>();
  attributes->num_running = runs;
  const auto start = std::chrono::high_resolution_clock::now();
  attributes->current_timer = [start]() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start).count();
  };
  return attributes;
}

std::shared_ptr<ppc::core::TaskData> MakeTaskData(std::vector<double>& input, std::vector<double>& output) {
  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  data->outputs_count.emplace_back(static_cast<uint32_t>(output.size()));
  return data;
}

}  // namespace

TEST(yasakova_t_sort_seq, test_pipeline_run) {
  auto input = MakeData(300000);
  std::vector<double> output(input.size(), 0.0);

  auto task_data = MakeTaskData(input, output);
  auto task = std::make_shared<yasakova_t_sort_seq::SortTaskSequential>(task_data);
  auto perf_attr = MakePerfAttr(200);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 1; i < output.size(); ++i) {
    ASSERT_LE(output[i - 1], output[i]);
  }
}

TEST(yasakova_t_sort_seq, test_task_run) {
  auto input = MakeData(300000);
  std::vector<double> output(input.size(), 0.0);

  auto task_data = MakeTaskData(input, output);
  auto task = std::make_shared<yasakova_t_sort_seq::SortTaskSequential>(task_data);
  auto perf_attr = MakePerfAttr(200);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 1; i < output.size(); ++i) {
    ASSERT_LE(output[i - 1], output[i]);
  }
}
