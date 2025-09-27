#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/strakhov_a_double_radix_merge/include/ops_tbb.hpp"

TEST(strakhov_a_double_radix_merge_tbb, test_pipeline_run) {
  constexpr int kCount = 2000000;

  // Create data
  std::random_device randomizer;
  std::mt19937 gen(randomizer());
  std::vector<double> in(kCount, 0);
  std::vector<double> out(kCount, 0);
  std::uniform_real_distribution<double> dist(-125.0, 125.0);
  for (size_t i = 0; i < kCount; i++) {
    in[i] = dist(randomizer);
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_tbbuential = std::make_shared<strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb>(task_data_tbb);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbbuential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in.size(), out.size());
}

TEST(strakhov_a_double_radix_merge_tbb, test_task_run) {
  constexpr int kCount = 2000000;

  // Create data
  std::random_device randomizer;
  std::mt19937 gen(randomizer());
  std::vector<double> in(kCount, 0);
  std::vector<double> out(kCount, 0);
  std::uniform_real_distribution<double> dist(-125.0, 125.0);
  for (size_t i = 0; i < kCount; i++) {
    in[i] = dist(randomizer);
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_tbbuential = std::make_shared<strakhov_a_double_radix_merge_tbb::DoubleRadixMergeTbb>(task_data_tbb);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbbuential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in.size(), out.size());
}
