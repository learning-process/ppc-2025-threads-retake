#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/chastov_v_shell_sort_with_even_odd_batcher_merge/include/ops_tbb.hpp"

namespace {
std::vector<int> GenerateRandomArray(int array_size, std::pair<int, int> value_range) {
  if (array_size <= 0) {
    throw std::invalid_argument("Invalid array size");
  }

  std::random_device random_seed;
  std::mt19937 random_engine(random_seed());
  std::uniform_int_distribution<int> value_distribution(value_range.first, value_range.second);

  std::vector<int> random_array;
  random_array.reserve(array_size);
  for (int i = 0; i < array_size; i++) {
    random_array.push_back(value_distribution(random_engine));
  }
  return random_array;
}
}  // namespace

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge_seq, test_pipeline_run) {
  const int max_range_value = 1000;
  const int min_range_value = -1000;
  const int size = 50000;

  bool descending_flag = false;

  std::vector<int> in = GenerateRandomArray(size, {min_range_value, max_range_value});
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected_result = in;
  std::ranges::sort(expected_result);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&descending_flag));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskTBB>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected_result, out);
}

TEST(chastov_v_shell_sort_with_even_odd_batcher_merge_seq, test_task_run) {
  const int max_range_value = 1000;
  const int min_range_value = -1000;
  const int size = 120000;

  bool descending_flag = false;

  // Create data
  std::vector<int> in = GenerateRandomArray(size, {min_range_value, max_range_value});
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected_result = in;
  std::ranges::sort(expected_result);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&descending_flag));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<chastov_v_shell_sort_with_even_odd_batcher_merge::TestTaskTBB>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected_result, out);
}
