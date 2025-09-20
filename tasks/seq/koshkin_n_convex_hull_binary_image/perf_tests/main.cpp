#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/koshkin_n_convex_hull_binary_image/include/ops_seq.hpp"

TEST(koshkin_n_convex_hull_binary_image_seq, test_pipeline_run) {
  // Create data
  int sz = 8500;

  std::vector<int> in(sz * sz, 1);
  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> exp = {{0, 0}, {0, sz - 1}, {sz - 1, 0}, {sz - 1, sz - 1}};

  // Create task_data
  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> out(exp.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(sz);
  task_data_seq->inputs_count.emplace_back(sz);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage>(task_data_seq);

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
  ASSERT_EQ(out, exp);
}

TEST(koshkin_n_convex_hull_binary_image_seq, test_task_run) {
  // Create data
  int sz = 8500;

  std::vector<int> in(sz * sz, 1);
  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> exp = {{0, 0}, {0, sz - 1}, {sz - 1, 0}, {sz - 1, sz - 1}};

  // Create task_data
  std::vector<koshkin_n_convex_hull_binary_image_seq::Pt> out(exp.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(sz);
  task_data_seq->inputs_count.emplace_back(sz);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<koshkin_n_convex_hull_binary_image_seq::ConvexHullBinaryImage>(task_data_seq);

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
  ASSERT_EQ(out, exp);
}