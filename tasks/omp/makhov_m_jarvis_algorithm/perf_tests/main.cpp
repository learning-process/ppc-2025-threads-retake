#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/makhov_m_jarvis_algorithm/include/ops_omp.hpp"

TEST(makhov_m_jarvis_algorithm_omp, test_pipeline_run) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_omp::Point> in(3000000);

  for (size_t i = 0; i < in.size(); i++) {
    makhov_m_jarvis_algorithm_omp::Point point = makhov_m_jarvis_algorithm_omp::TaskOmp::GetRandomPoint(
        makhov_m_jarvis_algorithm_omp::XCoord(-10.0), makhov_m_jarvis_algorithm_omp::XCoord(10.0),
        makhov_m_jarvis_algorithm_omp::YCoord(-10.0), makhov_m_jarvis_algorithm_omp::YCoord(10.0));
    in[i] = point;
  }

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_omp::TaskOmp::ConvertPointsToByteArray(in, buffer_size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  auto task_sequential = std::make_shared<makhov_m_jarvis_algorithm_omp::TaskOmp>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_omp::Point> restored_points =
      makhov_m_jarvis_algorithm_omp::TaskOmp::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                       task_data_seq->outputs_count[0]);

  // Should have a convex hull with at least 3 points
  size_t expected_size = 3;
  ASSERT_GE(restored_points.size(), expected_size);

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}

TEST(makhov_m_jarvis_algorithm_omp, test_task_run) {
  // Create data
  std::vector<makhov_m_jarvis_algorithm_omp::Point> in(3000000);

  for (size_t i = 0; i < in.size(); i++) {
    makhov_m_jarvis_algorithm_omp::Point point = makhov_m_jarvis_algorithm_omp::TaskOmp::GetRandomPoint(
        makhov_m_jarvis_algorithm_omp::XCoord(-10.0), makhov_m_jarvis_algorithm_omp::XCoord(10.0),
        makhov_m_jarvis_algorithm_omp::YCoord(-10.0), makhov_m_jarvis_algorithm_omp::YCoord(10.0));
    in[i] = point;
  }

  // Convert points to byte array
  uint32_t buffer_size = 0;
  uint8_t* buffer = makhov_m_jarvis_algorithm_omp::TaskOmp::ConvertPointsToByteArray(in, buffer_size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(buffer);
  task_data_seq->inputs_count.emplace_back(buffer_size);

  // Create Task
  auto task_sequential = std::make_shared<makhov_m_jarvis_algorithm_omp::TaskOmp>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Restore result from task_data
  std::vector<makhov_m_jarvis_algorithm_omp::Point> restored_points =
      makhov_m_jarvis_algorithm_omp::TaskOmp::ConvertByteArrayToPoints(task_data_seq->outputs[0],
                                                                       task_data_seq->outputs_count[0]);

  // Should have a convex hull with at least 3 points
  size_t expected_size = 3;
  ASSERT_GE(restored_points.size(), expected_size);

  // Cleanup
  delete[] task_data_seq->outputs[0];
  delete[] buffer;
}