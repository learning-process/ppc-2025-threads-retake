#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/polyakov_a_mult_complex_matrix_CRS/include/ops_seq.hpp"

TEST(polyakov_a_mult_complex_matrix_crs_seq, test_pipeline_run) {
  constexpr size_t kN = 2500;

  // Create data
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS a =
      polyakov_a_mult_complex_matrix_crs_seq::GetRandomMatrixCRS(kN, kN, 5);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS b =
      polyakov_a_mult_complex_matrix_crs_seq::GetRandomMatrixCRS(kN, kN, 5);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS c(kN, kN);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  auto test_task_sequential =
      std::make_shared<polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential>(task_data_seq);

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
}

TEST(polyakov_a_mult_complex_matrix_crs_seq, test_task_run) {
  constexpr size_t kN = 2500;

  // Create data
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS a =
      polyakov_a_mult_complex_matrix_crs_seq::GetRandomMatrixCRS(kN, kN, 5);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS b =
      polyakov_a_mult_complex_matrix_crs_seq::GetRandomMatrixCRS(kN, kN, 5);
  polyakov_a_mult_complex_matrix_crs_seq::MatrixCRS c(kN, kN);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));

  // Create Task
  auto test_task_sequential =
      std::make_shared<polyakov_a_mult_complex_matrix_crs_seq::TestTaskSequential>(task_data_seq);

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
}
