#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "include/ops_tbb.hpp"

TEST(ivashchuk_v_tbb, test_pipeline_run) {
  constexpr int kCount = 100;

  // Create data - two 100x100 matrices with complex numbers
  std::vector<std::complex<double>> in(2 * kCount * kCount, 0.0);
  std::vector<std::complex<double>> out(kCount * kCount, 0.0);

  // Create identity matrices for testing
  for (size_t i = 0; i < kCount; i++) {
    in[(i * kCount) + i] = std::complex<double>(1.0, 0.0);                      // First matrix (identity)
    in[(kCount * kCount) + (i * kCount) + i] = std::complex<double>(1.0, 0.0);  // Second matrix (identity)
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  auto test_task_tbb = std::make_shared<ivashchuk_v_tbb::SparseMatrixComplexCRS>(task_data_tbb);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Check result
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      if (i == j) {
        EXPECT_NEAR(out[(i * kCount) + j].real(), 1.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * kCount) + j].real(), 0.0, 1e-10);
      }
    }
  }
}

TEST(ivashchuk_v_tbb, test_task_run) {
  constexpr int kCount = 100;

  // Create data - two 100x100 matrices with complex numbers
  std::vector<std::complex<double>> in(2 * kCount * kCount, 0.0);
  std::vector<std::complex<double>> out(kCount * kCount, 0.0);

  // Create identity matrices for testing
  for (size_t i = 0; i < kCount; i++) {
    in[(i * kCount) + i] = std::complex<double>(1.0, 0.0);                      // First matrix (identity)
    in[(kCount * kCount) + (i * kCount) + i] = std::complex<double>(1.0, 0.0);  // Second matrix (identity)
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  auto test_task_tbb = std::make_shared<ivashchuk_v_tbb::SparseMatrixComplexCRS>(task_data_tbb);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Check result
  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      if (i == j) {
        EXPECT_NEAR(out[(i * kCount) + j].real(), 1.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * kCount) + j].real(), 0.0, 1e-10);
      }
    }
  }
}