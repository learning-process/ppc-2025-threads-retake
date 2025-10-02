#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/polyakov_a_mult_complex_matrix_CRS/include/ops_tbb.hpp"

namespace pcrs = polyakov_a_mult_complex_matrix_crs_tbb;

pcrs::MatrixCRS pcrs::SequentialMatrixMultiply(pcrs::MatrixCRS& m1, pcrs::MatrixCRS& m2) {
  pcrs::MatrixCRS result(pcrs::Rows{m1.rows}, pcrs::Cols{m2.cols});

  double eps = 1e-9;

  for (size_t r = 0; r < m1.rows; r++) {
    std::vector<std::complex<double>> temp_row(result.cols, 0);

    for (size_t i = m1.row_ptr[r]; i < m1.row_ptr[r + 1]; i++) {
      std::complex<double> a_value = m1.values[i];
      size_t k = m1.col_ind[i];

      for (size_t j = m2.row_ptr[k]; j < m2.row_ptr[k + 1]; j++) {
        std::complex<double> b_value = m2.values[j];
        size_t t = m2.col_ind[j];
        temp_row[t] += a_value * b_value;
      }
    }

    for (size_t i = 0; i < result.cols; i++) {
      if (std::abs(temp_row[i]) > eps) {
        result.values.push_back(temp_row[i]);
        result.col_ind.push_back(i);
      }
    }
    result.row_ptr.push_back(result.values.size());
  }

  return result;
}

TEST(polyakov_a_mult_complex_matrix_crs_tbb, test_pipeline_run) {
  constexpr size_t kN = 3000;

  // Create data
  pcrs::MatrixCRS a = GetRandomMatrixCRS(pcrs::Rows{kN}, pcrs::Cols{kN}, 5);
  pcrs::MatrixCRS b = GetRandomMatrixCRS(pcrs::Rows{kN}, pcrs::Cols{kN}, 5);
  pcrs::MatrixCRS c(pcrs::Rows{kN}, pcrs::Cols{kN});

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&c));

  // Create Task
  auto test_task_tbb = std::make_shared<pcrs::TestTaskTBB>(task_data_tbb);

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
  EXPECT_EQ(pcrs::SequentialMatrixMultiply(a, b), c);
}

TEST(polyakov_a_mult_complex_matrix_crs_tbb, test_task_run) {
  constexpr size_t kN = 3000;

  // Create data
  pcrs::MatrixCRS a = GetRandomMatrixCRS(pcrs::Rows{kN}, pcrs::Cols{kN}, 5);
  pcrs::MatrixCRS b = GetRandomMatrixCRS(pcrs::Rows{kN}, pcrs::Cols{kN}, 5);
  pcrs::MatrixCRS c(pcrs::Rows{kN}, pcrs::Cols{kN});

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(&c));

  // Create Task
  auto test_task_tbb = std::make_shared<pcrs::TestTaskTBB>(task_data_tbb);

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
  EXPECT_EQ(pcrs::SequentialMatrixMultiply(a, b), c);
}