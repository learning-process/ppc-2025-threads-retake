#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/Ivashchuk_V_mult_crs/include/ops_omp.hpp"

namespace {

struct MatrixDimensions {
  int rows1;
  int cols1;
  int rows2;
  int cols2;
};

std::vector<std::complex<double>> GenerateRandomSparseMatrix(int size_rows, int size_cols, double fill_density) {
  std::vector<std::complex<double>> matrix(size_rows * size_cols, {0.0, 0.0});
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> value_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  for (int i = 0; i < size_rows; i++) {
    for (int j = 0; j < size_cols; j++) {
      if (prob_dist(gen) < fill_density) {
        matrix[(i * size_cols) + j] = {value_dist(gen), value_dist(gen)};
      }
    }
  }
  return matrix;
}

void SetupAndRunPerformanceTest(const MatrixDimensions& dims, double density, bool pipeline_run = true) {
  // Create sparse matrices
  auto in1 = GenerateRandomSparseMatrix(dims.rows1, dims.cols1, density);
  auto in2 = GenerateRandomSparseMatrix(dims.rows2, dims.cols2, density);
  std::vector<std::complex<double>> out(dims.rows1 * dims.cols2, {0.0, 0.0});

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{dims.rows1, dims.cols1};
  std::vector<int> dims2{dims.rows2, dims.cols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  auto test_task = std::make_shared<ivashchuk_v_mult_crs::SparseMatrixMultiplierOMP>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

  if (pipeline_run) {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
  } else {
    perf_analyzer->TaskRun(perf_attr, perf_results);
  }

  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

}  // namespace

TEST(Ivashchuk_V_mult_crs_omp, test_pipeline_run) {

  constexpr int kRows1 = 3000;
  constexpr int kCols1 = 3000;
  constexpr int kRows2 = 3000;
  constexpr int kCols2 = 3000;
  constexpr double kDensity = 0.8; 

  MatrixDimensions dims{.rows1 = kRows1, .cols1 = kCols1, .rows2 = kRows2, .cols2 = kCols2};
  SetupAndRunPerformanceTest(dims, kDensity, true);
}

TEST(Ivashchuk_V_mult_crs_omp, test_task_run) {
  constexpr int kRows1 = 3000;
  constexpr int kCols1 = 3000;
  constexpr int kRows2 = 3000;
  constexpr int kCols2 = 3000;
  constexpr double kDensity = 0.8;

  MatrixDimensions dims{.rows1 = kRows1, .cols1 = kCols1, .rows2 = kRows2, .cols2 = kCols2};
  SetupAndRunPerformanceTest(dims, kDensity, false);
}