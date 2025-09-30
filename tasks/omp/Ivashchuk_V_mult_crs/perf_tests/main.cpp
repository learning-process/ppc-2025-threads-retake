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

std::vector<std::complex<double>> GenerateRandomSparseMatrix(int rows, int cols, double density) {
  std::vector<std::complex<double>> matrix(rows * cols, {0.0, 0.0});
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> value_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (prob_dist(gen) < density) {
        matrix[(i * cols) + j] = {value_dist(gen), value_dist(gen)};
      }
    }
  }
  return matrix;
}

}  // namespace

TEST(Ivashchuk_V_mult_crs_omp, test_pipeline_run) {
  constexpr int kRows1 = 100;
  constexpr int kCols1 = 100;
  constexpr int kRows2 = 100;
  constexpr int kCols2 = 100;
  constexpr double kDensity = 0.1;  // 10% non-zero elements

  // Create sparse matrices
  auto in1 = GenerateRandomSparseMatrix(kRows1, kCols1, kDensity);
  auto in2 = GenerateRandomSparseMatrix(kRows2, kCols2, kDensity);
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  auto test_task = std::make_shared<Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP>(task_data);

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
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(Ivashchuk_V_mult_crs_omp, test_task_run) {
  constexpr int kRows1 = 150;
  constexpr int kCols1 = 150;
  constexpr int kRows2 = 150;
  constexpr int kCols2 = 150;
  constexpr double kDensity = 0.05;  // 5% non-zero elements

  // Create sparse matrices
  auto in1 = GenerateRandomSparseMatrix(kRows1, kCols1, kDensity);
  auto in2 = GenerateRandomSparseMatrix(kRows2, kCols2, kDensity);
  std::vector<std::complex<double>> out(kRows1 * kCols2, {0.0, 0.0});

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));

  std::vector<int> dims1{kRows1, kCols1};
  std::vector<int> dims2{kRows2, kCols2};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims2.data()));

  task_data->inputs_count.emplace_back(in1.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(in2.size() * sizeof(std::complex<double>));
  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(2);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  // Create Task
  auto test_task = std::make_shared<Ivashchuk_V_mult_crs::SparseMatrixMultiplierOMP>(task_data);

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
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}