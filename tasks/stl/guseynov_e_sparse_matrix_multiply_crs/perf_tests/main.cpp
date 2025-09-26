#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/guseynov_e_sparse_matrix_multiply_crs/include/ops_stl.hpp"

namespace {
struct MatrixParams {
  int rows;
  int cols;
  double density;
  int seed;
};

guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix GenerateRandomMatrix(const MatrixParams &params) {
  std::mt19937 gen;
  gen.seed(params.seed);
  std::uniform_real_distribution<double> random(-2.0, 2.0);
  std::bernoulli_distribution bernoulli(params.density);

  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix result;
  result.n_rows = params.rows;
  result.n_cols = params.cols;
  result.pointer.assign(result.n_rows + 1, 0);
  std::vector<std::vector<std::pair<int, double>>> temp(result.n_rows);
  for (int i = 0; i < params.rows; i++) {
    for (int j = 0; j < params.cols; j++) {
      if (bernoulli(gen)) {
        double val(random(gen));
        temp[i].emplace_back(j, val);
      }
    }
  }
  for (int i = 0; i < result.n_rows; i++) {
    result.pointer[i + 1] = result.pointer[i];
    for (auto &j : temp[i]) {
      result.col_indexes.push_back(j.first);
      result.non_zero_values.push_back(j.second);
      result.pointer[i + 1]++;
    }
  }

  return result;
}
}  // namespace

TEST(guseynov_e_sparse_matrix_multiply_crs_stl, test_pipeline_run) {
  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix a =
      GenerateRandomMatrix({.rows = 300, .cols = 300, .density = 0.6, .seed = 1993});
  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix b =
      GenerateRandomMatrix({.rows = 300, .cols = 300, .density = 0.6, .seed = 4325});
  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test_task = std::make_shared<guseynov_e_sparse_matrix_multiply_crs_stl::SparseMatMultSTL>(task_data);

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

TEST(guseynov_e_sparse_matrix_multiply_crs_stl, test_task_run) {
  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix a =
      GenerateRandomMatrix({.rows = 300, .cols = 300, .density = 0.6, .seed = 1993});
  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix b =
      GenerateRandomMatrix({.rows = 300, .cols = 300, .density = 0.6, .seed = 4325});
  guseynov_e_sparse_matrix_multiply_crs_stl::CRSMatrix result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test_task = std::make_shared<guseynov_e_sparse_matrix_multiply_crs_stl::SparseMatMultSTL>(task_data);

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