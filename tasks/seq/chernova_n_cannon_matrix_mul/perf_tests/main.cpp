#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/chernova_n_cannon_matrix_mul/include/ops_seq.hpp"

namespace {
bool compareMatrices(const std::vector<double> &matrix1, const std::vector<double> &matrix2, double tolerance = 1e-4) {
  if (matrix1.size() != matrix2.size()) {
    return false;
  }

  for (size_t i = 0; i < matrix1.size(); ++i) {
    if (std::fabs(matrix1[i] - matrix2[i]) > tolerance) {
      return false;
    }
  }

  return true;
}
std::vector<double> GetRandomMatrix(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-100.0, 100.0);

  std::vector<double> matrix(n * n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix[(i * n) + j] = dis(gen);
    }
  }

  return matrix;
}
}  // namespace
TEST(chernova_n_cannon_matrix_mul_seq, test_pipeline_run) {
  int n = 1000;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n);
  std::vector<double> out(n * n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_seq->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_seq->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  std::vector<double> res = chernova_n_cannon_matrix_mul_seq::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n);

  auto test_task_sequential = std::make_shared<chernova_n_cannon_matrix_mul_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}

TEST(chernova_n_cannon_matrix_mul_seq, test_task_run) {
  int n = 1000;

  std::vector<double> in_mtrx_a = GetRandomMatrix(n);
  std::vector<double> in_mtrx_b = GetRandomMatrix(n);
  std::vector<double> out(n * n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_a.data()));
  task_data_seq->inputs_count.emplace_back(in_mtrx_a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_mtrx_b.data()));
  task_data_seq->inputs_count.emplace_back(in_mtrx_b.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  std::vector<double> res = chernova_n_cannon_matrix_mul_seq::MultiplyMatrix(in_mtrx_a, in_mtrx_b, n);

  auto test_task_sequential = std::make_shared<chernova_n_cannon_matrix_mul_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(compareMatrices(res, out, 1e-4));
}
