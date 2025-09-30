#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../include/ops_tbb.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

void CreateIdentityMatrices(std::vector<std::complex<double>> &in, size_t count) {
  for (size_t i = 0; i < count; i++) {
    in[(i * count) + i] = std::complex<double>(1.0, 0.0);
    in[(count * count) + (i * count) + i] = std::complex<double>(1.0, 0.0);
  }
}

void VerifyIdentityResult(const std::vector<std::complex<double>> &out, size_t count) {
  for (size_t i = 0; i < count; i++) {
    for (size_t j = 0; j < count; j++) {
      if (i == j) {
        EXPECT_NEAR(out[(i * count) + j].real(), 1.0, 1e-10);
      } else {
        EXPECT_NEAR(out[(i * count) + j].real(), 0.0, 1e-10);
      }
    }
  }
}

std::shared_ptr<ppc::core::PerfAttr> CreatePerfAttr() {
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  return perf_attr;
}

TEST(ivashchuk_v_tbb, test_pipeline_run) {
  constexpr int kCount = 100;
  std::vector<std::complex<double>> in(2 * kCount * kCount, 0.0);
  std::vector<std::complex<double>> out(kCount * kCount, 0.0);

  CreateIdentityMatrices(in, kCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  auto task = std::make_shared<ivashchuk_v_tbb::SparseMatrixComplexCRS>(task_data);
  auto perf_attr = CreatePerfAttr();
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  VerifyIdentityResult(out, kCount);
}

TEST(ivashchuk_v_tbb, test_task_run) {
  constexpr int kCount = 100;
  std::vector<std::complex<double>> in(2 * kCount * kCount, 0.0);
  std::vector<std::complex<double>> out(kCount * kCount, 0.0);

  CreateIdentityMatrices(in, kCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size() * sizeof(std::complex<double>));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(std::complex<double>));

  auto task = std::make_shared<ivashchuk_v_tbb::SparseMatrixComplexCRS>(task_data);
  auto perf_attr = CreatePerfAttr();
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  VerifyIdentityResult(out, kCount);
}