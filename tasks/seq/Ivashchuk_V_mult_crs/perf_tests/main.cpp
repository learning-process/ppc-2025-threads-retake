#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(Ivashchuk_V_sparse_matrix_seq, TestPipelineRun) {
  constexpr int kCount = 100;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in1[(i * kCount) + i] = std::complex<double>(1.0, 0.0);
    in2[(i * kCount) + i] = std::complex<double>(1.0, 0.0);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

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

  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kCount; ++j) {
      if (i == j) {
        EXPECT_NEAR(out[i * kCount + j].real(), 1.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[i * kCount + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      }
    }
  }
}

TEST(Ivashchuk_V_sparse_matrix_seq, TestTaskRun) {
  constexpr int kCount = 100;

  std::vector<std::complex<double>> in1(kCount * kCount, 0);
  std::vector<std::complex<double>> in2(kCount * kCount, 0);
  std::vector<std::complex<double>> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in1[(i * kCount) + i] = std::complex<double>(1.0, 0.0);
    in2[(i * kCount) + i] = std::complex<double>(1.0, 0.0);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<Ivashchuk_V_sparse_matrix_seq::SparseMatrixMultiplier>(task_data_seq);

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

  for (size_t i = 0; i < kCount; ++i) {
    for (size_t j = 0; j < kCount; ++j) {
      if (i == j) {
        EXPECT_NEAR(out[i * kCount + j].real(), 1.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      } else {
        EXPECT_NEAR(out[i * kCount + j].real(), 0.0, 1e-10);
        EXPECT_NEAR(out[i * kCount + j].imag(), 0.0, 1e-10);
      }
    }
  }
}