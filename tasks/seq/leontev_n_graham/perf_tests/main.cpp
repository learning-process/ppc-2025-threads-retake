#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/leontev_n_graham/include/ops_seq.hpp"

namespace {
std::vector<float> GenVec(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> dist(-5, 5);
  std::vector<float> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
} // namespace

TEST(leontev_n_graham_seq, test_pipeline_run) {
  constexpr int kCount = 500000;

  // Create data
  std::vector<float> in_x = GenVec(kCount);
  std::vector<float> in_y = GenVec(kCount);
  std::vector<float> out_x(kCount, 0.0F);
  std::vector<float> out_y(kCount, 0.0F);
  int out_size = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_seq->inputs_count.emplace_back(in_x.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_seq->outputs_count.emplace_back(out_x.size());

  // Create Task
  auto graham_seq = std::make_shared<leontev_n_graham_seq::GrahamSeq>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(graham_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(leontev_n_graham_seq, test_task_run) {
  constexpr int kCount = 500000;

  // Create data
  std::vector<float> in_x = GenVec(kCount);
  std::vector<float> in_y = GenVec(kCount);
  std::vector<float> out_x(kCount, 0.0F);
  std::vector<float> out_y(kCount, 0.0F);
  int out_size = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_x.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_y.data()));
  task_data_seq->inputs_count.emplace_back(in_x.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_x.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_y.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_size));
  task_data_seq->outputs_count.emplace_back(out_x.size());


  // Create Task
  auto graham_seq = std::make_shared<leontev_n_graham_seq::GrahamSeq>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(graham_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
