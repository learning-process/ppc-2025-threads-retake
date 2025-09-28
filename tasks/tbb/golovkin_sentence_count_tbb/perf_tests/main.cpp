// Golovkins
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/golovkin_sentence_count_tbb/include/ops_tbb.hpp"

TEST(golovkin_sentence_count_tbb, test_pipeline_run) {
  const int text_size = 1000000000;
  std::string text;
  text.reserve(text_size);

  for (int i = 0; i < text_size; ++i) {
    if (i % 50 == 0 && i > 0) {
      text.push_back('.');
    } else if (i % 25 == 0 && i > 0) {
      text.push_back('?');
    } else if (i % 33 == 0 && i > 0) {
      text.push_back('!');
    } else {
      text.push_back(static_cast<char>('a' + (i % 26)));
    }
  }

  int result = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(text.data()));
  task_data->inputs_count.emplace_back(text.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<golovkin_sentence_count_tbb::SentenceCountParallel>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(golovkin_sentence_count_tbb, test_task_run) {
  const int text_size = 1000000000;
  std::string text;
  text.reserve(text_size);

  for (int i = 0; i < text_size; ++i) {
    if (i % 50 == 0 && i > 0) {
      text.push_back('.');
    } else if (i % 25 == 0 && i > 0) {
      text.push_back('?');
    } else if (i % 33 == 0 && i > 0) {
      text.push_back('!');
    } else {
      text.push_back(static_cast<char>('a' + (i % 26)));
    }
  }

  int result = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(text.data()));
  task_data->inputs_count.emplace_back(text.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<golovkin_sentence_count_tbb::SentenceCountParallel>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}