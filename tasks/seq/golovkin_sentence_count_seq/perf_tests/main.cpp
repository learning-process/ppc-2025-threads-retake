// Golovkin Maksim
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/golovkin_sentence_count_seq/include/ops_seq.hpp"

TEST(golovkin_sentence_count_seq, perf_test) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(32, 126);

  const int text_size = 1000000;
  std::string text;
  text.reserve(text_size);

  for (int i = 0; i < text_size; ++i) {
    text.push_back(static_cast<char>(dist(gen)));
  }

  int result = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(text.data()));
  task_data->inputs_count.emplace_back(text.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<nesterov_sentence_count_seq::SentenceCountSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    static auto start_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time)
        .count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(golovkin_sentence_count_seq, perf_test) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(32, 126);

  const int text_size = 1000000;
  std::string text;
  text.reserve(text_size);

  for (int i = 0; i < text_size; ++i) {
    text.push_back(static_cast<char>(dist(gen)));
  }

  int result = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(text.data()));
  task_data->inputs_count.emplace_back(text.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<nesterov_sentence_count_seq::SentenceCountSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    static auto start_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time)
        .count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}