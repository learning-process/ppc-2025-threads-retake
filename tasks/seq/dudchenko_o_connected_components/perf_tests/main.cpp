#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_connected_components/include/ops_seq.hpp"

TEST(dudchenko_o_connected_components_seq, test_pipeline_run) {
  int width = 100;
  int height = 100;
  std::vector<int> image_data(width * height, 1);

  std::vector<int> input_data;
  input_data.push_back(width);
  input_data.push_back(height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  std::vector<int> output_data(width * height);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  auto test_task_sequential = std::make_shared<dudchenko_o_connected_components::TestTaskSequential>(task_data_seq);

  ASSERT_TRUE(test_task_sequential->Validation());
  ASSERT_TRUE(test_task_sequential->PreProcessing());
  ASSERT_TRUE(test_task_sequential->Run());
  ASSERT_TRUE(test_task_sequential->PostProcessing());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [=] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0);
    return static_cast<double>(duration.count()) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_NE(output_data[0], 0);
}

TEST(dudchenko_o_connected_components_seq, test_task_run) {
  int width = 100;
  int height = 100;
  std::vector<int> image_data(width * height);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (size_t i = 0; i < image_data.size(); ++i) {
    image_data[i] = std::rand() % 2;
  }

  std::vector<int> input_data;
  input_data.push_back(width);
  input_data.push_back(height);
  input_data.insert(input_data.end(), image_data.begin(), image_data.end());

  std::vector<int> output_data(width * height);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  auto test_task_sequential = std::make_shared<dudchenko_o_connected_components::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50000;
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

  for (int i = 0; i < width * height; ++i) {
    if (image_data[i] == 0) {
      EXPECT_EQ(output_data[i], 0);
    } else {
      EXPECT_NE(output_data[i], 0);
    }
  }
}
