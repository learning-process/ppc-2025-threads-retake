#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/stroganov_m_HorizGaus3x3/include/ops_stl.hpp"

TEST(stroganov_m_HorizGaus3x3_stl, test_pipeline_run) {
  constexpr size_t kWidth = 15000;
  constexpr size_t kHeight = 15000;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};
  double sum = kernel[0] + kernel[1] + kernel[2];

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      input_image[(i * kWidth) + j] = (j % 3 == 0) ? 100.0 : 0.0;
    }
  }

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      if (j == 0) {
        expected[(i * kWidth) + j] =
            (kernel[1] * input_image[(i * kWidth) + j] + kernel[2] * input_image[(i * kWidth) + j + 1]) / sum;
      } else if (j == kWidth - 1) {
        expected[(i * kWidth) + j] =
            (kernel[0] * input_image[(i * kWidth) + j - 1] + kernel[1] * input_image[(i * kWidth) + j]) / sum;
      } else {
        expected[(i * kWidth) + j] =
            (kernel[0] * input_image[(i * kWidth) + j - 1] + kernel[1] * input_image[(i * kWidth) + j] +
             kernel[2] * input_image[(i * kWidth) + j + 1]) /
            sum;
      }
    }
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_image.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  task_data_stl->inputs_count.emplace_back(input_image.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_image.data()));
  task_data_stl->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task_stl = std::make_shared<stroganov_m_horiz_gaus3x3_stl::ImageFilterStl>(task_data_stl);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected[(i * kWidth) + j], 1e-6);
    }
  }
}

TEST(stroganov_m_HorizGaus3x3_stl, test_task_run) {
  constexpr size_t kWidth = 15000;
  constexpr size_t kHeight = 15000;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<double> expected(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};
  double sum = kernel[0] + kernel[1] + kernel[2];

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      input_image[(i * kWidth) + j] = (j % 3 == 0) ? 100.0 : 0.0;
    }
  }

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      if (j == 0) {
        expected[(i * kWidth) + j] =
            (kernel[1] * input_image[(i * kWidth) + j] + kernel[2] * input_image[(i * kWidth) + j + 1]) / sum;
      } else if (j == kWidth - 1) {
        expected[(i * kWidth) + j] =
            (kernel[0] * input_image[(i * kWidth) + j - 1] + kernel[1] * input_image[(i * kWidth) + j]) / sum;
      } else {
        expected[(i * kWidth) + j] =
            (kernel[0] * input_image[(i * kWidth) + j - 1] + kernel[1] * input_image[(i * kWidth) + j] +
             kernel[2] * input_image[(i * kWidth) + j + 1]) /
            sum;
      }
    }
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_image.data()));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  task_data_stl->inputs_count.emplace_back(input_image.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_image.data()));
  task_data_stl->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task_stl = std::make_shared<stroganov_m_horiz_gaus3x3_stl::ImageFilterStl>(task_data_stl);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output_image[(i * kWidth) + j], expected[(i * kWidth) + j], 1e-6);
    }
  }
}
