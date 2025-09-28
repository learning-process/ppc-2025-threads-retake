#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/vasenkov_a_vertical_gauss_3x3/include/ops_tbb.hpp"

namespace {

std::vector<uint8_t> GeneratePerformanceTestImage(int width, int height) {
  std::vector<uint8_t> image(width * height * 3);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;

      if ((x + y) % 2 == 0) {
        image[idx] = 200;
        image[idx + 1] = 100;
        image[idx + 2] = 50;
      } else {
        image[idx] = 50;
        image[idx + 1] = 150;
        image[idx + 2] = 200;
      }

      if (x == y) {
        image[idx] = 255;
        image[idx + 1] = 255;
        image[idx + 2] = 255;
      }

      if (x == width / 2) {
        image[idx] = 0;
        image[idx + 1] = 255;
        image[idx + 2] = 0;
      }
    }
  }
  return image;
}

std::vector<float> GenerateGaussianKernel() {
  return {1.0F / 16, 2.0F / 16, 1.0F / 16, 2.0F / 16, 4.0F / 16, 2.0F / 16, 1.0F / 16, 2.0F / 16, 1.0F / 16};
}

}  // namespace

TEST(vasenkov_a_vertical_gauss_3x3_tbb, test_pipeline_run) {
  constexpr int kWidth = 5000;
  constexpr int kHeight = 5000;

  auto input_image = GeneratePerformanceTestImage(kWidth, kHeight);
  auto output_image = input_image;
  auto kernel = GenerateGaussianKernel();

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(9);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data->outputs_count.emplace_back(output_image.size());

  auto gauss_task = std::make_shared<vasenkov_a_vertical_gauss_3x3_tbb::VerticalGauss>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gauss_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  bool changed = false;
  for (size_t i = 0; i < output_image.size(); ++i) {
    if (input_image[i] != output_image[i]) {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed);
}

TEST(vasenkov_a_vertical_gauss_3x3_tbb, test_task_run) {
  constexpr int kWidth = 5000;
  constexpr int kHeight = 5000;

  auto input_image = GeneratePerformanceTestImage(kWidth, kHeight);
  auto output_image = input_image;
  auto kernel = GenerateGaussianKernel();

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(9);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data->outputs_count.emplace_back(output_image.size());

  auto gauss_task = std::make_shared<vasenkov_a_vertical_gauss_3x3_tbb::VerticalGauss>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gauss_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  bool changed = false;
  for (size_t i = 0; i < output_image.size(); ++i) {
    if (input_image[i] != output_image[i]) {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed);
}