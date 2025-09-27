#include <gtest/gtest.h>
#include <omp.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/dudchenko_o_connected_components_omp/include/ops_omp.hpp"

using dudchenko_o_connected_components_omp::ConnectedComponentsOmp;

namespace {
constexpr int kLargeWidth = 1024;
constexpr int kLargeHeight = 1024;

std::vector<uint8_t> GenerateTestImage(int w, int h, double density) {
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 255);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < w * h; ++i) {
    if (dist(rng) < density) {
      img[i] = 0;
    }
  }
  return img;
}

std::vector<uint8_t> GenerateGridImage(int w, int h, int grid_size) {
  std::vector<uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 255);

  for (int y = 0; y < h; y += grid_size) {
    for (int x = 0; x < w; x += grid_size) {
      if (x < w && y < h) {
        img[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] = 0;
      }
    }
  }
  return img;
}

inline double NowSec() {
  using Clock = std::chrono::high_resolution_clock;
  static const auto kT0 = Clock::now();
  return std::chrono::duration<double>(Clock::now() - kT0).count();
}

struct ImageSpec {
  int width;
  int height;
};

std::shared_ptr<ppc::core::PerfAttr> MakePerfAttr(int runs) {
  auto attr = std::make_shared<ppc::core::PerfAttr>();
  attr->num_running = runs;
  attr->current_timer = [] { return NowSec(); };
  return attr;
}

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<uint8_t>& img, const ImageSpec& spec,
                                                  std::vector<int>& labels) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(const_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(static_cast<unsigned int>(img.size()));
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&spec.width)));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&spec.height)));
  td->inputs_count.emplace_back(1);

  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(labels.data()));
  td->outputs_count.emplace_back(static_cast<unsigned int>(labels.size()));

  return td;
}
}  // namespace

TEST(dudchenko_o_connected_components_omp, perf_pipeline_random_image) {
  const auto img = GenerateTestImage(kLargeWidth, kLargeHeight, 0.3);
  std::vector<int> labels(img.size());
  ImageSpec spec{.width = kLargeWidth, .height = kLargeHeight};

  auto td = MakeTaskData(img, spec, labels);
  auto task = std::make_shared<ConnectedComponentsOmp>(td);

  auto perf_attr = MakePerfAttr(5);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const unsigned n = td->outputs_count[0];
  ASSERT_EQ(n, labels.size());

  int max_label = 0;
  for (int label : labels) {
    if (label > max_label) max_label = label;
  }
  EXPECT_GT(max_label, 0);
}

TEST(dudchenko_o_connected_components_omp, perf_taskrun_grid_image) {
  const auto img = GenerateGridImage(kLargeWidth, kLargeHeight, 8);
  std::vector<int> labels(img.size());
  ImageSpec spec{.width = kLargeWidth, .height = kLargeHeight};

  auto td = MakeTaskData(img, spec, labels);
  auto task = std::make_shared<ConnectedComponentsOmp>(td);

  auto perf_attr = MakePerfAttr(5);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  ASSERT_TRUE(task->ValidationImpl());
  ASSERT_TRUE(task->PreProcessingImpl());

  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const unsigned n = td->outputs_count[0];
  ASSERT_EQ(n, labels.size());

  int unique_components = 0;
  std::vector<bool> component_seen(kLargeWidth * kLargeHeight, false);
  for (int label : labels) {
    if (label > 0 && !component_seen[label]) {
      component_seen[label] = true;
      unique_components++;
    }
  }
  EXPECT_GT(unique_components, 100);
}