#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/dudchenko_o_connected_components_tbb/include/ops_tbb.hpp"

using dudchenko_o_connected_components_tbb::ConnectedComponentsTbb;

namespace {
constexpr int kSmallSize = 256;
constexpr int kMediumSize = 512;

std::vector<uint8_t> CreateTestImage(int size) {
  std::vector<uint8_t> img(size * size, 0);
  
  for (int y = 0; y < size; y += size / 8) {
    for (int x = 0; x < size; x += size / 8) {
      int block_size = size / 16;
      for (int dy = 0; dy < block_size; ++dy) {
        for (int dx = 0; dx < block_size; ++dx) {
          if (y + dy < size && x + dx < size) {
            img[(y + dy) * size + (x + dx)] = 1;
          }
        }
      }
    }
  }
  return img;
}

std::shared_ptr<ppc::core::PerfAttr> MakePerfAttr(int runs) {
  auto a = std::make_shared<ppc::core::PerfAttr>();
  a->num_running = runs;
  a->current_timer = [] {
    using Clock = std::chrono::high_resolution_clock;
    static auto t0 = Clock::now();
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  return a;
}
}  // namespace

TEST(dudchenko_o_connected_components_tbb, perf_pipeline_small) {
  auto img = CreateTestImage(kSmallSize);
  std::vector<int> out(img.size());
  int w = kSmallSize, h = kSmallSize;

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(img.size());
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ConnectedComponentsTbb>(td);
  auto perf_attr = MakePerfAttr(5);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(dudchenko_o_connected_components_tbb, perf_task_run_small) {
  auto img = CreateTestImage(kSmallSize);
  std::vector<int> out(img.size());
  int w = kSmallSize, h = kSmallSize;

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(img.size());
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ConnectedComponentsTbb>(td);
  auto perf_attr = MakePerfAttr(5);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  task->ValidationImpl();
  task->PreProcessingImpl();
  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(dudchenko_o_connected_components_tbb, perf_pipeline_medium) {
  auto img = CreateTestImage(kMediumSize);
  std::vector<int> out(img.size());
  int w = kMediumSize, h = kMediumSize;

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(img.size());
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ConnectedComponentsTbb>(td);
  auto perf_attr = MakePerfAttr(3);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(dudchenko_o_connected_components_tbb, perf_task_run_medium) {
  auto img = CreateTestImage(kMediumSize);
  std::vector<int> out(img.size());
  int w = kMediumSize, h = kMediumSize;

  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  td->inputs_count.emplace_back(img.size());
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&w));
  td->inputs_count.emplace_back(1);
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
  td->inputs_count.emplace_back(1);
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ConnectedComponentsTbb>(td);
  auto perf_attr = MakePerfAttr(3);
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  task->ValidationImpl();
  task->PreProcessingImpl();
  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}