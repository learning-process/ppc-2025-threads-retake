#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/ersoz_b_hoare_sort_simple_merge/include/ops_tbb.hpp"

namespace {

bool IsNonDecreasing(const std::vector<int>& v) {
  if (v.empty()) {
    return true;
  }
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i - 1] > v[i]) {
      return false;
    }
  }
  return true;
}

void FillRandom(std::vector<int>& v, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);
  for (auto& x : v) {
    x = dist(gen);
  }
}

}  // namespace

TEST(ersoz_b_hoare_sort_simple_merge_tbb, test_pipeline_run) {
  const std::size_t n = 2'000'000;
  std::vector<int> in(n);
  FillRandom(in, 777);
  std::vector<int> out(n, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ersoz_b_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 6;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto tp = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(IsNonDecreasing(out));
}

TEST(ersoz_b_hoare_sort_simple_merge_tbb, test_task_run) {
  const std::size_t n = 2'000'000;
  std::vector<int> in(n);
  FillRandom(in, 1337);
  std::vector<int> out(n, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<ersoz_b_hoare_sort_simple_merge_tbb::HoareSortSimpleMergeTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 6;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto tp = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);
  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE(IsNonDecreasing(out));
}
