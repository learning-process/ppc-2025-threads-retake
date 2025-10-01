#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/kavtorev_d_batcher_sort/include/ops_tbb.hpp"

using kavtorev_d_batcher_sort_tbb::RadixBatcherSortTBB;

namespace {

bool IsSorted(const std::vector<double>& arr) {
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i] < arr[i - 1]) {
      return false;
    }
  }
  return true;
}

bool SameElements(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }

  std::vector<double> sorted_a = a;
  std::vector<double> sorted_b = b;
  std::ranges::sort(sorted_a);
  std::ranges::sort(sorted_b);

  for (size_t i = 0; i < sorted_a.size(); ++i) {
    if (std::abs(sorted_a[i] - sorted_b[i]) > 1e-12) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(kavtorev_d_batcher_sort_tbb, perf_pipeline_run) {
  constexpr size_t kCount = 500000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> d(-1e6, 1e6);
  for (auto& v : in) {
    v = d(gen);
  }

  std::vector<double> original = in;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<RadixBatcherSortTBB>(task_data_omp);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
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

  EXPECT_TRUE(IsSorted(out));
  EXPECT_TRUE(SameElements(original, out));

  std::vector<double> std_sorted = original;
  std::ranges::sort(std_sorted);

  bool matches_std_sort = true;
  for (size_t i = 0; i < out.size(); ++i) {
    if (std::abs(out[i] - std_sorted[i]) > 1e-12) {
      matches_std_sort = false;
      break;
    }
  }
  EXPECT_TRUE(matches_std_sort);
}

TEST(kavtorev_d_batcher_sort_tbb, perf_task_run) {
  constexpr size_t kCount = 500000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::mt19937 gen(321);
  std::uniform_real_distribution<double> d(-1e6, 1e6);
  for (auto& v : in) {
    v = d(gen);
  }

  std::vector<double> original = in;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  auto task = std::make_shared<RadixBatcherSortTBB>(task_data_omp);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(IsSorted(out));
  EXPECT_TRUE(SameElements(original, out));

  std::vector<double> std_sorted = original;
  std::ranges::sort(std_sorted);

  bool matches_std_sort = true;
  for (size_t i = 0; i < out.size(); ++i) {
    if (std::abs(out[i] - std_sorted[i]) > 1e-12) {
      matches_std_sort = false;
      break;
    }
  }
  EXPECT_TRUE(matches_std_sort);
}
