#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/muradov_k_histogram_stretch_tbb/include/ops_tbb.hpp"

namespace {
std::vector<uint8_t> StretchRef(const std::vector<uint8_t>& in) {
  if (in.empty()) {
    return {};
  }
  auto mm = std::ranges::minmax_element(in);
  int min_v = *mm.min;
  int max_v = *mm.max;
  if (min_v == max_v) {
    return std::vector<uint8_t>(in.size(), 0);
  }
  int range = max_v - min_v;
  std::vector<uint8_t> out(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    int val = in[i];
    int stretched = (val - min_v) * 255 / range;
    stretched = std::clamp(stretched, 0, 255);
    out[i] = static_cast<uint8_t>(stretched);
  }
  return out;
}
}  // namespace

TEST(muradov_k_histogram_stretch_tbb, uniform_image) {
  size_t n = 1024;
  std::vector<uint8_t> in(n, 77);
  std::vector<uint8_t> out(n, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_EQ(out, std::vector<uint8_t>(n, 0));
}

TEST(muradov_k_histogram_stretch_tbb, full_range) {
  std::vector<uint8_t> in(256);
  for (int i = 0; i < 256; i++) {
    in[static_cast<size_t>(i)] = static_cast<uint8_t>(i);
  }
  std::vector<uint8_t> out(256, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_EQ(out, in);
}

TEST(muradov_k_histogram_stretch_tbb, small_manual) {
  std::vector<uint8_t> in = {50, 100, 150, 200};
  std::vector<uint8_t> out(4, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto ref = StretchRef(in);
  EXPECT_EQ(out, ref);
}

TEST(muradov_k_histogram_stretch_tbb, random_image_consistency) {
  size_t n = 50000;
  std::vector<uint8_t> in(n);
  std::mt19937 gen(31337);
  std::uniform_int_distribution<int> dist(10, 180);
  for (size_t i = 0; i < n; i++) {
    in[i] = static_cast<uint8_t>(dist(gen));
  }
  std::vector<uint8_t> out(n, 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());
  muradov_k_histogram_stretch_tbb::HistogramStretchTBBTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto ref = StretchRef(in);
  EXPECT_EQ(out, ref);
}
