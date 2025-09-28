#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/muradov_k_histogram_stretch/include/ops_tbb.hpp"

namespace {
std::shared_ptr<ppc::core::TaskData> MakeTaskData(std::vector<int>& in, std::vector<int>& out) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  td->inputs_count.emplace_back(in.size());
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());
  return td;
}
}  // namespace

TEST(muradov_k_histogram_stretch_tbb, stretch_small_vector) {
  std::vector<int> in{10, 20, 30, 40, 50};
  std::vector<int> out(in.size(), -1);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch::HistogramStretchTBBTask task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  std::vector<int> expected{0, (10 * 255) / 40, (20 * 255) / 40, (30 * 255) / 40, 255};
  EXPECT_EQ(out, expected);
}

TEST(muradov_k_histogram_stretch_tbb, stretch_constant) {
  std::vector<int> in(100, 77);
  std::vector<int> out(in.size(), -1);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch::HistogramStretchTBBTask task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_TRUE(std::ranges::all_of(out, [](int v) { return v == 0; }));
}

TEST(muradov_k_histogram_stretch_tbb, validation_invalid_range) {
  std::vector<int> in{0, 10, 260};
  std::vector<int> out(in.size(), 0);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch::HistogramStretchTBBTask task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(muradov_k_histogram_stretch_tbb, stretch_full_range_preserve) {
  std::vector<int> in(256, 0);
  for (int i = 0; i < 256; ++i) {
    in[i] = i;
  }
  std::vector<int> out(in.size(), 0);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch::HistogramStretchTBBTask task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_EQ(out, in);
}

TEST(muradov_k_histogram_stretch_tbb, output_min_zero_max_255) {
  std::vector<int> in{50, 60, 200, 180, 55, 190};
  std::vector<int> out(in.size(), 0);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch::HistogramStretchTBBTask task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto mm = std::ranges::minmax_element(out);
  EXPECT_EQ(*mm.min, 0);
  EXPECT_EQ(*mm.max, 255);
}
