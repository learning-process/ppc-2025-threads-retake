#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/muradov_k_histogram_stretch/include/ops_seq.hpp"

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

TEST(muradov_k_histogram_stretch_seq, stretch_small_vector) {
  std::vector<int> in{10, 20, 30, 40, 50};
  std::vector<int> out(in.size(), -1);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch_seq::HistogramStretchSequential task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  // Ожидаемое преобразование
  std::vector<int> expected{0, (10 * 255) / 40, (20 * 255) / 40, (30 * 255) / 40, 255};
  EXPECT_EQ(out, expected);
}

TEST(muradov_k_histogram_stretch_seq, stretch_constant) {
  std::vector<int> in(100, 77);
  std::vector<int> out(in.size(), -1);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch_seq::HistogramStretchSequential task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  // Все должны стать 0
  EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](int v) { return v == 0; }));
}

TEST(muradov_k_histogram_stretch_seq, validation_invalid_range) {
  std::vector<int> in{0, 10, 260};
  std::vector<int> out(in.size(), 0);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch_seq::HistogramStretchSequential task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(muradov_k_histogram_stretch_seq, stretch_full_range_preserve) {
  std::vector<int> in(256, 0);
  for (int i = 0; i < 256; ++i) in[i] = i;  // полный диапазон
  std::vector<int> out(in.size(), 0);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch_seq::HistogramStretchSequential task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_EQ(out, in);  // линейное отображение тождественно
}

TEST(muradov_k_histogram_stretch_seq, output_min_zero_max_255) {
  std::vector<int> in{50, 60, 200, 180, 55, 190};
  std::vector<int> out(in.size(), 0);
  auto td = MakeTaskData(in, out);
  muradov_k_histogram_stretch_seq::HistogramStretchSequential task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto mm = std::minmax_element(out.begin(), out.end());
  EXPECT_EQ(*mm.first, 0);
  EXPECT_EQ(*mm.second, 255);
}
