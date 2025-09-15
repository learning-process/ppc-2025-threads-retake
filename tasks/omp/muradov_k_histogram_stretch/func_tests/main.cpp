#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/muradov_k_histogram_stretch/include/ops_omp.hpp"

namespace {
std::shared_ptr<ppc::core::TaskData> MakeTD(std::vector<int>& in, std::vector<int>& out) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  td->inputs_count.emplace_back(in.size());
  td->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.emplace_back(out.size());
  return td;
}
}  // namespace

TEST(muradov_k_histogram_stretch_omp, small_vector) {
  std::vector<int> in{10, 20, 30, 40, 50};
  std::vector<int> out(in.size(), -1);
  auto td = MakeTD(in, out);
  muradov_k_histogram_stretch_omp::HistogramStretchOpenMP task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  std::vector<int> exp{0, (10 * 255) / 40, (20 * 255) / 40, (30 * 255) / 40, 255};
  EXPECT_EQ(out, exp);
}

TEST(muradov_k_histogram_stretch_omp, constant_image) {
  std::vector<int> in(500, 77);
  std::vector<int> out(in.size(), -1);
  auto td = MakeTD(in, out);
  muradov_k_histogram_stretch_omp::HistogramStretchOpenMP task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_TRUE(std::ranges::all_of(out, [](int v) { return v == 0; }));
}

TEST(muradov_k_histogram_stretch_omp, invalid_range) {
  std::vector<int> in{0, 256, 10};
  std::vector<int> out(in.size(), 0);
  auto td = MakeTD(in, out);
  muradov_k_histogram_stretch_omp::HistogramStretchOpenMP task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(muradov_k_histogram_stretch_omp, full_range_identity) {
  std::vector<int> in(256);
  for (int i = 0; i < 256; ++i) {
    in[i] = i;
  }
  std::vector<int> out(in.size(), 0);
  auto td = MakeTD(in, out);
  muradov_k_histogram_stretch_omp::HistogramStretchOpenMP task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_EQ(out, in);
}

TEST(muradov_k_histogram_stretch_omp, boundaries_check) {
  std::vector<int> in{50, 60, 200, 180, 55, 190};
  std::vector<int> out(in.size(), 0);
  auto td = MakeTD(in, out);
  muradov_k_histogram_stretch_omp::HistogramStretchOpenMP task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto mm = std::ranges::minmax_element(out);
  EXPECT_EQ(*mm.min, 0);
  EXPECT_EQ(*mm.max, 255);
}

TEST(muradov_k_histogram_stretch_omp, random_parallel) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> in(10000);
  for (auto& v : in) {
    v = dist(gen);
  }
  std::vector<int> out(in.size(), 0);
  auto td = MakeTD(in, out);
  muradov_k_histogram_stretch_omp::HistogramStretchOpenMP task(td);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  auto mm = std::ranges::minmax_element(out);
  EXPECT_EQ(*mm.min, 0);
  EXPECT_EQ(*mm.max, 255);
}
