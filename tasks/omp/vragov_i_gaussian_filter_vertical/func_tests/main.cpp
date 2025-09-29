#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/vragov_i_gaussian_filter_vertical/include/filter.hpp"

namespace {
template <typename T>
std::vector<T> MakeVerticalGradient(int x, int y) {
  std::vector<T> img(x * y);
  for (int i = 0; i < x; ++i) {
    for (int j = 0; j < y; ++j) {
      img[(i * y) + j] = j * 10;
    }
  }
  return img;
}
}  // namespace

TEST(vragov_i_gaussian_filter_vertical_omp, IdentityOnConstantImage) {
  constexpr int kX = 5;
  constexpr int kY = 5;
  std::vector<int> in(kX * kY, 42);
  std::vector<int> out(kX * kY, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(kX);
  task_data->inputs_count.emplace_back(kY);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (int v : out) {
    EXPECT_NEAR(v, 27, 2);
  }
}

TEST(vragov_i_gaussian_filter_vertical_omp, VerticalGradientBlur) {
  constexpr int kX = 3;
  constexpr int kY = 7;
  auto in = MakeVerticalGradient<int>(kX, kY);
  std::vector<int> out(kX * kY, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(kX);
  task_data->inputs_count.emplace_back(kY);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (int i = 0; i < kX; ++i) {
    for (int j = 1; j < kY - 1; ++j) {
      int idx = (i * kY) + j;
      int expected = static_cast<int>(std::round((in[idx - 1] * 0.015 + in[idx] * 0.8 + in[idx + 1] * 0.015) /
                                                 (std::sqrt(2.0 * std::acos(-1.0)) * 0.5)));
      EXPECT_NEAR(out[idx], expected, 2);
    }
  }
}

TEST(vragov_i_gaussian_filter_vertical_omp, HandlesEmptyImage) {
  std::vector<int> in;
  std::vector<int> out;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(0);
  task_data->inputs_count.emplace_back(0);
  task_data->inputs_count.emplace_back(0);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(0);

  vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_TRUE(out.empty());
}

TEST(vragov_i_gaussian_filter_vertical_omp, InvalidDimensionsFailValidation) {
  constexpr int kX = 4;
  constexpr int kY = 5;
  std::vector<int> in(10, 1);  // Only 10 elements, but kX*kY = 20
  std::vector<int> out(kX * kY, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(kX);
  task_data->inputs_count.emplace_back(kY);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(vragov_i_gaussian_filter_vertical_omp, SingleElementImage) {
  constexpr int kX = 1;
  constexpr int kY = 1;
  std::vector<int> in = {123};
  std::vector<int> out(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(kX);
  task_data->inputs_count.emplace_back(kY);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  EXPECT_NEAR(out[0], 79, 2);
}

TEST(vragov_i_gaussian_filter_vertical_omp, RandomImageBlurAverage) {
  constexpr int kX = 20;
  constexpr int kY = 20;
  std::vector<int> in(kX * kY);
  std::vector<int> out(kX * kY, 0);

  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& v : in) {
    v = dist(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(kX);
  task_data->inputs_count.emplace_back(kY);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vragov_i_gaussian_filter_vertical_omp::GaussianFilterTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  double avg_in = 0.0;
  double avg_out = 0.0;
  for (int v : in) {
    avg_in += static_cast<double>(v);
  }
  for (int v : out) {
    avg_out += static_cast<double>(v);
  }
  avg_in /= static_cast<double>(in.size());
  avg_out /= static_cast<double>(out.size());

  EXPECT_NEAR(avg_out, avg_in * 0.64, avg_in * 0.05);
}